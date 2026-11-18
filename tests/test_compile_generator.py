# ===============
# trainer.trainer.compile_generator (sage/training/compile.py): the encoder and the decoder are compiled in
# place, so the model keeps its module tree and state_dict keys (no `_orig_mod.`), autoencoder.encode/decode
# run the compiled code, and the result matches the eager model (fp64, aot_eager on CPU: inductor is checked
# on GPU). A compile error or the recompile limit prints an error and runs the call eagerly. Config: the key
# is off by default, in speedup/legacy.yaml, on in speedup/generator_fast.yaml, off in every recipe; the old
# `trainer.trainer.compile` is a deprecated alias. Checkpoint keys of runs with that old option: test_init_from,
# smoke/test_smoke_compile.py.
# ===============
from pathlib import Path

import pytest
import torch
from torch import nn

import sage.training.compile as compile_mod
from sage.compat import strip_compile_prefix, strip_compile_prefixes
from sage.training.compile import CompiledCall, compile_generator, compile_generator_requested

REPO = Path(__file__).parents[1]
CONFIGS = REPO / "configs"
TINY = ["models.model.swin.embed_dim=32", "models.model.encoder.depths=[2,2,2]", "models.model.decoder.depths=[2,2,2]"]
FP64_RTOL = 1e-12         # Σ|compiled − eager| / Σ|eager|, fp64 (aot_eager runs the same ops: ~1e-17 measured)


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    """Dynamo caches live on the forward code objects, shared by every instance of a class: start clean."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _compose(*overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import train  # noqa: F401  (registers the ${mul:} resolver)
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return OmegaConf.to_container(compose("main", list(overrides)), resolve=True)


def _tiny_sage(seed=0):
    """The pretraining architecture at embedding 32, depths 2-2-2, in fp64, drop path off; and its config."""
    from sage.model.autoencoder import SAGEAutoencoder
    cfg = _compose("+experiment=pretrain", *TINY)
    torch.manual_seed(seed)
    ae = SAGEAutoencoder.from_config(cfg["models"]["model"]).double()
    for m in ae.modules():
        if hasattr(m, "drop_prob"):
            m.drop_prob = 0.0
    return ae, cfg


def _rel(a, b):
    return float((a - b).abs().sum() / b.abs().sum().clamp_min(1e-300))


def _step(ae, x):
    """Training-mode encode -> decode -> loss -> backward; returns the output and every parameter gradient."""
    ae.zero_grad(set_to_none=True)
    y = ae.decode(ae.encode(x, deterministic=True))
    y.pow(2).mean().backward()
    return y.detach().clone(), {n: p.grad.clone() for n, p in ae.named_parameters() if p.grad is not None}


def _eval_pass(ae, x):
    """Eval-mode encode -> decode under no_grad (the validation path)."""
    ae.eval()
    with torch.no_grad():
        out = ae.decode(ae.encode(x, deterministic=True))
    ae.train()
    return out


class _Enc(nn.Module):
    """A user-defined forward: dynamo does not trace the top-level frames of torch.nn's own modules."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class _Dec(_Enc):
    """Its own forward code, as SAGEDecoder vs SAGEEncoder: dynamo counts the graphs per code object."""

    def forward(self, x):
        return 2 * torch.tanh(self.linear(x))


class _TinyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Enc(4, 3)
        self.decoder = _Dec(3, 4)


def _counting_backend(calls):
    """A torch.compile backend that runs the captured graph eagerly and counts its executions."""
    def backend(gm, example_inputs):
        def run(*args):
            calls.append(len(example_inputs))
            return gm.forward(*args)
        return run
    return backend


# ── the tiny SAGE, compiled in place ─────────────────────────────────────────

def test_compiled_in_place_keeps_the_model_and_its_keys():
    ae, _ = _tiny_sage()
    keys, modules, params = list(ae.state_dict()), [n for n, _ in ae.named_modules()], list(ae.parameters())
    assert compile_generator(ae, backend="aot_eager") == ("encoder", "decoder")
    assert isinstance(ae.encoder._compiled_call_impl, CompiledCall)
    assert isinstance(ae.decoder._compiled_call_impl, CompiledCall)
    assert ae.bottleneck._compiled_call_impl is None
    assert list(ae.state_dict()) == keys and not any("_orig_mod" in k for k in keys)
    assert [n for n, _ in ae.named_modules()] == modules
    assert all(a is b for a, b in zip(ae.parameters(), params))


def test_compiled_generator_matches_eager_and_runs_the_compiled_modules():
    """encode/decode reach the compiled encoder and decoder; outputs and gradients equal the eager model's
    (fp64), in training mode with gradients and in eval mode under no_grad (two graphs per module)."""
    ae, cfg = _tiny_sage()
    x = torch.randn(2, int(cfg["data"]["model_channels"]), 1024, 128, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(1))
    ref_y, ref_g = _step(ae, x)
    ref_eval = _eval_pass(ae, x)

    compile_generator(ae, backend="aot_eager")
    calls = {"encoder": 0, "decoder": 0}
    for name in calls:
        wrapper = getattr(ae, name)._compiled_call_impl
        compiled = wrapper.compiled

        def counted(*a, _c=compiled, _n=name, **k):
            calls[_n] += 1
            return _c(*a, **k)
        wrapper.compiled = counted
    y, g = _step(ae, x)
    out_eval = _eval_pass(ae, x)
    assert calls == {"encoder": 2, "decoder": 2}
    from torch._dynamo.utils import counters
    assert counters["stats"]["unique_graphs"] == 4             # (train, grad) and (eval, no_grad) per module
    assert _rel(y, ref_y) < FP64_RTOL and _rel(out_eval, ref_eval) < FP64_RTOL
    assert g.keys() == ref_g.keys() and len(g) == sum(p.requires_grad for p in ae.parameters())
    assert max(_rel(g[k], ref_g[k]) for k in g) < FP64_RTOL


INDUCTOR_RTOL = 1e-4      # Σ|compiled − eager| / Σ|eager|, fp32 without TF32 (inductor reorders float ops)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="inductor on the GPU (CPU inductor is too slow here)")
def test_inductor_on_cuda_matches_eager_in_fp32():
    """The real backend: outputs and gradients of the compiled generator within fp32 rounding of the eager
    ones (a wrong graph, e.g. dynamic=True on this model, is off by O(1))."""
    ae, cfg = _tiny_sage()
    ae = ae.float().cuda()
    x = torch.randn(2, int(cfg["data"]["model_channels"]), 1024, 128, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(1))
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        ref_y, ref_g = _step(ae, x)
        ref_eval = _eval_pass(ae, x)
        compile_generator(ae)
        y, g = _step(ae, x)
        out_eval = _eval_pass(ae, x)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    assert isinstance(ae.encoder._compiled_call_impl, CompiledCall)          # no fallback to eager
    assert _rel(y, ref_y) < INDUCTOR_RTOL and _rel(out_eval, ref_eval) < INDUCTOR_RTOL
    flat = torch.cat([g[k].flatten() for k in sorted(g)])
    ref_flat = torch.cat([ref_g[k].flatten() for k in sorted(ref_g)])
    assert g.keys() == ref_g.keys() and _rel(flat, ref_flat) < INDUCTOR_RTOL


# ── eager fallback ───────────────────────────────────────────────────────────

def _errors(monkeypatch):
    messages = []
    monkeypatch.setattr(compile_mod, "err", lambda msg, prefix="": messages.append(msg))
    return messages


def test_a_compile_error_is_printed_and_the_module_runs_eagerly(monkeypatch):
    messages = _errors(monkeypatch)

    def broken(gm, example_inputs):
        raise RuntimeError("backend exploded")
    ae, x = _TinyAE(), torch.randn(5, 4)
    ref = ae.decoder(ae.encoder(x))
    compile_generator(ae, backend=broken)
    out = ae.decoder(ae.encoder(x))
    assert torch.equal(out, ref)
    assert ae.encoder._compiled_call_impl is None and ae.decoder._compiled_call_impl is None
    assert len(messages) == 2 and all("backend exploded" in m and "eagerly" in m for m in messages)
    ae.decoder(ae.encoder(x))
    assert len(messages) == 2                                  # eager from then on: no new attempt


def test_the_recompile_limit_is_printed_once_and_new_signatures_run_eagerly(monkeypatch):
    messages = _errors(monkeypatch)
    monkeypatch.setattr(compile_mod, "GENERATOR_RECOMPILE_LIMIT", 1)
    graph_runs = []
    ae = _TinyAE()
    compile_generator(ae, backend=_counting_backend(graph_runs))
    a, b, c = torch.randn(2, 4), torch.randn(3, 4), torch.randn(4, 4)
    assert torch.equal(ae.encoder(a), ae.encoder._call_impl(a)) and len(graph_runs) == 1
    for x in (b, c, b):                                        # beyond the limit: eager, one message in all
        assert torch.equal(ae.encoder(x), ae.encoder._call_impl(x))
    assert len(graph_runs) == 1 and len(messages) == 1 and "recompile limit (1 graphs)" in messages[0]
    ae.encoder(a)                                              # the graph of the first signature is kept
    assert len(graph_runs) == 2
    assert ae.decoder(torch.randn(2, 3)).shape == (2, 4) and len(graph_runs) == 3 and len(messages) == 1  # own limit


def test_compile_refused_leaves_the_module_eager(monkeypatch):
    messages = _errors(monkeypatch)

    def refuse(self, *args, **kwargs):
        raise RuntimeError("dynamo is not supported here")
    monkeypatch.setattr(nn.Module, "compile", refuse)
    ae = _TinyAE()
    assert compile_generator(ae, backend="aot_eager") == ()
    assert ae.encoder._compiled_call_impl is None and len(messages) == 2


# ── checkpoint keys of runs with the old trainer.trainer.compile ────────────

def test_strip_compile_prefix():
    assert strip_compile_prefix("engine.autoencoder._orig_mod.encoder.w") == "engine.autoencoder.encoder.w"
    assert strip_compile_prefix("_orig_mod.decoder._orig_mod.b") == "decoder.b"
    assert strip_compile_prefix("engine.autoencoder.encoder.w") == "engine.autoencoder.encoder.w"
    assert strip_compile_prefix("x._orig_modx.y") == "x._orig_modx.y"                 # whole components only
    sd = {"a._orig_mod.w": 1, "b": 2, "_orig_mod.c": 3}
    assert strip_compile_prefixes(sd) == 2 and list(sd.items()) == [("a.w", 1), ("b", 2), ("c", 3)]
    assert strip_compile_prefixes(sd) == 0
    with pytest.raises(ValueError, match="with and without"):
        strip_compile_prefixes({"_orig_mod.w": 1, "w": 2})


def test_on_load_checkpoint_strips_the_prefix_before_the_strict_load():
    """A resume (`ckpt_path=`): SAGELightningModule.on_load_checkpoint renames engine.autoencoder._orig_mod.*
    and stashes ema_autoencoder._orig_mod.* as plain EMA keys; the strict load then matches every key."""
    from sage.training.callbacks import EMACallback
    from sage.training.lightning_module import SAGELightningModule
    ae, cfg = _tiny_sage()
    ds = cfg["data"]["train_dataset"]
    stft = {k: ds[k] for k in ("sample_rate", "n_fft", "hop_length", "win_length", "center", "normalized")}
    wrapper = SAGELightningModule(autoencoder=ae.float(), loss_config={"bottleneck": {"weights": {"kl": 0.0}}},
                                  stft_params=stft, audio_channels=int(cfg["data"]["audio_channels"]))
    sd = wrapper.state_dict()
    ae_keys = [k for k in sd if k.startswith("engine.autoencoder.")]
    old = {k.replace("engine.autoencoder.", "engine.autoencoder._orig_mod.", 1): v for k, v in sd.items()}
    old.update({f"ema_autoencoder._orig_mod.{k[len('engine.autoencoder.'):]}": sd[k] + 1 for k in ae_keys
                if sd[k].is_floating_point()})
    checkpoint = {"state_dict": old, "gen_step": 7}
    wrapper.on_load_checkpoint(checkpoint)
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    ema = EMACallback()
    ema.on_load_checkpoint(None, wrapper, checkpoint)
    assert not any("_orig_mod" in k for k in ema.ema_state_dict)
    assert set(ema.ema_state_dict) == {k[len("engine.autoencoder."):] for k in ae_keys if sd[k].is_floating_point()}


def test_ema_load_strips_the_prefix_itself():
    from sage.training.callbacks import EMACallback
    ema = EMACallback()
    ema.on_load_checkpoint(None, None, {"_ema_state": {"_orig_mod.encoder.w": torch.ones(1)}})
    assert list(ema.ema_state_dict) == ["encoder.w"]


# ── config ───────────────────────────────────────────────────────────────────

RECIPES = ("pretrain", "decoder_ft", "stereo_proxy", "stereo_proxy_a1", "stereo_proxy_a3", "stereo_proxy_a3_fm",
           "stereo_proxy_ft")


def _trainer(*overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        return OmegaConf.to_container(compose("main", list(overrides)), resolve=False)


def test_the_key_is_off_by_default_and_in_legacy():
    from omegaconf import OmegaConf
    assert OmegaConf.load(CONFIGS / "trainer.yaml").trainer.compile_generator is False
    assert OmegaConf.load(CONFIGS / "speedup" / "legacy.yaml").trainer.trainer.compile_generator is False
    assert "compile_generator" not in OmegaConf.load(CONFIGS / "speedup" / "recipe.yaml").trainer.trainer


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipes_keep_the_generator_eager(recipe):
    """Not in speedup/recipe.yaml until the A/B passes; the old `compile` key is gone from the recipes."""
    t = _trainer(f"+experiment={recipe}")["trainer"]["trainer"]
    assert t["compile_generator"] is False and "compile" not in t


@pytest.mark.parametrize("recipe", ["pretrain", "decoder_ft", "stereo_proxy_a3_fm"])
def test_generator_fast_changes_only_compile_generator(recipe):
    ref = _trainer(f"+experiment={recipe}", "+speedup=legacy")
    fast = _trainer(f"+experiment={recipe}", "+speedup=[legacy,generator_fast]")
    assert fast["trainer"]["trainer"]["compile_generator"] is True
    fast["trainer"]["trainer"]["compile_generator"] = False
    assert fast == ref
    on_recipe = _trainer(f"+experiment={recipe}", "+speedup=generator_fast")
    assert on_recipe["trainer"]["trainer"]["compile_generator"] is True


def test_the_old_compile_key_is_a_deprecated_alias(monkeypatch):
    warnings = []
    monkeypatch.setattr(compile_mod, "warn", lambda msg, prefix="": warnings.append(msg))
    assert compile_generator_requested({}) is False
    assert compile_generator_requested({"compile_generator": False, "compile": False}) is False
    assert compile_generator_requested({"compile_generator": True}) is True and not warnings
    assert compile_generator_requested({"compile": True}) is True
    assert len(warnings) == 1 and "deprecated" in warnings[0] and "compile_generator" in warnings[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
