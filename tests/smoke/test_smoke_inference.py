# ===============
# Smoke tests of inference: the paper checkpoint rebuilds from its inference_config, loads the
# EMA weights, and reproduces the golden references bit for bit (SAGE_GOLDEN_REF, default
# golden_local): the 128-frame training crop and the 10 s MoisesDB mix always, the 30 s mix
# with `-m golden_full`. All three inputs x {tri2, off} are the 18 arrays of the regression:
#   pytest tests/smoke/test_smoke_inference.py -m "smoke or golden_full" -k golden
# ===============
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import GOLDEN_DIR, GOLDEN_REF, PAPER_CKPT, SR

pytestmark = pytest.mark.smoke
needs_ckpt = pytest.mark.skipif(not PAPER_CKPT.is_file(), reason=f"paper checkpoint not found: {PAPER_CKPT}")
needs_golden = pytest.mark.skipif(not GOLDEN_REF.is_dir(), reason=f"golden references not found: {GOLDEN_REF}")
GOLDEN_INPUTS = ["crop_train_len_128frames", "moises10s_mix",
                 pytest.param("moises30s_mix", marks=pytest.mark.golden_full)]


def _pad_for_swin(wav: torch.Tensor, hop: int, n_down: int) -> tuple[torch.Tensor, int]:
    """The padding of the paper evaluators (frozen in make_golden.py; sage.inference.pad_for_swin)."""
    orig = wav.shape[-1]
    mult = 2 ** n_down
    frames = orig // hop + 1
    target = max(orig, (frames + (mult - frames % mult) % mult - 1) * hop)
    return (F.pad(wav, (0, target - orig)) if target > orig else wav), orig


@pytest.fixture(scope="module")
def single_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)                      # golden_local was generated with 1 thread
    yield
    torch.set_num_threads(n)


def _load(varlen: str, sdpa: bool = False):
    from sage import SAGE
    return SAGE.from_checkpoint(PAPER_CKPT, device="cpu", varlen=varlen, sdpa=sdpa)   # original attention: bit-exact


@needs_ckpt
def test_checkpoint_loads_every_weight_by_name():
    import warnings
    with warnings.catch_warnings(record=True) as caught:        # missing/unexpected keys are warnings
        warnings.simplefilter("always")
        codec = _load("tri2")
    key_warnings = [str(w.message) for w in caught if "keys" in str(w.message)]
    assert not key_warnings, key_warnings
    if GOLDEN_REF.is_dir():
        ref = json.loads((GOLDEN_REF / "state_dict_keys.json").read_text())
        cur = {k: list(v.shape) for k, v in codec.autoencoder.state_dict().items()}
        assert cur == ref


@needs_ckpt
def test_encode_decode_shapes():
    codec = _load("tri2")
    x = 0.1 * torch.randn(1, 2, 65024)                          # 128 STFT frames, the training grid
    z = codec.encode(x, deterministic=True)
    y = codec.decode(z, target_length=x.shape[-1])
    assert tuple(z.shape) == (1, 16, 128) and tuple(y.shape) == (1, 2, 65024)
    assert torch.isfinite(y).all()


@needs_ckpt
@needs_golden
@pytest.mark.parametrize("name", GOLDEN_INPUTS)
@pytest.mark.parametrize("mode", ["tri2", "off"])
def test_bit_exact_against_golden(name, mode, single_thread):
    codec = _load(mode)
    hop = codec.autoencoder._stft_config.hop_length
    n_down = len(codec.autoencoder.encoder.depths) - 1
    wav = torch.from_numpy(np.load(GOLDEN_DIR / "inputs" / f"{name}.npy"))
    wav_p, orig = _pad_for_swin(wav, hop, n_down)
    batch = wav_p.unsqueeze(0)

    mu = codec.encode(batch, deterministic=True)
    rec = codec.decode(mu, target_length=wav_p.shape[-1])[0, ..., :orig]
    cases = {"latent_mu": mu, "recon_mu": rec}
    if mode == "tri2":                                          # sampled path, as the paper evaluator
        torch.manual_seed(0)
        z = codec.encode(batch, deterministic=False)
        cases |= {"latent_z": z, "recon_z": codec.decode(z, target_length=wav_p.shape[-1])[0, ..., :orig]}
    for case, t in cases.items():
        ref = np.load(GOLDEN_REF / f"{name}__{mode}__{case}.npy")
        assert np.array_equal(t.numpy(), ref), f"{name}/{mode}/{case}: max|Δ|={np.abs(t.numpy() - ref).max():.3e}"


@needs_ckpt
@needs_golden
@pytest.mark.parametrize("name", ["crop_train_len_128frames", "moises10s_mix"])
def test_reconstruct_matches_the_evaluator_path(name, single_thread):
    """SAGE.reconstruct (pad -> encode -> decode -> trim) is bit-identical to the golden evaluator path."""
    codec = _load("tri2")
    wav = torch.from_numpy(np.load(GOLDEN_DIR / "inputs" / f"{name}.npy"))
    rec_mu = codec.reconstruct(wav, deterministic=True)
    torch.manual_seed(0)
    rec_z = codec.reconstruct(wav)                                  # sampled z, as the paper metrics
    for case, t in {"recon_mu": rec_mu, "recon_z": rec_z}.items():
        ref = np.load(GOLDEN_REF / f"{name}__tri2__{case}.npy")
        assert t.shape == wav.shape and np.array_equal(t.numpy(), ref), f"{name}/{case}"


@needs_ckpt
@needs_golden
@pytest.mark.parametrize("name", ["crop_train_len_128frames", "moises10s_mix"])
def test_default_sdpa_matches_the_golden_up_to_rounding(name, single_thread):
    """The default load (SDPA attention) reproduces the golden latents and reconstructions up to rounding."""
    from sage import SAGE
    codec = SAGE.from_checkpoint(PAPER_CKPT, device="cpu")
    assert codec.sdpa and codec.sdpa_blocks > 0
    wav = torch.from_numpy(np.load(GOLDEN_DIR / "inputs" / f"{name}.npy"))
    mu = codec.encode(codec.pad(wav.unsqueeze(0))[0], deterministic=True)
    cases = {"latent_mu": mu, "recon_mu": codec.reconstruct(wav, deterministic=True)}
    for case, t in cases.items():
        ref = torch.from_numpy(np.load(GOLDEN_REF / f"{name}__tri2__{case}.npy"))
        snr = 10 * torch.log10(ref.pow(2).sum() / (t - ref).pow(2).sum().clamp_min(1e-30))
        assert t.shape == ref.shape and snr > 80, f"{name}/{case}: SNR {snr:.1f} dB"


# The padding of the three paper evaluators, verbatim from commit 4400ee9 (only renamed):
# evaluation/evaluate_swin_10s.py (clip sets), evaluate_swin_varT.py (FMA, variable length),
# maeb/swin_encoder.py (probing). SAGE.pad must match all of them for any input length.
def _pad_eval_10s(wav, hop_length, num_downsamples=2):
    orig = wav.shape[-1]
    mult = 2 ** num_downsamples
    W    = (orig // hop_length) + 1
    pad  = (mult - W % mult) % mult
    target = max(orig, (W + pad - 1) * hop_length)
    if target > orig:
        wav = F.pad(wav, (0, target - orig))
    return wav, orig


def _pad_eval_fma(wav, hop_length, num_downsamples=2):
    orig_samples = wav.shape[-1]
    w_multiple   = 2 ** num_downsamples
    W_current    = (orig_samples // hop_length) + 1
    pad_w        = (w_multiple - W_current % w_multiple) % w_multiple
    target_samples = max(orig_samples, (W_current + pad_w - 1) * hop_length)
    if target_samples > orig_samples:
        wav = F.pad(wav, (0, target_samples - orig_samples))
    return wav, orig_samples


def _pad_eval_maeb(wav, hop, num_downsamples):
    orig = wav.shape[-1]
    w_mul = 2 ** num_downsamples
    W = (orig // hop) + 1
    pad_w = (w_mul - W % w_mul) % w_mul
    target = max(orig, (W + pad_w - 1) * hop)
    if target > orig:
        wav = F.pad(wav, (0, target - orig))
    return wav, orig


def _lengths():
    hop, grid = 512, 512 * 4
    edges = [1, 2, hop - 1, hop, hop + 1, grid - 1, grid, grid + 1, 3 * grid - hop, 3 * grid - hop + 1,
             44100, 44100 + 7, 65024, 441000, 1323000, 30 * 44100 + 1]
    rng = np.random.default_rng(0)
    return edges + [int(n) for n in rng.integers(1, 60 * 44100, size=200)]


def test_pad_matches_the_paper_evaluators_for_any_length(tiny_pretrain_ckpt):
    from sage import SAGE
    codec = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu")
    hop, n_down = codec.autoencoder._stft_config.hop_length, len(codec.autoencoder.encoder.depths) - 1
    assert (hop, n_down) == (512, 2)                               # the released model's STFT hop and merges
    for n in _lengths():
        x = torch.zeros(1, 2, n)
        got, orig = codec.pad(x)
        frames = got.shape[-1] // hop + 1
        assert orig == n and got.shape[-1] >= n and frames % 4 == 0, n
        assert got.shape[-1] - n < 4 * hop, n                      # never more than one grid step of padding
        for ref_fn in (_pad_eval_10s, _pad_eval_fma, _pad_eval_maeb):
            ref, ref_orig = ref_fn(x, hop, n_down)
            assert ref_orig == orig and ref.shape == got.shape, (ref_fn.__name__, n)
        x = torch.randn(1, 2, min(n, 5000), generator=torch.Generator().manual_seed(n))
        assert torch.equal(codec.pad(x)[0], _pad_eval_fma(x, hop, n_down)[0])   # same values, zeros appended


@pytest.mark.parametrize("sdpa", [False, True])
@pytest.mark.parametrize("n", [1, 100, 511, 512, 513, 2047, 2049, 44100 + 7, 3 * 44100 + 123])
def test_reconstruct_and_encode_any_length(tiny_pretrain_ckpt, n, sdpa):
    """Any input length encodes (after pad) and reconstructs to the same length, batched or not."""
    from sage import SAGE
    codec = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", sdpa=sdpa)
    wav = 0.1 * torch.randn(2, n, generator=torch.Generator().manual_seed(n))
    padded, orig = codec.pad(wav.unsqueeze(0))
    z = codec.encode(padded, deterministic=True)
    assert z.shape[-1] == padded.shape[-1] // 512 + 1             # one latent frame per STFT frame
    rec = codec.reconstruct(wav, deterministic=True)
    assert rec.shape == wav.shape and torch.isfinite(rec).all()
    assert torch.equal(codec.reconstruct(wav.unsqueeze(0), deterministic=True)[0], rec)


@pytest.mark.parametrize("sdpa", [False, True])
def test_readme_usage_pad_encode_decode_equals_reconstruct(tiny_pretrain_ckpt, sdpa):
    """The README example: pad -> encode -> decode by hand is reconstruct(deterministic=True)."""
    from sage import SAGE
    codec = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", sdpa=sdpa)
    wav = 0.1 * torch.randn(1, 2, 3 * codec.sample_rate + 123, generator=torch.Generator().manual_seed(0))
    padded, n = codec.pad(wav)
    z = codec.encode(padded, deterministic=True)
    y = codec.decode(z, target_length=padded.shape[-1])[..., :n]
    assert z.shape[1] == 16 and padded.shape[-1] >= wav.shape[-1] and n == wav.shape[-1]
    assert torch.equal(y, codec.reconstruct(wav, deterministic=True))


def test_sdpa_is_the_default_and_matches_the_original_attention(tiny_pretrain_ckpt):
    """SAGE.from_checkpoint routes every real attention module through SDPA by default (sdpa=False opts out),
    with the exact reformulations that come with it (fused Q/V bias, shared-phase QKV, patch layers as one
    GEMM), and the same output as the original attention up to rounding."""
    from sage import SAGE
    from sage.model.swin.attention import WindowAttention
    from sage.model.swin.block import SwinTransformerBlock
    default = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", sdpa=False)
    fast = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu")
    assert (default.sdpa, default.sdpa_blocks) == (False, 0) and fast.sdpa and fast.sdpa_blocks > 0
    assert (default.patch_linear_layers, fast.patch_linear_layers) == (0, 2)      # PatchEmbed + PatchUnembed
    for codec, on in ((default, False), (fast, True)):
        modules = list(codec.autoencoder.modules())
        assert all(m.fused_bias is on for m in modules if isinstance(m, WindowAttention))
        assert all(m.shared_phases is on for m in modules if isinstance(m, SwinTransformerBlock))
    wav = 0.1 * torch.randn(2, 3 * 44100 + 123, generator=torch.Generator().manual_seed(0))
    a, b = default.reconstruct(wav, deterministic=True), fast.reconstruct(wav, deterministic=True)
    snr = 10 * torch.log10(a.pow(2).sum() / (a - b).pow(2).sum().clamp_min(1e-30))
    assert a.shape == b.shape and snr > 80, snr


def test_precision_default_and_options(tiny_pretrain_ckpt):
    """TF32 by default; bf16 autocasts encoder and decoder only (FP32 weights, latents and output), close to
    FP32; on CPU the TF32 switches change nothing, so tf32, fp32 and None give the same output."""
    from sage import SAGE
    codecs = {p: SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", precision=p) for p in ("tf32", "fp32", "bf16", None)}
    assert SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu").precision == "tf32"
    wav = 0.1 * torch.randn(2, 3 * 44100 + 123, generator=torch.Generator().manual_seed(0))
    out = {p: c.reconstruct(wav, deterministic=True) for p, c in codecs.items()}
    assert torch.equal(out["tf32"], out["fp32"]) and torch.equal(out["tf32"], out[None])
    bf16 = codecs["bf16"]
    assert all(p.dtype == torch.float32 for p in bf16.autoencoder.parameters())
    assert bf16.encode(bf16.pad(wav.unsqueeze(0))[0], deterministic=True).dtype == torch.float32
    assert out["bf16"].dtype == torch.float32 and not torch.equal(out["bf16"], out["fp32"])
    assert _snr(out["fp32"], out["bf16"]) > 20
    assert set(bf16.autoencoder.state_dict()) == set(codecs["fp32"].autoencoder.state_dict())
    bf16.set_precision("fp32")                                   # switchable after loading
    assert torch.equal(bf16.reconstruct(wav, deterministic=True), out["fp32"])
    with pytest.raises(ValueError):
        SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", precision="fp16")


@pytest.fixture
def aot_eager(monkeypatch):
    """torch.compile on the CPU of the tests: the aot_eager backend, the mode dropped (inductor needs a C++ build,
    CUDA graphs a GPU; the CUDA runs are in tests/test_flex_attention.py and the GPU benchmark)."""
    real = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn=None, *, mode=None, **kw: real(fn, backend="aot_eager", **kw))
    yield
    torch._dynamo.reset()


def test_compile_option(tiny_pretrain_ckpt, aot_eager):
    """compile=: eager by default, True means "default", anything else outside the modes is refused; encoder and
    decoder compiled in place after the SDPA bias cache (filled eagerly in FP32): same weights and state dict, the
    eager output, metadata kept for decode without tensors, no FlexAttention on CPU nor at head size 4; the
    evaluation adapter records the mode in its provenance."""
    from evaluation.codecs import SAGEAdapter
    from sage import SAGE
    from sage.compile import is_compiled
    from sage.model.swin.attention import WindowAttention
    eager = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu")
    assert eager.compile is None and not is_compiled(eager.autoencoder.encoder) and "compile" not in str(
        eager.speedup_options())
    with pytest.raises(ValueError):
        SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", compile="max-autotune")
    wav = 0.1 * torch.randn(2, 3 * 44100 + 123, generator=torch.Generator().manual_seed(0))
    ref = eager.reconstruct(wav, deterministic=True)
    codec = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", compile=True)
    ae = codec.autoencoder
    assert codec.compile == "default" and "compile=default" in repr(codec)
    assert is_compiled(ae.encoder) and is_compiled(ae.decoder)
    attns = [m for m in ae.modules() if isinstance(m, WindowAttention) and not m.is_complex]
    assert attns and all(m._sdpa_cpb.dtype == torch.float32 for m in attns)    # filled before the compile
    assert not any(m.flex for m in attns)                                     # head size 4: no fused kernel
    assert codec.speedup_options()[-1] == "compile:default" and "flex" not in codec.speedup_options()
    state, ref_state = ae.state_dict(), eager.autoencoder.state_dict()
    assert state.keys() == ref_state.keys() and all(torch.equal(state[k], v) for k, v in ref_state.items())
    out = codec.reconstruct(wav, deterministic=True)
    assert out.shape == ref.shape and _snr(ref, out) > 80, _snr(ref, out)
    assert ae._last_encoder_info and not any(isinstance(v, torch.Tensor) for v in ae._last_encoder_info.values())
    prov = {c: SAGEAdapter(device="cpu", checkpoint=str(tiny_pretrain_ckpt), compile=c).provenance()
            for c in (None, "reduce-overhead")}
    assert "compile" not in prov[None] and prov["reduce-overhead"]["compile"] == "reduce-overhead"
    assert prov[None]["sdpa_options"] == prov["reduce-overhead"]["sdpa_options"]
    assert "flex" not in prov[None]["sdpa_options"]


def test_compile_traces_the_precision_and_follows_set_precision(tiny_pretrain_ckpt, aot_eager):
    """The bf16 autocast wrapper is installed before the compile, so the compiled call runs it; set_precision
    afterwards takes effect too (the next call compiles again)."""
    from sage import SAGE
    wav = 0.1 * torch.randn(1, 2, 2 * 44100, generator=torch.Generator().manual_seed(1))
    ref = {p: SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", precision=p).reconstruct(wav, deterministic=True)
           for p in ("bf16", "fp32")}
    codec = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", precision="bf16", compile="default")
    out = codec.reconstruct(wav, deterministic=True)
    assert _snr(ref["bf16"], out) > _snr(ref["bf16"], ref["fp32"]) + 20, (_snr(ref["bf16"], out),
                                                                         _snr(ref["bf16"], ref["fp32"]))
    codec.set_precision("fp32")
    assert _snr(ref["fp32"], codec.reconstruct(wav, deterministic=True)) > 80


def test_compiled_cuda_graph_mode_keeps_the_sampled_noise_of_chunked_inference(tiny_pretrain_ckpt, aot_eager):
    """reduce-overhead (on CPU: the copy-out wrapper, no CUDA graphs): the copied outputs keep the encoder's
    strides, so chunked sampling with a seed draws the eager noise; latents encoded earlier survive later calls."""
    from sage import SAGE
    kw = dict(device="cpu", chunk_seconds=1.5, chunk_margin_seconds=6)
    eager = SAGE.from_checkpoint(tiny_pretrain_ckpt, **kw)
    codec = SAGE.from_checkpoint(tiny_pretrain_ckpt, compile="reduce-overhead", **kw)
    wav = 0.1 * torch.randn(2, 2, 4 * 44100 + 321, generator=torch.Generator().manual_seed(2))
    padded, n = eager.pad(wav)
    z = {}
    for name, c in (("eager", eager), ("compiled", codec)):
        torch.manual_seed(0)
        z[name] = c.encode(padded)
    assert z["compiled"].stride() == z["eager"].stride() and _snr(z["eager"], z["compiled"]) > 80
    z1 = z["compiled"].clone()
    codec.encode(padded.flip(-1), deterministic=True)                 # a later call
    assert torch.equal(z["compiled"], z1)
    rec = codec.decode(z["compiled"], target_length=padded.shape[-1])
    assert _snr(eager.decode(z["eager"], target_length=padded.shape[-1]), rec) > 80


@needs_ckpt
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_paper_model_on_cuda_with_cuda_graphs_keeps_every_output():
    """compile="reduce-overhead" on CUDA (inductor, FlexAttention, CUDA graphs), FP32: encode(x1), encode(x2),
    decode(z1), decode(z2) at one length (the same graphs replayed) give the eager results, so no tensor a caller
    keeps is overwritten by a later replay; chunked inference (chunks kept while the next ones run) and its
    sampled z (same seed) match eager too."""
    from sage import SAGE
    torch._dynamo.reset()
    g = torch.Generator().manual_seed(0)
    x1, x2 = (0.1 * torch.randn(1, 2, 3 * SR + 77, generator=g) for _ in range(2))
    eager = SAGE.from_checkpoint(PAPER_CKPT, device="cuda", precision="fp32")
    comp = SAGE.from_checkpoint(PAPER_CKPT, device="cuda", precision="fp32", compile="reduce-overhead")
    assert "flex" in comp.speedup_options() and comp.speedup_options()[-1] == "compile:reduce-overhead"
    (p1, _), (p2, _) = eager.pad(x1), eager.pad(x2)
    n = p1.shape[-1]
    ref = {}
    for name, c in (("eager", eager), ("comp", comp)):
        for _ in range(2 if name == "comp" else 1):                       # compiled: a recording, then replays
            z1 = c.encode(p1, deterministic=True)
            z2 = c.encode(p2, deterministic=True)
            r1 = c.decode(z1, target_length=n)
            r2 = c.decode(z2, target_length=n)
        ref[name] = [t.cpu() for t in (z1, z2, r1, r2)]
    for a, b in zip(ref["eager"], ref["comp"]):
        assert _snr(a, b) > 60, [_snr(a, b) for a, b in zip(ref["eager"], ref["comp"])]
    kw = dict(device="cuda", precision="fp32", chunk_seconds=3, chunk_margin_seconds=3)
    chunk_eager = SAGE.from_checkpoint(PAPER_CKPT, **kw)
    chunk_comp = SAGE.from_checkpoint(PAPER_CKPT, compile="reduce-overhead", **kw)
    wav = 0.1 * torch.randn(1, 2, 6 * SR, generator=g)                     # 3 chunks, 3 context lengths
    padded, _ = eager.pad(wav)
    out = {}
    for name, c in (("eager", chunk_eager), ("comp", chunk_comp)):
        torch.manual_seed(0)
        z = c.encode(padded)
        out[name] = (z.cpu(), c.decode(z, target_length=padded.shape[-1]).cpu())
    assert _snr(out["eager"][0], out["comp"][0]) > 60 and _snr(out["eager"][1], out["comp"][1]) > 60
    torch._dynamo.reset()


def _snr(ref: torch.Tensor, est: torch.Tensor) -> float:
    return (10 * torch.log10(ref.pow(2).sum() / (ref - est).pow(2).sum().clamp_min(1e-30))).item()


def test_chunking_defaults_and_grid(tiny_pretrain_ckpt):
    """300 s chunks with 6 s margins by default, both on the 128-frame grid of the coarsest window."""
    from sage import SAGE
    codec = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu")
    g = codec._geometry
    assert (g.hop, g.frames_per_step, g.bands, g.grid) == (512, 4, 4, 128)
    assert (codec.chunk_seconds, codec.chunk_margin_seconds) == (300.0, 6.0)
    assert (codec._chunk_frames, codec._margin_frames) == (202 * 128, 4 * 128)
    assert SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", chunk_seconds=None)._chunk_frames is None
    for bad in ({"chunk_seconds": 0}, {"chunk_seconds": -1.0}, {"chunk_margin_seconds": -1.0}):
        with pytest.raises(ValueError):
            SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", **bad)


def test_inputs_below_the_chunk_length_run_one_pass(tiny_pretrain_ckpt):
    """Below chunk_seconds nothing changes: bit-identical to chunking off, sampled z included."""
    from sage import SAGE
    on = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", chunk_seconds=10)
    off = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", chunk_seconds=None)
    wav = 0.1 * torch.randn(1, 2, 9 * 44100, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    a = on.reconstruct(wav)
    torch.manual_seed(0)
    assert torch.equal(a, off.reconstruct(wav))


@pytest.mark.parametrize("B", [1, 2])
def test_chunked_inference_matches_one_pass(tiny_pretrain_ckpt, B):
    """12 s in 1.5 s chunks with 6 s margins (8 chunks, 7 seams) against one pass: the posterior mean, the sampled
    z (same seed -> same noise), decode alone, and reconstruct, unbatched input included."""
    from sage import SAGE
    chunk = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", chunk_seconds=1.5, chunk_margin_seconds=6)
    one = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", chunk_seconds=None)
    wav = 0.1 * torch.randn(B, 2, 12 * 44100 + 321, generator=torch.Generator().manual_seed(B))
    padded, n = one.pad(wav)
    mu, mu_c = one.encode(padded, deterministic=True), chunk.encode(padded, deterministic=True)
    assert mu_c.shape == mu.shape and _snr(mu, mu_c) > 80, _snr(mu, mu_c)
    torch.manual_seed(0)
    z = one.encode(padded)
    torch.manual_seed(0)
    z_c = chunk.encode(padded)
    assert _snr(z, z_c) > 80, _snr(z, z_c)                       # a different noise draw would be ~0-20 dB
    y = one.decode(mu, target_length=padded.shape[-1])
    y_c = chunk.decode(mu, target_length=padded.shape[-1])
    assert y_c.shape == y.shape and _snr(y, y_c) > 80, _snr(y, y_c)
    assert chunk.decode(mu).shape[-1] == (mu.shape[-1] - 1) * 512
    rec, rec_c = one.reconstruct(wav, deterministic=True), chunk.reconstruct(wav, deterministic=True)
    assert rec_c.shape == wav.shape and _snr(rec, rec_c) > 80, _snr(rec, rec_c)
    assert torch.equal(chunk.reconstruct(wav[0], deterministic=True), chunk.reconstruct(wav[:1], deterministic=True)[0])


def test_chunked_inference_needs_its_margin(tiny_pretrain_ckpt):
    """Without context the seams show: the margin is what makes the chunked output the one-pass one."""
    from sage import SAGE
    bare = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", chunk_seconds=1.5, chunk_margin_seconds=0)
    one = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", chunk_seconds=None)
    wav = 0.1 * torch.randn(1, 2, 12 * 44100, generator=torch.Generator().manual_seed(0))
    snr = _snr(one.reconstruct(wav, deterministic=True), bare.reconstruct(wav, deterministic=True))
    assert snr < 60, snr


def test_small_margins_keep_one_grid_step_of_context(tiny_pretrain_ckpt):
    """A margin that would round to 0 frames on the 1.49 s grid keeps one grid step; margin 0 stays 0."""
    from sage import SAGE
    assert SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", chunk_margin_seconds=0.3)._margin_frames == 128
    assert SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", chunk_margin_seconds=0)._margin_frames == 0


def test_checkpoint_without_sample_rate_still_loads(tiny_pretrain_ckpt, tmp_path):
    """Chunking needs the sample rate; a checkpoint without one loads with chunking off, as before chunking."""
    from sage import SAGE
    ckpt = torch.load(tiny_pretrain_ckpt, map_location="cpu", weights_only=False)
    ckpt["inference_config"]["sample_rate"] = None
    torch.save(ckpt, tmp_path / "no_sr.ckpt")
    codec = SAGE.from_checkpoint(tmp_path / "no_sr.ckpt", device="cpu")
    assert codec.chunk_seconds is None and codec._chunk_frames is None
    wav = 0.1 * torch.randn(2, 44100, generator=torch.Generator().manual_seed(0))
    assert codec.reconstruct(wav, deterministic=True).shape == wav.shape


@needs_ckpt
@needs_golden
def test_paper_model_chunked_matches_one_pass():
    """The paper model on the 10 s MoisesDB mix in 1.5 s chunks with 6 s margins (7 seams): equal to one pass up to
    rounding, for the posterior mean and for the sampled z (same seed)."""
    from sage import SAGE
    chunk = SAGE.from_checkpoint(PAPER_CKPT, device="cpu", chunk_seconds=1.5, chunk_margin_seconds=6)
    one = SAGE.from_checkpoint(PAPER_CKPT, device="cpu", chunk_seconds=None)
    wav = torch.from_numpy(np.load(GOLDEN_DIR / "inputs" / "moises10s_mix.npy"))
    assert _snr(one.reconstruct(wav, deterministic=True), chunk.reconstruct(wav, deterministic=True)) > 100
    torch.manual_seed(0)
    z = one.reconstruct(wav)
    torch.manual_seed(0)
    assert _snr(z, chunk.reconstruct(wav)) > 100


@needs_ckpt
@needs_golden
def test_paper_model_postnet_slices_match_one_pass(monkeypatch):
    """The postnet in slices (here of 256 frames, so that 10 s has several) gives the single-pass output up to
    rounding, alone and inside the chunked path."""
    from sage import SAGE
    from sage.model.swin.postnet import ResidualPostNet
    one = SAGE.from_checkpoint(PAPER_CKPT, device="cpu", chunk_seconds=None)
    wav = torch.from_numpy(np.load(GOLDEN_DIR / "inputs" / "moises10s_mix.npy"))
    ref = one.reconstruct(wav, deterministic=True)
    monkeypatch.setattr(ResidualPostNet, "max_slice_frames", 256)
    chunk = SAGE.from_checkpoint(PAPER_CKPT, device="cpu", chunk_seconds=1.5, chunk_margin_seconds=6)
    n_chunks = -(-(wav.shape[-1] // 512) // 128)                        # chunks of 128 frames (lower bound)
    for codec, passes in ((one, 1), (chunk, n_chunks)):
        calls = []
        handle = codec.autoencoder.decoder.postnet.net[0].register_forward_hook(lambda *_: calls.append(1))
        rec = codec.reconstruct(wav, deterministic=True)
        handle.remove()
        assert len(calls) > passes                                     # more than one slice per pass
        assert _snr(ref, rec) > 100


def test_encode_decode_script(tiny_pretrain_ckpt, audio_dir, tmp_path):
    """scripts/encode_decode.py: encode + decode restores the input length and equals reconstruct."""
    import subprocess, sys
    import soundfile as sf
    from conftest import REPO
    wav_in = sorted(audio_dir.glob("*.wav"))[0]
    run = lambda *a: subprocess.run([sys.executable, str(REPO / "scripts" / "encode_decode.py"), *map(str, a),
                                     "--ckpt", str(tiny_pretrain_ckpt), "--device", "cpu", "--deterministic"],
                                    capture_output=True, text=True, timeout=300, check=True)
    run("encode", wav_in, tmp_path / "z.pt")
    run("decode", tmp_path / "z.pt", tmp_path / "a.wav")
    run("reconstruct", wav_in, tmp_path / "b.wav")
    run("reconstruct", wav_in, tmp_path / "c.wav", "--no-sdpa")                 # the original attention
    run("reconstruct", wav_in, tmp_path / "d.wav", "--chunk-seconds", "0.75", "--chunk-margin-seconds", "6")
    run("reconstruct", wav_in, tmp_path / "e.wav", "--chunk-seconds", "0.75", "--no-chunk")
    a, sr = sf.read(tmp_path / "a.wav")
    b, _ = sf.read(tmp_path / "b.wav")
    c, _ = sf.read(tmp_path / "c.wav")
    d, _ = sf.read(tmp_path / "d.wav")
    e, _ = sf.read(tmp_path / "e.wav")
    assert sr == 44100 and a.shape == sf.read(wav_in)[0].shape and np.array_equal(a, b)
    assert c.shape == b.shape and np.abs(c - b).max() <= 2 / 32768                # equal up to rounding (16-bit)
    assert d.shape == b.shape and np.abs(d - b).max() <= 2 / 32768                # 2 s in chunks of 1.49 s
    assert np.array_equal(e, b)                                                  # --no-chunk wins
