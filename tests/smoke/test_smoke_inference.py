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

from conftest import GOLDEN_DIR, GOLDEN_REF, PAPER_CKPT

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
    with the same output as the original attention up to rounding."""
    from sage import SAGE
    default = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu", sdpa=False)
    fast = SAGE.from_checkpoint(tiny_pretrain_ckpt, device="cpu")
    assert (default.sdpa, default.sdpa_blocks) == (False, 0) and fast.sdpa and fast.sdpa_blocks > 0
    wav = 0.1 * torch.randn(2, 3 * 44100 + 123, generator=torch.Generator().manual_seed(0))
    a, b = default.reconstruct(wav, deterministic=True), fast.reconstruct(wav, deterministic=True)
    snr = 10 * torch.log10(a.pow(2).sum() / (a - b).pow(2).sum().clamp_min(1e-30))
    assert a.shape == b.shape and snr > 80, snr


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
