# ===============
# Shared fixtures for the smoke tests: synthetic stereo audio and a tiny SAGE trained for a few
# batches with the paper's pretraining recipe (CPU, CLAP teacher replaced by a fake, no W&B).
# The resulting checkpoint feeds the fine-tuning, inference and evaluation smoke tests.
# ===============
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SR = 44100

# Real paper checkpoint and local golden references (refactor-only; not part of the release).
# The released checkpoint where the README puts it, else the training checkpoint of the development machine.
PAPER_CKPT = Path(os.environ.get("SAGE_CKPT") or next(
    (p for p in (REPO / "models" / "SAGE_FTe992.ckpt", REPO / "sage_release_context" / "SAGE_FTe992.ckpt") if p.is_file()),
    REPO / "models" / "SAGE_FTe992.ckpt"))
GOLDEN_DIR = Path(os.environ.get("SAGE_GOLDEN_DIR", REPO / "sage_release_context" / "sage_golden"))
# Reference arrays checked bit for bit (default: generated on the development machine). To check
# another machine, generate references there with the pre-release code and point SAGE_GOLDEN_REF at them.
GOLDEN_REF = Path(os.environ.get("SAGE_GOLDEN_REF", GOLDEN_DIR / "golden_local"))

# Tiny SAGE: the paper architecture (patch 64x1, window 4x32, depths 2-6-2, SwiGLU, XSA,
# res-post norm) with a narrow embedding, so a training step takes about a second on CPU.
TINY_MODEL = [
    "models.model.swin.embed_dim=32",
    "models.model.encoder.depths=[2,2,2]",
    "models.model.decoder.depths=[2,2,2]",
]
# Same discriminator family, loss and stereo folding as the paper (WavTokenizer, RpGAN, fold_lrms),
# but one sub-discriminator per branch: the full 43M-parameter critic needs ~2 min per step on CPU.
# The paper's exact values are checked by the recipe-composition test, not here.
SMOKE_DISC = [
    "++trainer.loss_config.discriminator.config.periods=[2]",
    "++trainer.loss_config.discriminator.config.fft_sizes=[512]",
    "++trainer.loss_config.discriminator.config.resolutions=[[512,128,512]]",
]
# Single-corpus data from a folder of wavs, no metadata provider, no worker processes.
def _data_overrides(audio_dir: Path) -> list[str]:
    out = []
    for split in ("train_dataset", "eval_dataset"):
        out += [
            f"data.{split}.audio_dir={audio_dir}",
            f"data.{split}.extensions=[.wav]",
            f"data.{split}.custom_metadata_module=null",
            f"data.{split}.custom_metadata_kwargs=null",
        ]
    for split in ("train_dataloader", "eval_dataloader"):
        out += [f"data.{split}.batch_size=1", f"data.{split}.num_workers=0"]
    return out + ["data.demo.max_demos=1"]

RUN = [
    "++trainer.device=cpu",
    "trainer.trainer.num_gpus=1",
    "trainer.trainer.strategy=auto",
    "trainer.trainer.epochs=1",
    "trainer.trainer.limit_train_batches=2",   # 1 generator + 1 discriminator update (G/D alternate)
    "trainer.trainer.limit_val_batches=1",
    "trainer.trainer.num_sanity_val_steps=0",
    "trainer.trainer.save_every_n_epochs=1",
    "trainer.wandb.use_wandb=false",
]


class FakeCLAPTeacher(nn.Module):
    """Stand-in for the frozen LAION-CLAP teacher: fixed random projection to a unit 512-d vector."""

    def __init__(self, model_dir: str = "", src_sr: int = SR):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer("proj", torch.randn(64, 512, generator=g))

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.mean(dim=1) if wav.ndim == 3 else wav                        # (B, N) mono
        feats = x.unfold(-1, 1024, 1024).abs().mean(-1)                       # (B, frames)
        feats = nn.functional.adaptive_avg_pool1d(feats.unsqueeze(1), 64).squeeze(1)  # (B, 64)
        return nn.functional.normalize(feats @ self.proj, dim=-1)            # (B, 512)


def write_music_like_wavs(folder: Path, n: int, seconds: float = 2.0, seed: int = 0) -> list[Path]:
    """Stereo 44.1 kHz clips: a few harmonic partials with a slow envelope, decorrelated L/R, light noise."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    paths = []
    for i in range(n):
        f0 = rng.uniform(110, 440)
        env = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(0.5, 2) * t))
        mono = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)) * env
        left = mono + 0.02 * rng.standard_normal(t.size)
        right = np.roll(mono, rng.integers(10, 200)) + 0.02 * rng.standard_normal(t.size)
        wav = 0.1 * np.stack([left, right], axis=1) / np.abs(mono).max()
        p = folder / f"clip_{i:02d}.wav"
        sf.write(p, wav.astype(np.float32), SR)
        paths.append(p)
    return paths


def run_training(overrides: list[str], out_dir: Path) -> Path:
    """Compose the Hydra config exactly as `python train.py <overrides>` would, run it, return the newest checkpoint."""
    from hydra import compose, initialize_config_dir
    import train                                                  # registers the ${mul:} resolver
    import sage.training.loss_manager as loss_manager

    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(config_name="main", overrides=overrides)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(train, "get_original_cwd", lambda: str(out_dir))   # runs/ goes under out_dir
        mp.setattr(loss_manager, "CLAPTeacher", FakeCLAPTeacher)
        mp.chdir(out_dir)
        train.main(cfg)
    ckpts = sorted(out_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    assert ckpts, f"no checkpoint written under {out_dir}"
    return ckpts[-1]


@pytest.fixture(scope="session")
def audio_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("audio")
    write_music_like_wavs(d, n=4)
    return d


PRETRAIN = [
    "+experiment=pretrain",                                        # paper pretraining recipe (Table 6)
    "data=fma",                                                    # single-corpus loader for the smoke test
    "trainer.loss_config.semantic_distill.detach_warmup_steps=0",  # gate open: gradient flows into the latent
    "trainer.wandb.name=smoke_pretrain",
]


@pytest.fixture(scope="session")
def tiny_pretrain_ckpt(audio_dir, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("pretrain")
    return run_training(PRETRAIN + TINY_MODEL + SMOKE_DISC + _data_overrides(audio_dir) + RUN, out)
