# ===============
# The unified reconstruction evaluator (evaluation/reconstruction.py, Hydra config
# configs/reconstruction.yaml) on synthetic audio with the identity codec and
# sdr_only=true (no embedding weights): both protocols
# write the per-file CSVs; a run sharded over two SLURM tasks plus --merge gives the
# same CSV as one task; the CLI rejects a SAGE run without a checkpoint. Also the
# paper tables (evaluation/tables.py) from fake evaluation outputs.
# ===============
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from hydra import compose, initialize_config_dir

from evaluation import reconstruction, tables

SR = 44100
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FAST = ["device=cpu", "num_workers=0"]


def _clip(path: Path, seconds: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    sf.write(path, (0.1 * rng.standard_normal((int(seconds * SR), 2))).astype(np.float32), SR)


@pytest.fixture
def clips(tmp_path):
    d = tmp_path / "clips"
    d.mkdir()
    for i in range(4):
        _clip(d / f"clip_{i}.wav", 1.0 + 0.25 * i, i)
    return d


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _cfg(overrides):
    """configs/reconstruction.yaml with the overrides, as `python -m evaluation.reconstruction <overrides>`."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return compose(config_name="reconstruction", overrides=list(overrides))


def _clips(d):
    return ["dataset=musiccaps", f"dataset.data_dir={d}"]


def _run(overrides, monkeypatch, rank=0, world=1):
    monkeypatch.setenv("SLURM_PROCID", str(rank))
    monkeypatch.setenv("SLURM_NTASKS", str(world))
    reconstruction.run(_cfg(overrides))


def test_clips_protocol_writes_the_metrics(clips, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _run(["model=identity", *_clips(clips), f"output_dir={out}", *FAST, "sdr_only=true", "compute_ms_metrics=true"],
         monkeypatch)
    metrics = out / "identity" / "metrics"
    spectral = _rows(metrics / "spectral.csv")
    assert [r["file"] for r in spectral] == [f"clip_{i}" for i in range(4)]
    assert all(float(r["si_sdr"]) > 60 and float(r["stft_loss"]) < 1e-3 for r in spectral)   # identity codec
    assert len(_rows(metrics / "ms_metrics.csv")) == 4 and (metrics / "_done").exists()
    assert not (metrics / "timing.csv").exists()


def test_loader_workers_give_the_same_metrics(clips, tmp_path, monkeypatch):
    """num_workers > 0 works under the spawn start method (macOS default) and changes no number."""
    common = ["model=identity", *_clips(clips), "device=cpu", "sdr_only=true"]
    _run(common + [f"output_dir={tmp_path / 'w0'}", "num_workers=0"], monkeypatch)
    _run(common + [f"output_dir={tmp_path / 'w2'}", "num_workers=2"], monkeypatch)
    spectral = [_rows(tmp_path / w / "identity" / "metrics" / "spectral.csv") for w in ("w0", "w2")]
    assert spectral[0] == spectral[1] and len(spectral[0]) == 4


def test_fma_protocol_selects_the_test_split(tmp_path, monkeypatch):
    root = tmp_path / "fma"
    (root / "000").mkdir(parents=True)
    for tid in (2, 5, 7):
        _clip(root / "000" / f"{tid:06d}.wav", 1.0, tid)
    cols = pd.MultiIndex.from_tuples([("set", "split"), ("track", "title")])
    pd.DataFrame([["test", "a"], ["test", "b"], ["training", "c"]], index=[2, 5, 7], columns=cols).to_csv(
        tmp_path / "tracks.csv")
    out = tmp_path / "out"
    _run(["model=identity", "dataset=fma", f"dataset.data_dir={root}", f"dataset.fma_csv={tmp_path / 'tracks.csv'}",
          "dataset.cache_dir=null", f"output_dir={out}", *FAST, "sdr_only=true"], monkeypatch)
    assert [r["file"] for r in _rows(out / "identity" / "metrics" / "spectral.csv")] == ["000002", "000005"]


def test_sharded_run_plus_merge_equals_a_single_run(clips, tmp_path, monkeypatch):
    common = ["model=identity", *_clips(clips), *FAST, "sdr_only=true"]
    _run(common + [f"output_dir={tmp_path / 'single'}"], monkeypatch)
    sharded = common + [f"output_dir={tmp_path / 'sharded'}"]
    for rank in (0, 1):
        _run(sharded, monkeypatch, rank=rank, world=2)
    assert not (tmp_path / "sharded" / "identity" / "metrics" / "_done").exists()   # waits for merge=true
    _run(sharded + ["merge=true"], monkeypatch)
    single = {r["file"]: r for r in _rows(tmp_path / "single" / "identity" / "metrics" / "spectral.csv")}
    merged = {r["file"]: r for r in _rows(tmp_path / "sharded" / "identity" / "metrics" / "spectral.csv")}
    assert single == merged


def test_resubmitting_skips_the_files_already_scored(clips, tmp_path, monkeypatch):
    argv = ["model=identity", *_clips(clips), f"output_dir={tmp_path}", *FAST, "sdr_only=true"]
    _run(argv, monkeypatch)
    parts = tmp_path / "identity" / "metrics" / "parts"
    before = (parts / "spectral.0.csv").read_text()
    _run(argv, monkeypatch)                                       # every stem is in done.0.txt
    assert (parts / "spectral.0.csv").read_text() == before


def test_sage_needs_an_existing_checkpoint(clips, tmp_path):
    with pytest.raises(SystemExit, match="checkpoint not found"):
        reconstruction._args_from_cfg(_cfg(["model=sage", *_clips(clips), f"checkpoint={tmp_path / 'none.ckpt'}"]))


def test_dataset_and_its_folder_are_required(tmp_path):
    with pytest.raises(SystemExit, match="dataset=<name> is required"):
        reconstruction.run(_cfg(["model=identity"]))
    with pytest.raises(SystemExit, match="data folder is not set"):
        reconstruction._args_from_cfg(_cfg(["model=identity", "dataset=musiccaps", "paths.musiccaps=null"]))


def test_dataset_configs_read_their_folders_from_paths(tmp_path):
    for name, key in (("fma", "fma_audio"), ("moisesdb_mix", "moisesdb_mix"), ("moisesdb_stems", "moisesdb_stems"),
                      ("musiccaps", "musiccaps"), ("song_describer", "song_describer")):
        args = reconstruction._args_from_cfg(_cfg(["model=identity", f"dataset={name}", f"paths.{key}={tmp_path}",
                                                   f"paths.eval_output={tmp_path / 'out'}"]))
        assert args.data_dir == tmp_path and args.protocol == ("fma" if name == "fma" else "clips")
        assert args.output_dir == tmp_path / "out" / f"recon_{name}"


# ── tables ───────────────────────────────────────────────────────────────────

def test_tables_print_reconstruction_and_probing(tmp_path, capsys):
    for model, sdr in (("sage", 5.0), ("same-s", 8.0)):
        m = tmp_path / "recon" / model / "metrics"
        m.mkdir(parents=True)
        (m / "spectral.csv").write_text(f"file,si_sdr,sdr,stft_loss,mel_loss\na,1,{sdr},0.9,1\nb,1,{sdr},1.1,1\n")
        (m / "fad_mert.csv").write_text("model,score\nMERT-v1-95M-4,0.1\n")
    maeb = tmp_path / "maeb" / "sage" / "sage__std" / "local"
    maeb.mkdir(parents=True)
    (maeb / "FMAGenreClassification.json").write_text(json.dumps({"scores": {"test": [{"main_score": 0.61}]}}))
    tables.main(["--recon", f"FMA test={tmp_path / 'recon'}", "--maeb", str(tmp_path / "maeb" / "sage")])
    out = capsys.readouterr().out
    assert "Reconstruction — FMA test" in out and "| sage |" in out and "| same-s |" in out
    assert "**8.000**" in out                                     # best SDR in bold
    assert "0.610" in out and "Average per block" in out


# ── random codecs, cached targets, resume safety (review fixes) ───────────────

class _NoisyCodec:
    """Stand-in for a codec that samples (SAGE's z, SAME's noise): output depends on the torch RNG."""
    sample_rate, audio_channels = SR, 2

    def __init__(self, **_):
        pass

    def reconstruct(self, wav):
        import torch
        return wav + 0.01 * torch.randn_like(wav)


def _fake_embedders(monkeypatch):
    """Cheap embedders; FAD-CLAP's embedder takes a random crop with np.random, as LAION-CLAP does."""
    import sys
    import types
    fake_loader = types.ModuleType("fadtk.model_loader")

    class _Model:
        def __init__(self, *a, **k):
            self.model = types.SimpleNamespace(to=lambda d: None)

        def load_model(self):
            pass

    fake_loader.MERTModel = fake_loader.CLAPLaionModel = _Model
    monkeypatch.setitem(sys.modules, "fadtk.model_loader", fake_loader)
    def feats(x, n):                                              # deterministic in the signal, full rank
        rng = np.random.default_rng(int(abs(float(x.sum())) * 1e6) % 2**32)
        return rng.standard_normal((n, 8))

    def gud(wav, sr, device, *a):
        x = wav.numpy().mean(0)
        start = np.random.randint(0, max(1, x.size - SR // 2))    # the random crop
        return feats(x[start:start + SR // 2], 4).astype(np.float16)

    monkeypatch.setattr(reconstruction, "embed_clap", lambda ml, w, sr, d, *a: feats(w.numpy(), 3).astype(np.float16))
    monkeypatch.setattr(reconstruction, "embed_mert_framewise", lambda ml, w, sr, d, *a: feats(w.numpy(), 5).astype(np.float16))
    monkeypatch.setattr(reconstruction, "embed_pann", lambda w, sr, d, *a: feats(w.numpy(), 2).astype(np.float16))
    monkeypatch.setattr(reconstruction, "embed_clap_gud", gud)
    monkeypatch.setattr(reconstruction, "get_pann_model", lambda d: None)


@pytest.fixture
def noisy(monkeypatch):
    monkeypatch.setitem(reconstruction.ADAPTERS, "noisy", _NoisyCodec)
    monkeypatch.setattr(reconstruction, "build_adapter", lambda name, **kw: reconstruction.ADAPTERS[name](**kw))


def test_random_codec_sharded_or_resumed_equals_one_run(clips, tmp_path, monkeypatch, noisy):
    base = ["model=noisy", *_clips(clips), *FAST, "sdr_only=true"]
    _run(base + [f"output_dir={tmp_path / 'one'}"], monkeypatch)
    for rank in (0, 1):
        _run(base + [f"output_dir={tmp_path / 'sharded'}"], monkeypatch, rank=rank, world=2)
    _run(base + [f"output_dir={tmp_path / 'sharded'}", "merge=true"], monkeypatch)
    _run(base + [f"output_dir={tmp_path / 'resumed'}", "max_files=2"], monkeypatch)   # killed run...
    parts = tmp_path / "resumed" / "noisy" / "metrics" / "parts"
    (parts / "run.json").unlink()                                 # ...relaunched on the whole set
    _run(base + [f"output_dir={tmp_path / 'resumed'}"], monkeypatch)
    read = lambda d: {r["file"]: r for r in _rows(tmp_path / d / "noisy" / "metrics" / "spectral.csv")}
    assert read("one") == read("sharded") == read("resumed")


def test_fad_clap_does_not_depend_on_the_target_cache(tmp_path, monkeypatch, noisy):
    _fake_embedders(monkeypatch)
    root = tmp_path / "fma" / "000"
    root.mkdir(parents=True)
    for tid in (2, 5, 9):
        _clip(root / f"{tid:06d}.wav", 2.0, tid)
    argv = ["model=noisy", "dataset=fma", f"dataset.data_dir={tmp_path / 'fma'}", "dataset.fma_csv=null",
            f"dataset.cache_dir={tmp_path / 'cache'}", *FAST, "skip_cdpam=true"]
    _run(argv + [f"output_dir={tmp_path / 'cold'}"], monkeypatch)      # fills the target cache
    _run(argv + [f"output_dir={tmp_path / 'warm'}"], monkeypatch)      # reads it
    for fad in ("fad_gudgud.csv", "fad_mert.csv", "fad_pann.csv"):
        assert (tmp_path / "cold" / "noisy" / "metrics" / fad).read_text() == \
               (tmp_path / "warm" / "noisy" / "metrics" / fad).read_text(), fad


def test_resuming_with_other_settings_is_refused(clips, tmp_path, monkeypatch):
    base = ["model=identity", *_clips(clips), f"output_dir={tmp_path}", *FAST, "sdr_only=true"]
    _run(base, monkeypatch)
    with pytest.raises(SystemExit, match="other settings"):
        _run(base + ["seed=1"], monkeypatch)


def test_a_file_whose_embedding_fails_is_retried(clips, tmp_path, monkeypatch, noisy):
    _fake_embedders(monkeypatch)
    real = reconstruction.embed_pann
    monkeypatch.setattr(reconstruction, "embed_pann", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    argv = ["model=noisy", *_clips(clips), f"output_dir={tmp_path}", *FAST, "skip_cdpam=true"]
    _run(argv, monkeypatch)
    parts = tmp_path / "noisy" / "metrics" / "parts"
    assert not (parts / "done.0.txt").exists()
    monkeypatch.setattr(reconstruction, "embed_pann", real)
    _run(argv, monkeypatch)
    assert len((parts / "done.0.txt").read_text().split()) == 4


def test_model_settings_are_validated(clips, tmp_path):
    for bad in (["model=sam-s"], ["model=same-s", "deterministic=true"], ["model=same-s", f"checkpoint={clips}"]):
        with pytest.raises(SystemExit):
            reconstruction._args_from_cfg(_cfg([*bad, *_clips(clips)]))


def test_tables_read_any_mteb_revision_and_never_bold_the_oracle(tmp_path, capsys):
    for encoder, revision, score in (("sage", "local", 0.60), ("clap", "music_audioset_epoch_15_esc_90.14", 0.90)):
        d = tmp_path / encoder / f"{encoder}-model" / revision
        d.mkdir(parents=True)
        (d / "FMAGenreClassification.json").write_text(json.dumps({"scores": {"test": [{"main_score": score}]}}))
    m = tmp_path / "recon" / "sage" / "metrics"
    m.mkdir(parents=True)
    (m / "ms_metrics.csv").write_text("file,width_bias,d_width,sisdr_s,sisdr_m,ref_mono\n"
                                      "a,-0.1,0.2,3,9,0\nb,0.5,0.5,0,0,1\n")
    tables.main(["--recon", f"S={tmp_path / 'recon'}", "--maeb", str(tmp_path / "sage"), "--maeb", str(tmp_path / "clap")])
    out = capsys.readouterr().out
    assert "| clap | 0.900" in out and "**0.600**" in out           # oracle shown, best autoencoder bolded
    assert "| sage | -0.100 | 0.200 | 3.000 | 9.000 |" in out.replace("**", "")   # mono reference excluded
