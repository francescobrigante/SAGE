# ===============
# channel=side of evaluation.reconstruction and evaluation.build_references: the embedders and
# CDPAM read (L-R)/2, every artefact carries the "-side" suffix (run folder, reference embeddings,
# FAD statistics), and the default channel=mid leaves the paper's names and calls unchanged.
# Synthetic clips, identity codec, recording test doubles instead of the embedding weights.
# ===============
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from hydra import compose, initialize_config_dir

from evaluation import build_references, reconstruction
from evaluation.common import channel_suffix

SR = 44100
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FAST = ["device=cpu", "num_workers=0"]


@pytest.fixture
def clips(tmp_path):
    d = tmp_path / "clips"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        sf.write(d / f"clip_{i}.wav", (0.1 * rng.standard_normal((SR, 2))).astype(np.float32), SR)
    return d


def _cfg(name, overrides):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return compose(config_name=name, overrides=list(overrides))


def _fake_models(monkeypatch):
    loader = types.ModuleType("fadtk.model_loader")

    class _Model:
        def __init__(self, *a, **k):
            self.model = SimpleNamespace(to=lambda d: None)

        def load_model(self):
            pass

    loader.MERTModel = loader.CLAPLaionModel = _Model
    monkeypatch.setitem(sys.modules, "fadtk.model_loader", loader)


def _recording_embedders(monkeypatch, module, calls):
    """Replace the embedders of `module` with doubles that record the channel they receive."""
    def emb(n):
        def f(*a):
            calls.append(a[-1])                                   # channel: last positional argument
            return np.random.default_rng(len(calls)).standard_normal((n, 8)).astype(np.float16)
        return f
    for fn, n in (("embed_clap", 3), ("embed_mert_framewise", 5), ("embed_pann", 2), ("embed_clap_gud", 4)):
        monkeypatch.setattr(module, fn, emb(n))
    monkeypatch.setattr(module, "get_pann_model", lambda d: None)


def test_channel_suffix():
    assert channel_suffix("mid") == "" and channel_suffix("side") == "-side"
    with pytest.raises(ValueError):
        channel_suffix("left")


def test_an_unknown_channel_is_refused(clips):
    with pytest.raises(SystemExit, match="channel"):
        reconstruction._args_from_cfg(_cfg("reconstruction", ["model=identity", "dataset=musiccaps",
                                                              f"dataset.data_dir={clips}", "channel=left"]))


@pytest.mark.parametrize("channel", ["mid", "side"])
def test_reconstruction_feeds_the_channel_and_looks_up_its_references(clips, tmp_path, monkeypatch, channel):
    _fake_models(monkeypatch)
    calls, refs, stats = [], [], []
    _recording_embedders(monkeypatch, reconstruction, calls)
    monkeypatch.setattr(reconstruction, "_load_ref_emb", lambda d, name, stem: refs.append(name))
    monkeypatch.setattr(reconstruction, "_fad_from_stats", lambda d, name, *a: stats.append(name))
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    reconstruction.run(_cfg("reconstruction", ["model=identity", "dataset=musiccaps", f"dataset.data_dir={clips}",
                                               f"output_dir={tmp_path}", *FAST, "skip_cdpam=true",
                                               f"channel={channel}"]))
    suffix = channel_suffix(channel)
    assert (tmp_path / f"identity{suffix}" / "metrics" / "_done").exists()
    assert calls and set(calls) == {channel}
    assert refs and all(n.endswith(suffix) for n in refs) and (suffix or not any("-side" in n for n in refs))
    assert sorted(stats) == sorted(n + suffix for n in reconstruction.FAD_CSV)
    manifest = json.loads((tmp_path / f"identity{suffix}" / "metrics" / "parts" / "run.json").read_text())
    assert manifest["channel"] == channel


def test_a_manifest_written_before_the_channel_key_is_a_mid_run(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"model": "identity", "seed": 0}))   # no "channel" key
    reconstruction._check_manifest(tmp_path, SimpleNamespace(model="identity", seed=0, channel="mid"), rank=0)
    with pytest.raises(SystemExit, match="other settings"):
        reconstruction._check_manifest(tmp_path, SimpleNamespace(model="identity", seed=0, channel="side"), rank=0)


def test_build_references_writes_side_folders(clips, monkeypatch):
    _fake_models(monkeypatch)
    calls = []
    _recording_embedders(monkeypatch, build_references, calls)
    build_references.run(_cfg("build_references", ["dataset=musiccaps", f"dataset.data_dir={clips}",
                                                   *FAST, "channel=side"]))
    assert calls and set(calls) == {"side"}
    for name in build_references.EMBEDDERS:
        assert len(list((clips / "embeddings" / f"{name}-side").glob("*.npy"))) == 3
        assert not (clips / "embeddings" / name).exists()
    for name in build_references.STATS_EMBEDDERS:
        assert (clips / "stats_ours" / f"{name}-side" / "mu.npy").exists()
