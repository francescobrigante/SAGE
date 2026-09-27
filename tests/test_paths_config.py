# ===============
# Machine-specific paths live in one place, configs/paths/default.yaml (environment
# variables with defaults): the other configs hold no absolute path and read every data
# and weight location from `paths.*`; no module reads a cluster path or the old
# repo-root config.py.
# ===============
import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import train  # noqa: F401  (registers the ${mul:} resolver)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
PATH_VARIABLES = sorted(set(re.findall(r"\$\{oc\.env:(\w+)", (CONFIGS / "paths" / "default.yaml").read_text())))


@pytest.fixture(autouse=True)
def _machine_without_paths(monkeypatch):
    """The expectations below assume no path variable is set: a configured machine exports them."""
    for name in PATH_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _compose(config_name, overrides=()):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return OmegaConf.to_container(compose(config_name=config_name, overrides=list(overrides)), resolve=True)


def _strings(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)
    elif isinstance(node, str):
        yield node


def test_no_config_outside_paths_holds_an_absolute_path():
    for f in CONFIGS.rglob("*.yaml"):
        if f.parent.name == "paths":
            continue
        raw = OmegaConf.to_container(OmegaConf.load(f), resolve=False)
        bad = [s for s in _strings(raw) if s.startswith(("/", "~"))]
        assert not bad, f"{f.relative_to(REPO)}: {bad}"


def test_paths_default_reads_environment_variables(monkeypatch):
    monkeypatch.setenv("SAGE_MODELS", "/w")
    monkeypatch.setenv("FMA_AUDIO", "/d/fma_large")
    paths = OmegaConf.to_container(OmegaConf.load(CONFIGS / "paths" / "default.yaml"), resolve=True)
    assert paths["fma_audio"] == "/d/fma_large" and paths["musiccaps"] is None
    assert paths["sage_checkpoint"] == "/w/SAGE_FTe992.ckpt"
    assert paths["clap_teacher"] == "/w/LAION_CLAP/music_audioset_epoch_15_esc_90.14.pt"
    assert paths["pann"] == "/w/PANN/Cnn14_16k_mAP=0.438.pth"


@pytest.mark.parametrize("recipe", ["pretrain", "decoder_ft"])
def test_training_data_and_teacher_come_from_paths(recipe):
    cfg = _compose("main", [f"+experiment={recipe}", "paths.fma_full_audio=/p/full", "paths.fma_audio=/p/large",
                            "paths.fma_metadata=/p/tracks.csv", "paths.jamendo_audio=/p/jam",
                            "paths.jamendo_split_tsv=/p/jam.tsv", "paths.m4singer_audio=/p/m4",
                            "paths.filelist_cache_dir=/p/cache", "paths.models_dir=/p/models"])
    data = cfg["data"]
    assert [c["audio_dir"] for c in data["corpora"]] == ["/p/full", "/p/jam", "/p/m4"]
    assert [c["filelist_cache"].rsplit("/", 1)[0] for c in data["corpora"]] == ["/p/cache"] * 3
    assert data["corpora"][0]["custom_metadata_kwargs"]["metadata_csv"] == "/p/tracks.csv"
    assert data["corpora"][1]["custom_metadata_kwargs"]["split_tsv"] == "/p/jam.tsv"
    assert data["eval_dataset"]["audio_dir"] == "/p/large"
    sem = cfg["trainer"]["loss_config"].get("semantic_distill")
    if recipe == "pretrain":
        assert sem["teacher_checkpoint"] == "/p/models/LAION_CLAP/music_audioset_epoch_15_esc_90.14.pt"
    else:
        assert sem is None                                        # no CLAP teacher in the fine-tuning


def test_eval_configs_read_weights_and_outputs_from_paths():
    cfg = _compose("maeb", ["encoder=sage", "paths.models_dir=/p/models", "paths.eval_output=/p/out"])
    assert cfg["paths"]["sage_checkpoint"] == "/p/models/SAGE_FTe992.ckpt"
    rec = _compose("reconstruction", ["model=identity", "dataset=fma", "paths.eval_cache=/p/cache",
                                      "paths.eval_output=/p/out"])
    assert rec["dataset"]["cache_dir"] == "/p/cache/fma" and rec["output_dir"] == "/p/out/recon_fma"


def test_no_module_reads_cluster_paths_or_the_old_root_config():
    pattern = re.compile(r"^\s*(import config\b|from config import)|\$FAST|\$WORK|leonardo|IscrC_|"
                         r"environ(\.get)?\(?\[?[\"'](FAST|WORK)[\"']", re.MULTILINE)
    offenders = []
    for root in ("src", "evaluation", "scripts", "configs"):
        for f in (REPO / root).rglob("*"):
            if f.suffix in (".py", ".yaml", ".sbatch", ".sh") and pattern.search(f.read_text(errors="ignore")):
                offenders.append(str(f.relative_to(REPO)))
    for f in ("train.py", "pyproject.toml"):
        if pattern.search((REPO / f).read_text()):
            offenders.append(f)
    assert not offenders, offenders
    assert not (REPO / "config.py").exists()
