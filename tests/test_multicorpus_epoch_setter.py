# ===============================================================
# test_multicorpus_epoch_setter.py
#
#   MultiCorpusEpochSetter must advance BOTH the rotating sampler and
#   the dataset on train epoch start (so the M4 chunk rotates AND the
#   crop window varies), tolerate a missing set_epoch on either side,
#   and no-op when there is no train dataloader yet. Validation never
#   triggers it, so val stays at epoch 0 (fixed crops).
# ===============================================================
import sys
import types
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# stub the losses package (login-node torchaudio ABI) before importing callbacks
_LOSSES = "ar_spectra.training.losses"
if _LOSSES not in sys.modules:
    _stub = types.ModuleType(_LOSSES)
    _stub.__path__ = [str(PROJECT_ROOT / "src/ar_spectra/training/losses")]
    _stub.__package__ = _LOSSES
    _stub.__spec__ = None
    sys.modules[_LOSSES] = _stub

from ar_spectra.training.callbacks import MultiCorpusEpochSetter


class _Recorder:
    def __init__(self):
        self.epoch = None
    def set_epoch(self, e):
        self.epoch = e


def _trainer(dataloader, epoch):
    return SimpleNamespace(train_dataloader=dataloader, current_epoch=epoch)


# ── advances both sampler and dataset with the current epoch ─────────────────
def test_sets_epoch_on_sampler_and_dataset():
    sampler, dataset = _Recorder(), _Recorder()
    dl = SimpleNamespace(sampler=sampler, dataset=dataset)
    MultiCorpusEpochSetter().on_train_epoch_start(_trainer(dl, 5), pl_module=None)
    assert sampler.epoch == 5
    assert dataset.epoch == 5


# ── tolerates a sampler without set_epoch (e.g. None) ────────────────────────
def test_tolerates_missing_sampler():
    dataset = _Recorder()
    dl = SimpleNamespace(sampler=None, dataset=dataset)
    MultiCorpusEpochSetter().on_train_epoch_start(_trainer(dl, 3), pl_module=None)
    assert dataset.epoch == 3


# ── no train dataloader yet → no crash, no-op ────────────────────────────────
def test_noop_without_dataloader():
    MultiCorpusEpochSetter().on_train_epoch_start(_trainer(None, 0), pl_module=None)  # must not raise


# ── distinct epochs propagate on successive calls ────────────────────────────
def test_epoch_advances():
    sampler, dataset = _Recorder(), _Recorder()
    dl = SimpleNamespace(sampler=sampler, dataset=dataset)
    cb = MultiCorpusEpochSetter()
    for e in range(4):
        cb.on_train_epoch_start(_trainer(dl, e), pl_module=None)
        assert sampler.epoch == e and dataset.epoch == e
