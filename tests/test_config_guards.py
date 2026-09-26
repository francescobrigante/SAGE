# =============================================================================
# Unit tests for config consistency guards (config_guards.py) and
# trainer.yaml scalar checks.
# =============================================================================

import math
import pytest
import yaml
from pathlib import Path

import sys
import os

from ar_spectra.utils.config_guards import check_cac_consistency


# ── Hotspot A: clip_grad_norm value ──────────────────────────────────────────

TRAINER_YAML = Path(__file__).parent.parent / "config" / "trainer.yaml"


class TestClipGradNorm:

    def test_clip_grad_norm_is_not_old_value(self):
        """Confirm the old value 1.0 is gone."""
        with open(TRAINER_YAML) as f:
            cfg = yaml.safe_load(f)
        val = cfg["trainer"]["clip_grad_norm"]
        assert val != pytest.approx(1.0), (
            "clip_grad_norm is still 1.0 — Hotspot A fix was not applied."
        )


# ── Hotspot B: cac/is_complex guard ──────────────────────────────────────────

class TestCacConsistencyGuard:

    # ── real models: no error regardless of cac ──────────────────────────────

    def test_real_model_cac_true_ok(self):
        """Real models (is_complex=False) accept cac=True without error."""
        check_cac_consistency(is_complex_model=False, train_cac=True, eval_cac=True, demo_cac=True)

    def test_real_model_cac_false_ok(self):
        check_cac_consistency(is_complex_model=False, train_cac=False, eval_cac=False, demo_cac=False)

    # ── complex model, correct config ────────────────────────────────────────

    def test_complex_model_cac_false_ok(self):
        """Complex model + cac=False is the correct config — no error."""
        check_cac_consistency(is_complex_model=True, train_cac=False, eval_cac=False, demo_cac=False)

    def test_complex_model_no_eval_no_demo_ok(self):
        """Optional eval/demo absent (None) does not trigger error."""
        check_cac_consistency(is_complex_model=True, train_cac=False, eval_cac=None, demo_cac=None)

    # ── complex model + mismatch → must raise ────────────────────────────────

    def test_complex_train_cac_true_raises(self):
        """Complex model + train_cac=True must raise ValueError."""
        with pytest.raises(ValueError, match="train_dataset.cac=True"):
            check_cac_consistency(is_complex_model=True, train_cac=True)

    def test_complex_eval_cac_true_raises(self):
        """Complex model + eval_cac=True must raise ValueError."""
        with pytest.raises(ValueError, match="eval_dataset.cac=True"):
            check_cac_consistency(is_complex_model=True, train_cac=False, eval_cac=True)

    def test_complex_demo_cac_true_raises(self):
        """Complex model + demo_cac=True must raise ValueError."""
        with pytest.raises(ValueError, match="demo.istft_params.cac=True"):
            check_cac_consistency(is_complex_model=True, train_cac=False, eval_cac=False, demo_cac=True)

    def test_error_message_contains_hint(self):
        """Error must include the CLI override hint so the user knows how to fix it."""
        with pytest.raises(ValueError, match="data.train_dataset.cac=false"):
            check_cac_consistency(is_complex_model=True, train_cac=True)

    def test_train_cac_takes_priority_over_eval(self):
        """train_cac is checked first; error mentions train_dataset, not eval_dataset."""
        with pytest.raises(ValueError, match="train_dataset"):
            check_cac_consistency(is_complex_model=True, train_cac=True, eval_cac=True)

    # ── data config default sanity ───────────────────────────────────────────

    @pytest.mark.parametrize("name", ["fma", "multicorpus"])
    def test_data_yaml_has_cac_field(self, name):
        """config/data/<name>.yaml must have an explicit cac field in train/eval datasets."""
        with open(Path(__file__).parent.parent / "config" / "data" / f"{name}.yaml") as f:
            cfg = yaml.safe_load(f)
        assert "cac" in cfg["train_dataset"], (
            "data.yaml train_dataset is missing the cac key — required for guard to work."
        )
        assert "cac" in cfg["eval_dataset"], (
            "data.yaml eval_dataset is missing the cac key — required for guard to work."
        )
