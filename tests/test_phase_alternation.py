# ===============
# Tests for the G/D phase alternation fix. The disc phase must fire ~every other
# batch regardless of how many optimizer.step() each phase performs (the bug: with
# an aux optimizer the gen phase did 2 steps → global_step stayed even → disc starved).
# ===============
import pytest
from ar_spectra.training.engine import select_training_phase


def _steps_for_phase(phase: str, has_aux: bool) -> int:
    """optimizer.step() count per batch: gen = opt_gen (+ opt_aux), disc = opt_disc."""
    if phase == "gen":
        return 2 if has_aux else 1
    return 1


def _simulate_old(n: int, has_aux: bool):
    """OLD logic: phase from global_step parity (the buggy one)."""
    gs, phases = 0, []
    for _ in range(n):
        phase = "disc" if (gs % 2 == 1) else "gen"   # old use_disc_phase (use_disc=True, adv)
        phases.append(phase)
        gs += _steps_for_phase(phase, has_aux)
    return phases


def _simulate_new(n: int, has_aux: bool):
    """NEW logic: per-batch boolean toggle + select_training_phase (mirrors wrapper:
    starts True so the first batch flips to gen, matching prior gen-first behaviour)."""
    disc_phase, phases = True, []
    for _ in range(n):
        disc_phase = not disc_phase
        phase = select_training_phase(use_disc=True, warmup_mode="adv",
                                      disc_phase=disc_phase, warmed_up=True)
        phases.append(phase)
    return phases


# ---- the bug, reproduced -----------------------------------------------------
def test_old_logic_works_without_aux():
    phases = _simulate_old(20, has_aux=False)
    assert phases.count("disc") == 10            # 1 step/batch → parity alternates ✓

def test_old_logic_starves_disc_with_aux():
    phases = _simulate_old(20, has_aux=True)
    assert phases.count("disc") == 0             # BUG: gen=+2 → global_step always even

# ---- the fix -----------------------------------------------------------------
def test_new_logic_alternates_without_aux():
    phases = _simulate_new(20, has_aux=False)
    assert phases == ["gen", "disc"] * 10        # unchanged vs working case

def test_new_logic_alternates_with_aux():
    phases = _simulate_new(20, has_aux=True)
    assert phases == ["gen", "disc"] * 10        # disc now fires every other batch ✓
    assert phases.count("disc") == 10

# ---- gating preserved --------------------------------------------------------
def test_no_disc_phase_when_use_disc_false():
    assert select_training_phase(False, "adv", True, True) == "gen"

def test_full_warmup_gating():
    assert select_training_phase(True, "full", True, False) == "gen"   # warmup not done
    assert select_training_phase(True, "full", True, True) == "disc"   # warmup done

def test_adv_mode_ignores_warmup():
    assert select_training_phase(True, "adv", True, False) == "disc"

def test_toggle_off_is_gen():
    assert select_training_phase(True, "adv", False, True) == "gen"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
