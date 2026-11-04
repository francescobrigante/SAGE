# ===============
# Stereo diagnostics of the critic (sage/training/stereo_diagnostics.py): the Side variants, the
# per-example scores of folded and joint branches (checked against single-example forwards), the
# report and the adversarial gradient share are read-only (no .grad, no RNG, no state change).
# ===============
from types import SimpleNamespace

import pytest
import torch

import sage.training.stereo_diagnostics as diag
from sage.nn.discriminators.wavtokenizer import WavTokenizerGANLoss

SMALL = dict(channels=2, sample_rate=44100, periods=[2, 3], resolutions=[(512, 128, 512)], fft_sizes=[512],
             preprocess=True, fold_lrms=True, loss_type="rpgan")
ARMS = {"C0": {}, "A1": dict(preprocess_gain="shared"),
        "A3": dict(preprocess_gain="shared", mpd_stereo="lrms", mrd_stereo="lrms", dac_stereo="lrms")}
T = 8192


def _batch(b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    m, s = torch.randn(b, T, generator=g), torch.randn(b, T, generator=g)
    x = 0.3 * torch.stack([m + 0.4 * s, m - 0.4 * s], dim=1)
    x[0, 1] = x[0, 0]                                                       # example 0 is mono
    return x


def _gan(arm):
    torch.manual_seed(0)
    return WavTokenizerGANLoss(**{**SMALL, **ARMS[arm]})


def test_side_variants_keep_the_mid_and_the_side_energy():
    x = _batch()
    mid = lambda y: (y[:, 0] + y[:, 1]) / 2                                 # noqa: E731
    side = lambda y: (y[:, 0] - y[:, 1]) / 2                                # noqa: E731
    f = diag.foreign_side(x)
    assert torch.allclose(mid(f), mid(x), atol=1e-6)
    assert diag.side_to_mid_db(x)[0] < diag.MONO_SIDE_DB < diag.side_to_mid_db(x)[1]
    assert torch.equal(side(f)[0], side(x)[0])                              # the mono example keeps its null Side
    for i in (1, 2):                                                        # stereo examples swap Sides with each other
        assert torch.allclose(side(f)[i].square().mean(), side(x)[i].square().mean(), rtol=1e-4)
        assert (side(f)[i] - side(x)[i]).abs().max() > 1e-3
    assert torch.equal(diag.foreign_side(x[:2]), x[:2])                     # one stereo example: nothing to swap


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_group_scores_map_rows_to_their_example(arm):
    disc, x = _gan(arm).discriminator, _batch()
    with torch.no_grad():
        batch = diag.group_scores(disc, x)
        for b in range(x.shape[0]):
            alone = diag.group_scores(disc, x[b:b + 1])
            for g in batch:
                assert batch[g].shape == (x.shape[0],)
                assert batch[g][b].item() == pytest.approx(alone[g][0].item(), rel=1e-4, abs=1e-5), (g, b)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_report_is_read_only(arm):
    gan, x = _gan(arm), _batch()
    before = {k: v.clone() for k, v in gan.state_dict().items()}
    rng = torch.get_rng_state()
    report = diag.critic_stereo_report(gan.discriminator, x)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in gan.state_dict().items())
    assert all(p.grad is None for p in gan.parameters())
    assert report["n_mono"] == 1.0
    for g in ("mpd", "mrd", "dac"):
        assert {f"{g}/real", f"{g}/foreign_side_gap", f"{g}/quiet_side_gap", f"{g}/mono_minus_stereo"} <= set(report)


def test_adversarial_gradient_share_per_example_and_without_grad():
    out = torch.randn(3, 4, requires_grad=True)
    w = torch.tensor([1.0, 2.0, 3.0]).view(3, 1)
    adv = (w * out).sum()                                                   # ∂/∂out = w
    rest = (out ** 2).sum() / 2                                             # ∂/∂out = out
    total = adv + rest + torch.tensor(5.0)                                  # a term that does not touch out
    reals = _batch()
    share = diag.adversarial_gradient_share(total, {"loss_adv": adv}, out, reals)
    expected = (w.expand(3, 4).norm(dim=1) / out.detach().norm(dim=1))
    assert share["adv_grad_share"] == pytest.approx(expected.mean().item(), rel=1e-5)
    assert share["adv_grad_share_mono"] == pytest.approx(expected[0].item(), rel=1e-5)
    assert share["adv_grad_share_stereo"] == pytest.approx(expected[1:].mean().item(), rel=1e-5)
    assert out.grad is None
    total.backward()                                                        # the graph is still usable
    assert out.grad is not None


def test_callback_reports_on_its_epochs_only(monkeypatch):
    logged = []
    monkeypatch.setattr(diag, "log_metric", lambda logger, key, value: logged.append(key))
    cb = diag.StereoCriticDiagnostics(every_n_epochs=5)
    module = SimpleNamespace(discriminator=_gan("A3"), device=torch.device("cpu"))
    cb.on_validation_batch_end(None, module, None, (None, _batch()), batch_idx=0)
    for epoch, expect in ((3, False), (4, True)):
        logged.clear()
        trainer = SimpleNamespace(sanity_checking=False, global_rank=0, current_epoch=epoch, logger=None)
        cb.on_validation_epoch_end(trainer, module)
        assert bool(logged) == expect
    assert "critic/dac/foreign_side_gap" in logged
