"""
CLAP semantic distillation loss tests (LatentCosineDistillLoss, SALAD arXiv:2510.07592).

Uses a tiny STUB teacher (no CLAP load) so the tests are fast and snapshot-free.

  - shapes: projector D→proj_dim, clip-level time pool, finite scalar in [0, 2].
  - loss → 0 when projected clip vector aligns with teacher clip vector.
  - detached warmup gates encoder grad; projector still receives gradient.
  - teacher output has no gradient (frozen CLAPTeacher contract).
  - a (B, 512) clip embedding (CLAP) and a (B, C, T) frame sequence give the same loss
    when the sequence is constant in time.
"""
import torch
from torch import nn

from sage.nn.losses.semantic import LatentCosineDistillLoss


class _StubTeacher(nn.Module):
    """Returns a fixed tensor — stands in for the frozen teacher."""
    def __init__(self, feat):
        super().__init__()
        self.feat = feat

    @torch.no_grad()
    def forward(self, wav):
        return self.feat


def _info(z, feature_shape=(4, 32), reals=None, step=10**9):
    if reals is None:
        reals = torch.randn(z.shape[0], 2, 8192)
    return {"latents": z, "reals": reals, "feature_shape": feature_shape, "global_step": step}


# ---------------------------------------------------------------------------
# LatentCosineDistillLoss (SALAD clip-level cosine distillation)
# ---------------------------------------------------------------------------

def test_distill_shapes_and_range():
    """Loss is a finite scalar in [0, 2]; projector D→proj_dim applied at clip level."""
    B, C, F_lat, T_lat = 2, 16, 4, 32
    z = torch.randn(B, C, F_lat * T_lat)                       # (B, 16, 128) flattened latent
    distill_proj = nn.Linear(C * F_lat, 768)                   # D=64 → 768
    teacher = _StubTeacher(torch.randn(B, 768, 110))           # frame sequence (any T), time-averaged by the loss
    loss = LatentCosineDistillLoss(distill_proj, teacher, weight=1.0)
    out = loss(_info(z, (F_lat, T_lat)))
    assert out.ndim == 0, "loss must be a scalar"
    assert torch.isfinite(out), "loss must be finite"
    assert 0.0 <= out.item() <= 2.0 + 1e-5, f"loss must be in [0, 2], got {out.item()}"


def test_distill_zero_when_aligned():
    """Loss → 0 when z_proj and t_avg point in the same direction."""
    B, C, F_lat, T_lat = 2, 16, 4, 32
    distill_proj = nn.Linear(C * F_lat, 768, bias=False)
    from sage.nn.losses.semantic import standardize_bottleneck
    z = torch.randn(B, C, F_lat * T_lat)
    # Compute the clip projection that the loss will produce, and make the teacher match it.
    with torch.no_grad():
        z4 = z.reshape(B, C, F_lat, T_lat)
        zf = standardize_bottleneck(z4)                        # (B, 64, T)
        z_proj = distill_proj(zf.mean(dim=-1))                 # (B, 768) — same as loss forward
    loss = LatentCosineDistillLoss(distill_proj, _StubTeacher(z_proj.unsqueeze(-1)), weight=1.0, detach_warmup_steps=0)
    out = loss(_info(z, (F_lat, T_lat)))
    assert torch.allclose(out, torch.zeros(()), atol=1e-5), f"aligned loss should be ~0, got {out.item()}"


def test_distill_warmup_gates_encoder_grad():
    """During warmup, encoder gradient is blocked; projector still receives gradient."""
    B, C, F_lat, T_lat = 2, 16, 4, 32
    distill_proj = nn.Linear(C * F_lat, 768)
    teacher = _StubTeacher(torch.randn(B, 768, T_lat))

    # warmup active → latent detached → no encoder grad, projector trains
    z = torch.randn(B, C, F_lat * T_lat, requires_grad=True)
    loss_w = LatentCosineDistillLoss(distill_proj, teacher, weight=1.0, detach_warmup_steps=100)
    loss_w(_info(z, (F_lat, T_lat), step=0)).backward()
    assert z.grad is None, "encoder grad must be None during warmup"
    assert any(p.grad is not None for p in distill_proj.parameters()), "projector must have grad during warmup"

    # no warmup → grad flows to encoder
    distill_proj.zero_grad()
    z2 = torch.randn(B, C, F_lat * T_lat, requires_grad=True)
    loss_n = LatentCosineDistillLoss(distill_proj, teacher, weight=1.0, detach_warmup_steps=0)
    loss_n(_info(z2, (F_lat, T_lat), step=0)).backward()
    assert z2.grad is not None and torch.isfinite(z2.grad).all(), "encoder grad must flow post-warmup"


def test_distill_teacher_no_grad():
    """Even if the stub teacher returns a requires_grad tensor, the loss detaches it
    — so the raw teacher output tensor never receives gradients."""
    B, C, F_lat, T_lat = 2, 16, 4, 32
    distill_proj = nn.Linear(C * F_lat, 768)
    # Simulate a teacher whose output happens to require grad (e.g. not fully frozen).
    feat = torch.randn(B, 768, 50, requires_grad=True)
    teacher = _StubTeacher(feat)
    z = torch.randn(B, C, F_lat * T_lat, requires_grad=True)
    loss = LatentCosineDistillLoss(distill_proj, teacher, weight=1.0, detach_warmup_steps=0)
    out = loss(_info(z, (F_lat, T_lat)))
    out.backward()
    # The loss calls .detach() on the teacher output, so feat must have no grad.
    assert feat.grad is None, "loss must detach teacher output; grad must not reach feat"


def test_distill_clip_embedding_equals_constant_sequence():
    """CLAP returns one (B, 512) vector per clip; a time-constant (B, 512, T) sequence must match it."""
    torch.manual_seed(0)
    B, C, F_lat, T_lat = 2, 16, 4, 32
    distill_proj = nn.Linear(C * F_lat, 512)
    z = torch.randn(B, C, F_lat * T_lat)
    clip = torch.randn(B, 512)
    a = LatentCosineDistillLoss(distill_proj, _StubTeacher(clip), detach_warmup_steps=0)(_info(z))
    b = LatentCosineDistillLoss(distill_proj, _StubTeacher(clip.unsqueeze(-1).expand(B, 512, 7)),
                                detach_warmup_steps=0)(_info(z))
    assert torch.allclose(a, b, atol=1e-6)
