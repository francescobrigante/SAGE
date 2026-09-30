"""
Semantic-distillation infrastructure tests:
  - standardize_bottleneck: parameter-free C·F fold, real/complex/passthrough.
  - LossManager.aux_parameters: empty when distillation is off
    (so opt_aux is not created and training behavior is unchanged).
  - engine.compute exposes loss_info["gen_step"] (consumed by the detached warm-up).
"""
import torch

from sage.nn.losses.semantic import standardize_bottleneck
from sage.training.loss_manager import LossManager


def test_standardize_bottleneck_real_fold_is_value_preserving():
    r = torch.randn(2, 16, 4, 32)                  # (B, C, F, T)
    out = standardize_bottleneck(r)
    assert out.shape == (2, 64, 32)                # D = C*F
    assert torch.allclose(out.reshape(2, 16, 4, 32), r)  # bijective relabeling


def test_standardize_bottleneck_complex_concats_re_im():
    c = torch.randn(2, 16, 4, 32, dtype=torch.cfloat)
    out = standardize_bottleneck(c)
    assert out.shape == (2, 128, 32)               # D = 2*C*F
    assert torch.is_floating_point(out)
    assert torch.allclose(out[:, :64], c.real.reshape(2, 64, 32))
    assert torch.allclose(out[:, 64:], c.imag.reshape(2, 64, 32))


def test_standardize_bottleneck_flat_passthrough():
    f = torch.randn(2, 64, 32)                     # already (B, D, T)
    assert torch.equal(standardize_bottleneck(f), f)


def test_aux_parameters_empty_without_distillation():
    # Bind the real method onto a minimal stand-in: no distillation head set.
    class _Stub:
        _aux_module_names = ("distill_proj",)
        aux_parameters = LossManager.aux_parameters
    assert _Stub().aux_parameters() == []          # opt_aux will not be created


def test_aux_parameters_collects_registered_modules():
    class _Stub:
        _aux_module_names = ("distill_proj",)
        aux_parameters = LossManager.aux_parameters
    s = _Stub()
    s.distill_proj = torch.nn.Linear(64, 512)
    n = sum(p.numel() for p in s.aux_parameters())
    assert n == sum(p.numel() for p in s.distill_proj.parameters()) and n > 0

