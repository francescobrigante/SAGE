# ===============
# sage.precision: the TF32 switches are set for one block and restored, the bf16 autocast runs a block's
# network in bfloat16 and hands FP32 back, and None / unknown precisions behave as documented.
# ===============
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sage import precision as P


@pytest.fixture
def flags():
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _flags():
    return torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32


@pytest.mark.parametrize("before", [(False, True), (True, False), (False, False)])
@pytest.mark.parametrize("precision,inside", [("tf32", (True, True)), ("bf16", (True, True)), ("fp32", (False, False))])
def test_tf32_flags_set_inside_and_restored(flags, before, precision, inside):
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before
    with P.tf32_flags(precision):
        assert _flags() == inside
    assert _flags() == before


def test_tf32_flags_restored_after_an_exception(flags):
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = False, True
    with pytest.raises(RuntimeError):
        with P.tf32_flags("tf32"):
            raise RuntimeError
    assert _flags() == (False, True)


def test_none_leaves_the_flags_alone(flags):
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = False, True
    with P.tf32_flags(None):
        assert _flags() == (False, True)


@pytest.mark.parametrize("bad", ["fp16", "TF32", "high", ""])
def test_unknown_precision_raises(bad):
    with pytest.raises(ValueError):
        P.check_precision(bad)
    assert P.check_precision(None) is None and P.DEFAULT_PRECISION == "tf32"


class _Block(nn.Module):
    """Returns a tensor, a tuple with a dict inside, the dtype the linear layer saw."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x):
        y = self.lin(x)
        return y, {"y": y, "dtype": y.dtype}


def test_set_autocast_runs_bf16_and_returns_fp32():
    block, x = _Block(), torch.randn(4, 8)
    ref, _ = block(x)
    P.set_autocast(block, torch.bfloat16)
    y, info = block(x)
    assert info["dtype"] == torch.bfloat16                      # the network ran in bfloat16
    assert y.dtype == info["y"].dtype == torch.float32          # the caller gets FP32 back
    assert not torch.equal(y, ref) and torch.allclose(y, ref, atol=5e-2)
    assert all(p.dtype == torch.float32 for p in block.parameters())
    assert set(block.state_dict()) == {"lin.weight", "lin.bias"}
    P.set_autocast(block, None)                                 # back to the class forward
    assert "forward" not in block.__dict__ and torch.equal(block(x)[0], ref)


def test_set_autocast_twice_does_not_stack():
    block, x = _Block(), torch.randn(4, 8)
    P.set_autocast(block, torch.bfloat16)
    once = block(x)[0]
    P.set_autocast(block, torch.bfloat16)
    assert torch.equal(block(x)[0], once)
