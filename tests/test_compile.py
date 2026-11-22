# ===============================================================
# Tests for sage/compile.py (torch.compile of inference, SAGE `compile=`): the compile= values and their
# normalisation, the in-place compile (module tree, parameters and state dict unchanged, the module's own
# forward traced), back to eager with None, the recompile limits (raised, never lowered), and under CUDA graphs
# ("reduce-overhead") a new graph step per call with every output copied out of the graph's memory, strides
# kept. torch.compile is replaced by the aot_eager backend on CPU (no C++ build, no GPU); the end-to-end checks
# on a SAGE checkpoint are in tests/smoke/test_smoke_inference.py, the CUDA ones run on a GPU node.
# ===============================================================

import pytest
import torch
import torch.nn as nn

from sage import compile as C


@pytest.fixture
def aot_eager(monkeypatch):
    """torch.compile → the aot_eager backend with the mode dropped; returns the keyword arguments of each call."""
    real, calls = torch.compile, []

    def fake(fn=None, *, mode=None, **kwargs):
        calls.append({"mode": mode, **kwargs})
        return real(fn, backend="aot_eager", **kwargs)

    monkeypatch.setattr(torch, "compile", fake)
    yield calls
    torch._dynamo.reset()


class Net(nn.Module):
    """A block whose output is a transposed view, as the encoder's, plus a dict and a list."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(8, 6)
        self.register_buffer("scale", torch.tensor(2.0))

    def forward(self, x):
        y = self.lin(x) * self.scale                               # (B, T, 6)
        return y.transpose(1, 2), {"mean": y.mean()}, [y[:, 0]]    # (B, 6, T) non-contiguous


@pytest.mark.parametrize("value, mode", [(None, None), (False, None), (True, "default"), ("default", "default"),
                                         ("reduce-overhead", "reduce-overhead")])
def test_check_compile_normalises(value, mode):
    assert C.check_compile(value) == mode


@pytest.mark.parametrize("bad", ["max-autotune", "", "Default", 1, 0, "true"])
def test_check_compile_rejects_other_values(bad):
    with pytest.raises(ValueError):
        C.check_compile(bad)


def test_default_is_eager():
    from sage.constants import DEFAULT_COMPILE
    assert DEFAULT_COMPILE is None and C.DEFAULT_COMPILE is None


@pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
def test_compile_in_place_keeps_the_module_and_its_output(mode, aot_eager):
    net = Net().eval()
    x = torch.randn(2, 5, 8)
    ref = net(x)
    params = dict(net.named_parameters())
    state = {k: v.clone() for k, v in net.state_dict().items()}
    C.compile_module(net, mode)
    assert C.is_compiled(net) and aot_eager == [{"mode": mode, "dynamic": False}]
    assert all(p is params[n] for n, p in net.named_parameters())  # same objects: compiled in place
    assert net.state_dict().keys() == state.keys()
    assert all(torch.equal(net.state_dict()[k], v) for k, v in state.items())
    out = net(x)
    assert torch.equal(out[0], ref[0]) and out[0].stride() == ref[0].stride()
    assert torch.equal(out[1]["mean"], ref[1]["mean"]) and torch.equal(out[2][0], ref[2][0])
    C.compile_module(net, None)                                    # back to eager
    assert not C.is_compiled(net)
    assert torch.equal(net(x)[0], ref[0])


def test_compile_traces_a_forward_set_on_the_instance(aot_eager):
    """sage.precision installs its autocast wrapper as an instance ``forward``: the compiled call runs it."""
    net = Net().eval()
    x = torch.randn(2, 5, 8)
    plain = net.forward
    net.forward = lambda x: tuple([plain(x)[0] + 1.0, *plain(x)[1:]])
    C.compile_module(net, "default")
    assert torch.equal(net(x)[0], Net().eval()(x)[0] + 1.0)


def test_check_compile_runs_before_anything_is_compiled(aot_eager):
    net = Net()
    with pytest.raises(ValueError):
        C.compile_module(net, "max-autotune")
    assert not C.is_compiled(net) and aot_eager == []


def test_cuda_graph_calls_open_a_step_and_copy_every_output(monkeypatch):
    """reduce-overhead: each call first marks a new CUDA-graph step, and hands back copies of its outputs (same
    values and strides, own storage), so a later replay that rewrites the graph's memory cannot change them. The
    "graph memory" is simulated by a compiled call that returns views of one persistent tensor."""
    pool = torch.arange(24.0).view(2, 3, 4)
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: lambda *a, **k: (pool.transpose(1, 2), {"p": pool[0]}))
    steps = []
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: steps.append(1))
    net = Net()
    C.compile_module(net, "reduce-overhead")
    first = net(torch.randn(1, 5, 8))
    second = net(torch.randn(1, 5, 8))
    assert len(steps) == 2
    view, d = first
    assert torch.equal(view, pool.transpose(1, 2)) and view.stride() == pool.transpose(1, 2).stride()
    assert view.untyped_storage().data_ptr() != pool.untyped_storage().data_ptr()
    assert d["p"].untyped_storage().data_ptr() != pool.untyped_storage().data_ptr()
    pool.zero_()                                                   # the next replay overwrites the pool
    assert view.abs().sum() > 0 and second[0].abs().sum() > 0


def test_default_mode_hands_back_the_compiled_outputs(monkeypatch):
    """Without CUDA graphs nothing is copied and no step is marked."""
    out = torch.ones(3)
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: lambda *a, **k: out)
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: pytest.fail("step marked"))
    net = Net()
    C.compile_module(net, "default")
    assert net(torch.randn(1, 5, 8)) is out


def test_recompile_limits_are_raised_never_lowered(monkeypatch):
    config = torch._dynamo.config
    monkeypatch.setattr(config, "recompile_limit", 8)
    monkeypatch.setattr(config, "accumulated_recompile_limit", 100_000)
    C.raise_recompile_limit()
    assert config.recompile_limit == C.RECOMPILE_LIMIT
    assert config.accumulated_recompile_limit == 100_000


def test_cli_compile_flag_reaches_the_codec(monkeypatch, tmp_path):
    """scripts/encode_decode.py --compile: off by default, the mode passed to SAGE.from_checkpoint, others refused."""
    import runpy
    import sys
    from pathlib import Path
    from sage.inference import SAGE

    script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "encode_decode.py"))
    seen = []

    def stop(*args, **kwargs):
        seen.append(kwargs.get("compile", "missing"))
        raise RuntimeError("stop after loading")

    monkeypatch.setattr(SAGE, "from_checkpoint", stop)
    base = ["encode_decode.py", "encode", str(tmp_path / "a.wav"), str(tmp_path / "z.pt"), "--ckpt", "x.ckpt"]
    for extra in ([], ["--compile", "default"], ["--compile", "reduce-overhead"]):
        monkeypatch.setattr(sys, "argv", base + extra)
        with pytest.raises(RuntimeError, match="stop after loading"):
            script["main"]()
    assert seen == [None, "default", "reduce-overhead"]
    monkeypatch.setattr(sys, "argv", base + ["--compile", "max-autotune"])
    with pytest.raises(SystemExit):
        script["main"]()


def test_a_change_of_the_tf32_switches_compiles_again(monkeypatch):
    """FlexAttention fixes FP32 / TF32 when compiled: a call under other TF32 flags must not reuse that code."""
    from torch._dynamo.testing import CompileCounterWithBackend
    counter = CompileCounterWithBackend("aot_eager")
    real = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn=None, *, mode=None, **kw: real(fn, backend=counter, **kw))
    net = Net().eval()
    C.compile_module(net, "default")
    saved = torch.backends.cuda.matmul.allow_tf32
    try:
        for tf32 in (False, True, False):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            net(torch.randn(2, 5, 8))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved
        torch._dynamo.reset()
    assert counter.frame_count == 2                                # one per setting, the third call reuses
