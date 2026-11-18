# ===============
# torch.compile of the generator in training (trainer.trainer.compile_generator). The encoder and the
# decoder are compiled in place (nn.Module.compile): same module tree, parameters and state_dict keys, so
# the EMA, the checkpoints, a resume and +init_from see the plain model. A compile failure, or a module
# that reaches the recompile limit, is reported loudly and the call runs eagerly instead.
# ===============
from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from torch import nn
from torch._dynamo.exc import FailOnRecompileLimitHit, TorchDynamoException

from sage.constants import GENERATOR_COMPILE_MODE, GENERATOR_RECOMPILE_LIMIT
from sage.utils.console import err, ok, warn

GENERATOR_MODULES = ("encoder", "decoder")      # SAGEAutoencoder attributes compiled by compile_generator
_MAX_ERROR_CHARS = 600                          # compile errors can be pages long: the console gets the head


class CompiledCall:
    """What nn.Module.compile stores as ``module._compiled_call_impl``, with an eager fallback.

    nn.Module.__call__ runs ``_compiled_call_impl`` when it is set, so the module keeps its place in the
    model and the engine's ``autoencoder.encode/decode`` reach the compiled code unchanged.

    - The recompile limit (GENERATOR_RECOMPILE_LIMIT graphs per forward code object: SAGEEncoder and
      SAGEDecoder each have their own) raises at the first input signature beyond it: the error is printed
      once and that call runs eagerly. Later calls keep the graphs already built; dynamo runs any other
      new signature eagerly.
    - A compile error (dynamo or inductor) is printed and the module runs eagerly from then on. An error
      of the model itself then raises again from the eager call.
    Both hold whatever the global dynamo config says: some libraries turn on ``suppress_errors`` at import
    (stable_audio_3, the SAME baseline), which would hide the errors. Errors in the compilation of the
    backward graph (built at the first backward) are not caught.
    """

    def __init__(self, module: nn.Module, name: str, compiled: Callable[..., Any]):
        self.module = module            # the module compiled in place
        self.name = name                # its name in the autoencoder, for the messages
        self.compiled = compiled        # torch.compile(module._call_impl, ...)
        self.limit_hit = False          # True once a call reached the recompile limit

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            with torch._dynamo.config.patch(recompile_limit=GENERATOR_RECOMPILE_LIMIT, suppress_errors=False,
                                            fail_on_recompile_limit_hit=not self.limit_hit):
                return self.compiled(*args, **kwargs)
        except FailOnRecompileLimitHit:
            self.limit_hit = True
            err(f"compile_generator: the {self.name} reached the recompile limit ({GENERATOR_RECOMPILE_LIMIT} "
                "graphs). Inputs with a new shape, grad mode or train/eval mode now run eagerly (same values, "
                "slower); TORCH_LOGS=recompiles lists the reasons.", prefix="COMPILE")
        except TorchDynamoException as exc:
            self.module._compiled_call_impl = None
            err(f"compile_generator: torch.compile of the {self.name} failed "
                f"({type(exc).__name__}: {str(exc)[:_MAX_ERROR_CHARS]}). The {self.name} runs eagerly from now on.",
                prefix="COMPILE")
        return self.module._call_impl(*args, **kwargs)


def compile_in_place(module: nn.Module, name: str, *, backend: str = "inductor") -> bool:
    """Compile one module in place (static shapes, GENERATOR_COMPILE_MODE). Returns False if torch.compile
    refused it (the module stays eager)."""
    mode = GENERATOR_COMPILE_MODE if backend == "inductor" else None     # modes are inductor's
    try:
        module.compile(backend=backend, mode=mode, dynamic=False)
    except Exception as exc:                    # e.g. a Python version dynamo does not support
        err(f"compile_generator: torch.compile of the {name} refused ({type(exc).__name__}: {exc}); "
            "it stays eager.", prefix="COMPILE")
        return False
    module._compiled_call_impl = CompiledCall(module, name, module._compiled_call_impl)
    return True


def compile_generator(autoencoder: nn.Module, *, backend: str = "inductor") -> tuple[str, ...]:
    """Compile the encoder and the decoder of a SAGEAutoencoder in place; return the names compiled.

    Call it after the weights are loaded (+init_from) and before training: compilation itself is lazy (the
    first call of each input signature, a few minutes in all on a GPU). Not bit-exact with the eager model:
    inductor fuses and reorders float operations, and draws the drop-path masks from its own RNG stream.

    Args:
        autoencoder: The SAGEAutoencoder; its other parts (bottleneck, STFT, pre-transform) stay eager.
        backend:     torch.compile backend ("inductor"; tests use "aot_eager" on CPU).
    """
    done = tuple(name for name in GENERATOR_MODULES
                 if getattr(autoencoder, name, None) is not None
                 and compile_in_place(getattr(autoencoder, name), name, backend=backend))
    if done:
        mode = f", mode={GENERATOR_COMPILE_MODE}" if backend == "inductor" else ""
        ok(f"torch.compile of the {' and '.join(done)} in place (backend={backend}{mode}, static shapes, "
           f"recompile limit {GENERATOR_RECOMPILE_LIMIT}); compiled at their first calls.", prefix="COMPILE")
    return done


def compile_generator_requested(trainer_cfg: Mapping[str, Any]) -> bool:
    """Whether trainer.trainer asks for compile_generator, also through its deprecated alias ``compile``."""
    requested = bool(trainer_cfg.get("compile_generator", False))
    if bool(trainer_cfg.get("compile", False)):
        warn("trainer.trainer.compile is deprecated: it wrapped the whole autoencoder, which has no forward, "
             "so it compiled nothing and only renamed the checkpoint keys to _orig_mod.*. It now turns on "
             "trainer.trainer.compile_generator (encoder and decoder compiled in place, mode default); "
             "set that key instead.", prefix="COMPILE")
        requested = True
    return requested
