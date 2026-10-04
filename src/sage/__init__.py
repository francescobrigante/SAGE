"""SAGE: a Swin V2 variational autoencoder for stereo music.

    >>> from sage import SAGE
    >>> codec = SAGE.from_checkpoint("SAGE_FTe992.ckpt")
    >>> rec = codec.reconstruct(wav)                 # (B, 2, N) waveform at 44.1 kHz -> same shape
    >>> padded, n = codec.pad(wav)
    >>> z = codec.encode(padded)                     # (B, 16, frames) latent, one frame per 512 samples

Layout: ``sage.model`` is the SAGE architecture, built from the generic building blocks of
``sage.nn``; ``sage.training`` holds the two training phases; ``sage.inference`` loads checkpoints.
"""

__version__ = "0.1.1"


def __getattr__(name):
    # Lazy: `import sage.nn...` must not pull in the whole inference stack.
    if name == "SAGE":
        from sage import inference
        return getattr(inference, name)
    raise AttributeError(f"module 'sage' has no attribute {name!r}")
