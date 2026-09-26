"""SAGE: a Swin V2 variational autoencoder for stereo music.

    >>> from sage import SAGE
    >>> codec = SAGE.from_checkpoint("SAGE_FTe992.ckpt")
    >>> z = codec.encode(wav)                        # (B, 2, N) waveform -> (B, 16, N / 512) latent
    >>> rec = codec.decode(z, target_length=wav.shape[-1])

Layout: ``sage.model`` is the SAGE architecture, built from the generic building blocks of
``sage.nn``; ``sage.training`` holds the two training phases; ``sage.inference`` loads checkpoints.
"""

__version__ = "0.1.0"


def __getattr__(name):
    # Lazy: `import sage.nn...` must not pull in the whole inference stack.
    if name in ("SAGE", "EuleroEncodeDecode"):
        from sage import inference
        return getattr(inference, name)
    raise AttributeError(f"module 'sage' has no attribute {name!r}")
