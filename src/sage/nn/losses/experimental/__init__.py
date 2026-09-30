"""Losses explored while developing SAGE and not used by the paper recipes.

None of these is built by the recipes in ``configs/experiment``: the paper loss (eq. 2)
is only L_STFT, L_mel, L_SD, KL, the CLAP distillation and the adversarial and
feature-matching terms. They are kept for ablations and future work, and enter a run
through ``trainer.loss_config.extra``, a list of entries such as::

    extra:
      - name: stereo_coh              # key in the logged loss breakdown
        weight: 0.1
        input_key: decoded            # prediction: decoded / sp_decoded / sp_decoded_linear
        target_key: reals             # target: reals / encoder_input; omit for a loss of the input alone
        loss:
          _target_: sage.nn.losses.experimental.StereoCoherenceLoss
          fft_sizes: [2048, 1024, 512]

``decoded`` and ``reals`` are waveforms ``(B, C, T)``; ``sp_decoded`` and ``encoder_input``
are the power-compressed complex-as-channels spectrograms the model sees, and
``sp_decoded_linear`` is the decoded spectrogram before power compression.
"""
from sage.nn.losses.experimental.perceptual import HubertLoss
from sage.nn.losses.experimental.signal import StereoCoherenceLoss
from sage.nn.losses.experimental.spectral import (
    ComplexSpectralConvergence,
    InstantaneousFrequencyGroupDelayLoss,
    MRSTFTSame,
    MultiResolutionSpectrogramLoss,
    MultiResSpectralConvergence,
    NormalizedComplexDistanceLoss,
    PerceptualComplexMSE,
    PhaseCosineDistance,
    SideComplexMSE,
    SpectralContrastLoss,
    STFTConsistencyLoss,
    adaptive_log_mag,
    get_k_weight_curve,
)
