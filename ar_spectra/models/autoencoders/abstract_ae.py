from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple
import torch.nn as nn

class AbstractEncoder(nn.Module, ABC):
    """Minimal encoder contract.

    Encoders are expected to accept a spectrogram-like tensor and return
    ``(latents, info)`` where ``latents`` is the encoded representation and
    ``info`` is a dictionary that may contain shape metadata (e.g.,
    ``{"feature_shape": (freq, time)}``) used by decoders that need to
    rebuild spatial layouts. Returning just ``latents`` is allowed for legacy
    modules; the container will wrap an empty info dict in that case.
    """

    def __init__(self, *, input_size: int, is_complex: bool):
        super().__init__()
        self.input_size = input_size
        self.is_complex = is_complex
        # Optional spatial/temporal shape captured during forward (e.g., 2D feature map)
        self.last_feature_shape: Optional[Tuple[int, int]] = None


class AbastractDecoder(nn.Module, ABC):
    """Minimal decoder contract mirroring :class:`AbstractEncoder`.

    Decoders may optionally read ``feature_shape`` to reshape tokens back into
    2D feature maps (e.g., patch-based transformers). The autoencoder
    container populates this when the encoder reports it in ``info``.
    """

    def __init__(self, *, channels: int, is_complex: bool):
        super().__init__()
        self.channels = channels
        self.is_complex = is_complex
        # Decoders that need spatial shape can read this (populated by AutoEncoder.encode)
        self.feature_shape: Optional[Tuple[int, int]] = None

    


