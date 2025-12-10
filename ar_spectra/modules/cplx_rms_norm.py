import torch
from torch import nn
from typing import Union, List

class ComplexRMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-8,
        elementwise_affine: bool = True,
    ):
        """
        Complex-valued RMSNorm that normalizes jointly over Re/Im.

        Args:
            normalized_shape: last dimension(s) to normalize over,
                exactly like nn.LayerNorm.
            eps: numerical epsilon.
            elementwise_affine: if True, learn a real scale g with
                shape = normalized_shape, applied to both Re and Im.
        """
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            # real gain, shared between Re/Im
            self.weight = nn.Parameter(
                torch.ones(*normalized_shape, dtype=torch.float32)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: complex tensor, shape [..., *normalized_shape]
        """
        if not torch.is_complex(x):
            raise TypeError("ComplexRMSNorm expects a complex tensor as input")

        # assicuriamoci che le ultime dims corrispondano
        assert x.shape[-len(self.normalized_shape):] == self.normalized_shape, (
            f"Expected trailing dimensions {self.normalized_shape}, "
            f"got {x.shape[-len(self.normalized_shape):]}"
        )

        # assi su cui fare la RMS (come LayerNorm)
        axes = tuple(range(-len(self.normalized_shape), 0))

        # |x|^2 = Re^2 + Im^2
        mag_sq = x.real.pow(2) + x.imag.pow(2)

        # mean su feature, poi sqrt
        rms = torch.sqrt(mag_sq.mean(dim=axes, keepdim=True) + self.eps)

        # normalizzazione
        x_norm = x / rms

        if self.elementwise_affine:
            # weight is real, it needs to be broadcasted over the initial dims
            # ex: x.shape = [B, L, D], normalized_shape = [D]
            # -> shape weight_view = [1, 1, D]
            shape = (1,) * (x.ndim - len(self.normalized_shape)) + tuple(self.normalized_shape)
            w = self.weight.view(shape)  # real
            x_norm = x_norm * w          # scale Re and Im together
        return x_norm
