import torch
import torch.nn as nn

class ComplexDropout(nn.Module):
    """
    Dropout phase-preserving per tensori complessi.
    La maschera è reale e condivisa su parte reale/immaginaria.
    `broadcast_dims` specifica le dimensioni da comprimere a 1 per ottenere
    varianti tipo token-drop o channel-drop via broadcasting.
    """
    def __init__(self, p: float = 0.1, broadcast_dims: tuple[int, ...] = ()):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("p deve stare in [0,1).")
        self.p = float(p)
        self.broadcast_dims = tuple(broadcast_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = list(x.shape)
        for d in self.broadcast_dims:
            shape[d] = 1
        # maschera reale, stesso keep per R e I
        mask = (torch.rand(shape, device=x.device, dtype=x.real.dtype) < keep)
        mask = mask.to(x.real.dtype) / keep
        return x * mask.to(x.dtype)
