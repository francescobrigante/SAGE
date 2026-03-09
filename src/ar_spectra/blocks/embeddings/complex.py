# =============================================================================
# Specific positional encodings for complex tensors, preserving phase and magnitude properties.
# =============================================================================

from .positional import _pre_hook
import math
import torch
import torch.nn as nn
from ar_spectra.blocks.layers import ComplexDropout

class ComplexPositionalEncoding(nn.Module):
    """1D absolute complex-valued sinusoidal positional encoding.

    Args:
        d_model: Channel dimension for the latent tokens.
        dropout_rate: Dropout probability applied after adding PE.
        max_len: Minimum length used to preallocate the PE table.
        reverse: If True, positions are indexed from the sequence end.

    Input:  x ∈ ℂ^{B×L×d_model}
    Output: x' = sqrt(d_model) * x + pe[:, :L]

    For each position ``t`` and channel ``j``:
        ω_j = 10000^{-(j / d_model)}
        θ_{t,j} = t * ω_j
        pe_{t,j} = sin(θ_{t,j}) + i · cos(θ_{t,j})

    The encoding preserves analyticity by pairing sine (real) and cosine (imaginary)
    components in complex form, mirroring the standard Transformer PE but lifted to ℂ.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000, reverse: bool = False):
        super().__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = ComplexDropout(p=dropout_rate)
        self.pe: torch.Tensor | None = None
        # initialize a minimal buffer
        self.extend_pe(torch.zeros(1, max_len, dtype=torch.complex64))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x: torch.Tensor) -> None:
        """(Re)build the complex PE table if the current buffer is too short.

        Args:
            x: Complex or real tensor with shape (B, L, d_model) or (1, L).
        """
        seq_len = x.size(1)

        if self.pe is not None:
            if self.pe.size(1) >= seq_len:
                # only cast dtype/device when needed
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        # base real dtype (e.g., float32) for sin/cos construction
        if x.is_complex():
            real_dtype = x.real.dtype
        else:
            real_dtype = x.dtype

        # positions
        if self.reverse:
            position = torch.arange(
                seq_len - 1, -1, -1, dtype=real_dtype, device=x.device
            ).unsqueeze(1)  # (L,1)
        else:
            position = torch.arange(
                0, seq_len, dtype=real_dtype, device=x.device
            ).unsqueeze(1)  # (L,1)

        # log-spaced frequencies across channels
        dim = torch.arange(0, self.d_model, dtype=real_dtype, device=x.device)
        div_term = torch.exp(-math.log(10000.0) * dim / self.d_model)  # (d_model,)

        # angles shape (L, d_model)
        angles = position * div_term.unsqueeze(0)

        # build real and imaginary parts
        pe_real = torch.sin(angles)      # (L, d_model)
        pe_imag = torch.cos(angles)      # (L, d_model)

        pe = torch.complex(pe_real, pe_imag).unsqueeze(0)  # (1, L, d_model)
        self.pe = pe.to(dtype=x.dtype, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex positional encoding.

        Args:
            x: Tensor of shape (B, L, d_model), complex or real.

        Returns:
            Complex tensor with positional encoding added. Real inputs are
            promoted to complex to preserve compatibility with complex attention.
        """
        if not x.is_complex():
            # optional: promote to complex for consistency with complex attention
            x = x.to(torch.complex64)

        self.extend_pe(x)
        pe = self.pe[:, : x.size(1), :]  # (1, L, d_model)

        x = x * self.xscale + pe
        x = self.dropout(x)
        return x

class ComplexScaledPositionalEncoding(ComplexPositionalEncoding):
    """Complex sinusoidal positional encoding with learnable scaling.

    Applies ``x' = x + α · pe[:, :L]`` where ``α`` is a learnable real scalar.
    Inherits the complex sinusoidal construction from ``ComplexPositionalEncoding``
    while allowing the model to modulate the positional signal strength.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000, reverse: bool = False):
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len, reverse=reverse)
        # real alpha scales the complex PE amplitude
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.alpha.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            x = x.to(torch.complex64)

        self.extend_pe(x)
        pe = self.pe[:, : x.size(1), :]  # (1, L, d_model)

        # alpha è reale → viene broadcastato e castato a complesso in modo implicito
        alpha_c = self.alpha.to(x.dtype)
        x = x + alpha_c * pe
        x = self.dropout(x)
        return x
