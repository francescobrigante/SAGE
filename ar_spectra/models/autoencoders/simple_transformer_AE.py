import torch
from torch.nn import Module, ModuleList
import torch.nn as nn
from typing import Union
from ar_spectra.modules.cplx_attention import CMultiHeadedAttention
from ar_spectra.modules.attention import RelPositionMultiHeadedAttention
from ar_spectra.modules.normed_modules.conv import (
    NormConv2d, NormConvTranspose2d,
    SConv2d, SConvTranspose2d,
)
from ar_spectra.modules.normed_modules.norm import ComplexLayerNorm, ComplexBatchNorm1d
from ar_spectra.modules.activations import get_activation
from rich.console import Console
console = Console()

def ok(msg: str) -> None:
    console.print(msg, style="bold green")


def warn(msg: str) -> None:
    console.print(msg, style="bold yellow")


def err(msg: str) -> None:
    console.print(msg, style="bold red")


class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding per RelPositionMultiHeadedAttention.

    d_model: dimensione del modello usata dall'attenzione (qui: 2*feat_dim).
    max_len: massimo numero di frame T previsto.
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Embedding per offset in [-max_len+1, ..., max_len-1]
        self.emb = nn.Embedding(2 * max_len - 1, d_model)
        self.center = max_len - 1  # offset 0 -> indice center

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model) oppure (B, T, qualunque_dim), usiamo solo T e il device.
        Ritorna: pos_emb di shape (B, 2*T-1, d_model).
        """
        B, T = x.size(0), x.size(1)

        # offset relativi: -(T-1), ..., 0, ..., +(T-1)
        offsets = torch.arange(-(T - 1), T, device=x.device)  # (2T-1,)

        # mappiamo gli offset in indici di embedding [0, 2*max_len-2]
        idx = offsets + self.center            # ancora (2T-1,)
        pos = self.emb(idx)                    # (2T-1, d_model)

        # broadcast sul batch
        pos = pos.unsqueeze(0).expand(B, -1, -1)  # (B, 2T-1, d_model)
        return pos



class Transformer(Module):
    def __init__(self, n_heads: int, feat_dim: int, dropout_rate: float, use_complex: bool = False) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.use_complex = use_complex

        if use_complex:
            self.norm = ComplexLayerNorm(feat_dim)
            self.attn = CMultiHeadedAttention(
                n_heads, feat_dim, dropout_rate=dropout_rate
            )
            self.pos_enc = None  # o un'altra versione complessa se ti serve
        else:
            self.norm = nn.LayerNorm(2 * feat_dim)
            self.attn = RelPositionMultiHeadedAttention(
                n_heads, 2 * feat_dim, dropout_rate=dropout_rate
            )
            # Positional encoding relativo per il ramo reale
            self.pos_enc = RelPositionalEncoding(d_model=2 * feat_dim)

        self.norm2 = ComplexLayerNorm(feat_dim)
        self.linear1 = nn.Linear(feat_dim, feat_dim, dtype=torch.complex64)
        self.activation = get_activation("CRelu", is_complex=True)
        self.linear2 = nn.Linear(feat_dim, feat_dim, dtype=torch.complex64)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        x: (B, T, D) complesso, dove D = feat_dim = channels * frequency
        """
        D = self.feat_dim

        # --- Self-attention (pre-LN) ---
        if self.use_complex:
            # ramo complesso puro (se la tua C-attn non usa pos_emb)
            q = self.norm(x)
            k = q
            v = q
            x = x + self.attn(query=q, key=k, value=v, mask=mask)
        else:
            # CAC: C^D -> R^{2D}
            x_r = torch.cat([x.real, x.imag], dim=-1)  # (B, T, 2D)

            # pos_emb reale in R^{B x (2T-1) x 2D}
            pos_emb_r = self.pos_enc(x_r)  # usa solo B,T,device

            q = self.norm(x_r)
            k = q
            v = q

            x_r = x_r + self.attn(
                query=q,
                key=k,
                value=v,
                pos_emb=pos_emb_r,
                mask=mask,
            )

            # ritorno in C^D
            x = torch.complex(x_r[..., :D], x_r[..., D:])

        # --- Feed-forward complesso (pre-LN) ---
        residual = x
        x = self.linear1(self.norm2(x))
        x = self.activation(x)
        x = self.linear2(x)
        x = residual + x

        return x



class SimpleTransformerEncoder(Module):
    """A simple Transformer encoder for spectrograms."""

    def __init__(
        self,
        input_size: int,
        dim: int = 256,
        n_heads: int = 4,
        depth: int = 6,
        dropout_rate: float = 0.1,
        activation: str = "CRelu",) -> None:
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.n_heads = n_heads
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        self.conv0 = SConv2d(self.input_size, self.input_size, kernel_size=3, stride=1,is_complex=True)
        self.act0 = get_activation(self.activation, is_complex= True)
        self.conv1 = SConv2d(self.input_size, self.dim, kernel_size=8, stride=4, is_complex=True)
        # Lightning will move this anchor buffer together with the module; we re-use
        # it to place lazily-built attention stacks without manual device handling.
        self.register_buffer("_attn_anchor", torch.empty(0), persistent=False)
        self.attn = None
        self._attn_feat = None
        self.norm = ComplexLayerNorm(1024)
        self.linear = nn.Linear(1024, 1024, dtype=torch.complex64)

    def _ensure_attn(self, feat_dim: int):
        """Instantiate attention stack when feature size changes (Lightning handles device)."""
        if self.attn is None or self._attn_feat != feat_dim:
            self._attn_feat = feat_dim
            transformer = Transformer(
                n_heads=self.n_heads,
                feat_dim=feat_dim,
                dropout_rate=self.dropout_rate,)
            self.attn = transformer.to(self._attn_anchor.device)

    def forward(self, x: torch.Tensor, skip_attn: bool = False) -> torch.Tensor:
        x = self.conv0(x)                              # (B, dim, F, T)
        #print(f"Encoder after conv0 shape: {x.shape}")
        x = self.act0(x)
        x = self.conv1(x)                              # (B, dim, F, T)
        #print(f"Encoder after conv1 shape: {x.shape}")
        b, c, f, t = x.shape
        feat_dim = c * f                               # dim * freq
        self._ensure_attn(feat_dim)

        x = x.view(b, feat_dim, t).transpose(1, 2)     # (B, T, dim*freq)
        if not skip_attn:
            x = self.attn(x)
            #print(f"After attention block: {x.shape}")
        x = self.linear(self.norm(x))
        x = x.transpose(1, 2).contiguous()             # (B, dim*freq, T)
        #print(f"Encoder after attention stack shape: {x.shape}")
        return x

class SimpleTransformerDecoder(Module):
    """Simple Transformer decoder symmetric to ``SimpleTransformerEncoder`` for complex spectrograms.

    Args:
        output_size: Number of complex channels to reconstruct (matches encoder input_size).
        dim: Hidden dimension used by the encoder/decoder attention stack.
        n_heads: Number of attention heads.
        depth: Number of Transformer blocks.
        dropout_rate: Dropout rate applied in attention.
        activation: Name of the activation function after the transpose convolution.
    """
    def __init__(
        self,
        input_size: int,
        channels: int = "auto",
        n_heads: int = 4,
        depth: int = 6,
        dropout_rate: float = 0.1,
        activation: str = "CRelu",
        post_conv_kernel: int = (1, 1),
        post_conv_stride: int = (1, 1),
        post_conv_out_padding: Union[int, tuple] = ((0, 0), (0, 1)),
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.dim = input_size
        self.n_heads = n_heads
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.channels = channels
        self.activation = activation
        self.post_conv_kernel = post_conv_kernel
        self.post_conv_stride = post_conv_stride
        self.post_conv_out_padding = post_conv_out_padding

        self.register_buffer("_attn_anchor", torch.empty(0), persistent=False)
        self.attn = None
        self._attn_feat = None
        self.output_channels = self.dim if self.channels == "auto" else self.channels
        self.norm = ComplexLayerNorm(1024)
        self.linear = nn.Linear(1024, 1024, dtype=torch.complex64)
        
        # out_padding is set dynamically in forward to hit the requested target shape
        # Add frequency out_padding to recover the 1025 bins dropped in the encoder
        self.deconv = SConvTranspose2d(
            self.input_size,
            self.dim,
            kernel_size=8,
            stride=4,
            out_padding=((0, 1), (0, 0)),
            is_complex=True,
        )
        self.act = get_activation(self.activation, is_complex= True)
        # out_padding on the time axis recovers the single step dropped by the encoder
        self.final_conv = SConvTranspose2d(
            self.dim,
            self.output_channels,
            kernel_size=self.post_conv_kernel,
            stride=self.post_conv_stride,
            out_padding=self.post_conv_out_padding,
            is_complex=True,
        )

    def _ensure_attn(self, feat_dim: int):
        """Instantiate attention stack when feature size changes (Lightning handles device)."""
        if self.attn is None or self._attn_feat != feat_dim:
            self._attn_feat = feat_dim
            transformer = Transformer(
                n_heads=self.n_heads,
                feat_dim=feat_dim,
                dropout_rate=self.dropout_rate,)
            self.attn = transformer.to(self._attn_anchor.device)

    def forward(self, x, skip_attn=False) -> torch.Tensor:
        b, feat_dim, t = x.shape                       # feat_dim = dim * freq
        #print(f"Decoder input shape: {x.shape}")
        freq = feat_dim // self.dim
        self._ensure_attn(feat_dim)

        x = x.transpose(1, 2).contiguous()             # (B, T, dim*freq)
        x = self.linear(self.norm(x))
        #print(f"Decoder before attn shape: {x.shape}")
        if not skip_attn:
            x = self.attn(x)
        
        x = x.transpose(1, 2).contiguous()             # (B, dim*freq, T)
        #print(f"Decoder after attention stack shape: {x.shape}")
        x = x.view(b, self.dim, freq, t)               # (B, dim, F, T)
        #print(f"Decoder reshaped to: {x.shape}")

        x = self.deconv(x)
        #print(f"After deconv shape: {x.shape}")
        x = self.act(x)
        x = self.final_conv(x)
        #print(f"After final conv shape: {x.shape}")
        return x

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    dummy = torch.zeros((32, 2, 1025, 32), dtype=torch.complex64).to(device)

    # Encoder: input_size matches dummy channels (2).
    encoder = SimpleTransformerEncoder(
        input_size=2,
        dim=4,
        n_heads=4,
        depth=2,
        dropout_rate=0.1,
        activation="CRelu",
    ).to(device)

    # Decoder: channels=2 to match original input channels.
    decoder = SimpleTransformerDecoder(
        input_size=4,
        channels=2,
        n_heads=4,
        depth=2,
        dropout_rate=0.1,
        activation="CRelu",
        post_conv_out_padding=((0, 0), (0, 1)),
    ).to(device)

    with torch.no_grad():
        latent = encoder(dummy)
        recon = decoder(latent)

    print(f"Input shape:   {tuple(dummy.shape)}")
    print(f"Latent shape:  {tuple(latent.shape)}")
    print(f"Output shape:  {tuple(recon.shape)}")
