import torch
from torch.nn import Module, ModuleList
import torch.nn as nn
from typing import Sequence, Tuple, Union
from ar_spectra.modules.cplx_attention import CMultiHeadedAttention
from ar_spectra.modules.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from ar_spectra.modules.normed_modules.conv import (
    NormConv2d, NormConvTranspose2d,
    SConv2d, SConvTranspose2d,
)
from ar_spectra.modules.normed_modules.norm import ComplexLayerNorm, ComplexBatchNorm1d
from ar_spectra.modules.activations import get_activation
from ar_spectra.models.autoencoders.abstract_ae import AbstractEncoder, AbastractDecoder
from rich.console import Console
console = Console()

def ok(msg: str) -> None:
    console.print(msg, style="bold green")


def warn(msg: str) -> None:
    console.print(msg, style="bold yellow")


def err(msg: str) -> None:
    console.print(msg, style="bold red")
    

def pair(t):
    if isinstance(t, Sequence) and not isinstance(t, (str, bytes)):
        if len(t) != 2:
            raise ValueError("pair expects a scalar or a length-2 sequence")
        return tuple(t)
    return (t, t)


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
    def __init__(
        self,
        n_heads: int,
        feat_dim: int,
        dropout_rate: float,
        use_complex: bool = False,
        use_rel_pos: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.use_complex = use_complex
        self.use_rel_pos = use_rel_pos

        if use_complex:
            self.norm = ComplexLayerNorm(feat_dim)
            self.attn = CMultiHeadedAttention(
                n_heads, feat_dim, dropout_rate=dropout_rate
            )
            self.pos_enc = None  # o un'altra versione complessa se ti serve
        else:
            self.norm = nn.LayerNorm(2 * feat_dim)
            if use_rel_pos:
                self.attn = RelPositionMultiHeadedAttention(
                    n_heads, 2 * feat_dim, dropout_rate=dropout_rate
                )
                # Positional encoding relativo per il ramo reale
                self.pos_enc = RelPositionalEncoding(d_model=2 * feat_dim)
            else:
                self.attn = MultiHeadedAttention(
                    n_heads, 2 * feat_dim, dropout_rate=dropout_rate
                )
                self.pos_enc = None

        self.norm2 = ComplexLayerNorm(feat_dim)
        self.linear1 = nn.Linear(feat_dim, feat_dim, dtype=torch.complex64)
        self.activation = get_activation("CReLU", is_complex=True)
        self.linear2 = nn.Linear(feat_dim, feat_dim, dtype=torch.complex64)

    def forward(self, x: torch.Tensor, mask=None, debug=False) -> torch.Tensor:
        """
        x: (B, T, D) complesso, dove D = feat_dim = channels * frequency
        """
        D = self.feat_dim

        # self-attention (pre-LN)
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
            pos_emb_r = self.pos_enc(x_r) if self.pos_enc is not None else None

            q = self.norm(x_r)
            k = q
            v = q
            if pos_emb_r is not None:
                x_r = x_r + self.attn(
                    query=q,
                    key=k,
                    value=v,
                    pos_emb=pos_emb_r,
                    mask=mask,
                )
            else:
                x_r = x_r + self.attn(
                    query=q,
                    key=k,
                    value=v,
                    mask=mask,
                )

            # ritorno in C^D
            x = torch.complex(x_r[..., :D], x_r[..., D:])

        # Feed-forward complesso (pre-LN)
        residual = x
        ff_out = self.linear2(self.activation(self.linear1(self.norm2(x))))

        x = residual + ff_out

        return x

class SimpleTransformerEncoder(AbstractEncoder):
    """A simple Transformer encoder for spectrograms."""

    def __init__(
        self,
        input_size: int,
        dim: int = 4,
        inter_dim: int = 256,
        mlp_mult: int = 4,
        n_heads: int = 4,
        depth: int = 6,
        dropout_rate: float = 0.1,
        activation: str = "CReLU",
        image_size: Tuple[int, int] = (256, 32),
        patch_size: Union[int, Tuple[int, int]] = (8, 8),
        is_complex: bool = True,
        use_rel_pos: bool = True,
        ) -> None:
        super().__init__(input_size=input_size, is_complex=is_complex)
        self.input_size = input_size
        self.dim = dim # number of channels after initial conv
        self.n_heads = n_heads
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.inter_dim = inter_dim
        self.mlp_mult = mlp_mult
        self.use_rel_pos = use_rel_pos
        
        image_height, image_width = pair(image_size)
        self.image_size = (image_height, image_width)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = self.dim * patch_height * patch_width   # 32*8*8 = 2048

        self.conv0 = SConv2d(self.input_size, self.input_size, kernel_size=3, stride=1,is_complex=True)
        self.act0 = get_activation(self.activation, is_complex= True)
        self.conv1 = SConv2d(self.input_size, self.dim, kernel_size=5, stride=1, is_complex=True)
        
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.linear_proj = nn.Linear(patch_dim, self.inter_dim, dtype=torch.complex64)
        # Lightning will move this anchor buffer together with the module; we re-use
        # it to place lazily-built attention stacks without manual device handling.
        self.register_buffer("_attn_anchor", torch.empty(0), persistent=False)
        self.attn_blocks = ModuleList([
            Transformer(
                n_heads=self.n_heads,
                feat_dim=self.inter_dim,
                dropout_rate=self.dropout_rate,
                use_rel_pos=self.use_rel_pos,
            )
            for _ in range(self.depth)
        ])
        self._attn_feat = None
        self.norm = ComplexLayerNorm(self.inter_dim)
        self.linear = nn.Linear(self.inter_dim, self.inter_dim, dtype=torch.complex64)

    def forward(self, x: torch.Tensor, skip_attn: bool = False, debug: bool = False) -> torch.Tensor:
        x = self.conv0(x)
        if debug:
            console.log(f"Encoder after conv0 shape: {x.shape}")
        x = self.act0(x)
        x = self.conv1(x)                              # (B, D, F', T')
        if debug:
            console.log(f"Encoder after conv1 shape: {x.shape}")

        x = self.unfold(x)                         # (B, P, L)
        if debug:
            console.log(f"Encoder after unfold shape: {x.shape}")
        x = x.transpose(1, 2).contiguous()         # (B, L, P)
        x = self.linear_proj(x)                    # (B, L, D)
        if debug:
            console.log(f"Encoder after linear_proj shape: {x.shape}")

        if not skip_attn:
            for blk in self.attn_blocks:
                x = blk(x, debug=debug)                       # (B, L, D)
                if debug:
                    console.log(f"Encoder after attention block shape: {x.shape}")

        x = self.linear(self.norm(x))              # (B, L, D)
        if debug:
            console.log(f"Encoder after final norm+linear shape: {x.shape}")

        # se vuoi uscire come (B, D, L) per compatibilità col decoder:
        x = x.transpose(1, 2).contiguous()         # (B, D, L)
        return x

class SimpleTransformerDecoder(AbastractDecoder):
    """
    """

    def __init__(
        self,
        channels: int,                 
        dim: int = 4, 
        inter_dim: int = 128,                     
        n_heads: int = 4,
        depth: int = 6,
        dropout_rate: float = 0.1,
        activation: str = "CReLU",
        mlp_mult: int = 4,
        image_size: Tuple[int, int] = (256, 32),          # = (F1,T1) dopo conv1
        patch_size: Union[int, Tuple[int, int]] = (8, 8), # = encoder.patch_size
        deconv_out_padding: Union[int, tuple] = ((0, 1), (0, 0)),
        final_out_padding: Union[int, tuple] = ((0, 0), (0, 0)),
        is_complex: bool = True,
        use_rel_pos: bool = True,
    ) -> None:
        super().__init__(channels=channels, is_complex=is_complex)
        self.channels = channels
        self.dim = dim
        self.n_heads = n_heads
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.inter_dim = inter_dim
        self.mlp_mult = mlp_mult
        self.use_rel_pos = use_rel_pos

        F1, T1 = pair(image_size)
        self.image_size = (F1, T1)

        ph, pw = pair(patch_size)
        self.patch_size = (ph, pw)

        assert F1 % ph == 0 and T1 % pw == 0, "image_size (F1,T1) deve essere divisibile per patch_size"

        self.L_expected = (F1 // ph) * (T1 // pw)
        self.P = dim * ph * pw  # = dim*patch_area

        # Token stack (uguale all’encoder): (B,L,dim) -> (B,L,dim)
        self.norm_tok = ComplexLayerNorm(inter_dim)
        self.linear_tok = nn.Linear(inter_dim, inter_dim, dtype=torch.complex64)

        self.blocks = nn.ModuleList([
            Transformer(
                n_heads=n_heads,
                feat_dim=inter_dim,
                dropout_rate=dropout_rate,
                use_rel_pos=self.use_rel_pos,
            )
            for _ in range(depth)
        ])

        # unprojection: dim -> P per ogni token
        self.linear_unproj = nn.Linear(inter_dim, self.P, dtype=torch.complex64)

        # Fold: (B, P, L) -> (B, dim, F1, T1)
        self.fold = nn.Fold(output_size=self.image_size, kernel_size=self.patch_size, stride=self.patch_size)

        # Deconv per invertire conv1: (B, dim, F1, T1) -> (B, C, F, T) circa
        self.deconv1 = SConvTranspose2d(
            dim,
            channels,
            kernel_size=5, stride=1,
            out_padding=deconv_out_padding,
            is_complex=True,
        )
        self.act = get_activation(self.activation, is_complex=True)

        # Refinement finale (simmetrico a conv0, opzionale ma utile)
        self.final_conv = SConvTranspose2d(
            channels,
            channels,
            kernel_size=(3,3),
            stride=(1,1),
            out_padding=final_out_padding,
            is_complex=True,
        )

    def forward(self, z: torch.Tensor, skip_attn: bool = False, debug: bool = False) -> torch.Tensor:
        # z: (B, inter_dim, L) where inter_dim matches the token size used in the encoder
        if debug:
            console.log(f"Decoder input (latent) shape: {z.shape}")

        B, D, L = z.shape
        assert D == self.inter_dim, f"Decoder attende inter_dim={self.inter_dim}, ma ha ricevuto D={D}"
        assert L == self.L_expected, f"Decoder attende L={self.L_expected} (da image_size/patch), ma ha ricevuto L={L}"

        x = z.transpose(1, 2).contiguous()  # (B, L, dim)

        # norm+linear token-wise (feature-last)
        x = self.linear_tok(self.norm_tok(x))  # (B, L, dim)
        if debug:
            console.log(f"Decoder after token norm+linear shape: {x.shape}")

        # Transformer blocks
        if not skip_attn:
            for blk in self.blocks:
                x = blk(x, debug=debug)  # (B, L, dim)

            if debug:
                console.log(f"Decoder after attention stack shape: {x.shape}")

        # unproject tokens -> patch vectors
        x = self.linear_unproj(x)                # (B, L, P)
        x = x.transpose(1, 2).contiguous()       # (B, P, L)
        if debug:
            console.log(f"Decoder after unprojection shape: {x.shape}")

        # Fold back to feature map (B, dim, F1, T1)
        fm = self.fold(x)  # (B, dim, F1, T1)
        
        if debug:
            console.log(f"Decoder after fold shape: {fm.shape}")

        # Invert conv1
        y = self.deconv1(fm)
        y = self.act(y)
        y = self.final_conv(y)

        if debug:
            console.log(f"Decoder output shape: {y.shape}")

        return y


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    dummy = torch.zeros((32, 2, 1025, 128), dtype=torch.complex64).to(device)

    # Encoder: input_size matches dummy channels (2).
    encoder = SimpleTransformerEncoder(
        input_size=2,
        dim=4,
        n_heads=4,
        depth=2,
        dropout_rate=0.1,
        patch_size=(16, 16),
        activation="CRelu",
        inter_dim=128,
    ).to(device)

    # Decoder: channels=2 to match original input channels.
    decoder = SimpleTransformerDecoder(
        channels=2,
        dim=4,
        n_heads=4,
        depth=2,
        dropout_rate=0.1,
        activation="CRelu",
        image_size=(256, 32),     # = shape dopo conv1
        patch_size=(16, 16),
        inter_dim=128,
        deconv_out_padding=((0, 1), (0, 0)),  # lascia come avevi se ti serve recuperare 1025
    ).to(device)

    with torch.no_grad():
        latent = encoder(dummy, debug=True)
        recon = decoder(latent, debug=True)

    print(f"Input shape:   {tuple(dummy.shape)}")
    print(f"Latent shape:  {tuple(latent.shape)}")
    print(f"Output shape:  {tuple(recon.shape)}")
