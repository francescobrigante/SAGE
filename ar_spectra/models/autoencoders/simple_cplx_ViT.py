import torch
from torch import nn
from torch.nn import Module, ModuleList, Unfold

from einops import repeat as einops_repeat
import logging
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from collections.abc import Sequence
import torch
from torch import nn
from ar_spectra.models.autoencoders.AbsEncoder import AbsEncoder
from ar_spectra.modules.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from ar_spectra.modules.cplx_attention import CMultiHeadedAttention
from ar_spectra.modules.cplx_rms_norm import ComplexRMSNorm
from ar_spectra.modules.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
    IdentityPositionalEncoding,  # noqa: H301
)
from ar_spectra.modules.multi_layer_conv import Conv1dLinear
from ar_spectra.modules.multi_layer_conv import MultiLayeredConv1d
from ar_spectra.modules.nets_utils import make_pad_mask
from ar_spectra.modules.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from ar_spectra.modules.subsampling import Conv2dSubsampling
from ar_spectra.modules.subsampling import Conv2dSubsampling2
from ar_spectra.modules.subsampling import Conv2dSubsampling6
from ar_spectra.modules.subsampling import Conv2dSubsampling8
from ar_spectra.modules.subsampling import TooShortUttError
from ar_spectra.modules.subsampling import check_short_utt
from ar_spectra.modules.subsampling import Conv2dSubsamplingPad
from ar_spectra.modules.subsampling import Conv1dSubsampling2
from torch.nn import functional as F
from ar_spectra.modules.cplx_dropout import ComplexDropout
from ar_spectra.modules.normed_modules.norm import ComplexLayerNorm, ComplexBatchNorm1d
from ar_spectra.modules.normed_modules.conv import (
    NormConv1d, NormLinear, NormConv2d, NormConvTranspose1d, NormConvTranspose2d,
    SConv2d, SConvTranspose2d,
)
from ar_spectra.modules.activations import get_activation, _build_activation
from ar_spectra.models.autoencoders.SeaNET_AE import SEANetResnetBlock2d
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


class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0., is_complex=True):
        super().__init__()
        assert is_complex==True, "Only complex FeedForward is supported in cplx_ViT"
        self.net = nn.Sequential(
            ComplexLayerNorm(dim),
            NormLinear(dim, hidden_dim, is_complex=True, norm="none"),
            get_activation("CReLU", is_complex=True),
            ComplexDropout(dropout),
            NormLinear(hidden_dim, dim, is_complex=True, norm="none"),
            ComplexDropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0., bias: bool = True):
        super().__init__()
        self.norm = ComplexLayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                ComplexLayerNorm(dim),                 # ln1
                CMultiHeadedAttention(n_head=heads, n_feat=dim, dropout_rate=dropout, bias=bias),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, mask: Optional[torch.Tensor] = None, debug: bool = False):
        for ln1, attn, ff in self.layers:
            h = ln1(x)
            x = x + attn(query=h, key=h, value=h, mask=mask)
            x = x + ff(x)

        if debug:
            print(f"Transformer output shape (B, L, dim): {x.shape}")

        return self.norm(x)

class ViTEncoder(Module):
    """Complex-valued Vision Transformer encoder for spectrogram patches.

    The encoder slices an input spectrogram into possibly rectangular patches, projects
    them into a latent embedding space, optionally adds a learnable CLS token, applies
    a configurable positional encoding, and processes the resulting sequence with a
    stack of `Transformer` layers backed by complex-valued attention. It can return
    the entire token sequence (for reconstruction tasks) or a pooled representation
    (CLS or mean) for downstream classifiers.

    Args:
        image_size (Union[int, Tuple[int, int]]): Spectrogram size expressed either
            as a single integer (square) or a tuple `(time, frequency)`.
        patch_size (Union[int, Tuple[int, int]]): Patch size, same semantics as
            `image_size`; rectangular patches are supported when a tuple is used.
        dim (int): Embedding dimensionality of each patch token.
        depth (int): Number of stacked Transformer layers.
        heads (int): Number of attention heads per Transformer block.
        mlp_dim (int): Hidden dimensionality of the feed-forward sublayers.
        pool (str): One of `{'cls','mean','none'}` selecting the pooling strategy.
        channels (int): Number of complex-valued channels in the input spectrogram.
        dropout (float): Dropout probability applied within attention/MLP blocks.
        emb_dropout (float): Dropout probability applied after positional encoding.
        positional_encoding (str): `"none"`, `"identity"`, or `"learned"` selecting
            which positional encoding strategy to apply to the token sequence.
    """

    def __init__(  # type: ignore[override]
        self,
        *,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "none",
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        positional_encoding: str = "none",
        input_size: Optional[Union[int, Tuple[int, int]]] = None,
        is_complex: bool = True,
        bias: bool = True,
        use_pre_conv: bool = True,
        pre_conv_kernel: Union[int, Tuple[int, int]] = (3, 3),
        pre_conv_stride: Union[int, Tuple[int, int]] = 1,
        ln_debug: bool = False,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.image_size = (image_height, image_width)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.channels = input_size
        patch_dim = self.channels * patch_height * patch_width

        assert pool in {'cls', 'mean', 'none'}, "pool must be one of {'cls', 'mean', 'none'}"
        num_cls_tokens = 1 if pool == 'cls' else 0
        
        self.pre_patch_conv = (
            SConv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=pre_conv_kernel,
                stride=pre_conv_stride,
                norm="none",
                is_complex=True,
            )
            if use_pre_conv
            else nn.Identity()
        )

        self.resblock = SEANetResnetBlock2d(
            dim=self.channels,
            kernel_sizes=[(5,5), (1,1)],
            dilations=[(1,1), (1,1)],
            norm="weight_norm",
            is_complex=True,
            compress=1,
            activation="CReLU",
            activation_params={},
        )
        self.act_compress = get_activation("CReLU", is_complex=True)

        self.compress = SConv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=(4, 4),
            stride=(2, 2),
            norm="none",
            is_complex=True,)
        
        self.unfold = Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.to_patch_embedding = nn.Sequential(
            NormLinear(patch_dim, dim, is_complex=True, norm="none"),
        )

        self.cls_token = (
            nn.Parameter(torch.randn(num_cls_tokens, dim, dtype=torch.complex64))
            if num_cls_tokens == 1
            else None
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(num_patches + num_cls_tokens, dim, dtype=torch.complex64)
        )

        self.dropout = ComplexDropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, bias=bias)
        self.positional_encoding = positional_encoding.lower()
        allowed_positional_encodings = {"none", "identity", "learned"}
        if self.positional_encoding not in allowed_positional_encodings:
            valid_options = ", ".join(sorted(allowed_positional_encodings))
            warn(
                f"Unsupported positional encoding '{positional_encoding}'. Valid options: {valid_options}, defaulting to 'none'."
            )
            self.positional_encoding = "none"
        self.identity_pos_enc = IdentityPositionalEncoding(d_model=dim)

        self.pool = pool
        self._trim_warned = False
        self._ln_debug_handles: List[torch.utils.hooks.RemovableHandle] = []



        # Attach LN debug hooks if requested (prints at every forward/backward)
        if ln_debug:
            self._ln_debug_handles = attach_complex_ln_debug_hooks(
                self, name_prefix="encoder"
            )

    def forward(self, img: torch.Tensor, debug: bool = True) -> torch.Tensor:
        if debug:
            print(f"Input spectrogram shape (B, C, F, T): {img.shape}")
        batch = img.shape[0]
        img = self.pre_patch_conv(img)
        if debug:
            print(f"Post-pre-conv input shape (B, C, F, T): {img.shape}")
        img = self.resblock(img)
        if debug:
            print(f"Post-resblock input shape (B, C, F, T): {img.shape}")
        img = self.act_compress(img)
        img = self.compress(img)
        if debug:
            print(f"Post-compress input shape (B, C, F, T): {img.shape}")
        if img.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} input channels but received {img.shape[1]}. "
                "Set encoder.channels explicitly if your dataset differs."
            )
        patches = self.unfold(img)
        if debug:
            print(f"Unfolded patches shape (B, patch_dim, L): {patches.shape}")
        x = patches.transpose(1, 2)
        if debug:
            print(f"Unfolded patches shape (B, L, patch_dim): {x.shape}")
        x = self.to_patch_embedding(x)
        if debug:
            print(f"Projected patch embeddings shape (B, L, dim): {x.shape}")

        if self.cls_token is not None:
            cls_tokens = einops_repeat(self.cls_token, '... d -> b ... d', b = batch)
            x = torch.cat((cls_tokens, x), dim = 1)

        x = self._apply_positional_encoding(x)
        if debug:
            print(f"Positional encoded tokens shape (B, L, dim): {x.shape}")
        x = self.dropout(x)

        x = self.transformer(x)
        if debug:
            print(f"Transformer output shape (B, L, dim): {x.shape}")
        x = x.reshape(batch, -1, x.shape[-2]) # (B, L, dim) -> (B, dim, L)
        if debug:
            print(f"Reshaped output shape (B, dim, L): {x.shape}")
        if self.pool == 'cls' and self.cls_token is not None:
            return x[:, 0]
        if self.pool == 'mean':
            return x.mean(dim=1)
        
        return x

    def _apply_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.shape[1]
        if self.positional_encoding == "learned":
            return x + self.pos_embedding[:seq]
        if self.positional_encoding == "identity":
            return self.identity_pos_enc(x)
        if self.positional_encoding == "none":
            return x
        raise ValueError(f"Unexpected positional encoding value: {self.positional_encoding}")


class ViTDecoder(Module):
    """Transformer-based inverse of the complex ViT encoder.

    The decoder expects the same number of tokens produced by the encoder, enriches
    them with positional information, refines them with a `Transformer` stack, and
    projects each token back to its corresponding spectrogram patch before folding
    the sequence into the original time–frequency layout.

    Args:
        image_size (Union[int, Tuple[int, int]]): Output spectrogram size, mirroring
            the encoder's `image_size`.
        patch_size (Union[int, Tuple[int, int]]): Patch geometry used during encoding
            and decoding.
        dim (int): Dimensionality of incoming token embeddings.
        channels (int): Number of channels to reconstruct (e.g., magnitude/phase).
        depth (int): Number of Transformer blocks applied in the decoder.
        heads (int): Number of attention heads inside each decoder block.
        mlp_dim (int): Hidden size of decoder feed-forward sublayers.
        dropout (float): Dropout probability inside decoder attention/MLP blocks.
        positional_encoding (str): `"none"`, `"identity"`, or `"learned"` positional
            encoding applied to decoder tokens prior to the Transformer stack.
    """

    def __init__(  # type: ignore[override]
        self,
        *,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        channels: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        positional_encoding: str = "none",
        input_size: Optional[Tuple[int, int]] = None,
        is_complex: bool = True,
        bias: bool = True,
        use_post_conv: bool = True,
        post_conv_kernel: Union[int, Tuple[int, int]] = (3, 3),
        post_conv_stride: Union[int, Tuple[int, int]] = 1,
        post_conv_out_padding: Union[int, Tuple[Tuple[int, int], Tuple[int, int]]] = ((1, 0), (0, 0)),
        ln_debug: bool = False,
    ):
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        self.channels = channels
        self.input_size = input_size

        img_h, img_w = self.image_size
        patch_h, patch_w = self.patch_size
        assert img_h % patch_h == 0 and img_w % patch_w == 0, "Image dimensions must be divisible by the patch size"

        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        patch_dim = channels * patch_h * patch_w

        self.to_patch = nn.Sequential(
            NormLinear(input_size, patch_dim, is_complex=True, norm="none"),
        )
        self.fold = nn.Fold(
            output_size=self.image_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.transformer = Transformer(input_size, depth, heads, mlp_dim, dropout, bias=bias)

        self.positional_encoding = positional_encoding.lower()
        allowed_positional_encodings = {"none", "identity", "learned"}
        if self.positional_encoding not in allowed_positional_encodings:
            valid_options = ", ".join(sorted(allowed_positional_encodings))
            warn(
                f"Unsupported decoder positional encoding '{positional_encoding}'. Valid options: {valid_options}, defaulting to 'none'."
            )
            self.positional_encoding = "none"
        self.identity_pos_enc = IdentityPositionalEncoding(d_model=input_size)
        self.pos_embedding = None
        if self.positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(
                torch.randn(self.num_patches, input_size, dtype=torch.complex64)
            )

        
        self.resblock = SEANetResnetBlock2d(
            dim=self.channels,
            kernel_sizes=[(5,5), (1,1)],
            dilations=[(1,1), (1,1)],
            norm="weight_norm",
            is_complex=True,
            compress=1,
            activation="CReLU",
            activation_params={},
        )
        self.act_compress = get_activation("CReLU", is_complex=True)

        self.compress = SConvTranspose2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=(4, 4),
            stride=(2, 2),
            norm="none",
            is_complex=True,)
        self.post_fold_conv = (
            SConvTranspose2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=post_conv_kernel,
                stride=post_conv_stride,
                norm="none",
                is_complex=True,
                out_padding=post_conv_out_padding,
            )
            if use_post_conv
            else nn.Identity()
        )
        self._ln_debug_handles: List[torch.utils.hooks.RemovableHandle] = []
        if ln_debug:
            self._ln_debug_handles = attach_complex_ln_debug_hooks(
                self, name_prefix="decoder"
            )

    def forward(self, tokens: torch.Tensor, debug: bool = True) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(err(f"tokens must have shape (batch, seq_len, dim)"))
        if debug:
            print(f"Decoder input tokens shape (B, L, dim): {tokens.shape}")
            
        tokens = tokens.transpose(1, 2).contiguous()  # (B, dim, L)
        if debug:
            print(f"Transposed tokens shape (B, dim, L): {tokens.shape}")
            
        if tokens.size(1) != self.num_patches:
            raise ValueError(
                err(f"Expected {self.num_patches} tokens but received {tokens.size(1)}")
            )
        if debug:
            print(f"Number of tokens verified: {tokens.size(1)}")
        x = self._apply_positional_encoding(tokens)
        x = self.transformer(x)
        if debug:
            print(f"Transformer output shape (B, dim, L): {x.shape}")

        patches = self.to_patch(x)  # (B, L, patch_dim)
        if debug:
            print(f"Projected patches shape (B, L, patch_dim): {patches.shape}")

        patches = patches.transpose(1, 2).contiguous()  # (B, patch_dim, L)
        if debug:
            print(f"Transposed patches shape (B, patch_dim, L): {patches.shape}")

        recon = self.fold(patches)
        
        recon = self.resblock(recon)
        if debug:
            print(f"Post-resblock recon shape (B, C, F, T): {recon.shape}")
        recon = self.act_compress(recon)
        # upsample back to the original spectrogram resolution
        recon = self.compress(recon)
        recon = self.post_fold_conv(recon)
        if debug:
            print(f"Reconstructed spectrogram shape (B, C, F, T): {recon.shape}")
        return recon

    def _apply_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if self.positional_encoding == "learned" and self.pos_embedding is not None:
            return x + self.pos_embedding.unsqueeze(0)
        if self.positional_encoding == "identity":
            return self.identity_pos_enc(x)
        if self.positional_encoding == "none":
            return x
        raise ValueError(
            f"Unexpected decoder positional encoding value: {self.positional_encoding}"
        )


def attach_complex_ln_debug_hooks(module: nn.Module, name_prefix: str = ""):
    """Attach forward/backward hooks to all ComplexLayerNorm layers to print stats."""
    handles = []

    def _fmt_cplx_stats(t: torch.Tensor) -> str:
        is_nan = torch.isnan(t).any().item()
        is_inf = torch.isinf(t).any().item()
        if torch.is_complex(t):
            mag = t.abs()
            return (f"mean=({t.real.mean():.3e}+j{t.imag.mean():.3e}) "
                    f"max_abs={mag.max():.3e} nan={is_nan} inf={is_inf} "
                    f"std_abs={mag.std():.3e} "
                    f"real_std={t.real.std():.3e} imag_std={t.imag.std():.3e}")
        return (f"mean={t.mean():.3e} max={t.max():.3e} "
                f"nan={is_nan} inf={is_inf}"
                f" std={t.std():.3e}"
                f"real_std={t.real.std():.3e} imag_std={t.imag.std():.3e}")

    def register_ln(m: nn.Module, path: str):
        if isinstance(m, ComplexLayerNorm) or isinstance(m, ComplexRMSNorm):
            def fwd_hook(_, __, output):
                print(f"[LN {path} fwd] {_fmt_cplx_stats(output)}")

            def bwd_hook(_, grad_input, grad_output):
                go = grad_output[0]
                if go is not None:
                    print(f"[LN {path} bwd] {_fmt_cplx_stats(go)}")

            handles.append(m.register_forward_hook(fwd_hook))
            handles.append(m.register_full_backward_hook(bwd_hook))

    for n, m in module.named_modules():
        register_ln(m, f"{name_prefix}{n}" if name_prefix else n)

    return handles

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ok(f"Using device: {device}")
    vit = ViTEncoder(
        image_size=(1024, 64),
        patch_size=(4, 8),
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=512,
        pool='none',
        channels=2,
        dropout=0.1,
        emb_dropout=0.1,
        positional_encoding="learned",
    ).to(device)
    decoder = ViTDecoder(
        image_size=(1024, 64),
        patch_size=(4, 8),
        input_size=64,
        channels=2,
        depth=6,
        heads=8,
        mlp_dim=64,
        dropout=0.1,
        positional_encoding="learned",
    ).to(device)

    # Attach LN debug hooks
    ln_hooks = []
    ln_hooks += attach_complex_ln_debug_hooks(vit, name_prefix="encoder.")
    ln_hooks += attach_complex_ln_debug_hooks(decoder, name_prefix="decoder.")

    dummy = torch.randn(2, 2, 1024, 64, dtype=torch.complex64, device=device)
    ok(f"Input shape: {dummy.shape}")
    tokens = vit(dummy)
    ok(f"Encoder output shape (tokens): {tokens.shape}")
    recon = decoder(tokens)
    ok(f"End-to-end reconstruction shape: {recon.shape}")

    # Clean up hooks if needed
    for h in ln_hooks:
        h.remove()