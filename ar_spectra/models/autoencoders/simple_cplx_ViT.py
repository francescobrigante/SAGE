import torch
from torch import nn
from torch.nn import Module, ModuleList, Unfold

from einops import repeat as einops_repeat
import logging
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn
from ar_spectra.models.autoencoders.AbsEncoder import AbsEncoder
from ar_spectra.modules.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from ar_spectra.modules.cplx_attention import CMultiHeadedAttention
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
from ar_spectra.modules.normed_modules.conv import NormConv1d, NormLinear
from ar_spectra.modules.activations import get_activation, _build_activation
from rich.console import Console
console = Console()

def ok(msg: str) -> None:
    console.print(msg, style="bold green")


def warn(msg: str) -> None:
    console.print(msg, style="bold yellow")


def err(msg: str) -> None:
    console.print(msg, style="bold red")
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0., is_complex=True):
        super().__init__()
        assert is_complex==True, "Only complex FeedForward is supported in cplx_ViT"
        self.net = nn.Sequential(
            ComplexLayerNorm(dim),
            NormLinear(dim, hidden_dim, is_complex=True, norm="none"),
            get_activation("CGELU", is_complex=True),
            ComplexDropout(dropout),
            NormLinear(hidden_dim, dim, is_complex=True, norm="none"),
            ComplexDropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = ComplexLayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                CMultiHeadedAttention(n_feat=dim, n_head=heads, dropout_rate = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        for attn, ff in self.layers:
            x = attn(query=x, key=x, value=x, mask=mask) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(Module):
    def __init__(self, *, image_size, patch_size,  
                 dim, depth, heads, mlp_dim, pool = 'none', channels = 3, 
                 dim_head = 64, dropout = 0., emb_dropout = 0., positional_encoding: str = "none"):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean', 'none'}, "pool must be one of {'cls', 'mean', 'none'}"
        num_cls_tokens = 1 if pool == 'cls' else 0

        self.unfold = Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.to_patch_embedding = nn.Sequential(
            ComplexLayerNorm(patch_dim),
            NormLinear(patch_dim, dim, is_complex=True, norm="none"),
            ComplexLayerNorm(dim),
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

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
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

    def forward(self, img):
        batch = img.shape[0]
        patches = self.unfold(img)
        x = patches.transpose(1, 2)
        x = self.to_patch_embedding(x)

        if self.cls_token is not None:
            cls_tokens = einops_repeat(self.cls_token, '... d -> b ... d', b = batch)
            x = torch.cat((cls_tokens, x), dim = 1)

        x = self._apply_positional_encoding(x)
        x = self.dropout(x)

        x = self.transformer(x)

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
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        dim,
        channels,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        positional_encoding: str = "none",
    ):
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        self.channels = channels

        img_h, img_w = self.image_size
        patch_h, patch_w = self.patch_size
        assert img_h % patch_h == 0 and img_w % patch_w == 0, "Image dimensions must be divisible by the patch size"

        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        patch_dim = channels * patch_h * patch_w

        self.to_patch = nn.Sequential(
            ComplexLayerNorm(dim),
            NormLinear(dim, patch_dim, is_complex=True, norm="none"),
            ComplexLayerNorm(patch_dim),
        )
        self.fold = nn.Fold(
            output_size=self.image_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.positional_encoding = positional_encoding.lower()
        allowed_positional_encodings = {"none", "identity", "learned"}
        if self.positional_encoding not in allowed_positional_encodings:
            valid_options = ", ".join(sorted(allowed_positional_encodings))
            warn(
                f"Unsupported decoder positional encoding '{positional_encoding}'. Valid options: {valid_options}, defaulting to 'none'."
            )
            self.positional_encoding = "none"
        self.identity_pos_enc = IdentityPositionalEncoding(d_model=dim)
        self.pos_embedding = None
        if self.positional_encoding == "learned":
            self.pos_embedding = nn.Parameter(
                torch.randn(self.num_patches, dim, dtype=torch.complex64)
            )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(err(f"tokens must have shape (batch, seq_len, dim)"))
        if tokens.size(1) != self.num_patches:
            raise ValueError(
                err(f"Expected {self.num_patches} tokens but received {tokens.size(1)}")
            )

        x = self._apply_positional_encoding(tokens)
        x = self.transformer(x)

        patches = self.to_patch(x)  # (B, L, patch_dim)
        patches = patches.transpose(1, 2).contiguous()  # (B, patch_dim, L)
        recon = self.fold(patches)
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


if __name__ == "__main__":
    torch.manual_seed(0)
    vit = ViT(
        image_size=(32, 64),
        patch_size=(4, 8),
        dim=64,
        depth=2,
        heads=4,
        mlp_dim=64,
        pool='none',
        channels=2,
        dropout=0.1,
        emb_dropout=0.1,
        positional_encoding="learned",
    )
    dummy = torch.randn(2, 2, 32, 64, dtype=torch.complex64)
    output = vit(dummy)
    ok(f"Encoder output shape (tokens): {output.shape}")

    decoder = ViTDecoder(
        image_size=(32, 64),
        patch_size=(4, 8),
        dim=64,
        channels=2,
        depth=2,
        heads=4,
        mlp_dim=64,
        dropout=0.1,
        positional_encoding="learned",
    )

    random_tokens = torch.randn(16, 1025, 64, dtype=torch.complex64)
    random_recon = decoder(random_tokens)
    ok(f"Decoder-alone reconstruction shape: {random_recon.shape}")

    recon = decoder(output)
    ok(f"End-to-end reconstruction shape: {recon.shape}")