import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, List, Tuple
from ar_spectra.modules.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
    IdentityPositionalEncoding,  # noqa: H301
)
import logging
from ar_spectra.modules.normed_modules.conv import SConv1d, SConv2d
from ar_spectra.modules.normed_modules.conv import SConvTranspose1d, SConvTranspose2d, NormLinear
from ar_spectra.modules.activations import get_activation, _build_activation
from rich.console import Console
console = Console()

def ok(msg: str) -> None:
    console.print(msg, style="bold green")


def warn(msg: str) -> None:
    console.print(msg, style="bold yellow")


def err(msg: str) -> None:
    console.print(msg, style="bold red")
    

class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, hdim, odim, dropout_rate, patch_size=(16, 32), pos_enc=None, norm:str="none", is_complex=True):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            SConv2d(1, hdim, 4, 2, is_complex=is_complex, norm=norm, pad_mode="reflect"),
            get_activation("relu"),
            SConv2d(hdim, hdim, 4, 2, is_complex=is_complex, norm=norm, pad_mode="reflect"),
            get_activation("relu"),
        )
        self.patch_fn = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        from ar_spectra.modules.embedding import PositionalEncoding
        self.out = torch.nn.Sequential(
            NormLinear(hdim * patch_size[0] * patch_size[1], odim, is_complex=True, norm=norm),
            IdentityPositionalEncoding(odim),
        )

    def forward(self, x):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)    # (b, c, floor(t/2), floor(f/2))
        patches = self.patch_fn(x)  # (b, c, n)
        x = self.out(patches.transpose(1, 2))

        return x


class ViTEncoder(nn.Module):
    def __init__(
            self,
            input_size: int,
            vit_input_layer: str = "conv2d2",
            sequence_model_type: str = "conformer",
            path_size: Union[int, List[int], Tuple[int]] = (16, 32),
            conv_norm: str = "none",
            **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        output_size = kwargs.get("output_size", 512)
        positional_dropout_rate = kwargs.get("positional_dropout_rate", 0.1)

        rel_pos_type = kwargs.get("rel_pos_type", "legacy")
        pos_enc_layer_type = kwargs.get("pos_enc_layer_type", "rel_pos")
        selfattention_layer_type = kwargs.get("selfattention_layer_type", "rel_selfattn")

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            warn(f"Unknown rel_pos_type {rel_pos_type}. Using legacy settings: identityPE")
            

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            warn(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            pos_enc_class = IdentityPositionalEncoding
            warn(
                f"Unknown pos_enc_layer_type {pos_enc_layer_type}. Using IdentityPositionalEncoding."
            )

        if vit_input_layer == "conv2d2":
            self.vit_input_layer = Conv2dSubsampling2(
                idim=input_size,
                hdim=input_size * 4,
                odim=output_size,
                dropout_rate=kwargs.get("dropout_rate", 0.1),
                patch_size=path_size,
                pos_enc=pos_enc_class(output_size, positional_dropout_rate),
                norm=conv_norm,
            )
        else:
            raise TypeError(f"Unknown input layer type {vit_input_layer}.")

        self._patch_kernel = tuple(self.vit_input_layer.patch_fn.kernel_size)
        self._patch_stride = tuple(self.vit_input_layer.patch_fn.stride)
        self._conv_downsample_factor = 4  # two stride-2 convolutions along time
        self._freq_patch_count = self._compute_freq_patch_count(self.input_size)

        self.sequence_model = None
        if sequence_model_type == "conformer":
            from ar_spectra.models.autoencoders.conformer_base import ConformerEncoder
            self.sequence_model = ConformerEncoder(
                input_size=output_size,
                **kwargs,
            )
            
    def forward(self, x, ilens=None, ctc=None):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (batch, time, freq), got {x.shape} instead.")

        batch, time_steps, _ = x.shape
        device = x.device
        if ilens is None:
            ilens = torch.full((batch,), time_steps, dtype=torch.long, device=device)
        else:
            if ilens.dim() != 1 or ilens.size(0) != batch:
                raise ValueError(
                    f"ilens must be 1D with length {batch}, got {tuple(ilens.shape)} instead."
                )
            ilens = ilens.to(device=device, dtype=torch.long)

        vit_tokens = self.vit_input_layer(x)
        projected_ilens = self._project_lengths(ilens)

        if self.sequence_model is None:
            return vit_tokens, projected_ilens, None

        return self.sequence_model(
            vit_tokens,
            ilens=projected_ilens,
            prev_states=None,
            ctc=ctc,
        )

    def _project_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Map original time lengths to patch sequence lengths."""
        if lengths.numel() == 0:
            return lengths

        reduced = (lengths + (self._conv_downsample_factor - 1)) // self._conv_downsample_factor

        patch_time, patch_freq = self._patch_kernel
        stride_time, _ = self._patch_stride

        time_tokens = torch.zeros_like(reduced)
        valid = reduced >= patch_time
        if valid.any():
            time_tokens_valid = ((reduced[valid] - patch_time) // stride_time) + 1
            time_tokens[valid] = time_tokens_valid

        return time_tokens * self._freq_patch_count

    def _compute_freq_patch_count(self, freq_bins: int) -> int:
        freq_reduced = (freq_bins + (self._conv_downsample_factor - 1)) // self._conv_downsample_factor

        patch_freq = self._patch_kernel[1]
        stride_freq = self._patch_stride[1]
        if freq_reduced < patch_freq:
            raise ValueError(
                err(f"Patch frequency size {patch_freq} is larger than reduced freq dimension {freq_reduced}.")
            )
        return ((freq_reduced - patch_freq) // stride_freq) + 1
    
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch, time_steps, freq_bins = 16, 1024, 128
    ok("Testing Conv2dSubsampling2...")
    ok("Input shape: " + str((batch, time_steps, freq_bins)))
    # complex64 tensor to exercise the complex convolution path
    dummy = torch.randn(batch, time_steps, freq_bins, dtype=torch.complex64).to(device)
    conv = Conv2dSubsampling2(
        idim=freq_bins,
        hdim=freq_bins * 4,
        odim=256,
        dropout_rate=0.1,
        patch_size=(16, 32),
        norm="none",
    ).to(device)
    with torch.inference_mode():
        out = conv(dummy)
    ok(f"Conv2dSubsampling2 output shape: {out.shape}")

    encoder = ViTEncoder(
        input_size=freq_bins,
        vit_input_layer="conv2d2",
        sequence_model_type="conformer",
        path_size=(16, 32),
        conv_norm="none",
        attention_heads=8,
        linear_units=freq_bins,
        num_blocks=3,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        input_layer="linear",
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=5,
        macaron_style=False,
        rel_pos_type="none",
        pos_enc_layer_type="none",
        selfattention_layer_type="selfattn",
        activation_type="CReLU",
        use_cnn_module=True,
        zero_triu=False,
        cnn_module_kernel=5,
    ).to(device)
    ok("Testing ViTEncoder with ConformerEncoder...")
    with torch.inference_mode():
        ok(f"allocated before ViT = {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")

        out_enc = encoder.vit_input_layer(dummy)
        new_ilens = torch.full((batch,), out_enc.size(1), dtype=torch.long)
        y3, olens, _ = encoder.sequence_model(
            out_enc,
            ilens=new_ilens,
            prev_states=None,
            ctc=None,
        )
        ok(f"allocated after ViT = {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
        ok(f"Max memory allocated {torch.cuda.max_memory_allocated(device) / 1024**2:.1f} MB")
    ok(f"ViTEncoder output shape: {y3.shape}")

