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

    def __init__(self, idim, hdim, odim, dropout_rate, patch_size=(16, 32), pos_enc=None, norm:str="none"):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            SConv2d(1, hdim, 4, 2, padding=1, padding_mode="reflect", is_complex=True, norm=norm),
            get_activation("relu"),
            SConv2d(hdim, hdim, 4, 2, padding=1, padding_mode="reflect", is_complex=True, norm=norm),
            get_activation("relu"),
        )
        self.patch_fn = nn.Unfold(kernel_size=patch_size, stride=patch_size, is_complex=True)
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
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            pos_enc_class = IdentityPositionalEncoding
            logging.warning(
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

        self.sequence_model = None
        if sequence_model_type == "conformer":
            from models.autoencoders.ConformerAE import ConformerEncoder
            self.sequence_model = ConformerEncoder(
                input_size=output_size,
                **kwargs,
            )
            
            