
# =============================================================================
# Subsampling modules based on 2D convolutions (ideal for spectrogram-like representations).
# =============================================================================

import torch
from ar_spectra.blocks.embeddings import IdentityPositionalEncoding
from ar_spectra.blocks.conv import SConv2d
from ar_spectra.blocks.conv import NormLinear
from ar_spectra.blocks.activations import get_activation
import math
from .helpers import sequence_mask


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, is_complex: bool = True, conv_norm: str = "none",
                 act: str = "relu"):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            SConv2d(1, odim, 3, 2, is_complex=is_complex, norm=conv_norm),
            get_activation(act),
            SConv2d(odim, odim, 3, 2, is_complex=is_complex, norm=conv_norm),
            get_activation(act),
        )
        self.out = torch.nn.Sequential(
            NormLinear(odim * (((idim - 1) // 2 - 1) // 2), odim, is_complex=is_complex, norm=conv_norm),
            pos_enc if pos_enc is not None else IdentityPositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class Conv2dSubsamplingPad(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, is_complex: bool = True,
                 conv_norm: str = "none", act: str = "relu"):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsamplingPad, self).__init__()
        self.conv = torch.nn.Sequential(
            SConv2d(1, odim, 3, 2, is_complex=is_complex, norm=conv_norm, pad_mode='constant'),
            get_activation(act),
            SConv2d(odim, odim, 3, 2, is_complex=is_complex, norm=conv_norm, pad_mode='constant'),
            get_activation(act),
        )
        self.out = torch.nn.Sequential(
            NormLinear(odim * (((idim - 1) // 2 - 1) // 2), odim, is_complex=is_complex, norm=conv_norm),
            pos_enc if pos_enc is not None else IdentityPositionalEncoding(odim, dropout_rate),
        )
        self.pad_fn = torch.nn.ConstantPad1d((0, 4), 0.0)

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.transpose(1, 2)
        x = self.pad_fn(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        x_len = torch.sum(x_mask[:, 0, :], dim=-1)
        x_len = (x_len - 1) // 2 + 1
        x_len = (x_len - 1) // 2 + 1
        mask = sequence_mask(x_len, None, x_len.dtype, x[0].device)
        return x, mask[:, None, :]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, is_complex: bool = True,
                 conv_norm: str = "none", act: str = "relu"):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            SConv2d(1, odim, 3, 2, is_complex=is_complex, norm=conv_norm),
            get_activation(act),
            SConv2d(odim, odim, 3, 1, is_complex=is_complex, norm=conv_norm),
            get_activation(act),
        )
        self.out = torch.nn.Sequential(
            NormLinear(odim * (((idim - 1) // 2 - 2)), odim, is_complex=is_complex, norm=conv_norm),
            pos_enc if pos_enc is not None else IdentityPositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
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
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:1]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, is_complex: bool = True,
                 conv_norm: str = "none", act: str = "relu"):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = torch.nn.Sequential(
            SConv2d(1, odim, 3, 2, is_complex=is_complex, norm=conv_norm),
            get_activation(act),
            SConv2d(odim, odim, 5, 3, is_complex=is_complex, norm=conv_norm),
            get_activation(act),
        )
        self.out = torch.nn.Sequential(
            NormLinear(odim * (((idim - 1) // 2 - 2) // 3), odim, is_complex=is_complex, norm=conv_norm),
            pos_enc if pos_enc is not None else IdentityPositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-4:3]

class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, is_complex: bool = True,
                 conv_norm: str = "none", act: str = "relu"):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = torch.nn.Sequential(
            SConv2d(1, odim, 3, 2, is_complex=is_complex, norm=conv_norm),
            get_activation(act),
            SConv2d(odim, odim, 3, 2, is_complex=is_complex, norm=conv_norm),
            get_activation(act),
            SConv2d(odim, odim, 3, 2, is_complex=is_complex, norm=conv_norm),
            get_activation(act),
        )
        self.out = torch.nn.Sequential(
            NormLinear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim, is_complex=is_complex, norm=conv_norm),
            pos_enc if pos_enc is not None else IdentityPositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]

class Conv2dSubsamplingWOPosEnc(torch.nn.Module):
    """Convolutional 2D subsampling.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        kernels (list): kernel sizes
        strides (list): stride sizes

    """

    def __init__(self, idim, odim, dropout_rate, kernels, strides):
        """Construct an Conv2dSubsamplingWOPosEnc object."""
        assert len(kernels) == len(strides)
        super().__init__()
        conv = []
        olen = idim
        for i, (k, s) in enumerate(zip(kernels, strides)):
            conv += [
                torch.nn.Conv2d(1 if i == 0 else odim, odim, k, s),
                torch.nn.ReLU(),
            ]
            olen = math.floor((olen - k) / s + 1)
        self.conv = torch.nn.Sequential(*conv)
        self.out = torch.nn.Linear(odim * olen, odim)
        self.strides = strides
        self.kernels = kernels

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        for k, s in zip(self.kernels, self.strides):
            x_mask = x_mask[:, :, : -k + 1 : s]
        return x, x_mask
