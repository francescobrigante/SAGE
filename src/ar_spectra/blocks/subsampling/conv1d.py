
# =============================================================================
# Subsampling modules based on 1D convolutions to reduce temporal resolution.
# =============================================================================

import numpy as np
import torch
import torch.nn.functional as F
from ar_spectra.blocks.embeddings import IdentityPositionalEncoding
import logging
from ar_spectra.blocks.conv import SConv1d
from ar_spectra.blocks.conv import NormLinear
from ar_spectra.blocks.activations import get_activation


class Conv1dSubsampling2(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/2 length) implement with Conv2d.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, is_complex: bool = True,
                 conv_norm: str = "none", act: str = "relu"):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv1dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            SConv1d(idim, odim, 3, 2, is_complex=is_complex, norm=conv_norm),
            get_activation(act),
            SConv1d(odim, odim, 3, 1, is_complex=is_complex, norm=conv_norm),
            get_activation(act),
        )
        self.out = torch.nn.Sequential(
            NormLinear(odim, odim, is_complex=is_complex, norm=conv_norm),
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
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.out(x.transpose(1, 2).contiguous())
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

class Conv1dSubsampling(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, kernel_size, stride, pad,
                 tf2torch_tensor_name_prefix_torch: str = "stride_conv",
                 tf2torch_tensor_name_prefix_tf: str = "seq2seq/proj_encoder/downsampling",
                 ):
        super(Conv1dSubsampling, self).__init__()
        self.conv = torch.nn.Conv1d(idim, odim, kernel_size, stride)
        self.pad_fn = torch.nn.ConstantPad1d(pad, 0.0)
        self.stride = stride
        self.odim = odim
        self.tf2torch_tensor_name_prefix_torch = tf2torch_tensor_name_prefix_torch
        self.tf2torch_tensor_name_prefix_tf = tf2torch_tensor_name_prefix_tf

    def output_size(self) -> int:
        return self.odim

    def forward(self, x, x_len):
        """Subsample x.

        """
        x = x.transpose(1, 2)  # (b, d ,t)
        x = self.pad_fn(x)
        x = F.relu(self.conv(x))
        x = x.transpose(1, 2)  # (b, t ,d)

        if x_len is None:

            return x, None
        x_len = (x_len - 1) // self.stride + 1
        return x, x_len

    def gen_tf2torch_map_dict(self):
        tensor_name_prefix_torch = self.tf2torch_tensor_name_prefix_torch
        tensor_name_prefix_tf = self.tf2torch_tensor_name_prefix_tf
        map_dict_local = {
            ## predictor
            "{}.conv.weight".format(tensor_name_prefix_torch):
                {"name": "{}/conv1d/kernel".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": (2, 1, 0),
                 },  # (256,256,3),(3,256,256)
            "{}.conv.bias".format(tensor_name_prefix_torch):
                {"name": "{}/conv1d/bias".format(tensor_name_prefix_tf),
                 "squeeze": None,
                 "transpose": None,
                 },  # (256,),(256,)
        }
        return map_dict_local

    def convert_tf2torch(self,
                         var_dict_tf,
                         var_dict_torch,
                         ):
    
        map_dict = self.gen_tf2torch_map_dict()
    
        var_dict_torch_update = dict()
        for name in sorted(var_dict_torch.keys(), reverse=False):
            names = name.split('.')
            if names[0] == self.tf2torch_tensor_name_prefix_torch:
                name_tf = map_dict[name]["name"]
                data_tf = var_dict_tf[name_tf]
                if map_dict[name]["squeeze"] is not None:
                    data_tf = np.squeeze(data_tf, axis=map_dict[name]["squeeze"])
                if map_dict[name]["transpose"] is not None:
                    data_tf = np.transpose(data_tf, map_dict[name]["transpose"])
                data_tf = torch.from_numpy(data_tf).type(torch.float32).to("cpu")
            
                var_dict_torch_update[name] = data_tf
            
                logging.info(
                    "torch tensor: {}, {}, loading from tf tensor: {}, {}".format(name, data_tf.size(), name_tf,
                                                                                  var_dict_tf[name_tf].shape))
        return var_dict_torch_update
