import torch
from torch.nn import Module, ModuleList
import torch.nn as nn
from typing import Sequence, Tuple, Union, Dict
from ar_spectra.modules.cplx_attention import CMultiHeadedAttention
from ar_spectra.modules.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from ar_spectra.modules.normed_modules.conv import (
    NormConv2d, NormConvTranspose2d,
    SConv2d, SConvTranspose2d,
)
import typing as tp
from ar_spectra.modules.cplx_embedding import ComplexPositionalEncoding, ComplexScaledPositionalEncoding
from ar_spectra.modules.normed_modules.norm import ComplexLayerNorm, ComplexBatchNorm1d
from ar_spectra.modules.activations import get_activation
from ar_spectra.models.autoencoders.abstract_ae import AbstractEncoder, AbastractDecoder
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
        use_complex: bool = True,
        use_pos_enc: bool = False,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.use_complex = use_complex
        self.use_pos = use_pos_enc
        self.pos_enc = None
        # Pre-norm Transformer block supporting complex or real tokens.

        if use_complex:
            self.norm = ComplexLayerNorm(feat_dim)
            if use_pos_enc:
                self.pos_enc = ComplexScaledPositionalEncoding(
                    d_model=feat_dim,
                    dropout_rate=dropout_rate,
                    max_len=5000,
                )
                
            self.attn = CMultiHeadedAttention(
                n_heads, feat_dim, dropout_rate=dropout_rate
            )
        else:
            self.norm = nn.LayerNorm(2 * feat_dim)
            if use_pos_enc:
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
        ok("Created transformer block with {} heads, feat_dim={}, use_complex={}, use_pos_enc={}".format(
            n_heads, feat_dim, use_complex, use_pos_enc
        ))

        self.norm2 = ComplexLayerNorm(feat_dim)
        self.linear1 = nn.Linear(feat_dim, feat_dim, dtype=torch.complex64)
        self.activation = get_activation("CReLU", is_complex=True)
        self.linear2 = nn.Linear(feat_dim, feat_dim, dtype=torch.complex64)

    def forward(self, x: torch.Tensor, mask=None, debug=False) -> torch.Tensor:
        """Apply self-attention followed by a complex feed-forward block.

        Parameters
        ----------
        x : torch.Tensor
            Tokens shaped (B, T, D) where D equals ``feat_dim``. Complex when
            ``use_complex`` is True, otherwise real.
        mask : torch.Tensor, optional
            Attention mask broadcastable to (B, heads, T, T).
        debug : bool
            If True, prints intermediate shapes.
        """
        D = self.feat_dim

        # self-attention (pre-LN)
        if self.use_complex:
            if self.use_pos:
                x_in = self.pos_enc(x)
            else:
                x_in = x
            q = self.norm(x_in)
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
    """Vision Transformer-style encoder for complex-valued spectrograms.

    Parameters
    ----------
    input_size : int
        Number of input channels in the spectrogram. For mono audio with
        complex STFT representation, this is typically 1 (single complex
        channel) or 2 (separate real/imaginary channels depending on upstream
        preprocessing).
    dim : int, default=4
        Number of feature channels after the initial convolutional stem. This
        determines the capacity of the convolutional feature extractor before
        tokenization. Higher values increase model capacity at the cost of
        computational overhead.
    inter_dim : int, default=256
        Dimensionality of the Transformer token embeddings. This is the hidden
        size used throughout the self-attention layers. Controls the
        representational capacity of the latent space.
    n_heads : int, default=4
        Number of attention heads in the multi-head self-attention mechanism.
        Must evenly divide ``inter_dim``. More heads enable the model to attend
        to different representation subspaces jointly.
    depth : int, default=6
        Number of stacked Transformer blocks. Each block consists of layer
        normalization, multi-head self-attention, and a feed-forward network.
        Deeper networks can capture more complex dependencies but require more
        computation.
    dropout_rate : float, default=0.1
        Dropout probability applied within attention and feed-forward layers
        for regularization during training.
    activation : str, default="CReLU"
        Activation function used in convolutional layers and residual blocks.
        For complex-valued networks, "CReLU" (Cardioid ReLU or Complex ReLU)
        is recommended.
    image_size : tuple[int, int], optional
        Expected spatial dimensions (freq_bins, time_frames) of the input
        spectrogram **including the Nyquist bin**. Used for validation and to
        communicate the original shape to the decoder for reconstruction.
        If None, the encoder operates in a resolution-agnostic mode.
    patch_size : int or tuple[int, int], default=(8, 8)
        Size of non-overlapping patches (height, width) for tokenization.
        The downsampled feature map dimensions must be divisible by patch_size.
        Smaller patches yield more tokens with finer spatial resolution;
        larger patches reduce sequence length but may lose detail.
    norm : str, default="none"
        Normalization scheme for convolutional layers. Options include "none",
        "weight_norm", "spectral_norm", or "layer_norm".
    is_complex : bool, default=True
        Whether the encoder operates on complex-valued tensors. When True, all
        convolutional and linear layers use complex arithmetic.
    use_pos_enc : bool, default=True
        Whether to apply positional encoding to tokens before self-attention.
        Enables the Transformer to leverage spatial position information.
    n_residual_layers : int, default=1
        Number of residual blocks in the convolutional stem. More layers
        increase the receptive field before tokenization.
    residual_kernel_size : int, default=3
        Kernel size for convolutions within residual blocks.
    dilation_base : int, default=2
        Base for exponential dilation growth in residual blocks. Layer j uses
        dilation = dilation_base^j, expanding the receptive field.
    norm_params : dict, default={}
        Additional parameters passed to the normalization layers.
    activation_params : dict, default={}
        Additional parameters passed to the activation functions.
    causal : bool, default=False
        Whether to use causal (left-only) padding in convolutions, suitable
        for autoregressive or streaming applications.
    true_skip : bool, default=False
        If True, residual blocks use identity skip connections; otherwise,
        a 1x1 convolution is applied to the skip path.
    compress : int, default=2
        Channel compression factor within residual blocks (hidden dimension
        is dim // compress).
    pad_mode : str, default='reflect'
        Padding mode for convolutional layers ('reflect', 'replicate', 'zeros').
    conv_group_ratio : int, default=-1
        Group convolution ratio. If positive, uses grouped convolutions with
        groups = channels // conv_group_ratio. -1 disables grouping.
    downsample_ratio : int or tuple[int, int], default=4
        Spatial downsampling factor (freq, time) applied before tokenization.
        Can be a single integer for isotropic downsampling or a tuple for
        anisotropic downsampling.

    Attributes
    ----------
    _last_feature_shape : tuple[int, int] or None
        Cached spatial shape (F1, T1) of the feature map after downsampling,
        before patch tokenization. Populated during forward pass.

    Returns
    -------
    latents : torch.Tensor
        Channel-first latent tensor of shape (B, inter_dim, L) where L is the
        number of tokens (determined by downsampled size and patch size).
    info : dict
        Metadata dictionary containing:
        - ``feature_shape``: (F1, T1) downsampled spatial dimensions
        - ``patch_size``: (ph, pw) patch dimensions
        - ``tokens``: number of tokens L
        - ``channels``: latent channel count (inter_dim)
        - ``image_size``: original input shape including Nyquist bin

    Notes
    -----
    The encoder stores `image_size` (original spectrogram shape with Nyquist)
    in the returned info dict. The decoder uses this to verify that its output
    matches the expected reconstruction dimensions.
    """

    def __init__(
        self,
        input_size: int,
        dim: int = 4,
        inter_dim: int = 256,
        n_heads: int = 4,
        depth: int = 6,
        dropout_rate: float = 0.1,
        activation: str = "CReLU",
        image_size: tp.Optional[Tuple[int, int]] = None,
        patch_size: Union[int, Tuple[int, int]] = (8, 8),
        norm: str = "none",
        is_complex: bool = True,
        use_pos_enc: bool = True,
        n_residual_layers: int = 1,
        residual_kernel_size: int = 3, 
        dilation_base: int = 2,
        norm_params: dict = {},
        activation_params: dict = {},
        causal: bool = False,
        true_skip: bool = False, 
        compress: int = 2,
        pad_mode: str = 'reflect',
        conv_group_ratio: int = -1,
        downsample_ratio: Union[int, Tuple[int, int]] = 4,
        ) -> None:
        super().__init__(input_size=input_size, is_complex=is_complex)
        self.input_size = input_size
        self.dim = dim # number of channels after initial conv
        self.n_heads = n_heads
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.inter_dim = inter_dim
        self.use_pos_enc = use_pos_enc
        
        self.downsample_ratio = pair(downsample_ratio)
        # image_size stores the ORIGINAL spectrogram shape INCLUDING Nyquist bin
        # This is used by the decoder to verify reconstruction dimensions
        image_height, image_width = pair(image_size) if image_size is not None else (None, None)
        self.image_size = (image_height, image_width) if image_size is not None else None
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        # Validate that the frequency dimension AFTER removing Nyquist is divisible by patch size
        if image_size is not None:
            freq_without_nyquist = image_height - 1  # Remove Nyquist bin
            assert freq_without_nyquist % patch_height == 0 and image_width % patch_width == 0, (
                f'Image dimensions after Nyquist removal ({freq_without_nyquist}, {image_width}) '
                f'must be divisible by patch size ({patch_height}, {patch_width}).'
            )

        patch_dim = self.dim * patch_height * patch_width   # 32*8*8 = 2048

        self.conv0 = SConv2d(self.input_size, self.input_size, kernel_size=3, stride=1,is_complex=True, norm=norm)
        self.act0 = get_activation(self.activation, is_complex= True)
        self.conv1 = SConv2d(self.input_size, self.dim, kernel_size=5, stride=1, is_complex=True, norm=norm)
        
        resnet : tp.List[nn.Module] = []
        for j in range(n_residual_layers): # This is always 1, parameter never gets changed from default anywhere
            resnet += [
                SEANetResnetBlock2d(self.dim,
                                    kernel_sizes=[(residual_kernel_size, residual_kernel_size), (1, 1)],
                                    dilations=[(1, dilation_base ** j), (1, 1)],
                                    norm=norm, norm_params=norm_params,
                                    activation=activation, activation_params=activation_params,
                                    causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip,
                                    conv_group_ratio=conv_group_ratio, is_complex=is_complex)]
            
        self.resnet = nn.Sequential(*resnet)
        downsampling: tp.List[nn.Module] = []
        ds_kernel = tuple(2 * s for s in self.downsample_ratio)
        downsampling += [
            SConv2d(
                self.dim,
                self.dim,
                kernel_size=ds_kernel,
                stride=self.downsample_ratio,
                is_complex=True,
                norm=norm,
            ),
            get_activation(self.activation, is_complex=True),
        ]
        self.downsampling = nn.Sequential(*downsampling)
        
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
                use_pos_enc=self.use_pos_enc,
            )
            for _ in range(self.depth)
        ])
        self._attn_feat = None
        self.norm = ComplexLayerNorm(self.inter_dim)
        self.linear = nn.Linear(self.inter_dim, self.inter_dim, dtype=torch.complex64)
        self._last_feature_shape: tp.Optional[Tuple[int, int]] = None

    def forward(
        self,
        x: torch.Tensor,
        skip_attn: bool = False,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, tp.Any]]:
        """Encode spectrograms and return latents plus metadata.

        The forward pass first removes the Nyquist frequency bin from the input
        spectrogram, then processes through convolutions, downsampling, patch
        tokenization, and Transformer blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input complex spectrogram of shape (B, C, F, T) where F includes
            the Nyquist bin (e.g., n_fft//2 + 1 = 1025 for n_fft=2048).
        skip_attn : bool, default=False
            If True, bypasses the Transformer attention blocks for ablation
            studies or faster inference with reduced quality.
        debug : bool, default=False
            If True, prints tensor shapes at each processing stage.

        Returns
        -------
        latents : torch.Tensor
            Channels-first latent tokens of shape (B, inter_dim, L).
        info : dict
            Metadata dictionary containing:
            - ``feature_shape``: (F1, T1) spatial dimensions after downsampling
            - ``patch_size``: (ph, pw) patch dimensions used for tokenization
            - ``tokens``: total number of tokens L
            - ``channels``: latent channel dimension (inter_dim)
            - ``image_size``: original input shape (F, T) INCLUDING Nyquist bin,
              required by the decoder to verify reconstruction dimensions
        """
        # Store original input shape INCLUDING Nyquist for decoder verification
        original_freq, original_time = x.shape[-2], x.shape[-1]
        original_image_size = (original_freq, original_time)
        
        # Remove Nyquist bin (last frequency bin) for even frequency dimensions
        # The Nyquist bin carries limited information and its removal ensures
        # compatibility with patch-based tokenization
        x = x[..., :-1, :]
        if debug:
            ok(f"Encoder after Nyquist removal: {x.shape} (original F={original_freq})")
        
        x = self.conv0(x)
        if debug:
            ok(f"Encoder after conv0 shape: {x.shape}")
        x = self.act0(x)
        x = self.conv1(x)                              # (B, D, F', T')
        if debug:
            ok(f"Encoder after conv1 shape: {x.shape}")
        x = self.resnet(x)                            # (B, D, F', T')
        if debug:
            ok(f"Encoder after resnet shape: {x.shape}")
        x = self.downsampling(x)                      # (B, D, F1, T1)
        if debug:
            ok(f"Encoder after downsampling shape: {x.shape}")
        F1, T1 = x.shape[-2:]
        self._last_feature_shape = (F1, T1)
        if F1 % self.patch_size[0] != 0 or T1 % self.patch_size[1] != 0:
            raise ValueError(
                f"Downsampled feature map {F1}x{T1} not divisible by patch_size {self.patch_size}. "
                "Adjust patch_size or downsample_ratio so the grid aligns."
            )
        x = self.unfold(x)                         # (B, P, L)
        if debug:
            ok(f"Encoder after unfold shape: {x.shape}")
        x = x.transpose(1, 2).contiguous()         # (B, L, P)
        x = self.linear_proj(x)                    # (B, L, D)
        if debug:
            ok(f"Encoder after linear_proj shape: {x.shape}")

        if not skip_attn:
            for blk in self.attn_blocks:
                x = blk(x, debug=debug)                       # (B, L, D)
                if debug:
                    ok(f"Encoder after attention block shape: {x.shape}")

        x = self.linear(self.norm(x))              # (B, L, D)
        if debug:
            ok(f"Encoder after final norm+linear shape: {x.shape}")
        # se vuoi uscire come (B, D, L) per compatibilità col decoder:
        x = x.transpose(1, 2).contiguous()         # (B, D, L)
        latent_info: Dict[str, tp.Any] = {
            "feature_shape": self._last_feature_shape,
            "patch_size": self.patch_size,
            "tokens": x.shape[-1],
            "channels": x.shape[1],
            "image_size": original_image_size,  # Original shape INCLUDING Nyquist for decoder
        }
        return x, latent_info


class SimpleTransformerDecoder(AbastractDecoder):
    """Vision Transformer-style decoder for complex-valued spectrogram reconstruction.

    Parameters
    ----------
    channels : int
        Number of output channels in the reconstructed spectrogram. Must match
        the encoder's ``input_size`` for consistent reconstruction.
    dim : int, default=4
        Number of feature channels in the convolutional layers. Must match the
        encoder's ``dim`` parameter for architectural symmetry.
    inter_dim : int, default=128
        Dimensionality of the Transformer token embeddings. Must match the
        encoder's ``inter_dim`` for compatible latent representations.
    n_heads : int, default=4
        Number of attention heads in the multi-head self-attention mechanism.
        Must evenly divide ``inter_dim``.
    depth : int, default=6
        Number of stacked Transformer blocks for token refinement before
        spatial reconstruction.
    dropout_rate : float, default=0.1
        Dropout probability applied within attention and feed-forward layers.
    activation : str, default="CReLU"
        Activation function used in convolutional and residual layers.
        For complex-valued networks, "CReLU" is recommended.
    feature_shape : tuple[int, int], optional
        Spatial dimensions (freq, time) of the downsampled feature map before
        patch tokenization in the encoder. Can be provided at init or passed
        dynamically during forward from the encoder's metadata.
    patch_size : int or tuple[int, int], default=(8, 8)
        Patch dimensions (height, width) used for token folding back to spatial
        feature maps. Must match the encoder's ``patch_size``.
    deconv_out_padding : int or tuple, default=((0, 1), (0, 0))
        Asymmetric output padding for the transposed convolution that inverts
        the encoder's conv1. Format: ((top, bottom), (left, right)). The default
        ((0, 1), (0, 0)) adds one frequency bin at the bottom to recover the
        Nyquist bin dimension (e.g., 1024 -> 1025).
    final_out_padding : int or tuple, default=((0, 0), (0, 0))
        Asymmetric output padding for the final transposed convolution layer.
    image_size : tuple[int, int], optional
        Expected output spectrogram dimensions (freq_bins, time_frames)
        INCLUDING the Nyquist bin. Used for assertion-based verification of
        reconstruction correctness. Should match the encoder's input shape.
    is_complex : bool, default=True
        Whether the decoder operates on complex-valued tensors.
    use_pos_enc : bool, default=True
        Whether to apply positional encoding to tokens before self-attention.
    n_residual_layers : int, default=1
        Number of residual blocks after upsampling.
    norm : str, default="none"
        Normalization scheme for convolutional layers.
    residual_kernel_size : int, default=3
        Kernel size for convolutions within residual blocks.
    dilation_base : int, default=2
        Base for exponential dilation growth in residual blocks.
    norm_params : dict, default={}
        Additional parameters for normalization layers.
    activation_params : dict, default={}
        Additional parameters for activation functions.
    causal : bool, default=False
        Whether to use causal padding in convolutions.
    true_skip : bool, default=False
        If True, residual blocks use identity skip connections.
    compress : int, default=2
        Channel compression factor within residual blocks.
    pad_mode : str, default='reflect'
        Padding mode for convolutional layers.
    conv_group_ratio : int, default=-1
        Group convolution ratio. -1 disables grouping.
    downsample_ratio : int or tuple[int, int], default=4
        Spatial upsampling factor (freq, time). Must match the encoder's
        ``downsample_ratio`` for symmetric reconstruction.
    upsampling_out_padding : int or tuple, default=0
        Output padding for the transposed convolution upsampling layer.

    Attributes
    ----------
    image_size : tuple[int, int] or None
        Expected output shape (F, T) including Nyquist. Set at init or
        dynamically from encoder metadata during forward.
    fold : nn.Fold or None
        Lazily instantiated Fold module for reconstructing spatial feature maps
        from token sequences.

    Returns
    -------
    reconstruction : torch.Tensor
        Reconstructed complex spectrogram of shape (B, channels, F, T) where
        (F, T) matches ``image_size`` (including Nyquist bin).

    Raises
    ------
    AssertionError
        If the output spatial dimensions do not match ``image_size``.

    Notes
    -----
    The decoder validates its output shape against ``image_size`` when provided.
    This assertion catches dimensional mismatches early, which is critical for
    ensuring correct STFT reconstruction and downstream iSTFT compatibility.
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
        feature_shape: tp.Optional[Tuple[int, int]] = None,          # (F1, T1) dopo downsampling encoder
        patch_size: Union[int, Tuple[int, int]] = (8, 8), # = encoder.patch_size
        deconv_out_padding: Union[int, tuple] = ((0, 1), (0, 0)),
        final_out_padding: Union[int, tuple] = ((0, 0), (0, 0)),
        image_size: tp.Optional[Tuple[int, int]] = None,
        is_complex: bool = True,
        use_pos_enc: bool = True,
        n_residual_layers: int = 1,
        norm: str = "none",
        residual_kernel_size: int = 3, 
        dilation_base: int = 2,
        norm_params: dict = {},
        activation_params: dict = {},
        causal: bool = False,
        true_skip: bool = False, 
        compress: int = 2,
        pad_mode: str = 'reflect',
        conv_group_ratio: int = -1,
        downsample_ratio: Union[int, Tuple[int, int]] = 4,
        upsampling_out_padding: Union[int, tuple] = 0,
    ) -> None:
        super().__init__(channels=channels, is_complex=is_complex)
        self.channels = channels
        self.dim = dim
        self.n_heads = n_heads
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.inter_dim = inter_dim
        self.use_pos_enc = use_pos_enc
        self.feature_shape = feature_shape
        self.downsample_ratio = pair(downsample_ratio)
        # image_size stores the expected OUTPUT spectrogram shape INCLUDING Nyquist bin
        # Used for assertion-based verification of reconstruction dimensions
        image_height, image_width = pair(image_size) if image_size is not None else (None, None)
        self.image_size = (image_height, image_width) if image_size is not None else None

        ph, pw = pair(patch_size)
        self.patch_size = (ph, pw)

        self.P = dim * ph * pw  # = dim*patch_area
        self._fold_output_size: tp.Optional[Tuple[int, int]] = None
        self.fold: tp.Optional[nn.Fold] = None

        # Token stack (uguale all’encoder): (B,L,dim) -> (B,L,dim)
        self.norm_tok = ComplexLayerNorm(inter_dim)
        self.linear_tok = nn.Linear(inter_dim, inter_dim, dtype=torch.complex64)

        self.blocks = nn.ModuleList([
            Transformer(
                n_heads=n_heads,
                feat_dim=inter_dim,
                dropout_rate=dropout_rate,
                use_pos_enc=self.use_pos_enc,
            )
            for _ in range(depth)
        ])

        # unprojection: dim -> P per ogni token
        self.linear_unproj = nn.Linear(inter_dim, self.P, dtype=torch.complex64)

        # Fold is built lazily once we know the feature map size coming from the encoder
        
        upsampling: tp.List[nn.Module] = []
        up_kernel = tuple(2 * s for s in self.downsample_ratio)
        upsampling += [
            SConvTranspose2d(
                self.dim,
                self.dim,
                kernel_size=up_kernel,
                stride=self.downsample_ratio,
                is_complex=True,
                norm=norm,
                out_padding=upsampling_out_padding,
            ),
            get_activation(self.activation, is_complex=True),
        ]
        self.upsampling = nn.Sequential(*upsampling)
        
        resnet : tp.List[nn.Module] = []
        for j in range(n_residual_layers): # This is always 1, parameter never gets changed from default anywhere
            resnet += [
                SEANetResnetBlock2d(self.dim,
                                    kernel_sizes=[(residual_kernel_size, residual_kernel_size), (1, 1)],
                                    dilations=[(1, dilation_base ** j), (1, 1)],
                                    norm=norm, norm_params=norm_params,
                                    activation=activation, activation_params=activation_params,
                                    causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip,
                                    conv_group_ratio=conv_group_ratio, is_complex=is_complex)]
            
        self.resnet = nn.Sequential(*resnet)
        

        # Deconv per invertire conv1: (B, dim, F1, T1) -> (B, C, F, T) circa
        self.deconv1 = SConvTranspose2d(
            dim,
            channels,
            kernel_size=5, stride=1,
            out_padding=deconv_out_padding,
            is_complex=True,
            norm=norm,
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
            norm=norm,
        )

    def forward(
        self,
        z: torch.Tensor,
        feature_shape: tp.Optional[Tuple[int, int]] = None,
        image_size: tp.Optional[Tuple[int, int]] = None,
        skip_attn: bool = False,
        debug: bool = False,
    ) -> torch.Tensor:
        """Decode tokens into a spectrogram.

        Parameters
        ----------
        z : torch.Tensor
            Latent tokens shaped (B, inter_dim, L) from the paired encoder.
        feature_shape : tuple[int, int], optional
            Downsampled spatial shape (freq, time) before patching. If not
            provided, uses ``self.feature_shape`` populated earlier.
        image_size : tuple[int, int], optional
            Expected output spectrogram shape (freq, time) INCLUDING Nyquist
            bin. Passed from encoder's info dict. Used for output verification.
            If not provided, uses ``self.image_size`` set at init.
        skip_attn : bool
            If True, bypasses the Transformer stack for ablation or speed.
        debug : bool
            If True, prints tensor shapes at key steps.

        Returns
        -------
        torch.Tensor
            Reconstructed spectrogram of shape (B, channels, F, T).

        Raises
        ------
        AssertionError
            If output shape does not match ``image_size`` when specified.
        """
        # Update image_size from forward argument if provided
        if image_size is not None:
            self.image_size = pair(image_size)
        
        if debug:
            ok(f"Decoder input (latent) shape: {z.shape}")

        B, D, L = z.shape
        assert D == self.inter_dim, f"Decoder expects inter_dim={self.inter_dim}, but received D={D}"

        target_shape = feature_shape or self.feature_shape
        if target_shape is None:
            raise ValueError(
                "Decoder needs the encoder feature_shape (F1, T1). "
                "Pass feature_shape=encoder._last_feature_shape or supply it on the first call."
            )
        F1, T1 = pair(target_shape)
        self.feature_shape = (F1, T1)

        if F1 % self.patch_size[0] != 0 or T1 % self.patch_size[1] != 0:
            raise ValueError(
                f"feature_shape {F1}x{T1} not divisible by patch_size {self.patch_size}."
            )

        L_expected = (F1 // self.patch_size[0]) * (T1 // self.patch_size[1])
        assert L == L_expected, (
            f"Decoder expects L={L_expected} tokens for feature_shape={F1}x{T1} and patch_size={self.patch_size}, "
            f"but received L={L}."
        )

        if self.fold is None or self._fold_output_size != (F1, T1):
            self.fold = nn.Fold(output_size=(F1, T1), kernel_size=self.patch_size, stride=self.patch_size)
            self._fold_output_size = (F1, T1)

        x = z.transpose(1, 2).contiguous()  # (B, L, dim)

        # norm+linear token-wise (feature-last)
        x = self.linear_tok(self.norm_tok(x))  # (B, L, dim)
        if debug:
            ok(f"Decoder after token norm+linear shape: {x.shape}")

        # Transformer blocks
        if not skip_attn:
            for blk in self.blocks:
                x = blk(x, debug=debug)  # (B, L, dim)

            if debug:
                ok(f"Decoder after attention stack shape: {x.shape}")

        # unproject tokens -> patch vectors
        x = self.linear_unproj(x)                # (B, L, P)
        x = x.transpose(1, 2).contiguous()       # (B, P, L)
        if debug:
            ok(f"Decoder after unprojection shape: {x.shape}")

        # Fold back to feature map (B, dim, F1, T1)
        fm = self.fold(x)  # (B, dim, F1, T1)
        
        if debug:
            ok(f"Decoder after fold shape: {fm.shape}")
            
        fm = self.upsampling(fm)                      # (B, dim, ~F, ~T)
        if debug:
            ok(f"Decoder after upsampling shape: {fm.shape}")
            
        fm = self.resnet(fm)                      # (B, dim, F1, T1)
        if debug:
            ok(f"Decoder after resnet shape: {fm.shape}")

        # Invert conv1
        y = self.deconv1(fm)
        y = self.act(y)
        y = self.final_conv(y)

        if debug:
            ok(f"Decoder output shape: {y.shape}")

        # Verify output dimensions match expected image_size (including Nyquist)
        if self.image_size is not None:
            expected_F, expected_T = self.image_size
            actual_F, actual_T = y.shape[-2], y.shape[-1]
            assert actual_F == expected_F and actual_T == expected_T, (
                f"Decoder output shape mismatch: expected ({expected_F}, {expected_T}) "
                f"but got ({actual_F}, {actual_T}). Check deconv_out_padding and "
                f"final_out_padding parameters to ensure correct Nyquist bin reconstruction."
            )

        return y


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    # Input spectrogram: (B, C, F, T) where F=1025 includes Nyquist bin
    dummy = torch.zeros((32, 2, 1025, 128), dtype=torch.complex64).to(device)

    # Encoder: input_size matches dummy channels (2).
    # image_size=(1025, 128) is the ORIGINAL shape INCLUDING Nyquist
    encoder = SimpleTransformerEncoder(
        input_size=2,
        dim=4,
        n_heads=4,
        depth=2,
        dropout_rate=0.1,
        patch_size=(16, 16),
        image_size=(1025, 128),  # Original shape with Nyquist (F-1=1024 must be divisible by patch_size)
        use_pos_enc=True,
        activation="CRelu",
        inter_dim=64,
        downsample_ratio=(1, 1),
    ).to(device)

    # Decoder: channels=2 to match original input channels.
    # image_size=(1025, 128) ensures output matches original spectrogram dimensions
    decoder = SimpleTransformerDecoder(
        channels=2,
        dim=4,
        n_heads=4,
        depth=2,
        dropout_rate=0.1,
        activation="CRelu",
        patch_size=(16, 16),
        inter_dim=64,
        deconv_out_padding=((0, 1), (0, 0)),  # Adds 1 freq bin to recover Nyquist (1024 -> 1025)
        image_size=(1025, 128),  # Expected output shape INCLUDING Nyquist
        downsample_ratio=(1, 1),
    ).to(device)

    with torch.no_grad():
        latent, info = encoder(dummy, debug=True)
        # Pass image_size from encoder info to decoder for verification
        recon = decoder(
            latent, 
            feature_shape=info["feature_shape"], 
            image_size=info["image_size"],
            debug=True
        )

    print(f"Input shape:   {tuple(dummy.shape)}")
    print(f"Latent shape:  {tuple(latent.shape)}")
    print(f"Output shape:  {tuple(recon.shape)}")
    print(f"Reconstruction matches input shape: {dummy.shape == recon.shape}")
