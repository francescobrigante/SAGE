# =============================================================================
# Mathematical and tensor-shape alignment utilities for frequency and time dimensions.
# =============================================================================

import logging
from typing import Dict
import numpy as np
import torch

def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def to_torch_tensor(x):
    """Converte in torch.Tensor (reale o complesso) da numpy/torch o da dict {'real','imag'}."""
    # numpy.ndarray -> torch.Tensor (supporta anche dtype complessi di numpy)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)

    # dict {'real': ..., 'imag': ...} -> tensore complesso nativo
    if isinstance(x, dict):
        if "real" not in x or "imag" not in x:
            raise ValueError("serve un dict con chiavi 'real' e 'imag', ricevuto: {}".format(list(x)))
        real = to_torch_tensor(x["real"])
        imag = to_torch_tensor(x["imag"])
        if not isinstance(real, torch.Tensor) or not isinstance(imag, torch.Tensor):
            raise ValueError("valori 'real' e 'imag' non convertibili in torch.Tensor")

        # se arrivano complessi, usa solo la parte reale
        if real.is_complex():
            real = real.real
        if imag.is_complex():
            imag = imag.real

        # broadcast shape e armonizza dtype
        real, imag = torch.broadcast_tensors(real, imag)

        # impone float32/float64 perché torch.complex richiede input floating
        def _to_float(t):
            if t.dtype in (torch.float64, torch.double):
                return t.to(torch.float64)
            # per qualsiasi altro dtype forza float32
            return t.to(torch.float32)

        real = _to_float(real)
        imag = _to_float(imag)

        # se uno è float64, porta entrambi a float64 per ottenere complex128
        tgt = torch.float64 if (real.dtype is torch.float64 or imag.dtype is torch.float64) else torch.float32
        real = real.to(tgt)
        imag = imag.to(tgt)

        return torch.complex(real, imag)

    # torch.Tensor -> restituito così com'è (reale o complesso)
    if isinstance(x, torch.Tensor):
        return x

    # tipi non supportati
    raise ValueError(
        "x deve essere numpy.ndarray, torch.Tensor oppure dict "
        "{'real': <tensor/ndarray>, 'imag': <tensor/ndarray>}, ma è {}".format(type(x))
    )


def get_subsample(train_args, mode, arch):
    """Parse the subsampling factors from the args for the specified `mode` and `arch`.

    Args:
        train_args: argument Namespace containing options.
        mode: one of ('asr', 'mt', 'st')
        arch: one of ('rnn', 'rnn-t', 'rnn_mix', 'rnn_mulenc', 'transformer')

    Returns:
        np.ndarray / List[np.ndarray]: subsampling factors.
    """
    if arch == "transformer":
        return np.array([1])

    elif mode == "mt" and arch == "rnn":
        # +1 means input (+1) and layers outputs (train_args.elayer)
        subsample = np.ones(train_args.elayers + 1, dtype=np.int)
        logging.warning("Subsampling is not performed for machine translation.")
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif (
            (mode == "asr" and arch in ("rnn", "rnn-t"))
            or (mode == "mt" and arch == "rnn")
            or (mode == "st" and arch == "rnn")
    ):
        subsample = np.ones(train_args.elayers + 1, dtype=np.int)
        if train_args.etype.endswith("p") and not train_args.etype.startswith("vgg"):
            ss = train_args.subsample.split("_")
            for j in range(min(train_args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                "Subsampling is not performed for vgg*. "
                "It is performed in max pooling layers at CNN."
            )
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif mode == "asr" and arch == "rnn_mix":
        subsample = np.ones(
            train_args.elayers_sd + train_args.elayers + 1, dtype=np.int
        )
        if train_args.etype.endswith("p") and not train_args.etype.startswith("vgg"):
            ss = train_args.subsample.split("_")
            for j in range(
                    min(train_args.elayers_sd + train_args.elayers + 1, len(ss))
            ):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                "Subsampling is not performed for vgg*. "
                "It is performed in max pooling layers at CNN."
            )
        logging.info("subsample: " + " ".join([str(x) for x in subsample]))
        return subsample

    elif mode == "asr" and arch == "rnn_mulenc":
        subsample_list = []
        for idx in range(train_args.num_encs):
            subsample = np.ones(train_args.elayers[idx] + 1, dtype=np.int)
            if train_args.etype[idx].endswith("p") and not train_args.etype[
                idx
            ].startswith("vgg"):
                ss = train_args.subsample[idx].split("_")
                for j in range(min(train_args.elayers[idx] + 1, len(ss))):
                    subsample[j] = int(ss[j])
            else:
                logging.warning(
                    "Encoder %d: Subsampling is not performed for vgg*. "
                    "It is performed in max pooling layers at CNN.",
                    idx + 1,
                )
            logging.info("subsample: " + " ".join([str(x) for x in subsample]))
            subsample_list.append(subsample)
        return subsample_list

    else:
        raise ValueError("Invalid options: mode={}, arch={}".format(mode, arch))


def rename_state_dict(
        old_prefix: str, new_prefix: str, state_dict: Dict[str, torch.Tensor]
):
    """Replace keys of old prefix with new prefix in state dict."""
    # need this list not to break the dict iterator
    old_keys = [k for k in state_dict if k.startswith(old_prefix)]
    if len(old_keys) > 0:
        logging.warning(f"Rename: {old_prefix} -> {new_prefix}")
    for k in old_keys:
        v = state_dict.pop(k)
        new_k = k.replace(old_prefix, new_prefix)
        state_dict[new_k] = v


def align_freq_bins(s_hat: torch.Tensor, s_ref: torch.Tensor) -> torch.Tensor:
    """
    Rende compatibili i tensori spettrali lungo l'asse delle frequenze:
    - se differenza di 1 bin, aggiunge/toglie l'ultimo bin (Nyquist)
    - altrimenti solleva errore
    Supporta tensori complessi o reali; assume F è l'asse -2.
    """
    Fh = s_hat.shape[-2]
    Fr = s_ref.shape[-2]
    if Fh == Fr:
        return s_hat
    if abs(Fh - Fr) == 1:
        if Fh < Fr:
            pad_shape = list(s_hat.shape)
            pad_shape[-2] = 1
            pad = torch.zeros(pad_shape, dtype=s_hat.dtype, device=s_hat.device)
            return torch.cat([s_hat, pad], dim=-2)  # ripristina Nyquist
        else:
            return s_hat[..., :Fr, :]  # rimuove Nyquist extra
    raise ValueError(
        f"Spectrogram freq dim mismatch > 1: pred {Fh} vs target {Fr}. "
        "Controlla n_fft/hop oppure normalizza l'output del decoder."
    )


def align_time_frames(s_hat: torch.Tensor, s_ref: torch.Tensor) -> torch.Tensor:
    """
    Aligns the temporal dimension (last axis) of decoded spectrogram to reference.
    
    The encoder-decoder may produce more frames than the input due to
    non-integer downsampling ratios. This function trims the decoded
    spectrogram to match the reference, preventing misalignment in
    waveform reconstruction that would cause waveform losses to fail.
    
    Args:
        s_hat: Decoded spectrogram (B, C, F, T) or (B, C, T)
        s_ref: Reference spectrogram with target time dimension
        
    Returns:
        s_hat trimmed to match s_ref's time dimension
        
    Raises:
        ValueError: If decoder produced fewer frames than expected (indicates a bug).
    """
    Th = s_hat.shape[-1]
    Tr = s_ref.shape[-1]
    if Th == Tr:
        return s_hat
    if Th > Tr:
        # Decoder produced more frames - trim to match reference
        return s_hat[..., :Tr]
    # Decoder produced fewer frames - this is a bug, should never happen
    raise ValueError(
        f"Decoder produced fewer time frames than expected: got {Th}, expected {Tr}. "
        "This indicates a bug in the encoder-decoder architecture."
    )

