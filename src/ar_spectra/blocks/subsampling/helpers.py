# =============================================================================
# Helper functions and validity checks (masks, temporal limits) for sub-sampler modules.
# =============================================================================

import torch

def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
	if maxlen is None:
		maxlen = lengths.max()
	row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
	matrix = torch.unsqueeze(lengths, dim=-1)
	mask = row_vector < matrix
	mask = mask.detach()

	return mask.type(dtype).to(device) if device is not None else mask.type(dtype)

class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling.

    Args:
        message (str): Message for error catch
        actual_size (int): the short size that cannot pass the subsampling
        limit (int): the limit size for subsampling

    """

    def __init__(self, message, actual_size, limit):
        """Construct a TooShortUttError for error handler."""
        super().__init__(message)
        self.actual_size = actual_size
        self.limit = limit

def check_short_utt(ins, size):
    """Check if the utterance is too short for subsampling."""
    if isinstance(ins, Conv2dSubsampling2) and size < 3:
        return True, 3
    if isinstance(ins, Conv2dSubsampling) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling6) and size < 11:
        return True, 11
    if isinstance(ins, Conv2dSubsampling8) and size < 15:
        return True, 15
    return False, -1
