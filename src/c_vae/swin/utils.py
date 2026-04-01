# ===============================================================
# Swin Transformer V2 — shared weight-initialisation utility.
# init_swin_weights: truncated-normal for Linear, ones/zeros for
# LayerNorm. Apply once after model construction via .apply().
# ===============================================================

import torch.nn as nn
from timm.layers import trunc_normal_


def init_swin_weights(m: nn.Module) -> None:
    """Swin V2 weight initialisation.

    - ``nn.Linear``: truncated normal (std=0.02), zero bias.
    - ``nn.LayerNorm``: weight=1, bias=0.

    Usage::

        model.apply(init_swin_weights)
    """
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
