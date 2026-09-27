# =============================================================================
# Self-supervised feature loss explored during development and NOT used by the
# paper recipes (downloads a torchaudio pipeline on first use).
# =============================================================================
import typing as tp

import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from sage.utils.audio import fold_channels_into_batch


class HubertLoss(nn.Module):
    def __init__(self,
        feature_ids: tp.Optional[tp.List[int]] = None,
        weight: float = 1.0,
        model_name: str = "HUBERT_LARGE"
    ):
        super().__init__()
        self.weight = weight
        self.feature_ids = feature_ids
        self.model_name = model_name
        
        if self.model_name == "WAVLM_LARGE":
            bundle = torchaudio.pipelines.WAVLM_LARGE
        elif self.model_name == "HUBERT_LARGE":
            bundle = torchaudio.pipelines.HUBERT_LARGE
        elif self.model_name == "WAV2VEC2_LARGE_LV60K":
            bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

        self.model = bundle.get_model()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = fold_channels_into_batch(x)
        y = fold_channels_into_batch(y)
        conv_features = (self.feature_ids is not None and len(self.feature_ids) == 1 and self.feature_ids[0] == -1)

        if conv_features:
            if self.model.normalize_waveform:
                x = nn.functional.layer_norm(x, x.shape)
                y = nn.functional.layer_norm(y, y.shape)
            x_list, _ = self.model.model.feature_extractor(x, None)
            y_list, _ = self.model.model.feature_extractor(y, None)
            x_list, y_list = [x_list], [y_list]
        else:
            x_list, _ = self.model.extract_features(x)
            y_list, _ = self.model.extract_features(y)

        loss, denom = 0, 0
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            if self.feature_ids is None or i in self.feature_ids or conv_features:
                loss += F.l1_loss(x, y) / (y.std() + 1e-5)
                denom += 1
        return self.weight * (loss / denom)
