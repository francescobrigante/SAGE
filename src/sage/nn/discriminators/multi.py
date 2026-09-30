
# =============================================================================
# Classic Multi-Scale (MSD) and Multi-Period (MPD) discriminators for audio/frequency validation.
# =============================================================================

from sage.nn.discriminators.types import IndividualDiscriminatorOut, TensorDict
from sage.nn.discriminators.encodec import SharedDiscriminatorConvNet
import torch
import torch.nn as nn
import typing as tp
from functools import reduce

class MultiScaleDiscriminator(nn.Module):

    def __init__(self,
                in_channels: int,
                n_scales: int,
                **conv_kwargs) -> None:
        super().__init__()
        layers = []
        for _ in range(n_scales):
            layers.append(SharedDiscriminatorConvNet(in_channels, nn.Conv1d, **conv_kwargs))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> IndividualDiscriminatorOut:
        score = 0
        features = []
        for layer in self.layers:
            s, f = layer(x)
            score = score + s
            features.extend(f)
            x = nn.functional.avg_pool1d(x, 2)
        return score, features

class MultiPeriodDiscriminator(nn.Module):

    def __init__(self,
                 in_channels: int,
                 periods: tp.Sequence[int],
                 **conv_kwargs) -> None:
        super().__init__()
        layers = []
        self.periods = periods

        for _ in periods:
            layers.append(SharedDiscriminatorConvNet(in_channels, nn.Conv2d, **conv_kwargs))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> IndividualDiscriminatorOut:
        score = 0
        features = []
        for layer, n in zip(self.layers, self.periods):
            s, f = layer(self.fold(x, n))
            score = score + s
            features.extend(f)
        return score, features

    def fold(self, x: torch.Tensor, n: int) -> torch.Tensor:
        pad = (n - (x.shape[-1] % n)) % n
        x = nn.functional.pad(x, (0, pad))
        return x.reshape(*x.shape[:2], -1, n)

class MultiDiscriminator(nn.Module):
    """
    Individual discriminators should take a single tensor as input (NxB C T) and
    return a tuple composed of a score tensor (NxB) and a Sequence of Features
    Sequence[NxB C' T'].
    """

    def __init__(self, discriminator_list: tp.Sequence[nn.Module],
                 keys: tp.Sequence[str]) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList(discriminator_list)
        self.keys = keys

    def unpack_tensor_to_dict(self, features: torch.Tensor) -> TensorDict:
        features = features.chunk(len(self.keys), 0)
        return {k: features[i] for i, k in enumerate(self.keys)}

    @staticmethod
    def concat_dicts(dict_a, dict_b):
        out_dict = {}
        keys = set(list(dict_a.keys()) + list(dict_b.keys()))
        for k in keys:
            out_dict[k] = []
            if k in dict_a:
                if isinstance(dict_a[k], list):
                    out_dict[k].extend(dict_a[k])
                else:
                    out_dict[k].append(dict_a[k])
            if k in dict_b:
                if isinstance(dict_b[k], list):
                    out_dict[k].extend(dict_b[k])
                else:
                    out_dict[k].append(dict_b[k])
        return out_dict

    @staticmethod
    def sum_dicts(dict_a, dict_b):
        out_dict = {}
        keys = set(list(dict_a.keys()) + list(dict_b.keys()))
        for k in keys:
            out_dict[k] = 0.
            if k in dict_a:
                out_dict[k] = out_dict[k] + dict_a[k]
            if k in dict_b:
                out_dict[k] = out_dict[k] + dict_b[k]
        return out_dict

    def forward(self, inputs: TensorDict) -> TensorDict:
        discriminator_input = torch.cat([inputs[k] for k in self.keys], 0)
        all_scores = []
        all_features = []

        for discriminator in self.discriminators:
            score, features = discriminator(discriminator_input)
            scores = self.unpack_tensor_to_dict(score)
            scores = {f"score_{k}": scores[k] for k in scores.keys()}
            all_scores.append(scores)

            features = map(self.unpack_tensor_to_dict, features)
            features = reduce(self.concat_dicts, features)
            features = {f"features_{k}": features[k] for k in features.keys()}
            all_features.append(features)

        all_scores = reduce(self.sum_dicts, all_scores)
        all_features = reduce(self.concat_dicts, all_features)

        inputs.update(all_scores)
        inputs.update(all_features)

        return inputs
