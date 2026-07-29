'''Discriminador multi-escala tipo MelGAN.'''

from __future__ import annotations
from torch.nn.utils.parametrizations import weight_norm

import torch
import torch.nn as nn
import torch.nn.functional as F


def _wn(module: nn.Module) -> nn.Module:
    return weight_norm(module)


class ScaleDiscriminator(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...] = (16, 64, 256, 512, 512),
        kernel_sizes: tuple[int, ...] = (15, 41, 41, 41, 5),
        strides: tuple[int, ...] = (1, 4, 4, 4, 1),
        groups: tuple[int, ...] = (1, 4, 16, 64, 1),
    ):
        super().__init__()
        n_layers = len(channels)
        if not (n_layers == len(kernel_sizes) == len(strides) == len(groups)):
            raise ValueError('channels, kernel_sizes, strides y groups deben tener el mismo largo')

        layers = []
        c_prev = 1
        for c, k, s, g in zip(channels, kernel_sizes, strides, groups):
            layers.append(_wn(nn.Conv1d(c_prev, c, kernel_size=k, stride=s, padding=k // 2, groups=g)))
            c_prev = c
        self.layers = nn.ModuleList(layers)
        self.out_conv = _wn(nn.Conv1d(c_prev, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor):
        features = []
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, 0.2)
            features.append(x)
        logit = self.out_conv(x)
        return features, logit


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_scales: int = 3, pool_kernel: int = 4, pool_stride: int = 2, scale_kwargs: dict | None = None):
        super().__init__()
        scale_kwargs = scale_kwargs or {}
        self.discriminators = nn.ModuleList([ScaleDiscriminator(**scale_kwargs) for _ in range(n_scales)])
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_kernel // 2)

    def forward(self, x: torch.Tensor):
        outputs = []
        y = x
        for d in self.discriminators:
            outputs.append(d(y))
            y = self.pool(y)
        return outputs
