from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseRefinementBlock(nn.Module):
    """A lightweight spatial refinement block suitable for small batches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.normalization = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.normalization(self.pointwise(self.depthwise(inputs))))
