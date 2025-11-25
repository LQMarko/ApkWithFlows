"""Contrastive encoder implementation.

The contrastive encoder takes as input the latent representation
produced by the contractive autoencoder and maps it through a stack of
convolutional layers to produce a feature map suitable for computing
SSIM and applying the contrastive loss.  Following the paper, we
implement four convolutional layers.  The stride and padding are
chosen to downsample the input progressively while retaining spatial
structure.  Batch normalisation and ReLU activations are used to
stabilise training.

The output of the contrastive encoder is a tensor of shape
``(batch_size, embed_channels, H, W)`` where ``H`` and ``W`` depend
on the input size and stride configuration.  To compute SSIM, these
channels can be collapsed (e.g. averaged) to produce a 2‑D matrix per
sample.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ContrastiveEncoder(nn.Module):
    """Convolutional encoder for producing contrastive features."""

    def __init__(self, in_channels: int = 64, embed_channels: int = 64) -> None:
        super().__init__()
        # Four convolutional layers.  Each halves the spatial dimensions
        # using stride=2.  The number of channels increases to allow
        # richer representations.  BatchNorm layers stabilise training.
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, embed_channels, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(embed_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = self.activation(self.bn1(self.conv1(x)))
        h = self.activation(self.bn2(self.conv2(h)))
        h = self.activation(self.bn3(self.conv3(h)))
        h = self.activation(self.bn4(self.conv4(h)))
        return h