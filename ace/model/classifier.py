"""Fully connected classifier used in the ACE model.

The classifier takes the flattened output of the contrastive encoder
and produces a probability for the APK being malicious.  It uses
several fully connected (dense) layers interleaved with ReLU
activations and dropout for regularisation.  The final layer applies
a sigmoid to produce a value in [0,1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class Classifier(nn.Module):
    """Simple feed‑forward network for binary classification."""

    def __init__(
        self,
        in_features: int,
        hidden_sizes: Optional[List[int]] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            # Default hidden layer sizes loosely inspired by the paper
            hidden_sizes = [512, 256, 128]
        layers: List[nn.Module] = []
        prev = in_features
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev, hs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hs
        # Final output layer for binary classification
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Flatten the input except batch dimension
        x = x.view(x.size(0), -1)
        logits = self.net(x).squeeze(-1)
        return logits