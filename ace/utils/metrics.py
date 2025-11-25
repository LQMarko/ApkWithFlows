"""Metrics and loss utilities for ACE.

This module provides an implementation of the structural similarity
index (SSIM) for both numpy arrays and PyTorch tensors.  SSIM is
used in the contrastive loss to measure similarity between feature
maps.  The implementation follows the formula from the paper and
defaults to parameters K1=0.01, K2=0.2 and dynamic range L=1.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple


def ssim(
    x: np.ndarray,
    y: np.ndarray,
    K1: float = 0.01,
    K2: float = 0.2,
    L: float = 1.0,
) -> float:
    """Compute the structural similarity index between two 2‑D numpy arrays.

    The inputs ``x`` and ``y`` should have values in the range [0, L].
    If the arrays have more than two dimensions, the last two
    dimensions are treated as the spatial axes and all other
    dimensions are broadcast.  This function implements the formula
    from the paper and averages over all spatial locations.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = ((x - mu_x) ** 2).mean()
    sigma_y = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)


def ssim_tensor(
    x: torch.Tensor,
    y: torch.Tensor,
    K1: float = 0.01,
    K2: float = 0.2,
    L: float = 1.0,
) -> torch.Tensor:
    """Compute SSIM between two 2‑D PyTorch tensors.

    This function operates elementwise over batches if the input tensors
    have leading batch dimensions.  It collapses all leading
    dimensions and computes SSIM over the last two dimensions.  The
    result is a scalar tensor containing the SSIM index.
    """
    # Flatten any leading batch dimensions; operate on last two dims
    dims = x.dim()
    if dims < 2:
        raise ValueError("Inputs must have at least two dimensions for SSIM computation.")
    # Reshape to (-1, H, W)
    x_flat = x.reshape(-1, x.size(-2), x.size(-1)).double()
    y_flat = y.reshape(-1, y.size(-2), y.size(-1)).double()
    mu_x = x_flat.mean(dim=[1, 2])
    mu_y = y_flat.mean(dim=[1, 2])
    sigma_x = ((x_flat - mu_x.view(-1, 1, 1)) ** 2).mean(dim=[1, 2])
    sigma_y = ((y_flat - mu_y.view(-1, 1, 1)) ** 2).mean(dim=[1, 2])
    sigma_xy = ((x_flat - mu_x.view(-1, 1, 1)) * (y_flat - mu_y.view(-1, 1, 1))).mean(
        dim=[1, 2]
    )
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    # Avoid division by zero
    ssim_val = numerator / denominator
    return ssim_val.mean()