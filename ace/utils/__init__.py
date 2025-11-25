"""Utility functions for ACE.

This package collects helper functions used throughout the ACE
implementation.  Currently it contains an implementation of the
structural similarity index (SSIM) adapted for use with PyTorch
and numpy arrays.
"""

from .metrics import ssim, ssim_tensor

__all__ = ["ssim", "ssim_tensor"]