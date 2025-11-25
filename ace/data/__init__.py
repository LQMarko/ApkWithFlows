"""Dataset loading and preprocessing for the ACE malware detector.

This subpackage contains utilities for reading APK files, extracting
their DEX sections, converting those sections into 2‑D matrices and
resizing them into fixed‑size images.  It also defines a PyTorch
``Dataset`` that yields image tensors and labels for training.
"""

from .dataset import DexDataset, load_labels
from .feature_dataset import FeatureDataset

__all__ = ["DexDataset", "load_labels", "FeatureDataset"]