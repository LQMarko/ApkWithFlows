"""Dataset for pre‑extracted features.

This dataset assumes that features have been extracted from APK files
and saved as numpy arrays in a directory.  Each feature file is
expected to have a filename corresponding to the SHA256 hash of the
original APK (e.g. ``89C49B6D... .npy``) and to contain a 2‑D
matrix (typically 512×512) representing the DEX image.  Labels are
provided via a CSV file with columns ``sha256`` and ``label``.

The dataset yields tuples ``(tensor, label)`` where ``tensor`` is a
1×H×W PyTorch tensor and ``label`` is an integer (0 or 1).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore
    transforms = None  # type: ignore


class FeatureDataset(Dataset):  # type: ignore[misc]
    """PyTorch dataset loading precomputed features from disk."""

    def __init__(
        self,
        feature_dir: Path,
        csv_path: Path,
        transform: Optional[Callable] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not available. Please install PyTorch to use FeatureDataset.")
        self.feature_dir = Path(feature_dir)
        self.df = pd.read_csv(csv_path)
        required_cols = {"sha256", "label"}
        if not required_cols.issubset(self.df.columns):
            missing = required_cols.difference(self.df.columns)
            raise ValueError(f"CSV missing required columns: {missing}")
        # Default transform converts numpy array to tensor and normalises to [0,1]
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:  # type: ignore[override]
        row = self.df.iloc[idx]
        sha256 = row["sha256"]
        label = int(row["label"])
        # Load feature file
        feature_path = self.feature_dir / f"{sha256}.npy"
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file {feature_path} not found")
        arr = np.load(feature_path)
        # Ensure 2‑D array
        if arr.ndim != 2:
            raise ValueError(f"Feature array {feature_path} has invalid shape {arr.shape}")
        img = arr.astype(np.uint8)
        tensor = self.transform(img)  # shape (1,H,W)
        return tensor, label