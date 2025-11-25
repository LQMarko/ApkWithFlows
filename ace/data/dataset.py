"""Dataset and preprocessing utilities for ACE.

The key challenge in building the ACE dataset is converting raw APK
files into a form consumable by a neural network.  The paper
proposes reading each APK's ``classes.dex`` section as a sequence of
bytes, mapping each byte to an integer in the range [0, 255], then
reshaping that sequence into a square matrix.  Because DEX files vary
in length, zero‑padding is applied and bilinear interpolation is used
to resize all matrices to a common size (e.g. 512×512).  This module
implements those steps and wraps them in a ``torch.utils.data.Dataset``.

In addition, a utility is provided to read the labels CSV supplied by
the user.  The CSV must contain at least ``apk_name`` and ``label``
columns.  Additional columns (such as malware family identifiers) are
ignored but preserved in the returned dataframe for downstream use if
desired.
"""

from __future__ import annotations

import csv
import io
import os
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import pandas as pd

# We only import torch conditionally because the container may not have
# it installed.  Downstream scripts should ensure that torch is
# available when actually executing training.
try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore
    transforms = None  # type: ignore


def load_labels(labels_csv: Path) -> pd.DataFrame:
    """Load the labels CSV into a pandas DataFrame.

    Parameters
    ----------
    labels_csv : Path
        Path to the CSV file containing at least ``apk_name`` and
        ``label`` columns.  The ``apk_name`` is expected to be a
        unique identifier for each APK in the dataset, while ``label``
        indicates whether the APK is benign (0) or malicious (1).  Any
        additional columns are preserved for downstream use (e.g. to
        identify malware families), but they are not required for basic
        binary classification.

    Returns
    -------
    DataFrame
        A pandas DataFrame with the contents of the CSV.  The
        ``apk_name`` column is left unchanged; clients are responsible
        for matching this identifier to filenames on disk.
    """
    df = pd.read_csv(labels_csv)
    required_cols = {"apk_name", "label"}
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        raise ValueError(f"Labels CSV missing required columns: {missing}")
    return df


def _dex_to_matrix(dex_bytes: bytes) -> np.ndarray:
    """Convert a sequence of bytes into a 2‑D square matrix.

    The DEX file is read as raw bytes.  Each byte is interpreted as
    an unsigned integer in the range [0, 255] and placed into a 1‑D
    array.  That array is then reshaped into the smallest possible
    square matrix by computing ``sqrt(len)`` and padding with zeros
    if necessary.  Finally, the matrix is returned as a 2‑D numpy
    array of dtype ``uint8``.

    Parameters
    ----------
    dex_bytes : bytes
        Raw contents of a DEX file.

    Returns
    -------
    np.ndarray
        A square matrix representing the DEX file, zero‑padded on the
        right and bottom as needed.
    """
    # Convert bytes into a 1‑D numpy array of uint8
    arr = np.frombuffer(dex_bytes, dtype=np.uint8)
    length = len(arr)
    # Compute the smallest integer >= sqrt(length)
    side = int(np.ceil(np.sqrt(length)))
    # Create zero‑padded array
    padded = np.zeros(side * side, dtype=np.uint8)
    padded[:length] = arr
    matrix = padded.reshape((side, side))
    return matrix


def _resize_matrix(mat: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a 2‑D matrix to ``target_size``×``target_size`` via bilinear interpolation.

    Parameters
    ----------
    mat : np.ndarray
        Input square matrix.
    target_size : int
        Desired side length of the output matrix.

    Returns
    -------
    np.ndarray
        The resized matrix as a 2‑D array of dtype ``uint8``.  Values
        are clipped to [0, 255] and rounded to the nearest integer.
    """
    img = Image.fromarray(mat)
    img = img.resize((target_size, target_size), resample=Image.BILINEAR)
    # Convert back to numpy array
    return np.array(img, dtype=np.uint8)


def _extract_dex_bytes(apk_path: Path) -> bytes:
    """Extract the ``classes.dex`` file from an APK archive.

    APK files are essentially ZIP archives.  This helper searches for
    the first file whose name ends with ``.dex`` and returns its
    contents as bytes.  If no DEX file is found, ``FileNotFoundError``
    is raised.

    Parameters
    ----------
    apk_path : Path
        Path to the APK file to extract from.

    Returns
    -------
    bytes
        The raw contents of the first ``.dex`` file in the APK.
    """
    with zipfile.ZipFile(apk_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(".dex"):
                with zf.open(name) as f:
                    return f.read()
    raise FileNotFoundError(f"No DEX file found in {apk_path}")


class DexDataset(Dataset):  # type: ignore[misc]
    """PyTorch dataset yielding DEX images and their labels.

    Each item produced by this dataset is a tuple ``(image, label, meta)``.
    ``image`` is a tensor of shape ``(1, H, W)`` with values in
    ``[0.0, 1.0]``.  ``label`` is an integer 0 or 1 indicating benign or
    malicious.  ``meta`` is a dictionary containing auxiliary
    information such as the original APK name and any additional
    columns from the labels CSV.
    """

    def __init__(
        self,
        data_dir: Path,
        labels_df: pd.DataFrame,
        image_size: int = 512,
        transform: Optional[Callable] = None,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is not available. Please install PyTorch to use DexDataset."
            )
        self.data_dir = Path(data_dir)
        self.labels_df = labels_df
        self.image_size = image_size
        # Use a default transform if none is provided
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Precompute a mapping from apk_name to filepath
        # We look for files matching the apk_name anywhere under data_dir.
        # Only files ending in .apk are considered; if multiple files match
        # the same apk_name, the first occurrence is used.
        self.name_to_path: Dict[str, Path] = {}
        for root, _dirs, files in os.walk(self.data_dir):
            for fname in files:
                if not fname.lower().endswith(".apk"):
                    continue
                stem = Path(fname).stem
                if stem not in self.name_to_path:
                    self.name_to_path[stem] = Path(root) / fname

        # Verify that all labels correspond to existing files
        missing = []
        for name in self.labels_df["apk_name"]:
            if name not in self.name_to_path:
                missing.append(name)
        if missing:
            raise FileNotFoundError(
                f"The following APKs listed in the labels CSV were not found in {data_dir}: {missing[:10]}"
            )

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, str]]:  # type: ignore[override]
        row = self.labels_df.iloc[idx]
        apk_name = row["apk_name"]
        label = int(row["label"])

        apk_path = self.name_to_path[apk_name]

        # Read classes.dex and convert to image
        dex_bytes = _extract_dex_bytes(apk_path)
        matrix = _dex_to_matrix(dex_bytes)
        matrix = _resize_matrix(matrix, self.image_size)
        # Normalise to [0,1]
        img = Image.fromarray(matrix)
        tensor = self.transform(img)  # shape (1,H,W)

        # Build metadata dictionary with all columns except label
        meta = {col: row[col] for col in row.index if col != "label"}

        return tensor, label, meta
