"""Configuration utilities for the ACE implementation.

This module defines defaults and helpers for configuring the
preprocessing and training pipeline.  A configuration object can be
constructed from environment variables or command‑line arguments to
specify locations of datasets, caching behaviour and model
hyperparameters.  Centralising configuration in a single module
makes it easy to adapt the implementation to new datasets or
experiment settings without modifying the core logic.
"""
from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Optional


@dataclass
class ACEConfig:
    """Holds configuration parameters for the ACE model pipeline.

    Attributes
    ----------
    data_dir: Path
        Directory containing the APK files to be processed.  Each
        APK's name must match the ``apk_name`` column in the
        provided labels CSV file.  The code expects that every
        sample listed in the labels CSV has a corresponding file in
        this directory.  The directory itself can be arbitrarily
        nested; only files ending in ``.apk`` will be considered.

    labels_csv: Path
        Path to a CSV file containing at least two columns:
        ``apk_name`` and ``label``.  ``apk_name`` is a unique
        identifier which should correspond to the filename (without
        extension) of the APK to process.  ``label`` is an integer
        indicating whether the APK is benign (0) or malicious (1).

    image_size: int
        Target height and width for the converted DEX images.  The
        paper reports that 512×512 images strike a good balance
        between detection performance and computational cost.  This
        can be overridden for experimentation.

    batch_size: int
        Number of samples per minibatch during training.

    learning_rate: float
        Initial learning rate for the optimiser.

    lambda_contractive: float
        Weight applied to the contractive regularisation term in the
        autoencoder loss.

    device: str
        Name of the device to run the model on (e.g. ``"cpu"`` or
        ``"cuda:0"``).  Defaults to CPU and should be overridden
        externally when a GPU is available.
    """

    data_dir: Path
    labels_csv: Path
    image_size: int = 512
    batch_size: int = 64
    learning_rate: float = 8e-4
    lambda_contractive: float = 0.1
    device: str = "cpu"
    num_epochs: int = 10

    # Additional hyperparameters controlling the network architecture
    # and optimisation can be added here with sensible defaults.


def from_env() -> ACEConfig:
    """Create a configuration from environment variables.

    This helper reads the variables ``ACE_DATA_DIR`` and
    ``ACE_LABELS_CSV`` to locate the dataset and labels.  If either
    is missing, a ``ValueError`` is raised to alert the user to
    provide the necessary inputs.  Other parameters can similarly be
    overridden via environment variables if desired.
    """
    data_dir = os.getenv("ACE_DATA_DIR")
    labels_csv = os.getenv("ACE_LABELS_CSV")
    if not data_dir or not labels_csv:
        raise ValueError(
            "Both ACE_DATA_DIR and ACE_LABELS_CSV environment variables must be set."
        )
    return ACEConfig(
        data_dir=Path(data_dir),
        labels_csv=Path(labels_csv),
        image_size=int(os.getenv("ACE_IMAGE_SIZE", 512)),
        batch_size=int(os.getenv("ACE_BATCH_SIZE", 64)),
        learning_rate=float(os.getenv("ACE_LR", 8e-4)),
        lambda_contractive=float(os.getenv("ACE_LAMBDA", 0.1)),
        device=os.getenv("ACE_DEVICE", "cpu"),
        num_epochs=int(os.getenv("ACE_EPOCHS", 10)),
    )