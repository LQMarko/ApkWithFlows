"""Top level package for the ACE malware detection implementation.

This package implements the ACE model described in the paper
"ACE: A Static Android Malware Detection Method Based on
Supervised Contrastive Learning".  The implementation is broken
down into several submodules:

* ``config`` – configuration utilities and defaults.
* ``data`` – dataset loading, preprocessing, and augmentation.
* ``model`` – PyTorch modules implementing the contractive
  autoencoder, contrastive encoder, classifier and the combined ACE
  model.
* ``utils`` – helper functions such as SSIM computation and other
  metrics.

The code in this package is organised to follow the design of the
paper.  The preprocessing reads APK files, extracts their DEX
sections, converts them into 2‑D matrices and then resizes them
into 512×512 grayscale images.  The model itself consists of a
contractive autoencoder, a contrastive encoder and a classifier.
The training loop minimises a hierarchical loss consisting of
contractive, contrastive and binary cross‑entropy terms.

This file intentionally contains no logic; it exists only to mark
``ace`` as a Python package.
"""

__all__ = ["config", "data", "model", "utils"]