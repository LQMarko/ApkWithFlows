"""Model definitions for the ACE malware detector.

This subpackage contains implementations of the different modules
comprising the ACE architecture:

* ``autoencoder`` implements the contractive autoencoder used to
  compress raw 512×512 DEX images into a compact representation and
  apply contractive regularisation.
* ``contrastive_encoder`` implements the encoder used to produce
  latent representations for contrastive learning.
* ``classifier`` implements the fully connected layers used for
  classification.
* ``ace_model`` stitches together these components and defines
  forward and loss computation functions consistent with the
  hierarchical loss described in the paper.
"""

from .autoencoder import ContractiveAutoEncoder
from .contrastive_encoder import ContrastiveEncoder
from .classifier import Classifier
from .ace_model import ACEModel

__all__ = [
    "ContractiveAutoEncoder",
    "ContrastiveEncoder",
    "Classifier",
    "ACEModel",
]