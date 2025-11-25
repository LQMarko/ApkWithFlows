# ace/model/ace_model.py
"""Combined ACE model and hierarchical loss implementation.

This module brings together the contractive autoencoder, contrastive
encoder and classifier into a single PyTorch module. It also
provides functions for computing the hierarchical loss described in
the paper: reconstruction loss plus contractive penalty, contrastive
loss based on SSIM, and binary cross-entropy loss for classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .autoencoder import ContractiveAutoEncoder
from .contrastive_encoder import ContrastiveEncoder
from .classifier import Classifier
from ..utils.metrics import ssim_tensor


@dataclass
class ACEOutput:
    """Container for intermediate outputs of the ACE model."""
    recon: torch.Tensor
    latent: torch.Tensor
    contrastive: torch.Tensor
    logits: torch.Tensor


class ACEModel(nn.Module):
    """Full ACE model implementing forward pass and loss computation."""

    def __init__(
        self,
        image_channels: int = 1,
        latent_channels: int = 64,
        embed_channels: int = 64,
        lambda_contractive: float = 0.1,
    ) -> None:
        super().__init__()
        self.autoencoder = ContractiveAutoEncoder(
            in_channels=image_channels,
            latent_channels=latent_channels,
            lambda_contractive=lambda_contractive,
        )
        self.contrastive_encoder = ContrastiveEncoder(
            in_channels=latent_channels,
            embed_channels=embed_channels,
        )
        # Lazily instantiate the classifier when we know spatial size.
        self._classifier: Optional[Classifier] = None

        self.image_channels = image_channels
        self.embed_channels = embed_channels

    def _module_device(self) -> torch.device:
        """Return the current device of model parameters."""
        return next(self.parameters()).device

    def _ensure_classifier(self, contrastive_output: torch.Tensor) -> None:
        """Instantiate classifier if not yet created; place it on correct device."""
        if self._classifier is None:
            _, c, h, w = contrastive_output.shape
            in_features = c * h * w
            self._classifier = Classifier(in_features)
            self.add_module("classifier", self._classifier)
        # Keep classifier on the same device as features
        if self._classifier is not None:
            clf_dev = next(self._classifier.parameters()).device
            if clf_dev != contrastive_output.device:
                self._classifier.to(contrastive_output.device)

    def forward(self, x: torch.Tensor) -> ACEOutput:  # type: ignore[override]
        # Ensure input is on the same device as the model
        dev = self._module_device()
        if x.device != dev:
            x = x.to(dev, non_blocking=True)

        # Autoencoder forward
        recon, latent = self.autoencoder(x)

        # Contrastive encoder forward
        contrastive_feat = self.contrastive_encoder(latent)  # (B, C, H, W)

        # Classifier ensure & forward (flatten first!)
        self._ensure_classifier(contrastive_feat)
        assert self._classifier is not None
        B = contrastive_feat.shape[0]
        flat_feat = contrastive_feat.reshape(B, -1)  # (B, C*H*W)
        logits = self._classifier(flat_feat)

        return ACEOutput(recon, latent, contrastive_feat, logits)

    def compute_hierarchical_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the total loss and its components for a batch."""
        dev = self._module_device()
        if x.device != dev:
            x = x.to(dev, non_blocking=True)
        if labels.device != dev:
            labels = labels.to(dev, non_blocking=True)

        # Single forward
        out = self.forward(x)

        # Contractive + reconstruction
        l1 = self.autoencoder.contractive_loss(x, out.recon, out.latent)

        # BCE with logits
        bce = F.binary_cross_entropy_with_logits(out.logits, labels.float())

        # Contrastive SSIM-based loss
        l2 = self._contrastive_loss(out.contrastive, labels)

        total = l1 + l2 + bce
        return total, {"contractive": l1.detach(), "contrastive": l2.detach(), "bce": bce.detach()}

    def _contrastive_loss(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Contrastive loss based on SSIM of feature maps."""
        dev = feat.device
        feat_2d = feat.mean(dim=1)  # (B, H, W)
        B = feat_2d.size(0)

        sum_ssim1 = 0.0; count1 = 0  # benign-benign
        sum_ssim2 = 0.0; count2 = 0  # mal-mal
        sum_ssim3 = 0.0; count3 = 0  # cross

        for i in range(B):
            for j in range(i + 1, B):
                ssim_val = ssim_tensor(feat_2d[i], feat_2d[j])  # tensor scalar
                li = int(labels[i].item())
                lj = int(labels[j].item())
                if li == 0 and lj == 0:
                    sum_ssim1 += float(ssim_val.item()); count1 += 1
                elif li == 1 and lj == 1:
                    sum_ssim2 += float(ssim_val.item()); count2 += 1
                else:
                    sum_ssim3 += float(ssim_val.item()); count3 += 1

        SSIM1 = (sum_ssim1 / count1) if count1 > 0 else 0.0
        SSIM2 = (sum_ssim2 / count2) if count2 > 0 else 0.0
        SSIM3 = (sum_ssim3 / count3) if count3 > 0 else 0.0

        d1 = max(SSIM1 - SSIM3, 0.0)
        d2 = max(SSIM2 - SSIM3, 0.0)

        loss = (1.0 - d1) ** 2 + (1.0 - d2) ** 2
        return torch.tensor(loss, dtype=feat.dtype, device=dev)
