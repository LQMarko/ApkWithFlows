# ace/model/autoencoder.py
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, 3, 2, 1), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, 1, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ContractiveAutoEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_channels: int = 64, lambda_contractive: float = 0.1):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels)
        self.decoder = Decoder(latent_channels, in_channels)
        self.lambda_contractive = float(lambda_contractive)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)             # (B, C, H', W')
        recon = self.decoder(z)         # (B, in_channels, H, W)
        return recon, z

    def contractive_loss(self, x: torch.Tensor, recon: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss + contractive penalty.

        - 重构项直接用 forward 已算出的 recon 与原始 x；
        - contractive 惩罚项用一个 requires_grad=True 的 x 副本，仅走 encoder，
          确保可对 x 求 ∂z/∂x；并设置 retain_graph=True 以避免二次回溯报错。
        """
        # 1) 重构项
        recon_loss = F.mse_loss(recon, x)

        # 2) Contractive 项：对 z 关于 x 的雅可比范数做惩罚
        x_req = x.detach().to(x.device)
        x_req.requires_grad_(True)            # 允许对 x 求导
        z_req = self.encoder(x_req)           # 仅 encoder
        z_norm = torch.sum(z_req ** 2)

        grad = torch.autograd.grad(
            z_norm,
            x_req,
            retain_graph=True,               # ✅ 关键改动：保留图，供主损失 backward 使用
            create_graph=True,               # 需要把惩罚项接回 encoder 参数
            allow_unused=False,
        )[0]

        contractive_penalty = grad.pow(2).mean()
        return recon_loss + self.lambda_contractive * contractive_penalty
