# fusion_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ace.model.ace_model import ACEModel, ACEOutput

from flow_backbone import TrafficCBAMBackbone


@dataclass
class FusionOutput:
    ace_out: ACEOutput          # DEX 分支全部中间输出
    flow_logits: torch.Tensor   # (B,) Flow-only 头输出
    fusion_logits: torch.Tensor # (B,) 融合头输出
    apk_feat: torch.Tensor      # (B,C_ace)
    flow_feat: torch.Tensor     # (B,C_flow)


class APKFlowFusionBagNet(nn.Module):
    """
    APK+多流 融合模型（bag 级）:

      - DEX 分支: 使用 ACEModel (Contractive AE + ContrastiveEncoder + APK-only classifier)
      - Flow 分支: TrafficCBAMBackbone + 多流掩码平均聚合 + Flow-only classifier
      - Fusion 头: GAP(DEX特征) 与 GAP(Flow特征) 拼接，两层 MLP 输出融合 logit

    损失:
      L_total = w_ae * (L_recon+contractive + L_contrastive)
              + w_apk * BCE(logits_apk, y)
              + w_flow * BCE(logits_flow, y)
              + w_fusion * BCE(logits_fusion, y)
    """

    def __init__(
        self,
        n_packets: int = 8,
        bytes_per_packet: int = 100,
        ace_image_channels: int = 1,
        ace_latent_channels: int = 64,
        ace_embed_channels: int = 64,
        lambda_contractive: float = 0.1,
        fusion_hidden: int = 256,
        flow_base_channels: int = 32,
    ) -> None:
        super().__init__()

        # ---- DEX 分支 (ACE) ----
        self.ace = ACEModel(
            image_channels=ace_image_channels,
            latent_channels=ace_latent_channels,
            embed_channels=ace_embed_channels,
            lambda_contractive=lambda_contractive,
        )

        # ---- Flow 分支 backbone ----
        self.flow_backbone = TrafficCBAMBackbone(
            n_packets=n_packets,
            bytes_per_packet=bytes_per_packet,
            base_channels=flow_base_channels,
        )

        # GAP 后的维度
        self.apk_feat_dim = ace_embed_channels                    # GAP over contrastive: (B,C_ace,*,*) -> (B,C_ace)
        self.flow_feat_dim = self.flow_backbone.out_channels      # GAP over CBAM feature map

        # Flow-only 头: APK 级别聚合后的特征 -> 1 个 logit
        self.flow_head = nn.Linear(self.flow_feat_dim, 1)

        # Fusion 头: [apk_feat, flow_feat] 拼接 -> 1 个 logit
        fusion_in_dim = self.apk_feat_dim + self.flow_feat_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(
        self,
        dex_img: torch.Tensor,       # (B,1,H,W)
        flow_batch: torch.Tensor,    # (B,K,1,N,M)
        flow_mask: torch.Tensor,     # (B,K)
    ) -> FusionOutput:
        dev = next(self.parameters()).device
        dex_img = dex_img.to(dev, non_blocking=True)
        flow_batch = flow_batch.to(dev, non_blocking=True)
        flow_mask = flow_mask.to(dev, non_blocking=True)

        B, K, C_in, N, M = flow_batch.shape

        # ---- DEX 分支前向 ----
        ace_out: ACEOutput = self.ace(dex_img)

        # GAP 得到 APK 特征向量 (B,C_ace)
        apk_feat_map = ace_out.contrastive                      # (B,C_ace,hc,wc)
        apk_feat = F.adaptive_avg_pool2d(apk_feat_map, (1, 1))  # (B,C_ace,1,1)
        apk_feat = apk_feat.view(B, -1)

        # ---- Flow 分支: 多流 -> 每个流特征 -> bag 聚合 ----
        flow_flat = flow_batch.view(B * K, C_in, N, M)          # (B*K,1,N,M)
        flow_feat_map = self.flow_backbone(flow_flat)           # (B*K,C_out,hf,wf)
        C_out = flow_feat_map.size(1)
        hf, wf = flow_feat_map.size(2), flow_feat_map.size(3)

        flow_feat_map = flow_feat_map.view(B, K, C_out, hf, wf)  # (B,K,C_out,hf,wf)

        # 对每条流做 GAP -> (B,K,C_out)
        flow_gap = flow_feat_map.mean(dim=(3, 4))  # (B,K,C_out)

        # 使用 mask 做加权平均: 忽略 padding 流
        mask = flow_mask.unsqueeze(-1)             # (B,K,1)
        flow_gap_masked = flow_gap * mask         # (B,K,C_out)
        sum_feat = flow_gap_masked.sum(dim=1)     # (B,C_out)
        count = mask.sum(dim=1) + 1e-8            # (B,1)
        flow_feat = sum_feat / count              # (B,C_out)

        # Flow-only logit
        flow_logits = self.flow_head(flow_feat).squeeze(-1)  # (B,)

        # ---- Fusion 头 ----
        fusion_input = torch.cat([apk_feat, flow_feat], dim=1)  # (B,C_ace+C_out)
        fusion_logits = self.fusion_head(fusion_input).squeeze(-1)

        return FusionOutput(
            ace_out=ace_out,
            flow_logits=flow_logits,
            fusion_logits=fusion_logits,
            apk_feat=apk_feat,
            flow_feat=flow_feat,
        )

    def compute_multitask_loss(
        self,
        dex_img: torch.Tensor,
        flow_batch: torch.Tensor,
        flow_mask: torch.Tensor,
        labels: torch.Tensor,
        w_ae: float = 1.0,
        w_apk: float = 1.0,
        w_flow: float = 1.0,
        w_fusion: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多任务总损失：
          - DEX 分支: 自编码 + contractive + 对比（SSIM）
          - APK-only BCE
          - Flow-only BCE
          - Fusion BCE
        """
        dev = next(self.parameters()).device
        dex_img = dex_img.to(dev, non_blocking=True)
        flow_batch = flow_batch.to(dev, non_blocking=True)
        flow_mask = flow_mask.to(dev, non_blocking=True)
        labels = labels.to(dev, non_blocking=True)

        out = self.forward(dex_img, flow_batch, flow_mask)

        # ---- 自编码 + contractive ----
        ae_loss = self.ace.autoencoder.contractive_loss(
            dex_img, out.ace_out.recon, out.ace_out.latent
        )

        # ---- 对比损失 ----
        contrastive_loss = self.ace._contrastive_loss(out.ace_out.contrastive, labels)

        # ---- 三个 BCE 头 ----
        labels_f = labels.float()
        bce_apk = F.binary_cross_entropy_with_logits(out.ace_out.logits, labels_f)
        bce_flow = F.binary_cross_entropy_with_logits(out.flow_logits, labels_f)
        bce_fusion = F.binary_cross_entropy_with_logits(out.fusion_logits, labels_f)

        total = (
            w_ae * (ae_loss + contrastive_loss)
            + w_apk * bce_apk
            + w_flow * bce_flow
            + w_fusion * bce_fusion
        )

        loss_dict = {
            "total": total.detach(),
            "ae": ae_loss.detach(),
            "contrastive": contrastive_loss.detach(),
            "bce_apk": bce_apk.detach(),
            "bce_flow": bce_flow.detach(),
            "bce_fusion": bce_fusion.detach(),
        }
        return total, loss_dict
