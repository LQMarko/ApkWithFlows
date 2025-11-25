# flow_backbone.py
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        hidden_channels = max(in_channels // reduction_ratio, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)  # (B,C,1,1)

        # 分两步做最大池化，兼容你的 torch 版本
        max_pool, _ = torch.max(x, dim=2, keepdim=True)     # (B,C,1,W)
        max_pool, _ = torch.max(max_pool, dim=3, keepdim=True)  # (B,C,1,1)

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)           # (B,1,H,W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)         # (B,1,H,W)
        concat = torch.cat([avg_pool, max_pool], dim=1)         # (B,2,H,W)
        out = self.conv(concat)                                 # (B,1,H,W)
        scale = self.sigmoid(out)
        return x * scale


class CBAMBlock(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class TrafficCBAMBackbone(nn.Module):
    """
    流量特征提取骨干网络:
      输入: (B,1,N,M)
      输出: 特征图 (B,C_out,H',W')
    """

    def __init__(self, n_packets: int = 8, bytes_per_packet: int = 100, base_channels: int = 32) -> None:
        super().__init__()
        self.n_packets = n_packets
        self.bytes_per_packet = bytes_per_packet

        # Block1
        self.conv1 = nn.Conv2d(1, base_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam1 = CBAMBlock(base_channels, reduction_ratio=8)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block2
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.cbam2 = CBAMBlock(base_channels * 2, reduction_ratio=8)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block3
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.cbam3 = CBAMBlock(base_channels * 4, reduction_ratio=8)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 输出通道数
        self.out_channels = base_channels * 4

        # 记录池化后的空间大小（可选）
        h = n_packets
        w = bytes_per_packet
        for _ in range(3):
            h = max(h // 2, 1)
            w = max(w // 2, 1)
        self.out_h = h
        self.out_w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,N,M)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cbam1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.cbam2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.cbam3(x)
        x = self.pool3(x)

        return x  # (B,out_channels,out_h,out_w)
