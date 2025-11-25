# fusion_bag_dataset.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class APKFlowBagDataset(Dataset):
    """
    APK 级别的多模态数据集：一个 APK = 多条流 的一个 bag。

    数据来源：
      - flow_csv: 由流预处理脚本生成的 *_flows.csv
          必须包含列: ['sha256', 'label', 'npy_path', 'flow_index']
          每一行是一条流（会话级），npy_path 是 1×N×M 的特征文件。
      - dex_feature_dir: DEX 灰度图 .npy 所在目录，文件名为 <sha256>.npy，内容为 H×W 或 1×H×W 的 uint8 矩阵。

    返回：
      dex_tensor:  (1, H, W) float32, 归一化到 [0,1]
      flow_tensor: (max_flows, 1, N, M) float32, 归一化到 [0,1]，不足 max_flows 的部分全 0
      flow_mask:   (max_flows,) float32，真实流为 1.0，padding 为 0.0
      label:       int64, 0/1
    """

    def __init__(
        self,
        flow_csv: Path | str,
        dex_feature_dir: Path | str,
        max_flows: int = 16,
        sort_by: str = "flow_index",
    ) -> None:
        super().__init__()
        self.flow_csv = Path(flow_csv)
        self.dex_feature_dir = Path(dex_feature_dir)
        self.max_flows = max_flows
        self.sort_by = sort_by

        if not self.flow_csv.exists():
            raise FileNotFoundError(f"{self.flow_csv} not found")

        self.df = pd.read_csv(self.flow_csv)
        required_cols = {"sha256", "label", "npy_path"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(
                f"{self.flow_csv} 必须包含列: {required_cols}, 当前: {set(self.df.columns)}"
            )

        # 按 sha256 分组，一个 APK 就是一条样本
        self.items: List[Dict] = []
        for sha, grp in self.df.groupby("sha256"):
            grp = grp.sort_values(self.sort_by) if self.sort_by in grp.columns else grp
            label_vals = grp["label"].unique().tolist()
            if len(label_vals) != 1:
                raise ValueError(f"sha256={sha} 对应多种 label: {label_vals}")
            label = int(label_vals[0])
            paths = grp["npy_path"].tolist()

            # 截断到 max_flows
            if len(paths) > self.max_flows:
                paths = paths[: self.max_flows]

            self.items.append(
                {
                    "sha256": str(sha),
                    "label": label,
                    "paths": [str(p) for p in paths],
                }
            )

        if len(self.items) == 0:
            raise ValueError(f"{self.flow_csv} 中没有任何样本")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        sha = item["sha256"]
        label = item["label"]
        paths = item["paths"]

        # ---- DEX 分支：加载灰度图特征 ----
        dex_path = self.dex_feature_dir / f"{sha}.npy"
        if not dex_path.exists():
            raise FileNotFoundError(f"DEX feature not found: {dex_path}")
        dex_arr = np.load(dex_path)  # H×W 或 1×H×W
        if dex_arr.ndim == 2:
            dex_arr = dex_arr[None, :, :]  # (1,H,W)
        elif dex_arr.ndim == 3:
            pass
        else:
            raise ValueError(f"DEX array shape invalid: {dex_arr.shape} for {dex_path}")
        dex_arr = dex_arr.astype(np.float32) / 255.0
        dex_tensor = torch.from_numpy(dex_arr)  # (1,H,W)

        # ---- Flow 分支：打包多条流 ----
        if len(paths) == 0:
            raise ValueError(f"APK {sha} 对应的流数量为 0")

        # 先加载第一条流确定形状
        first_arr = np.load(paths[0]).astype(np.float32)  # (1,N,M)
        if first_arr.ndim != 3:
            raise ValueError(f"Flow array {paths[0]} shape invalid: {first_arr.shape}")

        _, N, M = first_arr.shape
        flows = np.zeros((self.max_flows, 1, N, M), dtype=np.float32)
        mask = np.zeros((self.max_flows,), dtype=np.float32)

        for i, p in enumerate(paths[: self.max_flows]):
            arr = np.load(p).astype(np.float32)
            if arr.shape != first_arr.shape:
                raise ValueError(
                    f"Flow array shape mismatch: {p} {arr.shape} vs {first_arr.shape}"
                )
            flows[i] = arr
            mask[i] = 1.0

        flows = flows / 255.0  # 归一化
        flow_tensor = torch.from_numpy(flows)  # (K,1,N,M)
        flow_mask = torch.from_numpy(mask)     # (K,)

        y = torch.tensor(label, dtype=torch.long)
        return dex_tensor, flow_tensor, flow_mask, y
