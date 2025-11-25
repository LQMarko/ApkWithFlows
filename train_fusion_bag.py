# train_fusion_bag.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from fusion_bag_dataset import APKFlowBagDataset
from fusion_model import APKFlowFusionBagNet


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    preds, labels: 0/1 tensor
    返回: {Acc, Pre, Re, F1, TP, TN, FP, FN}
    """
    preds = preds.view(-1)
    labels = labels.view(-1)
    TP = int(((preds == 1) & (labels == 1)).sum().item())
    TN = int(((preds == 0) & (labels == 0)).sum().item())
    FP = int(((preds == 1) & (labels == 0)).sum().item())
    FN = int(((preds == 0) & (labels == 1)).sum().item())
    eps = 1e-8
    total = TP + TN + FP + FN
    acc = (TP + TN) / max(total, 1)
    pre = TP / (TP + FP + eps) if (TP + FP) > 0 else 0.0
    re = TP / (TP + FN + eps) if (TP + FN) > 0 else 0.0
    f1 = 2 * pre * re / (pre + re + eps) if (pre + re) > 0 else 0.0
    return dict(Acc=acc, Pre=pre, Re=re, F1=f1, TP=TP, TN=TN, FP=FP, FN=FN)


@torch.no_grad()
def evaluate(model: APKFlowFusionBagNet, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    model.eval().to(device)
    all_labels = []
    all_apk = []
    all_flow = []
    all_fusion = []

    for dex, flows, mask, y in loader:
        dex = dex.to(device, non_blocking=True)
        flows = flows.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model.forward(dex, flows, mask)

        prob_apk = torch.sigmoid(out.ace_out.logits)
        prob_flow = torch.sigmoid(out.flow_logits)
        prob_fusion = torch.sigmoid(out.fusion_logits)

        pred_apk = (prob_apk >= threshold).long()
        pred_flow = (prob_flow >= threshold).long()
        pred_fusion = (prob_fusion >= threshold).long()

        all_labels.append(y.cpu())
        all_apk.append(pred_apk.cpu())
        all_flow.append(pred_flow.cpu())
        all_fusion.append(pred_fusion.cpu())

    labels = torch.cat(all_labels, dim=0)
    apk_preds = torch.cat(all_apk, dim=0)
    flow_preds = torch.cat(all_flow, dim=0)
    fusion_preds = torch.cat(all_fusion, dim=0)

    metrics = {
        "apk": compute_metrics(apk_preds, labels),
        "flow": compute_metrics(flow_preds, labels),
        "fusion": compute_metrics(fusion_preds, labels),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="APK+Flow 多模态融合训练 (bag 级)")
    parser.add_argument("--dex-feature-dir", type=str, required=True,
                        help="DEX 灰度特征 .npy 目录 (文件名为 <sha256>.npy)")
    parser.add_argument("--flow-feature-dir", type=str, required=True,
                        help="Flow 特征输出目录 (包含 train_flows.csv / val_flows.csv / test_flows.csv)")
    parser.add_argument("--max-flows", type=int, default=150,
                        help="每个 APK 最多采样多少条流 (默认 150)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-packets", type=int, default=8,
                        help="预处理时的 N (包数)")
    parser.add_argument("--bytes-per-packet", type=int, default=100,
                        help="预处理时的 M (每包字节数)")
    args = parser.parse_args()

    dex_dir = Path(args.dex_feature_dir)
    flow_dir = Path(args.flow_feature_dir)

    train_ds = APKFlowBagDataset(flow_dir / "train_flows.csv", dex_dir, max_flows=args.max_flows)
    val_ds   = APKFlowBagDataset(flow_dir / "val_flows.csv",   dex_dir, max_flows=args.max_flows)
    test_ds  = APKFlowBagDataset(flow_dir / "test_flows.csv",  dex_dir, max_flows=args.max_flows)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    model = APKFlowFusionBagNet(
        n_packets=args.n_packets,
        bytes_per_packet=args.bytes_per_packet,
        ace_image_channels=1,
        ace_latent_channels=64,
        ace_embed_channels=64,
        lambda_contractive=0.1,
        fusion_hidden=256,
        flow_base_channels=32,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_fusion_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        for dex, flows, mask, y in train_loader:
            optimizer.zero_grad()
            loss, losses = model.compute_multitask_loss(dex, flows, mask, y)
            loss.backward()
            optimizer.step()

            bs = dex.size(0)
            total_loss += loss.item() * bs
            seen += bs

        train_loss = total_loss / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device, threshold=0.5)
        fusion_f1 = val_metrics["fusion"]["F1"]

        print(
            f"[Epoch {epoch:03d}] TrainLoss={train_loss:.4f} | "
            f"Val_Fusion_F1={fusion_f1:.4f} "
            f"(APK_F1={val_metrics['apk']['F1']:.4f}, "
            f"Flow_F1={val_metrics['flow']['F1']:.4f})"
        )

        if fusion_f1 > best_fusion_f1:
            best_fusion_f1 = fusion_f1
            torch.save(model.state_dict(), flow_dir / "fusion_bag_best.pt")
            print("[INFO] Saved best fusion model to fusion_bag_best.pt")

    # 使用最佳模型在 test 上评估
    best_path = flow_dir / "fusion_bag_best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[INFO] Loaded best model from {best_path}")

    test_metrics = evaluate(model, test_loader, device, threshold=0.5)
    print("\n========== Test Set Results (APK-only) ==========")
    print(test_metrics["apk"])
    print("========== Test Set Results (Flow-only) =========")
    print(test_metrics["flow"])
    print("========== Test Set Results (Fusion) ==========")
    print(test_metrics["fusion"])


if __name__ == "__main__":
    main()
