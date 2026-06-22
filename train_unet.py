# -*- coding: utf-8 -*-
"""
train_unet.py
=============
Train U-Net (semantic segmentation) cho bài toán dò lỗi A26, dùng dataset sinh
bởi chia_data_unet.py (cặp ảnh + mask, mask 1 kênh pixel = class id, 0 = nền).

Dùng segmentation_models_pytorch (smp): U-Net với encoder PRETRAINED (ImageNet)
-> hội tụ nhanh, hợp dữ liệu ít. Loss = Dice + Focal để chống mất cân bằng pixel
(lỗi chiếm rất ít diện tích).

Cài đặt (trong WSL2, nên dùng venv đã có torch + CUDA):
    pip install segmentation-models-pytorch albumentations==0  # (albu KHÔNG bắt buộc)
    # tối thiểu:
    pip install segmentation-models-pytorch opencv-python pyyaml tqdm

Chạy:
    python train_unet.py
    python train_unet.py --encoder resnet34 --epochs 100 --batch 8
    python train_unet.py --arch UnetPlusPlus --encoder efficientnet-b0
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_params

DEFAULT_DATA = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A26/data_imgs_unet/dataset.yaml"
DEFAULT_PROJECT = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A26/results/unet"


# --------------------------------------------------------------------------- #
# Đọc ảnh unicode (Windows)
# --------------------------------------------------------------------------- #
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SegDataset(Dataset):
    """Đọc cặp (image, mask). mask 1 kênh, pixel = class id (0 = nền).

    augment=True -> lật/xoay 90 + đổi sáng nhẹ (tự cài, không cần albumentations).
    """

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(self, img_dir, mask_dir, mean, std, augment=False):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.augment = augment
        self.items = [p for p in sorted(self.img_dir.iterdir())
                      if p.suffix.lower() in self.IMG_EXTS]
        if not self.items:
            raise RuntimeError(f"Không có ảnh trong {img_dir}")

    def __len__(self):
        return len(self.items)

    def _aug(self, img, mask):
        if random.random() < 0.5:                     # lật ngang
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:                     # lật dọc
            img, mask = img[::-1, :], mask[::-1, :]
        k = random.randint(0, 3)                       # xoay 0/90/180/270
        if k:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
        if random.random() < 0.5:                     # đổi sáng/tương phản nhẹ
            a = random.uniform(0.85, 1.15)             # contrast
            b = random.uniform(-15, 15)                # brightness
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def __getitem__(self, i):
        img_path = self.items[i]
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        img = imread_unicode(img_path, cv2.IMREAD_COLOR)
        mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise RuntimeError(f"Lỗi đọc cặp: {img_path.name}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augment:
            img, mask = self._aug(img, mask)

        # chuẩn hoá theo encoder pretrained
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img.transpose(2, 0, 1))         # CxHxW
        mask = torch.from_numpy(mask.astype(np.int64))         # HxW class id
        return img, mask


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Trả về mIoU (mọi class) và IoU riêng của các class LỖI (id >= 1)."""
    model.eval()
    tp = fp = fn = tn = None
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(img)
        pred = logits.argmax(1)
        stat = smp.metrics.get_stats(pred, mask, mode="multiclass",
                                     num_classes=num_classes)
        cur = [s.sum(0) for s in stat]  # cộng dồn theo batch -> [num_classes]
        if tp is None:
            tp, fp, fn, tn = cur
        else:
            tp, fp, fn, tn = (tp + cur[0], fp + cur[1], fn + cur[2], tn + cur[3])

    iou = smp.metrics.iou_score(tp[None], fp[None], fn[None], tn[None],
                                reduction=None)[0]  # IoU mỗi class
    miou = float(iou.mean())
    defect_iou = float(iou[1:].mean()) if num_classes > 1 else miou
    return miou, defect_iou, iou.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA, help="Đường dẫn dataset.yaml")
    ap.add_argument("--arch", default="Unet",
                    help="Unet | UnetPlusPlus | FPN | DeepLabV3Plus ...")
    ap.add_argument("--encoder", default="resnet34",
                    help="resnet34 | resnet50 | efficientnet-b0 ...")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8,
                    help="VRAM 8GB + tile 512 -> 8. Giảm còn 4 nếu CUDA OOM")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--name", default="defect_unet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # --- Đọc dataset.yaml (do chia_data_unet.py sinh ra) ---
    with open(args.data, "r", encoding="utf-8") as f:
        ds = yaml.safe_load(f)
    base = Path(ds["path"])
    num_classes = int(ds["num_classes"])          # gồm cả nền (id 0)
    names = ds.get("names", {})
    print(f"[INFO] num_classes = {num_classes} (gồm nền). Lớp: {names}")

    # Tham số chuẩn hoá của encoder pretrained
    pp = get_preprocessing_params(args.encoder, pretrained="imagenet")
    mean = [m * 255.0 for m in pp["mean"]]         # ảnh ta để thang 0..255
    std = [s * 255.0 for s in pp["std"]]

    train_ds = SegDataset(base / ds["images"]["train"], base / ds["masks"]["train"],
                          mean, std, augment=True)
    val_ds = SegDataset(base / ds["images"]["val"], base / ds["masks"]["val"],
                        mean, std, augment=False)
    print(f"[INFO] train={len(train_ds)} tile, val={len(val_ds)} tile")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # --- Model: U-Net + encoder pretrained ---
    model = smp.create_model(
        args.arch, encoder_name=args.encoder, encoder_weights="imagenet",
        in_channels=3, classes=num_classes,
    ).to(device)

    # --- Loss: Dice + Focal (chống mất cân bằng pixel lỗi ít) ---
    dice = smp.losses.DiceLoss(mode="multiclass")
    focal = smp.losses.FocalLoss(mode="multiclass")

    def criterion(logits, target):
        return dice(logits, target) + focal(logits, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    out_dir = Path(args.project) / args.name
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img, mask in pbar:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(img)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        miou, defect_iou, iou_list = evaluate(model, val_loader, device, num_classes)
        avg_loss = running / max(1, len(train_loader))
        print(f"  -> loss={avg_loss:.4f} | val mIoU={miou:.4f} | "
              f"IoU lỗi={defect_iou:.4f} | per-class={['%.3f' % v for v in iou_list]}")

        # Lưu theo IoU của LỖI (quan trọng hơn nền)
        torch.save(model.state_dict(), out_dir / "weights" / "last.pt")
        if defect_iou > best_iou:
            best_iou = defect_iou
            torch.save(model.state_dict(), out_dir / "weights" / "best.pt")
            print(f"     [BEST] IoU lỗi mới = {best_iou:.4f} -> đã lưu best.pt")

    # Lưu cấu hình để predict nạp lại đúng kiến trúc
    cfg = {"arch": args.arch, "encoder": args.encoder, "num_classes": num_classes,
           "names": names, "mean": mean, "std": std, "tile": ds.get("tile")}
    with open(out_dir / "model_cfg.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"\n[HOÀN TẤT] best IoU lỗi = {best_iou:.4f}")
    print(f"Weights: {out_dir / 'weights' / 'best.pt'}")
    print(f"Cấu hình model: {out_dir / 'model_cfg.yaml'} (dùng cho script predict)")


if __name__ == "__main__":
    main()
