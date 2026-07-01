# -*- coding: utf-8 -*-
"""
train_unet.py
=============
Nâng cấp chuẩn Tech Lead:
1. Đồng bộ toán học: Ép Dice Loss bỏ qua nền, song sát cùng Weighted CE.
2. Sửa bug Logic: Đảo thứ tự cập nhật best_iou trước khi đóng gói checkpoint.
3. Phòng thủ hệ thống: Kiểm tra số lượng class_weights động bằng assert.
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

DEFAULT_DATA = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/data_imgs_unet/dataset.yaml"
DEFAULT_PROJECT = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/results/unet"


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


class SegDataset(Dataset):
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
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            img, mask = img[::-1, :], mask[::-1, :]
        k = random.randint(0, 3)
        if k:
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
        if random.random() < 0.5:
            a = random.uniform(0.85, 1.15)
            b = random.uniform(-15, 15)
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

        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask.astype(np.int64))
        return img, mask


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    tp = fp = fn = tn = None
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(img)
        pred = logits.argmax(1)
        stat = smp.metrics.get_stats(pred, mask, mode="multiclass", num_classes=num_classes)
        cur = [s.sum(0) for s in stat]
        if tp is None:
            tp, fp, fn, tn = cur
        else:
            tp, fp, fn, tn = (tp + cur[0], fp + cur[1], fn + cur[2], tn + cur[3])

    iou = smp.metrics.iou_score(tp[None], fp[None], fn[None], tn[None], reduction=None)[0]
    miou = float(iou.mean())
    defect_iou = float(iou[1:].mean()) if num_classes > 1 else miou
    return miou, defect_iou, iou.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA, help="Đường dẫn dataset.yaml")
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--name", default="defect_unet_260701")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="Bật cờ này để phục hồi train từ điểm gãy")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    with open(args.data, "r", encoding="utf-8") as f:
        ds = yaml.safe_load(f)
    base = Path(ds["path"])
    num_classes = int(ds["num_classes"])
    names = ds.get("names", {})
    print(f"[INFO] num_classes = {num_classes} (gồm nền). Lớp: {names}")

    pp = get_preprocessing_params(args.encoder, pretrained="imagenet")
    mean = [m * 255.0 for m in pp["mean"]]
    std = [s * 255.0 for s in pp["std"]]

    train_ds = SegDataset(base / ds["images"]["train"], base / ds["masks"]["train"], mean, std, augment=True)
    val_ds = SegDataset(base / ds["images"]["val"], base / ds["masks"]["val"], mean, std, augment=False)
    print(f"[INFO] train={len(train_ds)} tile, val={len(val_ds)} tile")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                             num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = smp.create_model(
        args.arch, encoder_name=args.encoder, encoder_weights="imagenet",
        in_channels=3, classes=num_classes,
    ).to(device)

    # --- ĐỘNG CƠ LOSS SỬA ĐỔI: PHÒNG THỦ VÀ ĐỒNG BỘ TOÁN HỌC ---
    weights_list = [0.2, 2.0, 2.0, 1.5, 2.0]
    assert len(weights_list) == num_classes, (
        f"Gãy logic: Khai báo {len(weights_list)} trọng số nhưng cấu hình hệ thống yêu cầu {num_classes} lớp!"
    )
    class_weights = torch.tensor(weights_list, dtype=torch.float32).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    # Ép Dice Loss bỏ qua lớp nền (class 0) để tránh lệch gradient do diện tích nền quá lớn
    dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=[i for i in range(1, num_classes)])

    def criterion(logits, target):
        return dice_loss(logits, target) + ce_loss(logits, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    out_dir = Path(args.project) / args.name
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint_path = weights_dir / "last.pt"
    start_epoch = 1
    best_iou = -1.0

    # --- LOGIC NẠP LẠI TRẠNG THÁI (RESUME) ---
    if args.resume:
        if last_checkpoint_path.exists():
            print(f"[INFO] Đang hồi sinh phiên làm việc từ file: {last_checkpoint_path}")
            checkpoint = torch.load(last_checkpoint_path, map_location=device)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint and (device.type == "cuda"):
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                
            start_epoch = checkpoint["epoch"] + 1
            best_iou = checkpoint["best_iou"]
            print(f"[SUCCESS] Đồng bộ thành công! Tiếp tục chạy từ Epoch {start_epoch} (Best IoU lỗi cũ: {best_iou:.4f})")
        else:
            print(f"[WARNING] Không tìm thấy file {last_checkpoint_path}. Chạy lại từ đầu.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img, mask in pbar:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
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

        # --- ĐẢO TRÌNH TỰ: CẬP NHẬT TRƯỚC KHI ĐÓNG GÓI CHECKPOINT ---
        if defect_iou > best_iou:
            best_iou = defect_iou
            torch.save(model.state_dict(), weights_dir / "best.pt")
            print(f"     [BEST] IoU lỗi mới = {best_iou:.4f} -> đã lưu best.pt")

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if (device.type == "cuda") else {},
            "best_iou": best_iou  # Giá trị đồng bộ chính xác tuyệt đối
        }
        torch.save(checkpoint_data, last_checkpoint_path)

    cfg = {"arch": args.arch, "encoder": args.encoder, "num_classes": num_classes,
           "names": names, "mean": mean, "std": std, "tile": ds.get("tile")}
    with open(out_dir / "model_cfg.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"\n[HOÀN TẤT] best IoU lỗi = {best_iou:.4f}")


if __name__ == "__main__":
    main()