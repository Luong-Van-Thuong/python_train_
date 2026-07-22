# -*- coding: utf-8 -*-
"""
train_kolektorsdd2.py
======================
Bản riêng của train_unet.py, dùng để tự học trên KolektorSDD2 (data công khai) mà
KHÔNG đụng vào train_unet.py/chia_data_unet.py sản xuất của công ty. Cùng kiến trúc,
cùng công tắc loss/metric đã học ở HOC_unet_loi_nho_6px.md Bài 1-9 — chỉ khác:

    weights_list SINH TỰ ĐỘNG theo num_classes (nền=0.2, mỗi lớp lỗi=2.0), thay vì
    hard-code 5 phần tử như train_unet.py (vốn cho đúng bài 4-lỗi công ty — dùng
    cứng số đó cho KolektorSDD2 (2 lớp) sẽ vỡ assert).

Chạy (sau khi đã có dataset.yaml từ chia_data_kolektorsdd2.py):
    python train_kolektorsdd2.py
    python train_kolektorsdd2.py --loss ftl_focal --best-metric f1_object --epochs 60
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

import sys as _sys, pathlib as _pl
_sys.path.insert(0, next(str(_p) for _p in _pl.Path(__file__).resolve().parents if (_p / "pathfix.py").exists()))
from pathfix import P

DEFAULT_DATA = Path(__file__).parent / "data_kolektorsdd2" / "dataset.yaml"
DEFAULT_PROJECT = Path(__file__).parent / "results_260721"


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
        # k = random.randint(0, 3)
        # if k:
        #     img = np.rot90(img, k)
        #     mask = np.rot90(mask, k)
        if random.random() < 0.5:
            a = random.uniform(0.85, 1.15)
            b = random.uniform(-15, 15)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            img = np.rot90(img, 2)
            mask = np.rot90(mask, 2)
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]  # Lật ngang (Horizontal Flip)
        if random.random() < 0.5:
            img, mask = img[::-1, :], mask[::-1, :]  # Lật dọc (Vertical Flip)
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


def _object_stats(pred, gt, num_classes):
    """Đếm ở mức 'CỤC LỖI' (blob), gộp mọi lớp lỗi (bỏ nền=0). Xem Bài 6."""
    tp = fp = fn = 0
    for c in range(1, num_classes):
        gt_c = (gt == c).astype(np.uint8)
        pr_c = (pred == c).astype(np.uint8)
        if gt_c.sum() == 0 and pr_c.sum() == 0:
            continue
        n_gt, lbl_gt = cv2.connectedComponents(gt_c, connectivity=8)
        n_pr, lbl_pr = cv2.connectedComponents(pr_c, connectivity=8)
        for g in range(1, n_gt):
            if pr_c[lbl_gt == g].any():
                tp += 1
            else:
                fn += 1
        for p in range(1, n_pr):
            if not gt_c[lbl_pr == p].any():
                fp += 1
    return tp, fp, fn


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    tp = fp = fn = tn = None
    o_tp = o_fp = o_fn = 0
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        mask_dev = mask.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(img)
        pred = logits.argmax(1)
        stat = smp.metrics.get_stats(pred, mask_dev, mode="multiclass", num_classes=num_classes)
        cur = [s.sum(0) for s in stat]
        if tp is None:
            tp, fp, fn, tn = cur
        else:
            tp, fp, fn, tn = (tp + cur[0], fp + cur[1], fn + cur[2], tn + cur[3])

        pred_np = pred.cpu().numpy()
        gt_np = mask.cpu().numpy()
        for i in range(pred_np.shape[0]):
            a, b, d = _object_stats(pred_np[i], gt_np[i], num_classes)
            o_tp += a; o_fp += b; o_fn += d

    iou = smp.metrics.iou_score(tp[None], fp[None], fn[None], tn[None], reduction=None)[0]
    miou = float(iou.mean())
    defect_iou = float(iou[1:].mean()) if num_classes > 1 else miou
    recall = o_tp / (o_tp + o_fn) if (o_tp + o_fn) > 0 else 0.0
    precision = o_tp / (o_tp + o_fp) if (o_tp + o_fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "miou": miou,
        "defect_iou": defect_iou,
        "iou_list": iou.tolist(),
        "obj_recall": recall,
        "obj_precision": precision,
        "obj_f1": f1,
        "obj_counts": (int(o_tp), int(o_fp), int(o_fn)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="Đường dẫn dataset.yaml")
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--project", default=str(DEFAULT_PROJECT))
    ap.add_argument("--name", default="kolektorsdd2_unet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--best-metric", default="f1_object",
                    choices=["iou_pixel", "recall_object", "f1_object"])
    ap.add_argument("--loss", default="dice_ce", choices=["dice_ce", "ftl_focal"])
    ap.add_argument("--tv-alpha", type=float, default=0.3)
    ap.add_argument("--tv-beta", type=float, default=0.7)
    ap.add_argument("--tv-gamma", type=float, default=1.333)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    with open(args.data, "r", encoding="utf-8") as f:
        ds = yaml.safe_load(f)
    base = Path(P(ds["path"]))
    num_classes = int(ds["num_classes"])
    names = ds.get("names", {})
    print(f"[INFO] num_classes = {num_classes} (gồm nền). Lớp: {names}")

    pp = get_preprocessing_params(args.encoder, pretrained="imagenet")
    mean = [m * 255.0 for m in pp["mean"]]
    std = [s * 255.0 for s in pp["std"]]

    train_ds = SegDataset(base / ds["images"]["train"], base / ds["masks"]["train"], mean, std, augment=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                             num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = None
    if ds["images"]["val"] and ds["masks"]["val"]:
        val_ds = SegDataset(base / ds["images"]["val"], base / ds["masks"]["val"], mean, std, augment=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        print(f"[INFO] train={len(train_ds)} ảnh, val={len(val_ds)} ảnh")
    else:
        print(f"[INFO] train={len(train_ds)} ảnh, val=KHÔNG CÓ (chọn best.pt theo train loss)")

    model = smp.create_model(
        args.arch, encoder_name=args.encoder, encoder_weights="imagenet",
        in_channels=3, classes=num_classes,
    ).to(device)

    # nền phạt nhẹ (0.2), mỗi lớp lỗi phạt nặng (2.0) -- xem Bài 3. Sinh tự động
    # theo num_classes thay vì hard-code, vì bài KolektorSDD2 chỉ có 2 lớp (nền+defect).
    weights_list = [0.2] + [2.0] * (num_classes - 1)
    defect_classes = list(range(1, num_classes))

    if args.loss == "dice_ce":
        class_weights = torch.tensor(weights_list, dtype=torch.float32).to(device)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=defect_classes)

        def criterion(logits, target):
            return dice_loss(logits, target) + ce_loss(logits, target)
    else:  # ftl_focal
        ftl_loss = smp.losses.TverskyLoss(
            mode="multiclass", classes=defect_classes,
            alpha=args.tv_alpha, beta=args.tv_beta, gamma=args.tv_gamma,
        )
        focal_loss = smp.losses.FocalLoss(mode="multiclass", gamma=args.focal_gamma)

        def criterion(logits, target):
            return ftl_loss(logits, target) + focal_loss(logits, target)

    if args.loss == "ftl_focal":
        print(f"[INFO] Loss = ftl_focal (Tversky α={args.tv_alpha} β={args.tv_beta} "
              f"γ={args.tv_gamma} + Focal γ={args.focal_gamma})")
    else:
        print("[INFO] Loss = dice_ce (Dice + CrossEntropy, baseline)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    out_dir = Path(args.project) / args.name
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint_path = weights_dir / "last.pt"
    start_epoch = 1
    best_score = -1.0
    best_loss = float("inf")
    print(f"[INFO] best.pt chọn theo tiêu chí: {args.best_metric}")

    if args.resume:
        if last_checkpoint_path.exists():
            print(f"[INFO] Đang hồi sinh từ: {last_checkpoint_path}")
            checkpoint = torch.load(last_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint and (device.type == "cuda"):
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_score = checkpoint.get("best_score", -1.0)
            best_loss = checkpoint.get("best_loss", float("inf"))
            print(f"[SUCCESS] Tiếp tục từ Epoch {start_epoch} (best {args.best_metric} cũ = {best_score:.4f})")
        else:
            print(f"[WARNING] Không tìm thấy {last_checkpoint_path}. Chạy lại từ đầu.")

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

        avg_loss = running / max(1, len(train_loader))

        if val_loader is not None:
            res = evaluate(model, val_loader, device, num_classes)
            score = {
                "iou_pixel": res["defect_iou"],
                "recall_object": res["obj_recall"],
                "f1_object": res["obj_f1"],
            }[args.best_metric]
            otp, ofp, ofn = res["obj_counts"]
            print(f"  -> loss={avg_loss:.4f} | IoU lỗi(pixel)={res['defect_iou']:.4f} "
                  f"per-class={['%.3f' % v for v in res['iou_list']]} | "
                  f"CỤC LỖI: R={res['obj_recall']:.3f} P={res['obj_precision']:.3f} "
                  f"F1={res['obj_f1']:.3f} (bắt={otp} nhầm={ofp} sót={ofn}) | "
                  f"best_by[{args.best_metric}]={score:.4f}")
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), weights_dir / "best.pt")
                print(f"     [BEST] {args.best_metric} mới = {best_score:.4f} -> đã lưu best.pt")
        else:
            print(f"  -> loss={avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), weights_dir / "best.pt")
                print(f"     [BEST] train loss mới = {best_loss:.4f} -> đã lưu best.pt")

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if (device.type == "cuda") else {},
            "best_score": best_score,
            "best_loss": best_loss,
        }
        torch.save(checkpoint_data, last_checkpoint_path)

    cfg = {"arch": args.arch, "encoder": args.encoder, "num_classes": num_classes,
           "names": names, "mean": mean, "std": std, "tile": ds.get("tile")}
    with open(out_dir / "model_cfg.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    if val_loader is not None:
        print(f"\n[HOÀN TẤT] best {args.best_metric} = {best_score:.4f}")
    else:
        print(f"\n[HOÀN TẤT] không có val, best.pt chọn theo train loss thấp nhất = {best_loss:.4f}")


if __name__ == "__main__":
    main()
