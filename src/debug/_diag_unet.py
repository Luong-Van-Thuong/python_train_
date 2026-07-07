# -*- coding: utf-8 -*-
"""Chẩn đoán vì sao UNet defect_unet2 không bắt được lỗi."""
import sys
from pathlib import Path
import numpy as np
import cv2
import yaml
import torch
import segmentation_models_pytorch as smp

CFG = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/results/unet/defect_unet/model_cfg.yaml"
WEIGHTS = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/results/unet/defect_unet2/weights/best.pt"
SRC_DIR = "/mnt/d/Images_/SIBV/A27/test3"


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def tile_origins(length, tile, overlap):
    if length <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, length - tile + 1, step))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


@torch.no_grad()
def predict_full(model, img_rgb, mean, std, tile, overlap, device, num_classes):
    H, W = img_rgb.shape[:2]
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    prob_sum = np.zeros((num_classes, H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    xs = tile_origins(W, tile, overlap)
    ys = tile_origins(H, tile, overlap)
    for y0 in ys:
        for x0 in xs:
            crop = img_rgb[y0:y0 + tile, x0:x0 + tile]
            th, tw = crop.shape[:2]
            if (th, tw) != (tile, tile):
                pad = np.zeros((tile, tile, 3), dtype=crop.dtype)
                pad[:th, :tw] = crop
                crop = pad
            x = (crop.astype(np.float32) - mean) / std
            x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0].float().cpu().numpy()
            prob_sum[:, y0:y0 + th, x0:x0 + tw] += prob[:, :th, :tw]
            count[y0:y0 + th, x0:x0 + tw] += 1.0
    count[count == 0] = 1.0
    prob_avg = prob_sum / count[None]
    pred = prob_avg.argmax(0).astype(np.int32)
    return pred, prob_avg


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    with open(CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    num_classes = int(cfg["num_classes"])
    names = cfg.get("names", {})
    tile = int(cfg.get("tile") or 512)
    mean, std = cfg["mean"], cfg["std"]
    print("[INFO]", cfg["arch"], cfg["encoder"], "classes", num_classes, "tile", tile)

    # Kiểm tra checkpoint best.pt là state_dict hay dict lồng
    ckpt = torch.load(WEIGHTS, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print("[WARN] best.pt là checkpoint dict (có model_state_dict). epoch:",
              ckpt.get("epoch"), "best_iou:", ckpt.get("best_iou"))
        state = ckpt["model_state_dict"]
    else:
        print("[INFO] best.pt là state_dict thuần.")
        state = ckpt

    model = smp.create_model(cfg["arch"], encoder_name=cfg["encoder"],
                             encoder_weights=None, in_channels=3,
                             classes=num_classes).to(device)
    model.load_state_dict(state)
    model.eval()

    files = sorted([p for p in Path(SRC_DIR).iterdir()
                    if p.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg", ".tif")])
    print("[INFO] số ảnh:", len(files))
    for f in files:
        img = imread_unicode(f, cv2.IMREAD_COLOR)
        if img is None:
            print("  [BỎ QUA] đọc lỗi:", f.name)
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred, prob_avg = predict_full(model, img_rgb, mean, std, tile, 0.2, device, num_classes)
        print("\n=== %s  shape=%s ===" % (f.name, img.shape))
        # thống kê xác suất tối đa của từng lớp lỗi trên toàn ảnh
        for cid in range(num_classes):
            cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
            pmax = float(prob_avg[cid].max())
            npix_pred = int((pred == cid).sum())
            print("  class %d %-10s : max_prob=%.4f  pixels_argmax=%d" % (cid, cname, pmax, npix_pred))
        # số pixel lỗi (non-bg) theo ngưỡng conf
        prob_max = prob_avg.max(0)
        defect_argmax = (pred != 0)
        for conf in (0.3, 0.5, 0.7, 0.9):
            n = int((defect_argmax & (prob_max >= conf)).sum())
            print("    pixel lỗi (argmax!=0 & prob>=%.1f): %d" % (conf, n))


if __name__ == "__main__":
    main()
