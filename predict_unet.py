# -*- coding: utf-8 -*-
"""
predict_unet.py
===============
Inference U-Net trên ẢNH GỐC độ phân giải cao bằng SLIDING WINDOW.
Đã loại bỏ phần xuất file mask thô để tối ưu tốc độ I/O.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
import torch
import segmentation_models_pytorch as smp

# ==============================================================================
# CONFIGURATION ZONE
# ==============================================================================
DEBUG_SOURCE = "/mnt/d/Images_/SIBV/A27/test/"
DEFAULT_CFG = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/results/unet/defect_unet/model_cfg.yaml"
DEFAULT_WEIGHTS = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/results/unet/defect_unet/weights/best.pt"
DEFAULT_OUT = "/mnt/d/Projects_/Cong_Ty/Python_/train/predict_out/folder_unet_AI_tra_ve"
IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

COLORS = [
    (0, 0, 0),        # 0 nền
    (0, 0, 255),      # 1 đỏ
    (0, 165, 255),    # 2 cam
    (0, 255, 0),      # 3 xanh lá
    (255, 0, 0),      # 4 xanh dương
    (255, 0, 255),    # 5 hồng
    (0, 255, 255),    # 6 vàng
]
# ==============================================================================

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)

def imwrite_unicode(path, img, ext=".png"):
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok

def gather_sources(source):
    p = Path(source)
    if p.is_dir():
        return [f for f in sorted(p.iterdir()) if f.suffix.lower() in IMG_EXTS]
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [p]
    return []

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
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(x)
            prob = torch.softmax(logits, dim=1)[0].float().cpu().numpy()

            prob_sum[:, y0:y0 + th, x0:x0 + tw] += prob[:, :th, :tw]
            count[y0:y0 + th, x0:x0 + tw] += 1.0

    count[count == 0] = 1.0
    prob_avg = prob_sum / count[None]
    pred = prob_avg.argmax(0).astype(np.int32)
    prob_max = prob_avg.max(0)
    return pred, prob_max

def render(img_bgr, pred, prob_max, names, conf, min_area):
    vis = img_bgr.copy()
    overlay = vis.copy()
    defects = []

    for cid in np.unique(pred):
        if cid == 0:
            continue
        color = COLORS[cid % len(COLORS)]
        cls_mask = ((pred == cid) & (prob_max >= conf)).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue

        n, lbl, stats, _ = cv2.connectedComponentsWithStats(cls_mask, 8)
        keep = np.zeros_like(cls_mask)
        for k in range(1, n):
            area = stats[k, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            keep[lbl == k] = 1
            x, y, w, h = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                          stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
            cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
            defects.append((cname, int(area), x, y, w, h))
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        if keep.sum() > 0:
            overlay[keep == 1] = color
            cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, color, 1)

    vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
    return vis, defects

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="Ảnh hoặc thư mục ảnh")
    ap.add_argument("--cfg", default=DEFAULT_CFG, help="model_cfg.yaml")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="best.pt")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Thư mục xuất kết quả")
    ap.add_argument("--tile", type=int, default=0, help="0 = lấy theo cfg")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    source = args.source if args.source is not None else DEBUG_SOURCE

    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        print(f"[LỖI] Không thấy model_cfg.yaml: {cfg_path}")
        return
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    w_path = Path(args.weights)
    if not w_path.exists():
        print(f"[LỖI] Không thấy weights: {w_path}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = int(cfg["num_classes"])
    names = cfg.get("names", {})
    tile = args.tile if args.tile > 0 else int(cfg.get("tile") or 512)
    mean, std = cfg["mean"], cfg["std"]

    print(f"[INFO] {cfg['arch']} + {cfg['encoder']} | classes={num_classes} | tile={tile}")
    model = smp.create_model(cfg["arch"], encoder_name=cfg["encoder"],
                             encoder_weights=None, in_channels=3,
                             classes=num_classes).to(device)
    model.load_state_dict(torch.load(w_path, map_location=device))
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = gather_sources(source)
    if not files:
        print(f"[LỖI] Không tìm thấy ảnh tại: {source}")
        return
    print(f"[INFO] Xử lý {len(files)} ảnh...")

    for f in files:
        img = imread_unicode(f, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[BỎ QUA] Lỗi đọc: {f.name}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred, prob_max = predict_full(model, img_rgb, mean, std, tile,
                                      args.overlap, device, num_classes)
        vis, defects = render(img, pred, prob_max, names, args.conf, args.min_area)

        # CHỈ LƯU ẢNH KẾT QUẢ VÀ FILE TXT THÔNG TIN LỖI
        imwrite_unicode(out_dir / f"{f.stem}_pred.png", vis, ".png")
        print(f"-> {f.name}: {len(defects)} vùng lỗi")
        
        if defects:
            with open(out_dir / f"{f.stem}.txt", "w", encoding="utf-8") as fh:
                for cname, area, x, y, w, h in defects:
                    fh.write(f"{cname} area={area} {x} {y} {x + w} {y + h}\n")

    print(f"\n[HOÀN TẤT] Kết quả tại: {out_dir}")

if __name__ == "__main__":
    main()