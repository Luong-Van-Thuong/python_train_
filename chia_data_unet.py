# -*- coding: utf-8 -*-
"""
chia_data_unet.py
=================
Chuẩn bị dữ liệu cho mô hình SEMANTIC SEGMENTATION (U-Net) từ nhãn labelme.
Tối ưu hóa bộ nhớ (Localized Mask Rendering) & Linh hoạt cấu hình Train/Val Split.

Chạy 100% Train (Không chia Val):
    python chia_data_unet.py --val-ratio 0 --val-prefix ""

Chạy mặc định (80% Train, 20% Val):
    python chia_data_unet.py --val-ratio 0.2 --tile 512 --preview
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Cấu hình mặc định (Đường dẫn chạy WSL2/Windows)
# --------------------------------------------------------------------------- #
DEFAULT_SRC = "/mnt/d/Images_/SIBV/A27/img_train/ng1"
DEFAULT_OK = "/mnt/d/Images_/SIBV/A27/img_train/ok_"
DEFAULT_OUT = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/data_imgs_unet"
DEFAULT_VAL_PREFIX = "pass"

IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

PREVIEW_COLORS = [
    (0, 0, 0),       # 0 nền
    (0, 0, 255),     # 1 đỏ
    (0, 165, 255),   # 2 cam
    (0, 255, 0),     # 3 xanh lá
    (255, 0, 0),     # 4 xanh dương
    (255, 0, 255),   # 5 hồng
    (0, 255, 255),   # 6 vàng
]

# --------------------------------------------------------------------------- #
# I/O Unicode Safe
# --------------------------------------------------------------------------- #
def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img, ext=".png"):
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok


# --------------------------------------------------------------------------- #
# Hình học & Parsing Labelme
# --------------------------------------------------------------------------- #
def shape_to_points(shape):
    pts = shape.get("points", [])
    st = shape.get("shape_type", "polygon")
    if st == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        ring = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    elif st == "polygon" and len(pts) >= 3:
        ring = [(float(x), float(y)) for x, y in pts]
    else:
        return None
    return np.array(ring, dtype=np.float64).round().astype(np.int32)


def extract_shapes(json_path, class_map):
    """Trích xuất tất cả các đa giác lỗi từ file JSON để chuẩn bị vẽ cục bộ."""
    if json_path is None or not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    shapes_list = []
    for sh in data.get("shapes", []):
        pts = shape_to_points(sh)
        if pts is None or sh["label"] not in class_map:
            continue
        shapes_list.append((class_map[sh["label"]], pts))
    return shapes_list


def build_tile_mask(tile_shape, x0, y0, shapes_list):
    """
    ROOT CAUSE FIX: Chỉ vẽ mask trên kích thước của TILE (Local Mask).
    Dịch chuyển tọa độ đa giác về gốc (x0, y0) giúp tiết kiệm tài nguyên RAM.
    """
    tile_mask = np.zeros(tile_shape, dtype=np.uint8)
    tile_poly_list = []
    
    for cid, pts in shapes_list:
        pts_shifted = pts - np.array([x0, y0], dtype=np.int32)
        # Bộ lọc sơ bộ xem đa giác lỗi có nằm trong phạm vi Tile hay không
        if np.any((pts_shifted[:, 0] >= 0) & (pts_shifted[:, 0] < tile_shape[1]) &
                  (pts_shifted[:, 1] >= 0) & (pts_shifted[:, 1] < tile_shape[0])):
            tile_poly_list.append((cv2.contourArea(pts_shifted), cid, pts_shifted))
            
    # Vẽ từ đa giác có diện tích lớn đến nhỏ để tránh đè lớp
    for _, cid, pts_s in sorted(tile_poly_list, key=lambda t: t[0], reverse=True):
        cv2.fillPoly(tile_mask, [pts_s], int(cid))
        
    return tile_mask


def collect_dir(dir_path, label=""):
    if not dir_path:
        return []
    src = Path(dir_path)
    if not src.exists():
        print(f"[CẢNH BÁO] Không tìm thấy thư mục {label}: {dir_path}")
        return []
    out = []
    for img_path in sorted(src.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        json_path = img_path.with_suffix(".json")
        out.append((img_path, json_path if json_path.exists() else None))
    return out


def tile_origins(length, tile, overlap):
    if length <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, length - tile + 1, step))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def colorize_mask(crop_img, mask_tile):
    vis = crop_img.copy()
    overlay = vis.copy()
    for cid in np.unique(mask_tile):
        if cid == 0:
            continue
        color = PREVIEW_COLORS[cid % len(PREVIEW_COLORS)]
        overlay[mask_tile == cid] = color
    return cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)


# --------------------------------------------------------------------------- #
# Xử lý Cắt & Lọc dữ liệu lỗi/nền
# --------------------------------------------------------------------------- #
def process_image(img_path, json_path, class_map, dirs, tile, overlap, 
                  bg_ratio, ok_tiles_per_img, min_defect_px, img_ext, save_preview, rng):
    img = imread_unicode(img_path)
    if img is None:
        return {"pos": 0, "neg": 0, "defect_px": 0}

    H, W = img.shape[:2]
    shapes_list = extract_shapes(json_path, class_map)

    xs = tile_origins(W, tile, overlap)
    ys = tile_origins(H, tile, overlap)

    pos_tiles, neg_tiles = [], []

    # Giai đoạn 1: Xác định tọa độ và phân loại bằng Local Mask nhanh
    for y0 in ys:
        for x0 in xs:
            # Chỉ sinh tạm mask với kích thước tile để đếm số lượng pixel lỗi
            m_tile = build_tile_mask((tile, tile), x0, y0, shapes_list)
            n_def = int(np.count_nonzero(m_tile))
            
            if n_def >= min_defect_px:
                pos_tiles.append((x0, y0))
            else:
                neg_tiles.append((x0, y0))

    # Giai đoạn 2: Cân bằng tỷ lệ nền/lỗi theo yêu cầu hệ thống
    if json_path is None:
        if ok_tiles_per_img >= 0:
            rng.shuffle(neg_tiles)
            neg_tiles = neg_tiles[:ok_tiles_per_img]
    elif bg_ratio >= 0:
        keep = int(round(len(pos_tiles) * bg_ratio))
        rng.shuffle(neg_tiles)
        neg_tiles = neg_tiles[:keep]

    img_dir, mask_dir, prev_dir = dirs
    stem = img_path.stem
    defect_px = 0

    # Giai đoạn 3: Thực thi I/O ghi xuống đĩa các tile được chọn
    def save_tile(x0, y0):
        nonlocal defect_px
        crop = img[y0:y0 + tile, x0:x0 + tile]
        m = build_tile_mask((tile, tile), x0, y0, shapes_list)
        
        ch, cw = crop.shape[:2]
        if (ch, cw) != (tile, tile):
            pad_img = np.zeros((tile, tile, 3), dtype=img.dtype)
            pad_m = np.zeros((tile, tile), dtype=m.dtype)
            pad_img[:ch, :cw] = crop
            pad_m[:ch, :cw] = m
            crop, m = pad_img, pad_m

        name = f"{stem}__x{x0}_y{y0}"
        imwrite_unicode(img_dir / f"{name}{img_ext}", crop, img_ext)
        imwrite_unicode(mask_dir / f"{name}.png", m, ".png")
        
        if save_preview and prev_dir is not None:
            imwrite_unicode(prev_dir / f"{name}.png", colorize_mask(crop, m), ".png")
        defect_px += int(np.count_nonzero(m))

    for x0, y0 in pos_tiles: save_tile(x0, y0)
    for x0, y0 in neg_tiles: save_tile(x0, y0)

    return {"pos": len(pos_tiles), "neg": len(neg_tiles), "defect_px": defect_px}


# --------------------------------------------------------------------------- #
# Main Execution Pipeline
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="U-Net Data Splitter & Local Tiling Engine")
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--ok-src", default=DEFAULT_OK)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Tỷ lệ chia validation (0 -> 1.0). Thiết lập 0 nếu muốn dồn 100% vào Train")
    ap.add_argument("--val-prefix", default=DEFAULT_VAL_PREFIX, help="Tiền tố ép ảnh vào val. Đặt '' để tắt hoàn toàn luồng ép buộc")
    ap.add_argument("--group-sep", default="")
    ap.add_argument("--bg-ratio", type=float, default=2.0)
    ap.add_argument("--ok-tiles-per-img", type=int, default=10)
    ap.add_argument("--min-defect-px", type=int, default=10)
    ap.add_argument("--img-ext", default=".png", choices=[".png", ".jpg"])
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pairs = collect_dir(args.src, "NG")
    ok_pairs = collect_dir(args.ok_src, "OK")
    all_pairs = pairs + ok_pairs

    if not all_pairs:
        print("[LỖI] Không tìm thấy ảnh đầu vào!")
        return

    # Khởi tạo class map bằng alphabet
    labels = set()
    for _, j_path in all_pairs:
        if j_path is None: continue
        with open(j_path, "r", encoding="utf-8") as f:
            for sh in json.load(f).get("shapes", []): labels.add(sh["label"])
    classes = sorted(labels)
    class_map = {name: i + 1 for i, name in enumerate(classes)}

    # --------------------------------------------------------------------------- #
    # Tối ưu hóa phân tách dữ liệu linh hoạt (Train/Val Split Control)
    # --------------------------------------------------------------------------- #
    def split_pairs(plist):
        if not plist: return [], []
        
        # Nếu thiết lập tỉ lệ val-ratio = 0 và không bắt buộc prefix -> Đưa toàn bộ vào Train
        if args.val_ratio <= 0 and not args.val_prefix:
            return plist, []

        forced_val = [p for p in plist if args.val_prefix and p[0].shape[0] and p[0].stem.lower().startswith(args.val_prefix.lower())]
        rest = [p for p in plist if p not in forced_val]

        if args.val_ratio <= 0:
            return rest, forced_val

        # Gom nhóm theo cấu trúc linh kiện ('group-sep') tránh Leak dữ liệu
        groups = {}
        for pair in rest:
            stem = pair[0].stem
            g_key = stem.split(args.group_sep)[0] if args.group_sep and args.group_sep in stem else stem
            groups.setdefault(g_key, []).append(pair)

        keys = list(groups.keys())
        rng.shuffle(keys)
        
        n_val = max(1, int(round(len(keys) * args.val_ratio))) if keys else 0
        val_keys, train_keys = keys[:n_val], keys[n_val:]
        
        train = [p for k in train_keys for p in groups[k]]
        val = [p for k in val_keys for p in groups[k]] + forced_val
        return train, val

    train_ng, val_ng = split_pairs(pairs)
    train_ok, val_ok = split_pairs(ok_pairs)
    train_pairs = train_ng + train_ok
    val_pairs = val_ng + val_ok

    print(f"Tổng kết phân chia: Train = {len(train_pairs)} ảnh, Val = {len(val_pairs)} ảnh.")

    # Khởi tạo cấu trúc lưu trữ thư mục cứng
    out_base = Path(args.out)
    split_dirs = {}
    for split in ("train", "val"):
        if split == "val" and len(val_pairs) == 0:
            continue  # Nếu val_ratio = 0 thì không cần tạo/ghi thư mục val
        img_dir = out_base / "images" / split
        mask_dir = out_base / "masks" / split
        prev_dir = out_base / "preview" / split if args.preview else None
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        if prev_dir: prev_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = (img_dir, mask_dir, prev_dir)

    totals = {"train": {"pos": 0, "neg": 0, "defect_px": 0}, "val": {"pos": 0, "neg": 0, "defect_px": 0}}

    # Thực thi vòng lặp chính
    for split, plist in (("train", train_pairs), ("val", val_pairs)):
        if not plist: continue
        for img_path, json_path in tqdm(plist, desc=f"Cắt tile [{split}]"):
            st = process_image(img_path, json_path, class_map, split_dirs[split], 
                               tile=args.tile, overlap=args.overlap, bg_ratio=args.bg_ratio, 
                               ok_tiles_per_img=args.ok_tiles_per_img, min_defect_px=args.min_defect_px, 
                               img_ext=args.img_ext, save_preview=args.preview, rng=rng)
            for k in totals[split]: totals[split][k] += st[k]

    # Xuất thông tin file cấu hình YAML & TXT
    names = ["background"] + classes
    with open(out_base / "classes.txt", "w", encoding="utf-8") as f:
        for i, n in enumerate(names): f.write(f"{i} {n}\n")

    dataset = {
        "path": str(out_base),
        "images": {"train": "images/train", "val": "images/val" if len(val_pairs) > 0 else ""},
        "masks": {"train": "masks/train", "val": "masks/val" if len(val_pairs) > 0 else ""},
        "num_classes": len(names),
        "names": {i: n for i, n in enumerate(names)},
        "tile": args.tile,
    }
    with open(out_base / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset, f, allow_unicode=True, sort_keys=False)

    print("\n===== PIPELINE HOÀN TẤT SUỐN SẺ =====")


if __name__ == "__main__":
    main()