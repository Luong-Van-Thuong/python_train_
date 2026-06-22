# -*- coding: utf-8 -*-
"""
chia_data_unet.py
=================
Chuẩn bị dữ liệu cho mô hình SEMANTIC SEGMENTATION (U-Net) từ nhãn labelme
(polygon). Khác với YOLO (vẽ ô / toạ độ polygon), U-Net học theo từng PIXEL,
nên mỗi ảnh cần thêm 1 "ảnh mask" cùng kích thước:
    - pixel NỀN  -> 0
    - pixel LỖI  -> 1, 2, 3 ... (theo class id, bắt đầu từ 1)

Cách phân loại NG/OK (giống chia_data_seg_obd.py):
    - Ảnh CÓ file .json trùng tên  -> ảnh LỖI  (sinh mask theo polygon).
    - Ảnh KHÔNG có .json           -> ảnh NỀN  (mask toàn 0).
    -> NG/OK lẫn lộn vẫn đúng (phân theo .json, không theo tên thư mục).

Pipeline:
    1. Quét cả thư mục NG và OK, phân loại lỗi/nền theo .json.
    2. Gán class id ổn định theo alphabet (NỀN=0, lỗi=1..N).
    3. Chia train/val THEO ẢNH GỐC (tránh rò rỉ tile cùng ảnh sang 2 phía).
       Ảnh tên bắt đầu bằng 'pass' (đổi qua --val-prefix) LUÔN vào val.
    4. Với mỗi ảnh: vẽ mask cả ảnh (cv2.fillPoly) rồi CẮT TILE đồng thời ảnh
       và mask -> cặp (image, mask) cùng tên file.
    5. Cân bằng tile nền theo --bg-ratio / --ok-tiles-per-img.
    6. (Tuỳ chọn) sinh preview/ tô màu để kiểm tra mask bằng mắt.

Cấu trúc đầu ra (train U-Net thẳng được):
    <out>/images/{train,val}/<ten>.png      # ảnh tile
    <out>/masks/{train,val}/<ten>.png       # mask 1 kênh, pixel = class id
    <out>/preview/{train,val}/<ten>.png     # (tuỳ chọn) mask tô màu chồng ảnh
    <out>/classes.txt                       # 0=background, 1=<loi>, ...
    <out>/dataset.yaml                       # tóm tắt (num_classes, đường dẫn)

Cài thư viện:
    pip install opencv-python numpy pyyaml tqdm

Chạy:
    python chia_data_unet.py                 # dùng mặc định bên dưới
    python chia_data_unet.py --tile 512 --bg-ratio 2 --preview
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
# Cấu hình mặc định (sửa trực tiếp hoặc truyền qua dòng lệnh)
# Chạy WSL2: ổ D:\ map sang /mnt/d/
# --------------------------------------------------------------------------- #
DEFAULT_SRC = "/mnt/d/Images_/SIBV/A26/img_train/ng_"   # ảnh NG (có/không .json)
DEFAULT_OK = "/mnt/d/Images_/SIBV/A26/img_train/ok_"    # ảnh OK (thường không .json)
DEFAULT_OUT = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A26/data_imgs_unet"

# Ảnh có TÊN bắt đầu bằng tiền tố này -> LUÔN vào val (để test/đánh giá), KHÔNG
# bao giờ vào train. Đặt "" để tắt.
DEFAULT_VAL_PREFIX = "pass"

IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Bảng màu BGR cho preview (index 0 = nền -> không tô). Thêm màu nếu nhiều class.
PREVIEW_COLORS = [
    (0, 0, 0),       # 0 nền (không dùng)
    (0, 0, 255),     # 1 đỏ
    (0, 165, 255),   # 2 cam
    (0, 255, 0),     # 3 xanh lá
    (255, 0, 0),     # 4 xanh dương
    (255, 0, 255),   # 5 hồng
    (0, 255, 255),   # 6 vàng
]


# --------------------------------------------------------------------------- #
# Đọc/ghi ảnh an toàn với đường dẫn unicode (Windows)
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
# Labelme
# --------------------------------------------------------------------------- #
def shape_to_points(shape):
    """1 shape labelme -> mảng điểm np.int32 [[x,y],...]. Hỗ trợ polygon & rectangle."""
    pts = shape.get("points", [])
    st = shape.get("shape_type", "polygon")
    if st == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        ring = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    elif st == "polygon" and len(pts) >= 3:
        ring = [(float(x), float(y)) for x, y in pts]
    else:
        return None  # circle/line/point -> bỏ qua
    return np.array(ring, dtype=np.float64).round().astype(np.int32)


def collect_dir(dir_path, label=""):
    """Quét 1 thư mục -> list (img_path, json_path | None) theo SỰ TỒN TẠI .json."""
    if not dir_path:
        return []
    src = Path(dir_path)
    if not src.exists():
        print(f"[CẢNH BÁO] Không tìm thấy thư mục {label}: {dir_path}, bỏ qua.")
        return []
    out = []
    for img_path in sorted(src.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        json_path = img_path.with_suffix(".json")
        out.append((img_path, json_path if json_path.exists() else None))
    return out


def group_key(img_path, sep):
    """Khoá gom ảnh CÙNG 1 CON HÀNG: phần tên trước `sep`. '' -> mỗi ảnh 1 nhóm."""
    stem = img_path.stem
    if sep and sep in stem:
        return stem.split(sep)[0]
    return stem


def is_forced_val(img_path, prefix):
    """Ảnh tên bắt đầu bằng `prefix` (vd 'pass') -> luôn vào val. prefix '' -> tắt."""
    if not prefix:
        return False
    return img_path.stem.lower().startswith(prefix.lower())


def build_class_map(pairs):
    """Quét json -> class sắp alphabet -> {label: id}. NỀN=0, lỗi bắt đầu từ 1."""
    labels = set()
    for _, json_path in pairs:
        if json_path is None:
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sh in data.get("shapes", []):
            labels.add(sh["label"])
    classes = sorted(labels)
    return {name: i + 1 for i, name in enumerate(classes)}, classes


# --------------------------------------------------------------------------- #
# Mask + cắt tile
# --------------------------------------------------------------------------- #
def build_full_mask(img_shape, json_path, class_map):
    """Vẽ mask CẢ ẢNH: nền=0, lỗi=class id. Vẽ shape diện tích lớn trước để
    shape nhỏ (nếu chồng) không bị đè mất."""
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    if json_path is None:
        return mask
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = []
    for sh in data.get("shapes", []):
        pts = shape_to_points(sh)
        if pts is None or sh["label"] not in class_map:
            continue
        area = cv2.contourArea(pts)
        shapes.append((area, class_map[sh["label"]], pts))
    # lớn -> nhỏ
    for _, cid, pts in sorted(shapes, key=lambda t: t[0], reverse=True):
        cv2.fillPoly(mask, [pts], int(cid))
    return mask


def tile_origins(length, tile, overlap):
    """Toạ độ gốc tile dọc 1 chiều, phủ tới mép."""
    if length <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, length - tile + 1, step))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def colorize_mask(crop_img, mask_tile):
    """Trộn ảnh gốc với màu của từng class -> preview kiểm tra bằng mắt."""
    vis = crop_img.copy()
    overlay = vis.copy()
    for cid in np.unique(mask_tile):
        if cid == 0:
            continue
        color = PREVIEW_COLORS[cid % len(PREVIEW_COLORS)]
        overlay[mask_tile == cid] = color
    return cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)


def process_image(img_path, json_path, class_map, dirs, split,
                  tile, overlap, bg_ratio, ok_tiles_per_img,
                  min_defect_px, img_ext, save_preview, rng):
    """Cắt 1 ảnh thành cặp (image, mask) tile. Trả về thống kê."""
    img = imread_unicode(img_path)
    if img is None:
        print(f"[LỖI] Không đọc được ảnh {img_path.name}, bỏ qua.")
        return {"pos": 0, "neg": 0, "defect_px": 0}

    H, W = img.shape[:2]
    full_mask = build_full_mask(img.shape, json_path, class_map)

    xs = tile_origins(W, tile, overlap)
    ys = tile_origins(H, tile, overlap)

    # (x0, y0, là_lỗi)
    pos_tiles, neg_tiles = [], []
    for y0 in ys:
        for x0 in xs:
            m = full_mask[y0:y0 + tile, x0:x0 + tile]
            n_def = int(np.count_nonzero(m))
            (pos_tiles if n_def >= min_defect_px else neg_tiles).append((x0, y0))

    # Cân bằng tile nền
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

    def save_tile(x0, y0):
        nonlocal defect_px
        crop = img[y0:y0 + tile, x0:x0 + tile]
        m = full_mask[y0:y0 + tile, x0:x0 + tile]
        ch, cw = crop.shape[:2]
        if (ch, cw) != (tile, tile):  # pad mép
            pad_img = np.zeros((tile, tile, 3), dtype=img.dtype)
            pad_m = np.zeros((tile, tile), dtype=full_mask.dtype)
            pad_img[:ch, :cw] = crop
            pad_m[:ch, :cw] = m
            crop, m = pad_img, pad_m
        name = f"{stem}__x{x0}_y{y0}"
        imwrite_unicode(img_dir / f"{name}{img_ext}", crop, img_ext)
        imwrite_unicode(mask_dir / f"{name}.png", m, ".png")  # mask LUÔN .png
        if save_preview and prev_dir is not None:
            imwrite_unicode(prev_dir / f"{name}.png", colorize_mask(crop, m), ".png")
        defect_px += int(np.count_nonzero(m))

    for x0, y0 in pos_tiles:
        save_tile(x0, y0)
    for x0, y0 in neg_tiles:
        save_tile(x0, y0)

    return {"pos": len(pos_tiles), "neg": len(neg_tiles), "defect_px": defect_px}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Cắt tile + rasterize labelme -> dataset mask cho U-Net")
    ap.add_argument("--src", default=DEFAULT_SRC, help="Thư mục ảnh + json (NG)")
    ap.add_argument("--ok-src", default=DEFAULT_OK,
                    help="Thư mục ảnh OK (không json). '' để bỏ qua")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Thư mục dataset đầu ra")
    ap.add_argument("--tile", type=int, default=512, help="Kích thước tile (px)")
    ap.add_argument("--overlap", type=float, default=0.20, help="Tỉ lệ chồng lấn 0..1")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Tỉ lệ ảnh val")
    ap.add_argument("--val-prefix", default=DEFAULT_VAL_PREFIX,
                    help="Ảnh tên bắt đầu bằng tiền tố này LUÔN vào val. '' để tắt")
    ap.add_argument("--group-sep", default="",
                    help="Dấu phân tách gom ảnh CÙNG CON HÀNG. '' = chia theo từng ảnh")
    ap.add_argument("--bg-ratio", type=float, default=2.0,
                    help="Số tile nền / tile lỗi (mỗi ảnh NG). -1 = giữ tất cả")
    ap.add_argument("--ok-tiles-per-img", type=int, default=10,
                    help="Số tile nền lấy ngẫu nhiên mỗi ảnh OK. -1 = giữ tất cả")
    ap.add_argument("--min-defect-px", type=int, default=10,
                    help="Tile phải có >= ngần này pixel lỗi mới tính là tile LỖI")
    ap.add_argument("--img-ext", default=".png", choices=[".png", ".jpg"],
                    help="Định dạng ảnh tile xuất ra (mask luôn .png)")
    ap.add_argument("--preview", action="store_true",
                    help="Xuất thêm preview/ tô màu mask để kiểm tra bằng mắt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Quét 2 thư mục; phân loại lỗi/nền theo .json
    pairs = collect_dir(args.src, "NG")
    ok_pairs = collect_dir(args.ok_src, "OK")
    if not pairs and not ok_pairs:
        print(f"[LỖI] Không tìm thấy ảnh nào trong {args.src} hoặc {args.ok_src}")
        return

    all_pairs = pairs + ok_pairs
    n_defect = sum(1 for _, j in all_pairs if j is not None)
    print(f"Tìm thấy {len(all_pairs)} ảnh: {n_defect} ảnh LỖI (có .json), "
          f"{len(all_pairs) - n_defect} ảnh NỀN (không .json).")

    class_map, classes = build_class_map(all_pairs)
    print("Class map (NỀN=0, lỗi từ 1):")
    print("   0: background")
    for name, i in class_map.items():
        print(f"   {i}: {name}")

    # Chia train/val gom theo con hàng + ép 'pass*' vào val
    def split_pairs(plist):
        if not plist:
            return [], []
        forced_val = [p for p in plist if is_forced_val(p[0], args.val_prefix)]
        rest = [p for p in plist if not is_forced_val(p[0], args.val_prefix)]
        groups = {}
        for pair in rest:
            groups.setdefault(group_key(pair[0], args.group_sep), []).append(pair)
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
    if args.val_prefix:
        n_forced = sum(is_forced_val(p[0], args.val_prefix) for p in all_pairs)
        print(f"Ảnh tên bắt đầu '{args.val_prefix}' -> ép vào val: {n_forced} ảnh.")

    def dem(plist):
        d = sum(1 for _, j in plist if j is not None)
        return d, len(plist) - d
    tr_d, tr_b = dem(train_pairs)
    va_d, va_b = dem(val_pairs)
    print(f"Chia: train={len(train_pairs)} (lỗi={tr_d}, nền={tr_b}), "
          f"val={len(val_pairs)} (lỗi={va_d}, nền={va_b})")

    # Tạo cây thư mục
    out_base = Path(args.out)
    split_dirs = {}
    for split in ("train", "val"):
        img_dir = out_base / "images" / split
        mask_dir = out_base / "masks" / split
        prev_dir = out_base / "preview" / split if args.preview else None
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        if prev_dir is not None:
            prev_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = (img_dir, mask_dir, prev_dir)

    totals = {"train": {"pos": 0, "neg": 0, "defect_px": 0},
              "val": {"pos": 0, "neg": 0, "defect_px": 0}}

    for split, plist in (("train", train_pairs), ("val", val_pairs)):
        for img_path, json_path in tqdm(plist, desc=f"Cắt tile [{split}]"):
            st = process_image(
                img_path, json_path, class_map, split_dirs[split], split,
                tile=args.tile, overlap=args.overlap,
                bg_ratio=args.bg_ratio, ok_tiles_per_img=args.ok_tiles_per_img,
                min_defect_px=args.min_defect_px, img_ext=args.img_ext,
                save_preview=args.preview, rng=rng,
            )
            for k in totals[split]:
                totals[split][k] += st[k]

    # Ghi classes.txt + dataset.yaml
    names = ["background"] + classes  # index 0 = background
    with open(out_base / "classes.txt", "w", encoding="utf-8") as f:
        for i, n in enumerate(names):
            f.write(f"{i} {n}\n")

    dataset = {
        "path": str(out_base),
        "images": {"train": "images/train", "val": "images/val"},
        "masks": {"train": "masks/train", "val": "masks/val"},
        "num_classes": len(names),          # gồm cả nền
        "names": {i: n for i, n in enumerate(names)},
        "tile": args.tile,
        "ignore_index": None,
    }
    with open(out_base / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset, f, allow_unicode=True, sort_keys=False)

    # Báo cáo
    print("\n===== HOÀN TẤT =====")
    for split in ("train", "val"):
        tt = totals[split]
        tiles = tt["pos"] + tt["neg"]
        frac = (tt["defect_px"] / (tiles * args.tile * args.tile) * 100) if tiles else 0
        print(f"{split:5s}: {tt['pos']} tile lỗi, {tt['neg']} tile nền "
              f"(tỉ lệ pixel lỗi ~{frac:.3f}%)")
    print(f"\nDataset: {out_base}")
    print(f"  classes.txt + dataset.yaml (num_classes = {len(names)} gồm cả nền)")
    if args.preview:
        print("  preview/ : mask tô màu để kiểm tra bằng mắt")
    print("\nLƯU Ý train: pixel lỗi rất ít -> dùng loss Dice/Focal hoặc BCE có "
          "trọng số để chống mất cân bằng lớp.")


if __name__ == "__main__":
    main()
