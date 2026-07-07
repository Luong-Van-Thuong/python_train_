# -*- coding: utf-8 -*-
"""
chia_data.py
============
Chuẩn bị dữ liệu cho YOLO11-seg từ nhãn labelme (polygon) cho bài toán
phát hiện defect NHỎ (bien_dang / thieu_nhua) trên ảnh độ phân giải cao.

Pipeline:
    1. Quét toàn bộ cặp (.bmp/.png/.jpg + .json labelme) trong thư mục nguồn.
    2. Tự thu thập danh sách class -> gán id ổn định (sắp xếp alphabet).
    3. Chia train/val theo ẢNH GỐC (tránh rò rỉ giữa các tile cùng ảnh).
    4. Cắt mỗi ảnh thành tile (vd 640x640, overlap 20%).
    5. Clip polygon theo biên tile bằng Shapely (xử lý cả mảnh vỡ MultiPolygon).
    6. Giữ toàn bộ tile có lỗi; chỉ lấy một phần tile nền theo BG_RATIO.
    7. Ghi tile (PNG, lossless) + nhãn YOLO-seg (.txt toạ độ chuẩn hoá).
    8. Sinh data.yaml để train thẳng bằng Ultralytics.

Cài đặt thư viện:
    pip install opencv-python numpy shapely pyyaml tqdm

Chạy:
    python chia_data.py
(hoặc tuỳ chỉnh bằng tham số dòng lệnh, xem --help)
"""

import os
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Cấu hình mặc định (có thể sửa trực tiếp hoặc truyền qua dòng lệnh)
# --------------------------------------------------------------------------- #
# Chạy trong WSL2: ổ D:\ của Windows map sang /mnt/d/
DEFAULT_SRC = "/mnt/d/Images_/SIBV/A26/260615_0/ng_"
DEFAULT_OK = "/mnt/d/Images_/SIBV/A26/260615_0/ok_"  # ảnh OK, KHÔNG có .json
DEFAULT_OUT = "/mnt/d/Projects_/Cong_Ty/Python_/train/sibv/a26/data"

IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# Tiện ích đọc/ghi ảnh an toàn với đường dẫn Windows (unicode)
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
def shape_to_polygon(shape):
    """Chuyển 1 shape labelme thành list điểm [(x,y), ...]. Hỗ trợ polygon & rectangle."""
    pts = shape.get("points", [])
    st = shape.get("shape_type", "polygon")
    if st == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if st == "polygon" and len(pts) >= 3:
        return [(float(x), float(y)) for x, y in pts]
    # các loại khác (circle/line/point) bỏ qua
    return None


def collect_pairs(src_dir):
    """Tìm các cặp (ảnh, json) trong thư mục."""
    src = Path(src_dir)
    pairs = []
    for json_path in sorted(src.glob("*.json")):
        img_path = None
        for ext in IMG_EXTS:
            cand = json_path.with_suffix(ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            print(f"[CẢNH BÁO] Không tìm thấy ảnh cho {json_path.name}, bỏ qua.")
            continue
        pairs.append((img_path, json_path))
    return pairs


def collect_ok_images(ok_dir):
    """Thu thập ảnh OK (không có defect -> không có .json). Trả về list (img_path, None)."""
    if not ok_dir:
        return []
    src = Path(ok_dir)
    if not src.exists():
        print(f"[CẢNH BÁO] Không tìm thấy thư mục ảnh OK: {ok_dir}, bỏ qua.")
        return []
    pairs = []
    for img_path in sorted(src.iterdir()):
        if img_path.suffix.lower() in IMG_EXTS:
            pairs.append((img_path, None))  # None = không có nhãn (ảnh nền sạch)
    return pairs


def build_class_map(pairs):
    """Quét toàn bộ json -> danh sách class sắp xếp ổn định -> {label: id}."""
    labels = set()
    for _, json_path in pairs:
        if json_path is None:  # ảnh OK, không có nhãn
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sh in data.get("shapes", []):
            labels.add(sh["label"])
    classes = sorted(labels)
    return {name: i for i, name in enumerate(classes)}, classes


# --------------------------------------------------------------------------- #
# Cắt tile + clip nhãn
# --------------------------------------------------------------------------- #
def tile_origins(length, tile, overlap):
    """Danh sách toạ độ gốc của tile dọc theo 1 chiều, đảm bảo phủ tới mép."""
    if length <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, length - tile + 1, step))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def clip_polygon_to_tile(poly, tile_box):
    """Cắt polygon theo ô tile, trả về list các polygon hợp lệ (đã xử lý mảnh vỡ)."""
    if not poly.is_valid:
        poly = make_valid(poly)
    inter = poly.intersection(tile_box)
    if inter.is_empty:
        return []
    geoms = []
    gtype = inter.geom_type
    if gtype == "Polygon":
        geoms = [inter]
    elif gtype in ("MultiPolygon", "GeometryCollection"):
        geoms = [g for g in inter.geoms if g.geom_type == "Polygon"]
    return [g for g in geoms if g.area > 0]


def process_image(img_path, json_path, class_map, out_img_dir, out_lbl_dir,
                  tile, overlap, min_area_frac, min_area_px, bg_ratio,
                  img_ext, rng, ok_tiles_per_img=-1):
    """Cắt 1 ảnh thành tile, ghi tile + nhãn. Trả về dict thống kê.

    json_path = None -> ảnh OK (không có defect): mọi tile là nền, ghi .txt rỗng.
    """
    img = imread_unicode(img_path)
    if img is None:
        print(f"[LỖI] Không đọc được ảnh {img_path.name}, bỏ qua.")
        return {"pos": 0, "neg": 0, "inst": 0}

    H, W = img.shape[:2]

    if json_path is None:
        data = {"shapes": []}  # ảnh OK: không có annotation
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Chuẩn bị danh sách (class_id, shapely Polygon, area gốc)
    anns = []
    for sh in data.get("shapes", []):
        ring = shape_to_polygon(sh)
        if ring is None:
            continue
        cid = class_map[sh["label"]]
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty or poly.area <= 0:
            continue
        anns.append((cid, poly, poly.area))

    xs = tile_origins(W, tile, overlap)
    ys = tile_origins(H, tile, overlap)

    pos_tiles, neg_tiles = [], []  # mỗi phần tử: (x0, y0, [label_lines])

    for y0 in ys:
        for x0 in xs:
            tbox = box(x0, y0, x0 + tile, y0 + tile)
            lines = []
            for cid, poly, area0 in anns:
                if not poly.intersects(tbox):
                    continue
                for g in clip_polygon_to_tile(poly, tbox):
                    # Lọc mảnh quá nhỏ (sliver) do bị cắt
                    if g.area < min_area_px:
                        continue
                    if g.area < min_area_frac * area0:
                        continue
                    coords = list(g.exterior.coords)[:-1]  # bỏ điểm lặp cuối
                    if len(coords) < 3:
                        continue
                    norm = []
                    for px, py in coords:
                        nx = (px - x0) / tile
                        ny = (py - y0) / tile
                        nx = min(max(nx, 0.0), 1.0)
                        ny = min(max(ny, 0.0), 1.0)
                        norm.append(f"{nx:.6f} {ny:.6f}")
                    lines.append(f"{cid} " + " ".join(norm))

            if lines:
                pos_tiles.append((x0, y0, lines))
            else:
                neg_tiles.append((x0, y0, lines))

    # Cân bằng tile nền
    if json_path is None:
        # Ảnh OK: lấy mẫu theo ok_tiles_per_img (-1 = giữ tất cả)
        if ok_tiles_per_img >= 0:
            rng.shuffle(neg_tiles)
            neg_tiles = neg_tiles[:ok_tiles_per_img]
    elif bg_ratio >= 0:
        keep_neg = int(round(len(pos_tiles) * bg_ratio))
        rng.shuffle(neg_tiles)
        neg_tiles = neg_tiles[:keep_neg]

    stem = img_path.stem
    n_inst = 0

    def save_tile(x0, y0, lines):
        nonlocal n_inst
        crop = img[y0:y0 + tile, x0:x0 + tile]
        ch, cw = crop.shape[:2]
        if (ch, cw) != (tile, tile):  # pad nếu tràn mép (ảnh nhỏ hơn tile)
            padded = np.zeros((tile, tile, 3), dtype=img.dtype)
            padded[:ch, :cw] = crop
            crop = padded
        name = f"{stem}__x{x0}_y{y0}"
        imwrite_unicode(out_img_dir / f"{name}{img_ext}", crop, img_ext)
        with open(out_lbl_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        n_inst += len(lines)

    for x0, y0, lines in pos_tiles:
        save_tile(x0, y0, lines)
    for x0, y0, lines in neg_tiles:
        save_tile(x0, y0, lines)

    return {"pos": len(pos_tiles), "neg": len(neg_tiles), "inst": n_inst}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Cắt tile + convert labelme -> YOLO-seg")
    ap.add_argument("--src", default=DEFAULT_SRC, help="Thư mục chứa ảnh + json labelme (ảnh NG)")
    ap.add_argument("--ok-src", default=DEFAULT_OK,
                    help="Thư mục ảnh OK (không có json). '' để bỏ qua")
    ap.add_argument("--ok-tiles-per-img", type=int, default=-1,
                    help="Số tile nền lấy ngẫu nhiên mỗi ảnh OK. -1 = giữ tất cả")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Thư mục dataset đầu ra")
    ap.add_argument("--tile", type=int, default=640, help="Kích thước tile (px)")
    ap.add_argument("--overlap", type=float, default=0.20, help="Tỉ lệ chồng lấn 0..1")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Tỉ lệ ảnh dùng cho val")
    ap.add_argument("--bg-ratio", type=float, default=1.0,
                    help="Số tile nền giữ lại / số tile có lỗi (mỗi ảnh). -1 = giữ tất cả")
    ap.add_argument("--min-area-frac", type=float, default=0.10,
                    help="Giữ mảnh clip nếu diện tích >= frac * diện tích gốc")
    ap.add_argument("--min-area-px", type=float, default=8.0,
                    help="Diện tích tối thiểu (px^2) của mảnh sau khi clip")
    ap.add_argument("--img-ext", default=".png", choices=[".png", ".jpg"],
                    help="Định dạng tile xuất ra")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    pairs = collect_pairs(args.src)
    if not pairs:
        print(f"[LỖI] Không tìm thấy cặp ảnh+json nào trong {args.src}")
        return
    print(f"Tìm thấy {len(pairs)} cặp ảnh+nhãn (NG).")

    ok_pairs = collect_ok_images(args.ok_src)
    print(f"Tìm thấy {len(ok_pairs)} ảnh OK (nền sạch).")

    class_map, classes = build_class_map(pairs)
    print("Class map (id ổn định theo alphabet):")
    for name, i in class_map.items():
        print(f"   {i}: {name}")

    # Chia train/val theo ẢNH GỐC. Chia riêng NG và OK rồi gộp, để cả train và
    # val đều có tỉ lệ ảnh OK hợp lý (không bị dồn hết OK về một split).
    def split_pairs(plist):
        shuffled = plist[:]
        rng.shuffle(shuffled)
        if not shuffled:
            return [], []
        n_val = max(1, int(round(len(shuffled) * args.val_ratio)))
        return shuffled[n_val:], shuffled[:n_val]  # train, val

    train_ng, val_ng = split_pairs(pairs)
    train_ok, val_ok = split_pairs(ok_pairs)
    train_pairs = train_ng + train_ok
    val_pairs = val_ng + val_ok
    print(f"Chia: train={len(train_pairs)} ảnh (NG={len(train_ng)}, OK={len(train_ok)}), "
          f"val={len(val_pairs)} ảnh (NG={len(val_ng)}, OK={len(val_ok)})")

    out = Path(args.out)
    dirs = {}
    for split in ("train", "val"):
        dirs[(split, "img")] = out / "images" / split
        dirs[(split, "lbl")] = out / "labels" / split
        for d in (dirs[(split, "img")], dirs[(split, "lbl")]):
            d.mkdir(parents=True, exist_ok=True)

    totals = {"train": {"pos": 0, "neg": 0, "inst": 0},
              "val": {"pos": 0, "neg": 0, "inst": 0}}

    for split, plist in (("train", train_pairs), ("val", val_pairs)):
        for img_path, json_path in tqdm(plist, desc=f"Cắt tile [{split}]"):
            st = process_image(
                img_path, json_path, class_map,
                dirs[(split, "img")], dirs[(split, "lbl")],
                tile=args.tile, overlap=args.overlap,
                min_area_frac=args.min_area_frac, min_area_px=args.min_area_px,
                bg_ratio=args.bg_ratio, img_ext=args.img_ext, rng=rng,
                ok_tiles_per_img=args.ok_tiles_per_img,
            )
            for k in totals[split]:
                totals[split][k] += st[k]

    # Sinh data.yaml
    data_yaml = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for name, i in class_map.items()},
    }
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    # Báo cáo
    print("\n===== HOÀN TẤT =====")
    for split in ("train", "val"):
        t = totals[split]
        print(f"{split:5s}: {t['pos']} tile có lỗi, {t['neg']} tile nền, "
              f"{t['inst']} instance")
    print(f"\ndata.yaml -> {out / 'data.yaml'}")
    print("\nTrain thử (sau khi: pip install ultralytics):")
    print(f'   yolo segment train model=yolo11n-seg.pt data="{out / "data.yaml"}" '
          f'imgsz={args.tile} epochs=100 batch=16')


if __name__ == "__main__":
    main()
