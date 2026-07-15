# -*- coding: utf-8 -*-
"""
synth_defect.py
===============
Sinh MẪU LỖI GIẢ bằng kỹ thuật Copy-Paste (Cut-Paste augmentation) cho U-Net.

Ý tưởng: bóc "cục lỗi thật" theo polygon labelme -> dán lên ảnh OK (nền sạch)
ở vị trí ngẫu nhiên, kèm xoay / co giãn / lật / đổi sáng -> ra nhiều tile NG mới
CÓ MASK CHUẨN. Dùng để bơm thêm data TRAIN khi lỗi hiếm.

⚠️ CHỐNG LEAK (quan trọng): script TỰ ĐỘNG đọc thư mục val (images/val) để lấy
danh sách ảnh gốc thuộc val, rồi LOẠI TRỪ chúng khỏi kho bóc lỗi lẫn kho canvas.
=> Không bao giờ lấy lỗi/ảnh của tập test đem chế mẫu train (tự lừa mình).

⚠️ Chỉ dùng cho TRAIN. TUYỆT ĐỐI không trộn mẫu giả vào val/test.

Chạy (mặc định sinh vào thư mục synth riêng để soi trước, chưa đụng train):
    python synth_defect.py --n 300 --preview

Sau khi soi preview thấy ổn, gộp vào train:
    python synth_defect.py --n 300 --write-train
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

import sys as _sys, pathlib as _pl
_sys.path.insert(0, next(str(_p) for _p in _pl.Path(__file__).resolve().parents if (_p / "pathfix.py").exists()))
from pathfix import P

DEFAULT_NG = P("/mnt/d/Images_/SIBV/A27/260629_crop/img_train/ng")
DEFAULT_OK = P("/mnt/d/Images_/SIBV/A27/260629_crop/img_train/ok")
DEFAULT_DATA = P("/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/data_imgs_unet")

IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

PREVIEW_COLORS = [
    (0, 0, 0), (0, 0, 255), (0, 165, 255),
    (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255),
]


# --------------------------------------------------------------------------- #
# I/O Unicode-safe (đồng bộ với chia_data_unet.py)
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


# --------------------------------------------------------------------------- #
# Suy ra danh sách ảnh gốc thuộc VAL (để loại trừ chống leak)
# --------------------------------------------------------------------------- #
def val_source_stems(data_base):
    """Đọc images/val, cắt bỏ hậu tố __x{..}_y{..} -> tập tên ảnh gốc của val."""
    import re
    val_dir = Path(data_base) / "images" / "val"
    stems = set()
    if val_dir.exists():
        for p in val_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                stems.add(re.sub(r"__x\d+_y\d+$", "", p.stem))
    return stems


# --------------------------------------------------------------------------- #
# Kho lỗi: bóc từng cục lỗi thật (patch BGR + mask nhị phân theo polygon)
# --------------------------------------------------------------------------- #
def build_defect_bank(ng_dir, class_map, exclude_stems, min_area=4):
    bank = {cid: [] for cid in set(class_map.values())}
    ng = Path(ng_dir)
    for img_path in sorted(ng.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        if img_path.stem in exclude_stems:
            print(f"  [LEAK-GUARD] Bỏ ảnh val khỏi kho lỗi: {img_path.name}")
            continue
        jp = img_path.with_suffix(".json")
        if not jp.exists():
            continue
        img = imread_unicode(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        data = json.load(open(jp, encoding="utf-8"))
        for sh in data.get("shapes", []):
            if sh["label"] not in class_map:
                continue
            cid = class_map[sh["label"]]
            pts = shape_to_points(sh)
            if pts is None:
                continue
            x, y, w, h = cv2.boundingRect(pts)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            patch = img[y0:y1, x0:x1].copy()
            m = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.fillPoly(m, [pts - np.array([x0, y0])], 255)
            if int(np.count_nonzero(m)) < min_area:
                continue
            bank[cid].append((patch, m))
    return bank


# --------------------------------------------------------------------------- #
# Biến đổi cục lỗi: lật / xoay tự do / co giãn / đổi sáng
# --------------------------------------------------------------------------- #
def augment_patch(patch, mask, rng, scale_range=(0.7, 1.4)):
    if rng.random() < 0.5:
        patch, mask = patch[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        patch, mask = patch[::-1, :], mask[::-1, :]

    s = rng.uniform(*scale_range)
    if abs(s - 1.0) > 1e-3:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        patch = cv2.resize(patch, None, fx=s, fy=s, interpolation=interp)
        mask = cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)

    ang = rng.uniform(0, 360)
    h, w = mask.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    patch = cv2.warpAffine(patch, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)

    if rng.random() < 0.7:
        a = rng.uniform(0.85, 1.15)
        b = rng.uniform(-12, 12)
        patch = np.clip(patch.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(patch), np.ascontiguousarray(mask)


def random_ok_tile(ok_imgs, tile, rng, min_bright=35, tries=20):
    """Cắt 1 tile 512 ngẫu nhiên từ ảnh OK, ưu tiên vùng CÓ con hàng (không phải nền đen)."""
    for _ in range(tries):
        img = rng.choice(ok_imgs)
        H, W = img.shape[:2]
        if H < tile or W < tile:
            crop = cv2.resize(img, (tile, tile))
        else:
            x0 = rng.randint(0, W - tile)
            y0 = rng.randint(0, H - tile)
            crop = img[y0:y0 + tile, x0:x0 + tile].copy()
        if float(crop.mean()) >= min_bright:   # tránh tile toàn nền đen
            return crop
    return crop


def paste_defect(dst, dst_mask, patch, pmask, cid, rng, min_bright=35, tries=25):
    """Dán 1 cục lỗi lên tile, ưu tiên đặt trên vùng con hàng. Trả True nếu dán được."""
    th, tw = dst.shape[:2]
    ph, pw = pmask.shape[:2]
    if ph >= th or pw >= tw:
        return False
    for _ in range(tries):
        x0 = rng.randint(0, tw - pw)
        y0 = rng.randint(0, th - ph)
        roi = dst[y0:y0 + ph, x0:x0 + pw]
        # chỉ dán khi vùng đích nằm trên con hàng (đủ sáng), tránh dán lên nền đen
        if float(roi[pmask > 0].mean()) < min_bright:
            continue
        m3 = (pmask > 0)
        center = (x0 + pw // 2, y0 + ph // 2)
        try:
            blended = cv2.seamlessClone(patch, dst, pmask, center, cv2.NORMAL_CLONE)
            dst[:] = blended
        except cv2.error:
            roi[m3] = patch[m3]        # fallback: dán cứng
        dst_mask[y0:y0 + ph, x0:x0 + pw][m3] = cid
        return True
    return False


def colorize(img, mask):
    overlay = img.copy()
    for cid in np.unique(mask):
        if cid == 0:
            continue
        overlay[mask == cid] = PREVIEW_COLORS[cid % len(PREVIEW_COLORS)]
    return cv2.addWeighted(overlay, 0.45, img, 0.55, 0)


def main():
    ap = argparse.ArgumentParser(description="Copy-Paste synthetic defect generator (U-Net)")
    ap.add_argument("--ng-dir", default=DEFAULT_NG)
    ap.add_argument("--ok-dir", default=DEFAULT_OK)
    ap.add_argument("--data", default=DEFAULT_DATA, help="Gốc data_imgs_unet (để đọc dataset.yaml + val)")
    ap.add_argument("--n", type=int, default=300, help="Số tile giả cần sinh")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--max-per-tile", type=int, default=3, help="Số cục lỗi tối đa dán mỗi tile")
    ap.add_argument("--preview", action="store_true", help="Xuất thêm ảnh overlay để soi mắt")
    ap.add_argument("--write-train", action="store_true",
                    help="Ghi THẲNG vào images/train + masks/train (prefix 'synth_'). Mặc định ghi ra thư mục synth riêng để soi trước.")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data_base = Path(args.data)

    # class_map name->id lấy từ dataset.yaml (đồng bộ tuyệt đối với lúc train)
    ds = yaml.safe_load(open(data_base / "dataset.yaml", encoding="utf-8"))
    names = ds["names"]  # {0: background, 1: ban, ...}
    class_map = {n: int(i) for i, n in names.items() if int(i) != 0}
    print(f"[INFO] class_map = {class_map}")

    exclude = val_source_stems(data_base)
    print(f"[INFO] Ảnh gốc thuộc VAL sẽ bị loại (chống leak): {sorted(exclude)}")

    bank = build_defect_bank(args.ng_dir, class_map, exclude)
    id2name = {int(i): n for i, n in names.items()}
    print("[INFO] Kho lỗi bóc được (chỉ từ ảnh train):")
    usable = []
    for cid in sorted(bank):
        print(f"    lop {cid} {id2name[cid]:10s}: {len(bank[cid])} cục")
        if bank[cid]:
            usable.append(cid)
    if not usable:
        print("[LỖI] Kho lỗi rỗng sau khi loại val. Dừng.")
        return

    # canvas OK (loại ảnh OK thuộc val)
    ok_imgs = []
    for p in sorted(Path(args.ok_dir).iterdir()):
        if p.suffix.lower() not in IMG_EXTS or p.stem in exclude:
            if p.stem in exclude:
                print(f"  [LEAK-GUARD] Bỏ ảnh OK val khỏi canvas: {p.name}")
            continue
        im = imread_unicode(p)
        if im is not None:
            ok_imgs.append(im)
    if not ok_imgs:
        print("[LỖI] Không có ảnh OK làm canvas. Dừng.")
        return
    print(f"[INFO] Canvas OK: {len(ok_imgs)} ảnh")

    # Trọng số lấy mẫu: OVERSAMPLE lớp hiếm (ít cục -> hay được chọn hơn)
    inv = {cid: 1.0 / max(1, len(bank[cid])) for cid in usable}
    tot = sum(inv.values())
    weights = [inv[c] / tot for c in usable]

    if args.write_train:
        img_out = data_base / "images" / "train"
        mask_out = data_base / "masks" / "train"
        prev_out = data_base / "preview" / "synth"
        prefix = "synth_"
    else:
        img_out = data_base.parent / "data_imgs_unet_synth" / "images"
        mask_out = data_base.parent / "data_imgs_unet_synth" / "masks"
        prev_out = data_base.parent / "data_imgs_unet_synth" / "preview"
        prefix = ""
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    if args.preview:
        prev_out.mkdir(parents=True, exist_ok=True)

    made = Counter_zero = 0
    per_class = {cid: 0 for cid in usable}
    for i in tqdm(range(args.n), desc="Sinh mẫu giả"):
        tile = random_ok_tile(ok_imgs, args.tile, rng)
        mask = np.zeros((args.tile, args.tile), dtype=np.uint8)
        k = rng.randint(1, args.max_per_tile)
        placed = 0
        for _ in range(k):
            cid = rng.choices(usable, weights=weights, k=1)[0]
            patch, pmask = rng.choice(bank[cid])
            patch, pmask = augment_patch(patch, pmask, rng)
            if paste_defect(tile, mask, patch, pmask, cid, rng):
                placed += 1
                per_class[cid] += 1
        if placed == 0:
            Counter_zero += 1
            continue
        name = f"{prefix}synth_{i:05d}"
        imwrite_unicode(img_out / f"{name}.png", tile, ".png")
        imwrite_unicode(mask_out / f"{name}.png", mask, ".png")
        if args.preview:
            imwrite_unicode(prev_out / f"{name}.png", colorize(tile, mask), ".png")
        made += 1

    print(f"\n[HOÀN TẤT] Đã sinh {made} tile giả (bỏ {Counter_zero} tile không dán được cục nào).")
    print("  Số cục lỗi đã dán theo lớp:")
    for cid in usable:
        print(f"    lop {cid} {id2name[cid]:10s}: {per_class[cid]} cục")
    print(f"  Ảnh -> {img_out}")
    print(f"  Mask -> {mask_out}")
    if args.preview:
        print(f"  Preview -> {prev_out}")
    if not args.write_train:
        print("\n  (Đang ở chế độ SOI TRƯỚC. Xem preview ok thì chạy lại với --write-train để gộp vào train.)")


if __name__ == "__main__":
    main()
