# -*- coding: utf-8 -*-
"""
chia_data_obd_1folder.py — labelme (1 FOLDER CHUNG) -> dataset YOLO detect.
================================================================================
Danh cho DETECT THO nguyen anh (KHONG cat tile). Ban chi can DO TAT CA anh da
label + file .json vao CHUNG 1 FOLDER, script tu:
    1. Quet folder: anh CO .json (rectangle/polygon) -> sinh nhan;
                    anh KHONG .json / shapes rong     -> MAU AM (nhan rong).
       (Khong can tach OK/NG — object detection khong phan biet theo folder.)
    2. Tu gan class id on dinh theo alphabet tu cac label tim thay.
    3. Chia train/val, GOM frame gan-trung 1 con hang de tranh ro ri
       ('Cam0_0001_...a' & 'Cam0_0001_...b' -> cung 1 phia).
    4. Dua anh vao dataset bang HARDLINK (cung o -> tuc thi, khong ton dung luong;
       khac o thi tu dong copy). Sinh data.yaml.

Cau truc xuat ra (train thang bang YOLO / train_obd.py):
    <out>/images/{train,val}
    <out>/labels/{train,val}
    <out>/data.yaml

Chay (WSL hoac Windows Anaconda deu duoc nho pathfix):
    python src/data/chia_data_obd_1folder.py
    python src/data/chia_data_obd_1folder.py --src <folder> --out <dataset> --val-ratio 0.2
"""
import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
import yaml

import sys as _sys, pathlib as _pl
_sys.path.insert(0, next(str(_p) for _p in _pl.Path(__file__).resolve().parents
                         if (_p / "pathfix.py").exists()))
from pathfix import P

# --------------------------------------------------------------- cau hinh
DEFAULT_SRC = P("/mnt/d/Images_/JeaYoung/MLCC/Data/tu")
DEFAULT_OUT = P("/mnt/d/Projects_/Cong_Ty/Python_/train/JeaYoung/MLCC/data_tu")
IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


# --------------------------------------------------------------- tien ich
def group_key(stem):
    """Gom cac frame gan-trung 1 con hang de tranh RO RI train<->val.
    'Cam0_0001_...a' & 'Cam0_0001_...b' -> cung khoa 'Cam0_0001'.
    Khong khop mau nay -> lay nguyen ten (moi anh 1 nhom)."""
    m = re.match(r"(Cam\d+_\d+)", stem)
    return m.group(1) if m else stem


def linkorcopy(src, dst):
    """Hardlink (cung o -> tuc thi, 0 byte them); that bai -> copy."""
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)          # cung volume: khong ton dung luong
    except OSError:
        shutil.copy2(src, dst)     # khac volume / khong ho tro link


def shape_bbox(sh):
    """1 shape labelme -> (x0,y0,x1,y1) px. Ho tro rectangle & polygon."""
    st = sh.get("shape_type", "polygon")
    pts = np.array(sh.get("points", []), float)
    if pts.size == 0:
        return None
    if st == "rectangle" and len(pts) == 2:
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        return float(x0), float(y0), float(x1), float(y1)
    if st == "polygon" and len(pts) >= 3:
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        return float(x0), float(y0), float(x1), float(y1)
    return None   # circle/line/point -> bo qua


# --------------------------------------------------------------- quet folder
def scan(src):
    """Quet 1 folder -> list (img_path, json_path|None). Phan loai theo .json."""
    items = []
    for p in sorted(Path(src).iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        jp = p.with_suffix(".json")
        items.append((p, jp if jp.exists() else None))
    return items


def build_class_map(items):
    """Quet json -> {label: id} on dinh theo alphabet."""
    labels = set()
    for _, jp in items:
        if jp is None:
            continue
        for sh in json.load(open(jp, encoding="utf-8")).get("shapes", []):
            if shape_bbox(sh) is not None:
                labels.add(sh["label"])
    classes = sorted(labels)
    return {name: i for i, name in enumerate(classes)}, classes


def yolo_lines(jp, class_map):
    """json -> list dong YOLO 'cid cx cy w h' (chuan hoa). None json -> []."""
    if jp is None:
        return []
    j = json.load(open(jp, encoding="utf-8"))
    W = float(j["imageWidth"]); H = float(j["imageHeight"])
    out = []
    for sh in j.get("shapes", []):
        bb = shape_bbox(sh)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        cx = (x0 + x1) / 2 / W; cy = (y0 + y1) / 2 / H
        w = (x1 - x0) / W;       h = (y1 - y0) / H
        if w <= 0 or h <= 0:
            continue
        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
        w, h = min(w, 1), min(h, 1)
        out.append(f"{class_map[sh['label']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="labelme 1 folder chung -> dataset YOLO detect (nguyen anh)")
    ap.add_argument("--src", default=DEFAULT_SRC, help="folder chung anh + json")
    ap.add_argument("--out", default=DEFAULT_OUT, help="folder dataset xuat ra")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-group", action="store_true",
                    help="chia theo tung anh, KHONG gom con hang")
    args = ap.parse_args()

    items = scan(args.src)
    if not items:
        print(f"[LOI] Khong thay anh nao trong {args.src}")
        return
    n_pos = sum(1 for _, j in items if j is not None)
    print(f"[quet] {len(items)} anh: {n_pos} co nhan, {len(items) - n_pos} mau am")

    class_map, classes = build_class_map(items)
    if not classes:
        print("[LOI] Khong tim thay label nao trong json.")
        return
    print("[class] " + ", ".join(f"{i}:{c}" for c, i in class_map.items()))

    # chia train/val theo NHOM con hang
    groups = {}
    for ip, jp in items:
        key = ip.stem if args.no_group else group_key(ip.stem)
        groups.setdefault(key, []).append((ip, jp))
    keys = list(groups); random.Random(args.seed).shuffle(keys)
    n_val = max(1, round(len(keys) * args.val_ratio))
    val_keys = set(keys[:n_val])
    print(f"[chia] {len(keys)} nhom -> val {len(val_keys)} nhom (ti le {args.val_ratio})")

    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cnt = {"train": [0, 0], "val": [0, 0]}   # [so anh, so instance]
    for key, group in groups.items():
        split = "val" if key in val_keys else "train"
        for ip, jp in group:
            lines = yolo_lines(jp, class_map)
            linkorcopy(str(ip), str(out / "images" / split / ip.name))
            with open(out / "labels" / split / f"{ip.stem}.txt", "w",
                      encoding="utf-8") as f:
                f.write("\n".join(lines))     # rong = mau am
            cnt[split][0] += 1
            cnt[split][1] += len(lines)

    data_yaml = out / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "path": str(out),
            "train": "images/train",
            "val": "images/val",
            "names": {i: c for c, i in class_map.items()},
        }, f, allow_unicode=True, sort_keys=False)

    print("\n===== HOAN TAT =====")
    print(f"train: {cnt['train'][0]} anh, {cnt['train'][1]} box")
    print(f"val  : {cnt['val'][0]} anh, {cnt['val'][1]} box")
    print(f"data.yaml -> {data_yaml}")
    print("\nTrain thu (nhe/nhanh):")
    print(f'  yolo detect train model=yolo11n.pt data="{data_yaml}" '
          f"imgsz=512 batch=16 epochs=120 cache=False")


if __name__ == "__main__":
    main()
