# -*- coding: utf-8 -*-
"""Đếm số pixel & số tile chứa mỗi lớp trong mask train/val. Soi mất cân bằng lớp.
Dùng: python count_classes.py <thu_muc_dataset>  (mặc định data_imgs_unet)."""
import sys, numpy as np, cv2, yaml
from pathlib import Path

sys.path.insert(0, next(str(_p) for _p in Path(__file__).resolve().parents if (_p / "pathfix.py").exists()))
from pathfix import P
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    P("/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/data_imgs_unet"))
with open(BASE / "dataset.yaml", "r", encoding="utf-8") as f:
    ds = yaml.safe_load(f)
NAMES = ds.get("names", {i: str(i) for i in range(int(ds["num_classes"]))})
NC = int(ds["num_classes"])

def imread_u(p):
    d = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(d, cv2.IMREAD_UNCHANGED)

print(f"[DATASET] {BASE}  (num_classes={NC}, names={NAMES})")
for split in ("train", "val"):
    mdir = BASE / "masks" / split
    files = sorted([p for p in mdir.iterdir() if p.suffix.lower() in (".png", ".bmp")])
    px = np.zeros(NC, dtype=np.int64)
    tiles = np.zeros(NC, dtype=np.int64)
    for f in files:
        m = imread_u(f)
        if m is None:
            continue
        if m.ndim == 3:
            m = m[..., 0]
        vals, cnts = np.unique(m, return_counts=True)
        for v, n in zip(vals, cnts):
            if 0 <= v < NC:
                px[v] += n
                tiles[v] += 1
    total = px.sum()
    print(f"\n===== {split.upper()} ({len(files)} tiles) =====")
    print(f"{'lop':<12}{'so_tile_chua':>14}{'tong_pixel':>16}{'ti_le_%':>12}")
    for c in range(NC):
        print(f"{NAMES[c]:<12}{tiles[c]:>14}{px[c]:>16}{100*px[c]/total:>11.4f}%")
