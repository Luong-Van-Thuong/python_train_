# -*- coding: utf-8 -*-
"""
probe_vungtu_cv.py — Thu tim "vung tu" MLCC bang OpenCV THUAN (KHONG train),
doi chieu box ground-truth (labelme) de do IoU / ty le bat dung.

Y tuong khai thac chinh dac diem quang hoc cua bo anh:
  * DOF mong  => vung tu la CHU THE NET nhat, vien tu CRISP; nen bi mo bokeh.
  * Than tu   => vung TOI (khe dien cuc).
  * Ben trong => thuong co dom SPECULAR sang.
Ket hop: (vung NET) ∩ (vung TOI) ~ than tu. Sinh ung vien tu mask do,
cham diem theo shape-prior + nang luong vien + specular + do toi, chon 1 box
tot nhat (moi anh chi 1 tu) roi so IoU voi GT.

Muc dich: RA CON SO de quyet dinh classical du hay bat buoc YOLO.
KHONG phai giai phap cuoi — chi la baseline do luong.

Xuat DEBUG: moi anh 1 panel 9 o (goc -> gray -> focus E -> focus_mask ->
dark_mask -> specular -> cand -> ung vien -> GT vs pred) vao folder --debug-dir
de nhin tung buoc bien doi.

Chay:
  WSL:  python src/measure/probe_vungtu_cv.py --n 30
  Win:  python src\\measure\\probe_vungtu_cv.py --n 30
  (tat debug:  --no-debug)
"""
import cv2, numpy as np, json, glob, os, argparse

import sys as _sys, pathlib as _pl
_sys.path.insert(0, next(str(_p) for _p in _pl.Path(__file__).resolve().parents
                         if (_p / "pathfix.py").exists()))
from pathfix import P

# ---------------------------------------------------------------- cau hinh
DATA   = P("/mnt/d/Images_/JeaYoung/MLCC/Data/vung_tu")
DBG    = P("/mnt/d/Images_/JeaYoung/MLCC/Data/vung_tu_debug")   # folder debug
OUT    = P("/mnt/c/Users/ThuonngLV/AppData/Local/Temp/claude/"
           "D--Projects--Cong-Ty-Python--train/"
           "ae248baa-123a-4256-84f3-438a3c052f30/scratchpad")
WORK_H = 800          # chuan hoa chieu cao xu ly (px) de tham so on dinh moi resolution

# prior hinh dang cua tu (do tu 567 nhan that, dung dang FRACTION theo anh)
AR_LO, AR_HI      = 0.28, 1.00     # w/h : tu thon DUNG (mean ~0.53)
AREAF_LO, AREAF_HI= 0.008, 0.16    # dien tich box / dien tich anh
# trong so cham diem (evidence)
W_EDGE, W_DARK, W_SPEC = 0.45, 0.35, 0.20


# ---------------------------------------------------------------- tien ich
def load_gt(jp):
    """Doc box rectangle nhan 'd' tu file labelme -> (x0,y0,x1,y1) hoac None."""
    j = json.load(open(jp, encoding="utf-8"))
    for s in j.get("shapes", []):
        if s.get("shape_type") == "rectangle":
            p = np.array(s["points"], float)
            x0, y0 = p.min(0); x1, y1 = p.max(0)
            return (float(x0), float(y0), float(x1), float(y1))
    return None


def find_image(base):
    for ext in (".bmp", ".jpg", ".png"):
        if os.path.exists(base + ext):
            return base + ext
    return None


def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def norm01(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return v
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


# ---------------------------------------------------------------- loi giai
def detect(img):
    """Tra ve (box_native, dbg). box_native trong toa do ANH GOC.
    dbg chua toan bo anh trung gian (work-space) de xuat debug."""
    H0, W0 = img.shape[:2]
    s = WORK_H / H0
    im = cv2.resize(img, (int(W0 * s), WORK_H))
    H, W = im.shape[:2]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
    grayf = gray.astype(np.float32)

    # (1) nang luong VIEN NET: |Sobel| gom cuc bo bang box-filter
    gx = cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    E = cv2.boxFilter(mag, -1, (25, 25))              # nang luong net cuc bo
    focus_mask = (E > np.percentile(E, 82)).astype(np.uint8)

    # (2) vung TOI (than tu) — nguong theo percentile de tu thich nghi anh sang
    dark_mask = (gray < np.percentile(gray, 22)).astype(np.uint8)

    # (3) specular sang (dom loi trong tu)
    spec = (gray > np.percentile(gray, 99)).astype(np.uint8)

    # (than tu) = TOI  ∩  gan-NET  -> giao voi focus da gian no
    focus_d = cv2.dilate(focus_mask, np.ones((15, 15), np.uint8))
    cand = cv2.bitwise_and(dark_mask, focus_d)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

    dbg = dict(im=im, gray=gray, E=E, mag=mag, focus_mask=focus_mask,
               dark_mask=dark_mask, spec=spec, cand=cand, s=s, raw=[], best=None)

    cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgA = float(W * H)
    raw = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 6 or h < 6:
            continue
        ar = w / h
        af = (w * h) / imgA
        vertical = h >= w * 0.9
        # bang chung: nang luong vien quanh box + do toi trong box + co specular
        ring = np.zeros((H, W), np.uint8)
        cv2.rectangle(ring, (x, y), (x + w, y + h), 255, thickness=9)
        edge_v = float(E[ring > 0].mean()) if (ring > 0).any() else 0.0
        roi = gray[y:y + h, x:x + w]
        dark_v = float(255 - roi.mean())
        spec_v = float(spec[y:y + h, x:x + w].mean() > 0)
        gate = (AR_LO <= ar <= AR_HI) and (AREAF_LO <= af <= AREAF_HI) and vertical
        raw.append(dict(box=(x, y, w, h), edge=edge_v, dark=dark_v,
                        spec=spec_v, gate=gate, af=af, ar=ar))

    dbg["raw"] = raw
    if not raw:
        return None, dbg

    ne = norm01([r["edge"] for r in raw])
    nd = norm01([r["dark"] for r in raw])
    for i, r in enumerate(raw):
        r["score"] = (W_EDGE * ne[i] + W_DARK * nd[i] + W_SPEC * r["spec"])

    passed = [r for r in raw if r["gate"]]
    pool = passed if passed else raw          # neu khong ai qua gate -> van doan
    best = max(pool, key=lambda r: r["score"])
    dbg["best"] = best

    x, y, w, h = best["box"]
    box_native = (x / s, y / s, (x + w) / s, (y + h) / s)
    return box_native, dbg


# ---------------------------------------------------------------- ve debug
def _bgr(mask_or_gray, colormap=None):
    """gray/mask -> BGR de ghep panel. mask (0/1) -> 0/255."""
    m = mask_or_gray
    if m.dtype != np.uint8:
        m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif m.max() <= 1:
        m = m * 255
    if colormap is not None:
        return cv2.applyColorMap(m, colormap)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def _title(tile, text):
    """Dan thanh tieu de den phia tren tile."""
    h, w = tile.shape[:2]
    bar = np.zeros((26, w, 3), np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def render_debug(dbg, gt, pred, iou_val, save_path):
    """Xuat panel 9 o cho 1 anh (work-space)."""
    im = dbg["im"]; H, W = im.shape[:2]; s = dbg["s"]

    # o 8: tat ca ung vien (vang) / qua gate (cyan) / chon (do)
    cand_vis = im.copy()
    for r in dbg["raw"]:
        x, y, w, h = r["box"]
        col = (255, 255, 0) if r["gate"] else (0, 200, 255)   # cyan neu qua gate
        cv2.rectangle(cand_vis, (x, y), (x + w, y + h), col, 2)
    if dbg["best"]:
        x, y, w, h = dbg["best"]["box"]
        cv2.rectangle(cand_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # o 9: GT (xanh la) vs pred (do) trong work-space
    final = im.copy()
    gx0, gy0, gx1, gy1 = [int(v * s) for v in gt]
    cv2.rectangle(final, (gx0, gy0), (gx1, gy1), (0, 255, 0), 2)
    if pred:
        px0, py0, px1, py1 = [int(v * s) for v in pred]
        cv2.rectangle(final, (px0, py0), (px1, py1), (0, 0, 255), 2)

    tiles = [
        _title(im,                                    "1.goc"),
        _title(_bgr(dbg["gray"]),                     "2.gray"),
        _title(_bgr(dbg["E"], cv2.COLORMAP_JET),      "3.focus E (nang luong net)"),
        _title(_bgr(dbg["focus_mask"]),               "4.focus_mask (>p82)"),
        _title(_bgr(dbg["dark_mask"]),                "5.dark_mask (<p22)"),
        _title(_bgr(dbg["spec"]),                     "6.specular (>p99)"),
        _title(_bgr(dbg["cand"]),                     "7.cand = dark ∩ focus"),
        _title(cand_vis,                              "8.ung vien (do=chon)"),
        _title(final,                                 f"9.GT xanh / pred do  IoU={iou_val:.2f}"),
    ]
    # dong nhat kich thuoc o
    cw = 300
    tiles = [cv2.resize(t, (cw, int(t.shape[0] * cw / t.shape[1]))) for t in tiles]
    hh = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=0) for t in tiles]
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)]
    panel = np.vstack(rows)
    cv2.imwrite(save_path, panel)


# ---------------------------------------------------------------- chay probe
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="so anh de thu")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--debug-dir", default=DBG)
    ap.add_argument("--no-debug", action="store_true", help="tat xuat panel debug")
    args = ap.parse_args()

    jsons = sorted(glob.glob(os.path.join(args.data, "*.json")))
    items = []
    for jp in jsons:
        gt = load_gt(jp)
        ip = find_image(os.path.splitext(jp)[0])
        if gt and ip:
            items.append((ip, gt))
    # lay mau trai deu tren toan bo (tron lan cac resolution)
    idx = np.linspace(0, len(items) - 1, min(args.n, len(items))).astype(int)
    sample = [items[i] for i in sorted(set(idx))]
    print(f"[probe] tong cap hop le={len(items)} | thu {len(sample)} anh")

    if not args.no_debug:
        os.makedirs(args.debug_dir, exist_ok=True)
        print(f"[debug] panel tung buoc -> {args.debug_dir}")
    print()

    tiles, ious = [], []
    for ip, gt in sample:
        img = cv2.imread(ip)
        if img is None:
            continue
        pred, dbg = detect(img)
        j = iou(pred, gt) if pred else 0.0
        ious.append(j)
        name = os.path.basename(ip)
        print(f"  IoU={j:5.3f}  {name}")

        if not args.no_debug:
            stem = os.path.splitext(name)[0]
            render_debug(dbg, gt, pred, j,
                         os.path.join(args.debug_dir, f"{stem}_steps.png"))

        vis = img.copy()
        gx0, gy0, gx1, gy1 = map(int, gt)
        cv2.rectangle(vis, (gx0, gy0), (gx1, gy1), (0, 255, 0), 3)   # GT xanh la
        if pred:
            px0, py0, px1, py1 = map(int, pred)
            cv2.rectangle(vis, (px0, py0), (px1, py1), (0, 0, 255), 3)  # pred do
        cv2.putText(vis, f"IoU={j:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255) if j >= 0.5 else (0, 0, 255), 3)
        tiles.append(cv2.resize(vis, (300, 260)))

    ious = np.array(ious)
    print("\n================= KET QUA =================")
    print(f"  so anh          : {len(ious)}")
    print(f"  IoU trung binh  : {ious.mean():.3f}")
    print(f"  IoU trung vi    : {np.median(ious):.3f}")
    print(f"  hit @IoU>=0.5   : {(ious >= 0.5).mean()*100:5.1f}%  ({int((ious>=0.5).sum())}/{len(ious)})")
    print(f"  hit @IoU>=0.7   : {(ious >= 0.7).mean()*100:5.1f}%  ({int((ious>=0.7).sum())}/{len(ious)})")
    print(f"  truot @IoU<0.3  : {(ious <  0.3).mean()*100:5.1f}%  ({int((ious<0.3).sum())}/{len(ious)})")
    print("===========================================")

    # montage tong hop GT(xanh) vs Pred(do)
    os.makedirs(OUT, exist_ok=True)
    cols = 6
    while len(tiles) % cols:
        tiles.append(np.zeros((260, 300, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    mont = np.vstack(rows)
    outp = os.path.join(OUT, "probe_vungtu_result.png")
    cv2.imwrite(outp, mont)
    print(f"\n[montage] {outp}  {mont.shape}")


if __name__ == "__main__":
    main()
