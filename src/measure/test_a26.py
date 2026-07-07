"""
================================================================================
 ĐO KHOẢNG CÁCH 2 ĐƯỜNG THẲNG (caliper-pair) - CHẠY HÀNG LOẠT
--------------------------------------------------------------------------------
 Ý tưởng: mỗi vị trí cần đo = 1 ZONE. Trong ZONE có 2 "caliper" (cửa sổ tìm
 cạnh). Mỗi caliper quét gradient ngang -> tìm 1 cạnh dọc (sub-pixel).
 Khoảng cách = xB - xA. Tự miễn nhiễm xê dịch nhẹ nhờ ROI/caliper đủ rộng.

 - Ảnh production là ảnh XÁM sạch (không có vạch annotation).
 - Nếu ảnh còn vạch xanh/đỏ (ảnh mẫu) -> tự inpaint xoá đi trước khi đo.
================================================================================
"""
import cv2, numpy as np, csv, glob, os

# ============================== CẤU HÌNH =====================================
INPUT_DIR    = "/mnt/d/Images_/SIBV/A26/260615_0/tesst"
OUTPUT_DIR   = "/mnt/d/Images_/SIBV/A26/260615_0/tesst/ket_qua"
MM_PER_PIXEL = 1.0            # <-- THAY bằng hệ số hiệu chuẩn thật (mm/pixel)

# Mỗi zone:
#   roi      = (x, y, w, h) vùng chứa phép đo (toạ độ trên ảnh GỐC)
#   eA / eB  = (col_lo, col_hi, polarity) caliper tìm cạnh, cột TƯƠNG ĐỐI trong roi
#              polarity = -1 : cạnh sáng->tối | +1 : cạnh tối->sáng
#   ng_mm    = (min, max) ngưỡng đạt; ngoài khoảng -> NG
ZONES = [
    dict(name="tab_trai", roi=(85, 1289, 92, 391),
         eA=(22, 38, -1), eB=(39, 55, +1), ng_mm=(0, 9999)),
    dict(name="day_phai", roi=(1846, 1911, 82, 171),
         eA=(25, 40, +1), eB=(41, 58, +1), ng_mm=(0, 9999)),
    dict(name="day_trai", roi=(285, 1938, 69, 160),
         eA=(18, 33, -1), eB=(34, 50, -1), ng_mm=(0, 9999)),
]
# =============================================================================


def strip_annotation(img_bgr):
    """Nếu ảnh có vạch xanh/đỏ (ảnh mẫu) thì xoá đi, trả về ảnh xám sạch."""
    B, G, R = (img_bgr[:, :, 0].astype(int),
               img_bgr[:, :, 1].astype(int),
               img_bgr[:, :, 2].astype(int))
    green = (G > 140) & (R < 110) & (B < 110)
    red   = (R > 150) & (G < 90) & (B < 90)
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if green.sum() + red.sum() > 50:
        mask = cv2.dilate((green | red).astype(np.uint8), np.ones((5, 5), np.uint8))
        gray = cv2.inpaint(gray, mask, 5, cv2.INPAINT_TELEA)
    return gray


def find_edge_subpixel(sub, col_lo, col_hi, polarity):
    """Quét gradient ngang trong cửa sổ [col_lo..col_hi], trả về vị trí cạnh
    (cột, sub-pixel) theo phân cực yêu cầu. None nếu cạnh quá yếu."""
    gx = cv2.Sobel(cv2.GaussianBlur(sub, (3, 3), 0).astype(np.float32),
                   cv2.CV_32F, 1, 0, ksize=3)
    prof = gx.mean(axis=0) * polarity
    col_lo = max(1, col_lo)
    col_hi = min(len(prof) - 2, col_hi)
    if col_hi <= col_lo:
        return None, 0.0
    win = prof[col_lo:col_hi + 1]
    k = int(np.argmax(win)) + col_lo
    strength = float(prof[k])
    a, b, c = prof[k - 1], prof[k], prof[k + 1]
    den = (a - 2 * b + c)
    shift = 0.5 * (a - c) / den if den != 0 else 0.0
    return k + shift, strength


def measure_zone(gray, z):
    x, y, w, h = z["roi"]
    H, W = gray.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return None                      # ROI vuot khung anh -> bo qua
    sub = gray[y:y + h, x:x + w]
    if sub.size == 0:
        return None
    xa, sa = find_edge_subpixel(sub, *z["eA"])
    xb, sb = find_edge_subpixel(sub, *z["eB"])
    if xa is None or xb is None:
        return None
    dist_px = abs(xb - xa)
    return dict(dist_px=dist_px, dist_mm=dist_px * MM_PER_PIXEL,
                A=(int(x + xa), y, h), B=(int(x + xb), y, h),
                roi=(x, y, w, h), sA=sa, sB=sb)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Production = ảnh .bmp. (Đổi/THÊM đuôi nếu cần: "*.png", "*.jpg")
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.bmp")))

    rows = []
    for f in files:
        bgr = cv2.imread(f, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = strip_annotation(bgr)
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rec = {"file": os.path.basename(f)}
        verdict = "OK"

        for z in ZONES:
            r = measure_zone(gray, z)
            if r is None:
                rec[z["name"]] = "NO_EDGE"; verdict = "CHECK"; continue
            lo, hi = z["ng_mm"]
            ng = not (lo <= r["dist_mm"] <= hi)
            if ng:
                verdict = "NG"
            rec[z["name"]] = round(r["dist_mm"], 3)
            rec[z["name"] + "_kq"] = "NG" if ng else "OK"

            # ---- vẽ overlay ----
            col = (0, 0, 255) if ng else (0, 200, 0)
            x, yy, ww, hh = r["roi"]
            cv2.rectangle(vis, (x, yy), (x + ww, yy + hh), (255, 180, 0), 2)
            for (px, py, ph) in (r["A"], r["B"]):
                cv2.line(vis, (px, py), (px, py + ph), col, 1)
            ya = r["A"][1] + r["A"][2] // 2
            cv2.line(vis, (r["A"][0], ya), (r["B"][0], ya), col, 2)
            cv2.putText(vis, f'{z["name"]}:{r["dist_mm"]:.2f}',
                        (x, yy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        rec["KET_LUAN"] = verdict
        rows.append(rec)
        out = os.path.join(OUTPUT_DIR, "vis_" + os.path.basename(f) + ".png")
        cv2.imwrite(out, vis)
        print(rec)

    if rows:
        keys = ["file"] + [k for k in rows[0].keys() if k != "file"]
        # đảm bảo đủ cột cho mọi dòng
        allkeys = []
        for r in rows:
            for k in r:
                if k not in allkeys:
                    allkeys.append(k)
        with open(os.path.join(OUTPUT_DIR, "ket_qua.csv"), "w",
                  newline="", encoding="utf-8-sig") as fo:
            w = csv.DictWriter(fo, fieldnames=allkeys)
            w.writeheader(); w.writerows(rows)
        print("\n[XONG] CSV + anh overlay luu tai:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
