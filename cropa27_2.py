# -*- coding: utf-8 -*-
"""
cropa27_2.py
============
Cắt (crop) con hàng A27 đã được CHỤP GIỮA NỀN TRẮNG (đèn ring).

Bố cục mỗi ảnh (5068x5068):
  - GIỮA  : con hàng kim loại XÁM (có cửa sổ vuông SÁNG ở chính giữa).
  - VÒNG  : đĩa sáng TRẮNG (đèn) bao quanh con hàng.
  - 4 GÓC : tối ĐEN (vignette) -> luôn CHẠM biên ảnh.

Thuật toán (OpenCV thuần, không cần AI):
  Nền có CẢ trắng (đĩa) lẫn đen (góc) nên không tách bằng "tối/sáng" đơn thuần
  được. Mẹo: con hàng là vùng TỐI nằm LỌT GỌN bên trong đĩa sáng.
  1) Tách ĐĨA SÁNG = vùng sáng lớn nhất (gray > BRIGHT_THR).
  2) Lấp lỗ của đĩa -> phần lỗ bên trong đĩa chính là CON HÀNG
     (4 góc đen nằm NGOÀI đĩa nên tự bị loại).
  3) Lấy đốm lỗ lớn nhất = con hàng -> bounding-box.
  4) Nới thêm lề (MARGIN) rồi cắt và lưu.

Cách dùng:
  - Sửa INPUT_DIR / OUTPUT_DIR bên dưới.
  - Chạy:  python cropa27_2.py
  - Quét đệ quy mọi *.bmp trong INPUT_DIR (giữ nguyên cấu trúc thư mục con
    ban/ bavia/ nut/ ... ở OUTPUT_DIR).
  - Ảnh cắt: <tên ảnh>.png ; ảnh kiểm tra (khung đỏ) ở "_debug".
"""

import os
import glob
import cv2
import numpy as np

# ============================================================
# 1) CẤU HÌNH  -  chỉ cần chỉnh ở đây
# ============================================================
INPUT_DIR  = "/mnt/d/Images_/SIBV/A27/260629"
OUTPUT_DIR = "/mnt/d/Images_/SIBV/A27/260629_crop"

# Ảnh gốc rất to -> thu nhỏ về DET_DS để dò cho nhanh rồi quy ngược toạ độ.
DET_DS     = 1000      # bề rộng ảnh dò
BRIGHT_THR = 205       # gray > ngưỡng này -> coi là 'đĩa sáng' (nền đèn ring).
                       # HẠ (vd 195) nếu đĩa sáng bị thủng nhiều; TĂNG nếu con
                       # hàng có chỗ sáng bị tính nhầm vào nền.
MIN_AREA_F = 0.010     # diện tích đốm tối thiểu (theo DET_DS^2) -> loại đốm vụn
MARGIN_F   = 0.06      # nới thêm lề quanh con hàng = 6% cạnh khung (mỗi phía)

INPUT_EXT = "*.bmp"
SAVE_EXT  = ".png"
SHOW_DEBUG = True       # lưu ảnh kiểm tra (khung đỏ) ở thư mục _debug


# ============================================================
# 2) DÒ KHUNG CON HÀNG
# ============================================================
def _do_lon_nhat(mask):
    """Trả về (area, x, y, w, h) của đốm trắng LỚN NHẤT trong mask, hoặc None."""
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax([stats[k, cv2.CC_STAT_AREA] for k in range(1, n)]))
    return (stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT])


def tim_khung_con_hang(img):
    """Trả về (x1, y1, x2, y2) theo toạ độ ảnh GỐC, hoặc None nếu không thấy."""
    h0, w0 = img.shape[:2]
    s = DET_DS / float(w0)
    det_h = int(round(h0 * s))
    small = cv2.resize(img, (DET_DS, det_h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 1) Đĩa sáng = vùng sáng LỚN NHẤT (nền đèn ring). Góc đen bị loại ở đây.
    bright = (gray > BRIGHT_THR).astype(np.uint8) * 255
    disc = _do_lon_nhat(bright)
    if disc is None:
        return None
    dn, dx, dy, dw, dh = disc
    disc_mask = np.zeros_like(bright)
    disc_mask[dy:dy + dh, dx:dx + dw] = bright[dy:dy + dh, dx:dx + dw]

    # 2) Lấp lỗ của đĩa: floodFill từ (0,0) (luôn nằm NGOÀI đĩa) tô vùng ngoài;
    #    phần đĩa chưa tô + lỗ bên trong = đĩa đặc. Lỗ = con hàng.
    ff = disc_mask.copy()
    m = np.zeros((det_h + 2, DET_DS + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 255)
    filled = disc_mask | cv2.bitwise_not(ff)
    obj = cv2.bitwise_and(filled, cv2.bitwise_not(bright))

    # 3) Khử rìa lẻ tẻ rồi lấy đốm LỖ lớn nhất = con hàng.
    kc = max(5, DET_DS // 140)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, ker)
    best = _do_lon_nhat(obj)
    if best is None or best[0] < MIN_AREA_F * DET_DS * DET_DS:
        return None

    _, x, y, ww, hh = best

    # 4) Nới lề rồi quy ngược về toạ độ ảnh gốc.
    mx = int(round(ww * MARGIN_F))
    my = int(round(hh * MARGIN_F))
    x1 = int(round((x - mx) / s)); y1 = int(round((y - my) / s))
    x2 = int(round((x + ww + mx) / s)); y2 = int(round((y + hh + my) / s))
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w0, x2); y2 = min(h0, y2)
    return (x1, y1, x2, y2)


# ============================================================
# 3) CẮT VÀ LƯU CHO 1 ẢNH
# ============================================================
def cat_mot_anh(duong_dan_anh, out_path, debug_path):
    img = cv2.imread(duong_dan_anh)
    if img is None:
        print(f"  [LOI] Khong doc duoc anh: {duong_dan_anh}")
        return

    box = tim_khung_con_hang(img)
    ten = os.path.basename(duong_dan_anh)
    if box is None:
        print(f"  [BO QUA] Khong tim thay con hang: {ten}")
        return

    x1, y1, x2, y2 = box
    cv2.imwrite(out_path, img[y1:y2, x1:x2])
    print(f"  {ten}: cat ({x2 - x1}x{y2 - y1}) -> {os.path.basename(out_path)}")

    if SHOW_DEBUG and debug_path:
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 12)
        cv2.imwrite(debug_path, cv2.resize(vis, (1000, 1000)))


# ============================================================
# 4) CHẠY CHÍNH  (quét đệ quy, giữ nguyên cấu trúc thư mục con)
# ============================================================
def main():
    danh_sach = sorted(glob.glob(os.path.join(INPUT_DIR, "**", INPUT_EXT),
                                 recursive=True))
    print(f"Tim thay {len(danh_sach)} anh trong: {INPUT_DIR}\n")

    for duong_dan in danh_sach:
        rel = os.path.relpath(duong_dan, INPUT_DIR)
        rel_no_ext = os.path.splitext(rel)[0]
        out_path = os.path.join(OUTPUT_DIR, rel_no_ext + SAVE_EXT)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        debug_path = None
        if SHOW_DEBUG:
            debug_path = os.path.join(OUTPUT_DIR, "_debug", rel_no_ext + ".png")
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)

        cat_mot_anh(duong_dan, out_path, debug_path)

    print(f"\nXONG. Anh cat luu tai: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
