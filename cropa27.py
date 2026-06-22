# -*- coding: utf-8 -*-
"""
cropa27.py
==========
Cắt (crop) từng con hàng ra khỏi ảnh khay A27 bằng OpenCV thuần (không cần AI).

Khay A27 là lưới 2x2 (tối đa 4 con hàng), có thể thiếu con (ô rỗng -> bỏ qua).
Hỗ trợ 2 loại con hàng (chọn bằng PART_TYPE):

  - "round"  : con hàng TRÒN (vòng cao su tròn quanh chip vuông giữa).
               -> DÒ TÂM bằng HoughCircles trên TOÀN ảnh: rất ổn định, KHÔNG
                  phụ thuộc con hàng nằm chính giữa ô hay bị lệch vị trí.
  - "square" : connector VUÔNG (giữa có cửa sổ tối).
               -> chia lưới 2x2, mỗi ô dò khối "nhiều cạnh" (kim loại) làm tâm;
                  ô mật độ cạnh thấp = hốc rỗng -> bỏ qua.

Con hàng KHÔNG đổi kích thước -> có TÂM rồi cắt 1 khung CỐ ĐỊNH (rộng x cao)
quanh tâm: tròn dùng PART_ROUND, vuông dùng PART_SQUARE.

Cách dùng:
  - Sửa INPUT_DIR / OUTPUT_DIR và PART_TYPE bên dưới cho đúng.
  - Chạy:  python cropa27.py
  - Ảnh cắt ra: <tên ảnh>_1.png, _2.png, ... trong OUTPUT_DIR.
  - Ảnh kiểm tra (khung đỏ) ở "_debug"; ảnh các bước ở "_steps".
"""

import os
import glob
import cv2
import numpy as np

# ============================================================
# 1) CẤU HÌNH  -  chỉ cần chỉnh ở đây
# ============================================================

# Đường dẫn kiểu WSL2: ổ Windows D:\ -> /mnt/d, C:\ -> /mnt/c ...
INPUT_DIR  = "/mnt/d/Images_/SIBV/A27/img_train/burr"
OUTPUT_DIR = "/mnt/d/Images_/SIBV/A27/img_train/burr_crop"

# Loại con hàng: "round" (tròn - dùng Hough) hoặc "square" (vuông - dùng lưới)
PART_TYPE = "round"

# Lưới con hàng: 2 hàng x 2 cột (tối đa 4 con/khay)
GRID_ROWS = 2
GRID_COLS = 2
MAX_PARTS = GRID_ROWS * GRID_COLS

# ---- KÍCH THƯỚC KHUNG CẮT (CỐ ĐỊNH; con hàng không đổi kích thước) ----
# (rộng, cao) px. >>> Đổi khi kích thước thực của con hàng khác đi. <<<
PART_ROUND  = (2000, 2000)   # con hàng TRÒN
PART_SQUARE = (2000, 2000)   # connector VUÔNG

# ---- THAM SỐ DÒ TRÒN (HoughCircles) ----
# Ảnh gốc 5064px rất to -> thu nhỏ về HOUGH_DS cho nhanh rồi quy ngược toạ độ.
# Các tỉ lệ tính theo HOUGH_DS nên KHÔNG phụ thuộc kích thước ảnh gốc.
HOUGH_DS      = 1000
HOUGH_DP      = 1.2
HOUGH_MINDIST = 0.12     # khoảng cách tối thiểu giữa 2 tâm (theo HOUGH_DS)
HOUGH_PARAM1  = 120
HOUGH_PARAM2  = 45
HOUGH_RMIN    = 0.05     # bán kính nhỏ nhất (theo HOUGH_DS)
HOUGH_RMAX    = 0.13     # bán kính lớn nhất (theo HOUGH_DS)
RADIUS_TOL    = (0.6, 1.6)   # giữ vòng có bán kính trong [lo, hi]*trung vị

# ---- NGƯỠNG Ô RỖNG cho chế độ "square" (theo mật độ cạnh trung bình) ----
EMPTY_GRAD_ABS = 15.0
EMPTY_GRAD_REL = 0.5

# Đuôi file ảnh đầu vào và định dạng lưu ra
INPUT_EXT = "*.bmp"
SAVE_EXT  = ".png"

# Có lưu ảnh minh hoạ (debug + steps) không?
SHOW_STEPS = True


# ============================================================
# 2A) DÒ CON HÀNG TRÒN BẰNG HoughCircles (toàn ảnh)
# ============================================================
def _do_tron(img):
    """Trả về danh sách TÂM (cx, cy) theo toạ độ ảnh GỐC của các con hàng tròn."""
    h, w = img.shape[:2]
    s = HOUGH_DS / float(w)
    small = cv2.resize(img, (HOUGH_DS, int(round(h * s))))
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 7)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=HOUGH_DP,
        minDist=int(HOUGH_DS * HOUGH_MINDIST),
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=int(HOUGH_DS * HOUGH_RMIN),
        maxRadius=int(HOUGH_DS * HOUGH_RMAX))
    if circles is None:
        return []

    circ = list(circles[0])                 # đã sắp theo độ "chắc" (accumulator)
    rmed = float(np.median([c[2] for c in circ]))
    circ = [c for c in circ
            if RADIUS_TOL[0] * rmed <= c[2] <= RADIUS_TOL[1] * rmed]
    circ = circ[:MAX_PARTS]                  # khay tối đa MAX_PARTS con

    pts = [(int(cx / s), int(cy / s)) for cx, cy, _ in circ]
    pts.sort(key=lambda p: (round(p[1] / (h / 3.0)), p[0]))  # trên->dưới, trái->phải
    return pts


# ============================================================
# 2B) DÒ CONNECTOR VUÔNG THEO LƯỚI 2x2 (mật độ cạnh)
# ============================================================
def _grad_o(cell):
    """Mật độ cạnh của 1 ô: kim loại nhiều chi tiết -> cao; nền/nhựa -> thấp."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    k = max(5, (cell.shape[1] // 200) | 1)
    return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _tam_vuong_trong_o(cell, grad):
    """Tâm con hàng vuông trong 1 ô = tâm bao của khối 'nhiều cạnh' lớn nhất."""
    cw = cell.shape[1]
    dens = cv2.GaussianBlur(grad, (0, 0), cw / 40.0)
    _, m = cv2.threshold(dens, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kk = max(7, cw // 40)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return (x + w // 2, y + h // 2)


def _do_vuong(img):
    """Trả về danh sách TÂM (cx, cy) theo toạ độ ảnh GỐC, bỏ ô rỗng."""
    h_img, w_img = img.shape[:2]
    ch, cw = h_img // GRID_ROWS, w_img // GRID_COLS
    cells = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y0, x0 = r * ch, c * cw
            cell = img[y0:y0 + ch, x0:x0 + cw]
            g = _grad_o(cell)
            cells.append((x0, y0, cell, g, float(g.mean())))
    gmax = max(c[4] for c in cells) if cells else 0.0
    pts = []
    for x0, y0, cell, g, gm in cells:
        if gm < EMPTY_GRAD_ABS or gm < EMPTY_GRAD_REL * gmax:
            continue                       # hốc rỗng -> bỏ qua
        res = _tam_vuong_trong_o(cell, g)
        if res is not None:
            pts.append((x0 + res[0], y0 + res[1]))
    return pts


# ============================================================
# 2C) GỘP CHUNG: TÂM -> KHUNG CẮT CỐ ĐỊNH
# ============================================================
def _khung_co_dinh(tam_x, tam_y, W, H, w_img, h_img):
    """Khung W x H đặt giữa tại (tam_x, tam_y); dời vào trong nếu tràn biên."""
    x1 = min(max(0, tam_x - W // 2), max(0, w_img - W))
    y1 = min(max(0, tam_y - H // 2), max(0, h_img - H))
    return (x1, y1, x1 + W, y1 + H)


def tim_cac_linh_kien(img):
    """Dò tâm con hàng (theo PART_TYPE) rồi cắt khung cố định quanh mỗi tâm.
    Trả về danh sách khung (x1, y1, x2, y2) theo toạ độ ảnh GỐC."""
    h_img, w_img = img.shape[:2]
    if PART_TYPE == "round":
        centers, (W, H) = _do_tron(img), PART_ROUND
    else:
        centers, (W, H) = _do_vuong(img), PART_SQUARE
    return [_khung_co_dinh(cx, cy, W, H, w_img, h_img) for cx, cy in centers]


# ============================================================
# 3) ẢNH KIỂM TRA CÁC BƯỚC  (chỉ để xem - tắt bằng SHOW_STEPS)
# ============================================================
def luu_cac_buoc(img, boxes, ten_file, steps_dir):
    """Ghép 4 panel: 1. Gốc  2. Xám  3. Tâm (chấm vàng)  4. Kết quả (khung đỏ)."""
    tam = img.copy()
    ket_qua = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.circle(tam, ((x1 + x2) // 2, (y1 + y2) // 2), 30, (0, 255, 255), -1)
        cv2.rectangle(ket_qua, (x1, y1), (x2, y2), (0, 0, 255), 12)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def panel(im, nhan, la_mau=False):
        im = cv2.resize(im, (450, 450))
        if not la_mau:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.putText(im, nhan, (12, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        return im

    hang = np.hstack([panel(img, "1. Goc", True),
                      panel(gray, "2. Xam"),
                      panel(tam, "3. Tam", True),
                      panel(ket_qua, "4. Ket qua", True)])
    out_path = os.path.join(steps_dir, f"{ten_file}_steps.png")
    cv2.imwrite(out_path, hang)
    print(f"  >> Da luu anh cac buoc: {out_path}")


# ============================================================
# 4) CẮT VÀ LƯU CHO 1 ẢNH
# ============================================================
def cat_mot_anh(duong_dan_anh, debug_dir, steps_dir):
    img = cv2.imread(duong_dan_anh)
    if img is None:
        print(f"  [LOI] Khong doc duoc anh: {duong_dan_anh}")
        return

    ten_file = os.path.splitext(os.path.basename(duong_dan_anh))[0]
    boxes = tim_cac_linh_kien(img)

    vis = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{ten_file}_{i}{SAVE_EXT}"),
                    img[y1:y2, x1:x2])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 12)
        cv2.putText(vis, str(i), (x1 + 30, y1 + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 255), 14)

    print(f"  {ten_file}: cat duoc {len(boxes)} con hang")
    cv2.imwrite(os.path.join(debug_dir, f"{ten_file}_debug.png"),
                cv2.resize(vis, (1000, 1000)))
    if SHOW_STEPS:
        luu_cac_buoc(img, boxes, ten_file, steps_dir)


# ============================================================
# 5) CHẠY CHÍNH
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    debug_dir = os.path.join(OUTPUT_DIR, "_debug")
    steps_dir = os.path.join(OUTPUT_DIR, "_steps")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(steps_dir, exist_ok=True)

    danh_sach_anh = sorted(glob.glob(os.path.join(INPUT_DIR, INPUT_EXT)))
    print(f"Loai con hang: {PART_TYPE}")
    print(f"Tim thay {len(danh_sach_anh)} anh trong: {INPUT_DIR}\n")

    for duong_dan in danh_sach_anh:
        cat_mot_anh(duong_dan, debug_dir, steps_dir)

    print(f"\nXONG. Anh cat luu tai: {OUTPUT_DIR}")
    print(f"Anh kiem tra (debug)  : {debug_dir}")


if __name__ == "__main__":
    main()
