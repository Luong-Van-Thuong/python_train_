# -*- coding: utf-8 -*-
"""
cropa27.py
==========
Cắt (crop) từng con hàng ra khỏi ảnh khay A27 bằng OpenCV thuần (không cần AI).

Khay A27 là lưới cao su đen MỊN nhiều ô; con hàng (kim loại sáng) nằm RẢI RÁC ở
vài ô bất kỳ, số lượng thay đổi (0..4). Con hàng có thể VUÔNG (giữa có cửa sổ tối)
hoặc TRÒN (vành kim loại). Dò chung 1 cách, KHÔNG cần chọn loại:

  DÒ "ĐỐM CẠNH DÀY ĐẶC" trên TOÀN ảnh:
    - Con hàng = cụm cạnh tương phản DÀY ĐẶC 2D (kim loại sáng xen khe/cửa sổ tối)
      -> mật độ cạnh CAO.
    - Lưới cao su = đường MẢNH, thưa  -> mật độ cạnh thấp (sau khi nhoè) -> loại.
    - Nền nhựa trơn (dù sáng) = ít cạnh -> loại.
  Sau khi ngưỡng mật độ + lấp lỗ, mỗi con hàng thành 1 KHỐI đặc; tâm = tâm
  bounding-box của khối (ổn định cho cả con vuông lẫn vành tròn hở).

Con hàng KHÔNG đổi kích thước -> có TÂM rồi cắt 1 khung CỐ ĐỊNH (PART_BOX) quanh tâm.

Cách dùng:
  - Sửa INPUT_DIR / OUTPUT_DIR bên dưới cho đúng.
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
INPUT_DIR  = "/mnt/d/Images_/SIBV/A27/img_train/260622/test"
OUTPUT_DIR = "/mnt/d/Images_/SIBV/A27/test"

# Số con hàng tối đa trên 1 khay (nếu dò ra nhiều hơn -> giữ các đốm lớn nhất)
MAX_PARTS = 4

# ---- KÍCH THƯỚC KHUNG CẮT (CỐ ĐỊNH; con hàng không đổi kích thước) ----
# (rộng, cao) px. >>> Đổi khi kích thước thực của con hàng khác đi. <<<
PART_BOX = (2000, 2000)

# ---- THAM SỐ DÒ "ĐỐM CẠNH DÀY ĐẶC" ----
# Ảnh gốc 5064px rất to -> thu nhỏ về DET_DS cho nhanh rồi quy ngược toạ độ.
# Các tỉ lệ tính theo DET_DS nên KHÔNG phụ thuộc kích thước ảnh gốc.
DET_DS    = 1200     # bề rộng ảnh dò
RANGE_K   = 9        # cửa sổ tính tương phản cục bộ (range = max-min)
RANGE_THR = 35       # range >= ngưỡng này -> coi là 'có cạnh'. HẠ xuống (vd 28)
                     # để bắt thêm con TRÒN nhẵn (cạnh yếu); TĂNG (vd 45) nếu
                     # bị nhận nhầm nền/lưới (ít false-positive hơn).
DENS_THR  = 0.42     # tỉ lệ cạnh trong cửa sổ -> coi là vùng kim loại đặc
MIN_AREA_F = 0.012   # diện tích đốm tối thiểu (theo DET_DS^2) -> loại đốm vụn
MAX_AREA_F = 0.18    # diện tích đốm tối đa  (theo DET_DS^2) -> loại mảng nền lớn
EXTENT_MIN = 0.50    # area/bbox: con hàng ĐẶC -> cao; nhánh lưới/nhiễu -> thấp
ASPECT_TOL = (0.55, 1.8)  # rộng/cao của đốm: con hàng gần vuông -> loại vệt dài
BRIGHT_MIN = 95      # độ sáng tb của đốm: con hàng = kim loại SÁNG; ô rỗng (oval
                     # lưới) thì TỐI -> loại. Hạ nếu sót con hàng tối màu.

# Đuôi file ảnh đầu vào và định dạng lưu ra
INPUT_EXT = "*.bmp"
SAVE_EXT  = ".png"

# Có lưu ảnh minh hoạ (debug + steps) không?
SHOW_STEPS = True


# ============================================================
# 2A) DÒ "ĐỐM CẠNH DÀY ĐẶC" (toàn ảnh, dùng chung cho vuông & tròn)
# ============================================================
def _lap_lo(mask):
    """Lấp lỗ bên trong (cửa sổ tối ở giữa con hàng) -> khối đặc."""
    ff = mask.copy()
    h, w = mask.shape
    m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 255)          # tô nền ngoài
    return mask | cv2.bitwise_not(ff)          # phần còn lại = lỗ -> lấp


def _do_dom(img):
    """Trả về danh sách TÂM (cx, cy) theo toạ độ ảnh GỐC của các con hàng."""
    h, w = img.shape[:2]
    s = DET_DS / float(w)
    small = cv2.resize(img, (DET_DS, int(round(h * s))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 1) Cạnh tương phản cục bộ: range = max-min trong cửa sổ nhỏ.
    ker_r = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RANGE_K, RANGE_K))
    rng = cv2.dilate(gray, ker_r).astype(np.int16) - cv2.erode(gray, ker_r)
    edges = (rng >= RANGE_THR).astype(np.float32)

    # 2) Mật độ cạnh: con hàng = cụm cạnh DÀY ĐẶC 2D -> cao;
    #    lưới = đường MẢNH thưa -> thấp (sau khi nhoè) -> bị loại.
    dens = cv2.GaussianBlur(edges, (0, 0), DET_DS / 25.0)
    mask = (dens >= DENS_THR).astype(np.uint8) * 255

    kc = max(5, DET_DS // 50)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    mask = _lap_lo(mask)

    # 3) Mỗi khối đặc đủ lớn = 1 con hàng; tâm = tâm bounding-box.
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    A = DET_DS * DET_DS
    blobs = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < MIN_AREA_F * A or a > MAX_AREA_F * A:
            continue
        x, y, ww, hh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        if a / float(ww * hh) < EXTENT_MIN:          # đốm rỗng/vệt lưới -> bỏ
            continue
        if not (ASPECT_TOL[0] <= ww / float(hh) <= ASPECT_TOL[1]):
            continue
        if float(gray[lbl == i].mean()) < BRIGHT_MIN:  # ô rỗng (oval tối) -> bỏ;
            continue                                   # con hàng là kim loại SÁNG
        blobs.append((a, int((x + ww / 2.0) / s), int((y + hh / 2.0) / s)))

    blobs.sort(key=lambda b: -b[0])             # đốm lớn nhất trước
    blobs = blobs[:MAX_PARTS]                    # khay tối đa MAX_PARTS con
    pts = [(cx, cy) for _, cx, cy in blobs]
    pts.sort(key=lambda p: (round(p[1] / (h / 3.0)), p[0]))  # trên->dưới, trái->phải
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
    """Dò tâm con hàng rồi cắt khung cố định quanh mỗi tâm.
    Trả về danh sách khung (x1, y1, x2, y2) theo toạ độ ảnh GỐC."""
    h_img, w_img = img.shape[:2]
    W, H = PART_BOX
    return [_khung_co_dinh(cx, cy, W, H, w_img, h_img) for cx, cy in _do_dom(img)]


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
    print(f"Tim thay {len(danh_sach_anh)} anh trong: {INPUT_DIR}\n")

    for duong_dan in danh_sach_anh:
        cat_mot_anh(duong_dan, debug_dir, steps_dir)

    print(f"\nXONG. Anh cat luu tai: {OUTPUT_DIR}")
    print(f"Anh kiem tra (debug)  : {debug_dir}")


if __name__ == "__main__":
    main()
