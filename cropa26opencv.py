# -*- coding: utf-8 -*-
"""
cropa26opencv.py
================
Cắt (crop) từng linh kiện ra khỏi ảnh khay A26 bằng OpenCV thuần (không cần AI).

Ý tưởng rất đơn giản, dễ hình dung:
  1. Ảnh gốc là 1 khay chụp từ trên xuống, có 4 linh kiện sáng (kim loại)
     xếp theo lưới 2 hàng x 2 cột.
  2. CHIA ẢNH THÀNH 4 Ô (2x2) trước -> mỗi ô chắc chắn chứa đúng 1 linh kiện.
     (Cách này luôn ra đủ 4 linh kiện, kể cả khi 2 linh kiện sát/dính nhau.)
  3. Trong từng ô: tìm TÂM con hàng bằng cách dò 'hốc tối ở giữa' (cửa sổ cảm
     biến) - một vùng tối nhỏ gọn nằm gần giữa ô. Cách này KHÔNG bị các cầu nối
     kim loại sáng làm lệch tâm (ổn định cho mọi loại hàng).
  4. Vì con hàng cố định nên có kích thước cố định: cắt 1 hình chữ nhật ĐÚNG
     kích thước PART_WIDTH x PART_HEIGHT quanh tâm -> tránh nhiễu làm méo
     kích thước, mọi ảnh cắt ra đều bằng nhau.
     (Nếu ô nào dò hỏng thì lấy tâm ô làm tâm để không bị sót.)

Cách dùng:
  - Sửa 2 đường dẫn INPUT_DIR / OUTPUT_DIR bên dưới cho đúng máy của bạn.
  - Chạy:  python cropa26opencv.py
  - Ảnh cắt ra nằm trong OUTPUT_DIR, mỗi linh kiện 1 file:
        <tên ảnh>_1.png, <tên ảnh>_2.png, ...
  - Ảnh kiểm tra (vẽ khung đỏ lên ảnh thu nhỏ) nằm trong thư mục con "_debug"
    để bạn nhìn xem cắt có đúng không.
"""

import os
import glob
import cv2
import numpy as np

# ============================================================
# 1) CẤU HÌNH  -  chỉ cần chỉnh ở đây
# ============================================================

# Đường dẫn theo kiểu WSL2: ổ Windows D:\ nằm ở /mnt/d, C:\ ở /mnt/c ...
# (Nếu chạy trên Windows thuần thì đổi lại thành r"D:/Images_/SIBV/...")
#
# Thư mục chứa TẤT CẢ ảnh cần cắt
INPUT_DIR  = "/mnt/d/Images_/SIBV/A26/img_train/OK"

# Thư mục để lưu ảnh đã cắt (tự tạo nếu chưa có)
OUTPUT_DIR = "/mnt/d/Images_/SIBV/A26/img_train/OK_crop"

# Lưới linh kiện trên mỗi ảnh: 2 hàng x 2 cột = 4 linh kiện
GRID_ROWS = 2
GRID_COLS = 2
EXPECTED_COUNT = GRID_ROWS * GRID_COLS

# ---- KÍCH THƯỚC CỐ ĐỊNH CỦA CON HÀNG (pixel) ----
# Con hàng cố định nên có kích thước cố định: ta dò TÂM con hàng rồi cắt
# 1 hình chữ nhật ĐÚNG kích thước này quanh tâm. Nhờ vậy mọi khung cắt đều
# bằng nhau, KHÔNG bị nhiễu làm to/nhỏ thất thường.
# >>> Đổi 2 số này khi dùng cho loại hàng / bài toán khác. <<<
PART_WIDTH  = 2200   # chiều rộng (ngang) con hàng
PART_HEIGHT = 2200   # chiều cao  (dọc)  con hàng

# ---- CÁCH TÌM TÂM CON HÀNG: dò "hốc tối ở giữa" (cửa sổ cảm biến) ----
# Hốc này tối, nhỏ gọn, nằm giữa con hàng -> dùng làm tâm rất ổn định, KHÔNG bị
# các cầu nối kim loại sáng làm lệch (như khi lấy trọng tâm cả vùng sáng).
DARK_THRESH = 50          # pixel xám TỐI hơn ngưỡng này coi là "tối" (0..255)
CAVITY_MIN_RATIO = 0.012  # hốc phải to hơn 1.2% diện tích ô (bỏ đốm nhỏ)
CAVITY_MAX_RATIO = 0.20   # và nhỏ hơn 20% diện tích ô (bỏ mảng tối lớn)


# Đuôi file ảnh đầu vào và định dạng lưu ra
INPUT_EXT = "*.bmp"
SAVE_EXT  = ".png"     # .png lưu không mất chất lượng (tốt cho train)

# Có lưu ảnh minh hoạ các bước biến đổi (ghép 5 bước thành 1 file) không?
# True  = có lưu vào thư mục con _steps để mở xem (KHÔNG bật cửa sổ nên không lỗi GUI).
# False = tắt (chạy nhanh, không lưu).  -> sau này bạn để False hoặc comment.
SHOW_STEPS = True


# ============================================================
# 2) HÀM TÌM CÁC LINH KIỆN TRONG 1 ẢNH
# ============================================================
def _mask_hoc_toi(cell):
    """Tạo mask vùng TỐI (hốc cảm biến ở giữa con hàng) cho 1 ô."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    # Pixel tối hơn DARK_THRESH -> trắng (255) trong mask
    _, dark = cv2.threshold(blur, DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
    # Xoá các đốm tối nhỏ li ti cho sạch
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((35, 35), np.uint8))
    return dark


def _tim_tam_con_hang(cell):
    """
    Tìm TÂM con hàng trong 1 ô bằng cách dò 'hốc tối ở giữa' (cửa sổ cảm biến):
    một vùng tối, nhỏ gọn, nằm gần giữa ô. Cách này ổn định cho mọi loại hàng
    (kể cả khi xung quanh có cầu nối kim loại sáng).
    Trả về (cx, cy) theo toạ độ RIÊNG của ô, hoặc None nếu không tìm thấy.
    """
    ch, cw = cell.shape[:2]
    cell_area = cw * ch

    dark = _mask_hoc_toi(cell)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_dist = None
    for c in contours:
        area = cv2.contourArea(c)
        # Hốc phải có diện tích vừa phải (không quá nhỏ, không quá lớn)
        if area < CAVITY_MIN_RATIO * cell_area or area > CAVITY_MAX_RATIO * cell_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h)
        if aspect < 0.4 or aspect > 2.5:      # bỏ vệt dài (không phải hốc)
            continue
        cx, cy = x + w // 2, y + h // 2
        # Con hàng nằm gần giữa ô -> chọn hốc GẦN TÂM Ô nhất
        dist = (cx - cw // 2) ** 2 + (cy - ch // 2) ** 2
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (cx, cy)
    return best


def _khung_co_dinh(tam_x, tam_y, W, H, w_img, h_img):
    """
    Tạo khung chữ nhật ĐÚNG kích thước W x H, đặt giữa tại (tam_x, tam_y).
    Nếu khung lú ra ngoài ảnh thì DỜI vào trong (giữ nguyên kích thước),
    nhờ vậy mọi ảnh cắt ra đều có đúng cùng kích thước.
    """
    x1 = tam_x - W // 2
    y1 = tam_y - H // 2
    # Dời vào trong nếu tràn biên (vẫn giữ đúng W, H)
    x1 = min(max(0, x1), max(0, w_img - W))
    y1 = min(max(0, y1), max(0, h_img - H))
    return (x1, y1, x1 + W, y1 + H)


def tim_cac_linh_kien(img):
    """
    Chia ảnh thành lưới GRID_ROWS x GRID_COLS, dò tâm con hàng trong từng ô,
    rồi tạo khung cắt CỐ ĐỊNH (PART_WIDTH x PART_HEIGHT) quanh tâm đó.
    Trả về danh sách khung (x1, y1, x2, y2) theo toạ độ ảnh GỐC,
    đã sắp theo thứ tự: hàng trên xuống dưới, trái sang phải.
    """
    h_img, w_img = img.shape[:2]
    ch, cw = h_img // GRID_ROWS, w_img // GRID_COLS

    boxes = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y0, x0 = r * ch, c * cw            # góc trên-trái của ô
            cell = img[y0:y0 + ch, x0:x0 + cw]

            tam = _tim_tam_con_hang(cell)
            if tam is not None:
                cx, cy = tam                   # tâm con hàng (toạ độ trong ô)
            else:
                cx, cy = cw // 2, ch // 2      # dò hỏng -> lấy tâm ô làm tâm

            # Đổi tâm sang toạ độ ảnh gốc rồi cắt khung cố định quanh tâm
            box = _khung_co_dinh(x0 + cx, y0 + cy,
                                 PART_WIDTH, PART_HEIGHT, w_img, h_img)
            boxes.append(box)
    return boxes


# ============================================================
# 3) HÀM GHÉP & LƯU CÁC BƯỚC BIẾN ĐỔI  (chỉ để xem - sau có thể comment cả phần này)
# ============================================================
def luu_cac_buoc(img, boxes, ten_file, steps_dir):
    """
    Ghép 5 ảnh thành 1 bảng để thấy code biến đổi thế nào, rồi LƯU RA FILE:
        1. Ảnh gốc   2. Ảnh xám   3. Mask vùng TỐI (hốc cảm biến)
        4. Tâm dò được (chấm đỏ)  5. Kết quả khung cắt (khung đỏ)

    Dùng lưu-ra-file thay cho cv2.imshow vì cửa sổ GUI trong WSL hay lỗi
    "Qt platform plugin xcb". Mở file trong thư mục _steps bằng Windows để xem.
    """
    h_img, w_img = img.shape[:2]
    ch, cw = h_img // GRID_ROWS, w_img // GRID_COLS

    # Dựng lại mask vùng TỐI theo TỪNG Ô + chấm tâm dò được (đúng như lúc dò)
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark_full = np.zeros((h_img, w_img), np.uint8)
    tam_full = img.copy()
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y0, x0 = r * ch, c * cw
            cell = img[y0:y0 + ch, x0:x0 + cw]
            dark_full[y0:y0 + ch, x0:x0 + cw] = _mask_hoc_toi(cell)
            tam = _tim_tam_con_hang(cell)
            cx, cy = tam if tam is not None else (cw // 2, ch // 2)
            cv2.circle(tam_full, (x0 + cx, y0 + cy), 40, (0, 0, 255), -1)

    # Ảnh kết quả: vẽ khung đỏ lên ảnh gốc
    ket_qua = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(ket_qua, (x1, y1), (x2, y2), (0, 0, 255), 12)

    # Hàm phụ: thu nhỏ về 450x450 + ghi nhãn, mask xám đổi sang 3 kênh để ghép chung
    def panel(im, nhan, la_mau=False):
        im = cv2.resize(im, (450, 450))
        if not la_mau:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.putText(im, nhan, (12, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)
        return im

    o_trong = np.zeros((450, 450, 3), np.uint8)   # ô trống cho đủ lưới 2x3
    hang1 = np.hstack([panel(img, "1. Goc", True),
                       panel(gray_full, "2. Xam"),
                       panel(dark_full, "3. Vung toi")])
    hang2 = np.hstack([panel(tam_full, "4. Tam do duoc", True),
                       panel(ket_qua, "5. Ket qua", True),
                       o_trong])
    bang = np.vstack([hang1, hang2])

    out_path = os.path.join(steps_dir, f"{ten_file}_steps.png")
    cv2.imwrite(out_path, bang)
    print(f"  >> Da luu anh cac buoc bien doi: {out_path}")


# ============================================================
# 4) HÀM CẮT VÀ LƯU CHO 1 ẢNH
# ============================================================
def cat_mot_anh(duong_dan_anh, debug_dir, steps_dir):
    img = cv2.imread(duong_dan_anh)
    if img is None:
        print(f"  [LOI] Khong doc duoc anh: {duong_dan_anh}")
        return

    ten_file = os.path.splitext(os.path.basename(duong_dan_anh))[0]

    boxes = tim_cac_linh_kien(img)

    # Ảnh debug (thu nhỏ) để xem khung cắt có đúng không
    vis = img.copy()

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        crop = img[y1:y2, x1:x2]
        out_path = os.path.join(OUTPUT_DIR, f"{ten_file}_{i}{SAVE_EXT}")
        cv2.imwrite(out_path, crop)

        # Vẽ khung + số thứ tự lên ảnh debug
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 12)
        cv2.putText(vis, str(i), (x1 + 30, y1 + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 255), 14)

    print(f"  {ten_file}: cat duoc {len(boxes)} linh kien")

    # Lưu ảnh debug thu nhỏ
    vis_small = cv2.resize(vis, (1000, 1000))
    cv2.imwrite(os.path.join(debug_dir, f"{ten_file}_debug.png"), vis_small)

    # ----- Lưu ảnh minh hoạ các bước biến đổi (có thể comment khi không cần) -----
    if SHOW_STEPS:
        luu_cac_buoc(img, boxes, ten_file, steps_dir)


# ============================================================
# 5) CHẠY CHÍNH: duyệt toàn bộ ảnh trong INPUT_DIR
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
    if SHOW_STEPS:
        print(f"Anh cac buoc bien doi : {steps_dir}")


if __name__ == "__main__":
    main()
