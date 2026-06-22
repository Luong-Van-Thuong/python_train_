# -*- coding: utf-8 -*-
"""
cropa27_oring.py
================
Thuật toán cải tiến: Dò cấu trúc Vành lồng nhau (Nested O-Ring)
Trị dứt điểm lỗi nhận nhầm bóng đổ khay trống và sót hàng kim loại nhám.
"""

import os
import glob
import cv2
import numpy as np

# ============================================================
# 1) CẤU HÌNH VÀ THAM SỐ LỌC HÌNH HỌC O-RING
# ============================================================
INPUT_DIR  = "/mnt/d/Images_/SIBV/A27/img_train/260622/test"
OUTPUT_DIR = "/mnt/d/Images_/SIBV/A27/test"

MAX_PARTS = 4
PART_BOX = (2000, 2000) # (rộng, cao) px trên ảnh gốc

DET_DS         = 1200   # Bề rộng ảnh xử lý để tối ưu Cycle Time (<5ms)
DARK_THR       = 55     # Ngưỡng cắt vùng tối (Vành cao su đen của con hàng)
MIN_AREA_F     = 0.010  # Tỉ lệ diện tích tối thiểu so với DET_DS^2
MAX_AREA_F     = 0.080  # Tỉ lệ diện tích tối đa
ASPECT_TOL     = (0.80, 1.25) # Giới hạn tỷ lệ W/H (Ép form vuông/tròn để loại vệt bóng đổ)
CIRCULARITY_MIN = 0.50  # Độ tròn toán học: Chặn đứng bóng đổ dạng đường hở của vách khay

INPUT_EXT = "*.bmp"
SAVE_EXT  = ".png"


# ============================================================
# 2) HELPER - Tính toán Bounding Box cố định sát biên
# ============================================================
def _calculate_fixed_box(cx, cy, img_w, img_h):
    W, H = PART_BOX
    x = min(max(0, cx - W // 2), max(0, img_w - W))
    y = min(max(0, cy - H // 2), max(0, img_h - H))
    w = min(W, img_w - x)
    h = min(H, img_h - y)
    return (x, y, x + w, y + h)


# ============================================================
# 3) PIPELINE XỬ LÝ CHÍNH CHO 1 ẢNH
# ============================================================
def cat_mot_anh_oring(duong_dan_anh, debug_dir):
    img = cv2.imread(duong_dan_anh)
    if img is None:
        print(f"  [LOI] Khong doc duoc anh: {duong_dan_anh}")
        return

    ten_file = os.path.splitext(os.path.basename(duong_dan_anh))[0]
    orig_h, orig_w = img.shape[:2]
    scale = DET_DS / float(orig_w)
    det_h = int(round(orig_h * scale))

    # Bước 1: Hạ phân giải + Chuyển xám (Tối ưu RAM/CPU)
    small = cv2.resize(img, (DET_DS, det_h))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Bước 2: Nhị phân ngược trích xuất vùng tối (O-Ring cao su đen)
    _, thresh = cv2.threshold(gray, DARK_THR, 255, cv2.THRESH_BINARY_INV)

    # Bước 3: Morphology Opening khử các đường bóng đổ mảnh nối từ vách khay vào ô trống
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Bước 4: Tìm đường biên (Contours)
    contours, _ = cv2.findContours(morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    valid_candidates = []
    total_image_area = DET_DS * det_h
    vis_debug = small.copy() # Ảnh minh họa báo cáo (Step Filter)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Bộ lọc 1: Diện tích hình học
        if area < MIN_AREA_F * total_image_area or area > MAX_AREA_F * total_image_area:
            continue

        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = float(ww) / hh
        
        # Bộ lọc 2: Tỉ lệ khung Bounding Box (Chỉ giữ hình tiệm cận vuông/tròn)
        if not (ASPECT_TOL[0] <= aspect <= ASPECT_TOL[1]):
            continue

        # Bộ lọc 3: Độ tròn toán học (Circularity = 4 * PI * Area / Perimeter^2)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        is_valid = circularity >= CIRCULARITY_MIN
        color = (0, 255, 0) if is_valid else (0, 0, 255) # Xanh lá = Chọn, Đỏ = Loại bóng đổ
        
        # Vẽ trực quan hóa phục vụ báo cáo giải trình cho Sếp
        cv2.rectangle(vis_debug, (x, y), (x + ww, y + hh), color, 2)

        if is_valid:
            cx = int(x + ww / 2.0)
            cy = int(y + hh / 2.0)
            valid_candidates.append((area, cx, cy))

    # Sắp xếp diện tích giảm dần, lấy tối đa MAX_PARTS con hàng rõ nhất
    valid_candidates.sort(key=lambda b: -b[0])
    valid_candidates = valid_candidates[:MAX_PARTS]

    # Sắp xếp tọa độ từ trên xuống dưới, trái sang phải để thứ tự lưu file không bị đảo lộn
    valid_candidates.sort(key=lambda p: (round(p[2] / (det_h / 3.0)), p[1]))

    # Bước 5: Khôi phục tọa độ gốc và tiến hành Crop ảnh sản lượng 2000x2000
    vis_final = img.copy()
    for idx, (_, cx, cy) in enumerate(valid_candidates, start=1):
        # Quy đổi ngược tọa độ về ảnh gốc ban đầu
        orig_cx = int(round(cx / scale))
        orig_cy = int(round(cy / scale))

        x1, y1, x2, y2 = _calculate_fixed_box(orig_cx, orig_cy, orig_w, orig_h)

        # Thực hiện Crop ảnh sản lượng thực tế
        cropped = img[y1:y2, x1:x2]
        crop_path = os.path.join(OUTPUT_DIR, f"{ten_file}_{idx}{SAVE_EXT}")
        cv2.imwrite(crop_path, cropped)

        # Vẽ kết quả lên ảnh tổng kiểm tra
        cv2.rectangle(vis_final, (x1, y1), (x2, y2), (0, 0, 255), 12)
        cv2.putText(vis_final, str(idx), (x1 + 30, y1 + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 255), 12)

    print(f"  {ten_file}: Tim thay & cat thanh cong {len(valid_candidates)} con hang.")

    # Lưu ảnh phục vụ làm slide/báo cáo kỹ thuật
    cv2.imwrite(os.path.join(debug_dir, f"{ten_file}_report_filter.png"), vis_debug)
    cv2.imwrite(os.path.join(debug_dir, f"{ten_file}_final.png"), cv2.resize(vis_final, (1000, 1000)))


# ============================================================
# 4) KHỞI CHẠY HỆ THỐNG
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    debug_dir = os.path.join(OUTPUT_DIR, "_debug")
    os.makedirs(debug_dir, exist_ok=True)

    danh_sach_anh = sorted(glob.glob(os.path.join(INPUT_DIR, INPUT_EXT)))
    print(f"Tim thay {len(danh_sach_anh)} file anh thô đầu vào. Đang chạy pipeline...\n")

    for duong_dan in danh_sach_anh:
        cat_mot_anh_oring(duong_dan, debug_dir)

    print(f"\n[XONG] Hoàn tất báo cáo. Kiểm tra thư mục: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()