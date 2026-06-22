# -*- coding: utf-8 -*-
"""
cropa27_3_fixed.py
===================
Cắt linh kiện khay A27 bằng thuật toán dò tâm dựa trên Cửa Sổ Hình Chữ Nhật.
Triệt tiêu hoàn toàn nhiễu từ gờ nhựa và bóng đổ trong hốc ô số 3.
"""

import os
import glob
import cv2
import numpy as np

# ============================================================
# 1) CONFIGURATION - CẤU HÌNH HỆ THỐNG
# ============================================================
INPUT_DIR  = "/mnt/d/Images_/SIBV/A27/img_train/NG"
OUTPUT_DIR = "/mnt/d/Images_/SIBV/A27/img_train/NG_crop_2"

GRID_ROWS = 2
GRID_COLS = 2

# Khung cắt cố định hình học quanh tâm cấu kiện
PART_ROUND  = (2000, 2000)   # (w, h) px
PART_SQUARE = (2000, 2000)   # (w, h) px
BLOB_SPLIT  = 1250           # Ngưỡng phân loại Model hình tròn / vuông

# Ngưỡng động phân biệt túi trống (Empty Pocket) và cấu kiện
EMPTY_GRAD_ABS = 15.0
EMPTY_GRAD_REL = 0.5

INPUT_EXT = "*.bmp"
SAVE_EXT  = ".png"
SHOW_STEPS = True


# ============================================================
# 2) CORE ENGINE - THUẬT TOÁN XỬ LÝ HÌNH ẢNH MỚI
# ============================================================
def _grad_o(cell):
    """Tính mật độ cạnh trên toàn bộ ô để xác định ô có hàng."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    k = max(5, (cell.shape[1] // 200) | 1)
    return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _tam_con_hang(cell):
    """Dò tìm tâm thực của cấu kiện dựa trên Cửa Sổ Hình Chữ Nhật ở giữa."""
    h_cell, w_cell = cell.shape[:2]
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    
    # 1. Nhị phân hóa để lấy các vùng tối (như cửa sổ chữ nhật)
    # Dùng ngưỡng cố định thấp để chỉ lấy vùng tối nhất
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Xử lý đóng hình thái học để lấp đầy cửa sổ
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ker)
    
    # 3. Tìm Contour của cửa sổ
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return None
        
    # 4. Lọc Contour: Tìm cái nào gần hình chữ nhật nhất và có diện tích phù hợp
    best_cnt = None
    best_score = 0
    
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 5000: # Bỏ qua các đốm nhiễu nhỏ
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        extent = float(area)/rect_area # Tỷ lệ diện tích contour trên diện tích rect bao
        aspect_ratio = float(w)/h
        
        # Điểm số dựa trên độ "chữ nhật" và tỷ lệ khung hình
        score = extent * (1.0 / (abs(aspect_ratio - 1.5) + 1.0)) # Giả định tỷ lệ cửa sổ là 1.5
        
        if score > best_score:
            best_score = score
            best_cnt = cnt
            
    if best_cnt is None:
        return None
        
    # 5. Lấy tâm của hình chữ nhật bao của contour tốt nhất
    x, y, w, h = cv2.boundingRect(best_cnt)
    return (x + w // 2, y + h // 2, max(w, h))


def _khung_co_dinh(tam_x, tam_y, W, H, w_img, h_img):
    """Tạo bounding box cố định kích thước, tự động bù lề nếu chạm biên vật lý cảm biến."""
    x1 = min(max(0, tam_x - W // 2), max(0, w_img - W))
    y1 = min(max(0, tam_y - H // 2), max(0, h_img - H))
    return (x1, y1, x1 + W, y1 + H)


def tim_cac_linh_kien(img):
    """Phân tích lưới, cô lập tâm lõi cấu kiện và xuất danh sách tọa độ khung cắt."""
    h_img, w_img = img.shape[:2]
    ch, cw = h_img // GRID_ROWS, w_img // GRID_COLS

    cells_data = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y0, x0 = r * ch, c * cw
            cell = img[y0:y0 + ch, x0:x0 + cw]
            g = _grad_o(cell)
            
            cells_data.append({
                'x0': x0, 'y0': y0,
                'cell': cell,
                'mean_grad': float(g.mean())
            })
            
    gmax = max(item['mean_grad'] for item in cells_data) if cells_data else 0.0

    found = []   # Toạn độ đích (tam_x, tam_y, size) quy về ảnh gốc
    for item in cells_data:
        # Bộ lọc túi trống dựa trên mật độ cạnh
        if item['mean_grad'] < EMPTY_GRAD_ABS or item['mean_grad'] < EMPTY_GRAD_REL * gmax:
            continue
            
        res = _tam_con_hang(item['cell'])
        if res is not None:
            cx_cell, cy_cell, sz = res
            # CỘNG BÙ TỌA ĐỘ NGƯỢC VỀ ẢNH GỐC (Gốc ô + Tọa độ trong ô)
            global_cx = item['x0'] + cx_cell
            global_cy = item['y0'] + cy_cell
            found.append((global_cx, global_cy, sz))

    if not found:
        return []

    # Định biên kích thước hình học đồng nhất cho cả mẻ ảnh
    sizes = sorted(s for _, _, s in found)
    med_blob = sizes[len(sizes) // 2]
    W, H = PART_SQUARE if med_blob >= BLOB_SPLIT else PART_ROUND

    return [_khung_co_dinh(tx, ty, W, H, w_img, h_img) for tx, ty, _ in found]


# ============================================================
# 3) PIPELINE EXECUTION - CHẠY HỆ THỐNG TRỰC TIẾP
# ============================================================
def cat_mot_anh(duong_dan_anh, debug_dir):
    img = cv2.imread(duong_dan_anh)
    if img is None:
        print(f"  [ERROR] Không đọc được tệp tin: {duong_dan_anh}")
        return

    ten_file = os.path.splitext(os.path.basename(duong_dan_anh))[0]
    boxes = tim_cac_linh_kien(img)

    vis = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        crop = img[y1:y2, x1:x2]
        out_path = os.path.join(OUTPUT_DIR, f"{ten_file}_{i}{SAVE_EXT}")
        cv2.imwrite(out_path, crop)

        # Vẽ bounding box chuẩn hóa màu đỏ kiểm thử hệ thống
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 12)
        cv2.putText(vis, str(i), (x1 + 30, y1 + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 255), 14)

    print(f"  {ten_file}: Đã xuất thành công {len(boxes)} cấu kiện.")

    vis_small = cv2.resize(vis, (1000, 1000))
    cv2.imwrite(os.path.join(debug_dir, f"{ten_file}_debug.png"), vis_small)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    debug_dir = os.path.join(OUTPUT_DIR, "_debug")
    os.makedirs(debug_dir, exist_ok=True)

    danh_sach_anh = sorted(glob.glob(os.path.join(INPUT_DIR, INPUT_EXT)))
    print(f"Khởi chạy Pipeline. Tìm thấy {len(danh_sach_anh)} ảnh đầu vào.\n")

    for duong_dan in danh_sach_anh:
        cat_mot_anh(duong_dan, debug_dir)

    print(f"\n[SUCCESS] Hoàn thành. Kết quả lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()