import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

class YoloObbExtractor:
    def __init__(self, model_path):
        """
        Khởi tạo mô hình YOLO OBB
        """
        print(f"[INFO] Loading YOLO Model từ: {model_path} ...")
        self.model = YOLO(model_path)

    def process_and_crop(self, source_dir, dest_dir, conf_threshold=0.6):
        """
        Quét thư mục, dùng YOLO OBB tìm vật thể, xoay thẳng trục và cắt hàng loạt
        """
        # 1. Khởi tạo thư mục đích nếu chưa tồn tại
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            print(f"[INFO] Đã tạo thư mục lưu kết quả: {dest_dir}")

        # 2. Quét toàn bộ file .bmp trong thư mục gốc
        search_path = os.path.join(source_dir, "*.bmp")
        img_files = glob.glob(search_path)
        
        if not img_files:
            print(f"[WARN] Không tìm thấy file .bmp nào tại đường dẫn: {source_dir}")
            return

        print(f"[INFO] Tìm thấy {len(img_files)} file ảnh gốc cần xử lý.")

        # 3. Chạy Pipeline tuần tự qua từng ảnh
        for img_path in img_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Đọc ảnh gốc (Giữ nguyên định dạng gốc)
            raw_img = cv2.imread(img_path)
            if raw_img is None:
                print(f"[ERROR] Không thể đọc ảnh: {img_path}")
                continue

            # Dự đoán bằng YOLO với ngưỡng tự tin (Confidence)
            results = self.model.predict(source=raw_img, conf=conf_threshold, verbose=False)
            result = results[0]

            # Kiểm tra xem mô hình có trả về Bounding Box không
            if result.boxes is None or len(result.boxes) == 0:
                print(f"[WARN] Không tìm thấy object nào trong ảnh: {base_name}")
                continue

            # Lấy tọa độ Bounding Box: định dạng xyxy (x1, y1, x2, y2) - pixel thực
            det_boxes = result.boxes.xyxy.cpu().numpy()
            img_h, img_w = raw_img.shape[:2]

            # Duyệt qua từng vật thể tìm được trong 1 tấm ảnh
            for obj_idx, box in enumerate(det_boxes):
                x1, y1, x2, y2 = box

                # Chuyển sang int và bẫy lỗi tràn viền
                x_start = max(0, int(x1))
                y_start = max(0, int(y1))
                x_end = min(img_w, int(x2))
                y_end = min(img_h, int(y2))

                # Tính diện tích object (pixel²) và bỏ qua nếu quá nhỏ
                obj_area = (x_end - x_start) * (y_end - y_start)
                if obj_area < 450000:
                    print(f"[SKIP] Object #{obj_idx} trong {base_name}: diện tích = {obj_area}px² (< 10000)")
                    continue

                # Cắt ảnh nhỏ trực tiếp (không cần xoay vì model detect dùng box ngang)
                cropped_obj = raw_img[y_start:y_end, x_start:x_end]

                if cropped_obj.size == 0:
                    continue

                # 4. Lưu kết quả dưới định dạng .bmp theo đúng yêu cầu của ông
                output_filename = f"{base_name}_obj_{obj_idx}.bmp"
                output_path = os.path.join(dest_dir, output_filename)
                
                cv2.imwrite(output_path, cropped_obj)
                print(f"[SUCCESS] Đã cắt -> Lưu: {output_filename}")

# ==========================================
# THỰC THI CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # Cấu hình chuẩn 100% định dạng WSL2 (Linux) - Đã sửa chữ 'd' viết thường và dấu sụt xuôi '/'
    MODEL_BEST_PT = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A26/yolo11n_960_260614_part1/weights/best.pt"
    
    SOURCE_DIRECTORY = "/mnt/d/Images_/SIBV/A26/260616/ng"
    DEST_DIRECTORY = "/mnt/d/Images_/SIBV/A26/260616/ng_"

    # Khởi chạy extractor
    extractor = YoloObbExtractor(model_path=MODEL_BEST_PT)
    
    print("[INFO] Bắt đầu chạy luồng trích xuất dữ liệu...")
    extractor.process_and_crop(
        source_dir=SOURCE_DIRECTORY, 
        dest_dir=DEST_DIRECTORY, 
        conf_threshold=0.6  # Ngưỡng bắt độ chính xác (Confidence)
    )
    print("[INFO] Hoàn thành toàn bộ tiến trình!")