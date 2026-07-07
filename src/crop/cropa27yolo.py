import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

class YoloObbExtractor:
    def __init__(self, model_path):
        """
        Khởi tạo mô hình YOLO
        """
        print(f"[INFO] Loading YOLO Model từ: {model_path} ...")
        self.model = YOLO(model_path)

    def process_and_crop(self, source_dir, dest_dir, conf_threshold=0.6):
        """
        Quét thư mục Windows, trích xuất vật thể và cắt hàng loạt
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
            
            raw_img = cv2.imread(img_path)
            if raw_img is None:
                print(f"[ERROR] Không thể đọc ảnh: {img_path}")
                continue

            # Dự đoán bằng YOLO với ngưỡng tự tin (Confidence)
            results = self.model.predict(source=raw_img, conf=conf_threshold, verbose=False)
            result = results[0]

            # -----------------------------------------------------------------
            # SENIOR REVIEW: BẪY CHECK MODEL LÀ OBB HAY DETECT THƯỜNG
            # -----------------------------------------------------------------
            if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                # TRƯỜNG HỢP 1: Nếu thực sự là model YOLO OBB (Box Xoay)
                # Chuyển đổi OBB sang bounding box ngang dạng xyxy bao quanh để cắt ảnh
                det_boxes = result.obb.xyxy.cpu().numpy()
            elif result.boxes is not None and len(result.boxes) > 0:
                # TRƯỜNG HỢP 2: Nếu là model YOLO Detect thông thường (Box Ngang)
                det_boxes = result.boxes.xyxy.cpu().numpy()
            else:
                print(f"[WARN] Không tìm thấy object nào trong ảnh: {base_name}")
                continue

            img_h, img_w = raw_img.shape[:2]

            # Duyệt qua từng vật thể tìm được trong 1 tấm ảnh
            for obj_idx, box in enumerate(det_boxes):
                x1, y1, x2, y2 = box

                # Ép kiểu dữ liệu và bẫy lỗi tràn biên ảnh (Out of bound)
                x_start = max(0, int(x1))
                y_start = max(0, int(y1))
                x_end = min(img_w, int(x2))
                y_end = min(img_h, int(y2))

                # Tính diện tích object (pixel²) để lọc nhiễu
                obj_area = (x_end - x_start) * (y_end - y_start)
                if obj_area < 900000:
                    print(f"[SKIP] Object #{obj_idx} trong {base_name}: diện tích = {obj_area}px² (< 900000)")
                    continue

                # Cắt ảnh nhỏ trực tiếp từ tọa độ box bao quanh
                cropped_obj = raw_img[y_start:y_end, x_start:x_end]

                if cropped_obj.size == 0:
                    continue

                # 4. Lưu kết quả dưới định dạng .bmp
                output_filename = f"{base_name}_obj_{obj_idx}.bmp"
                output_path = os.path.join(dest_dir, output_filename)
                
                cv2.imwrite(output_path, cropped_obj)
                print(f"[SUCCESS] Đã cắt -> Lưu: {output_filename}")

# ==========================================
# THỰC THI CHƯƠNG TRÌNH TRÊN WSL2 (ổ D:\ map sang /mnt/d/)
# ==========================================
if __name__ == "__main__":
    # Đường dẫn WSL: ổ D:\ của Windows map sang /mnt/d/
    MODEL_BEST_PT = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/yolo_odb_detect/weights/best.pt"

    SOURCE_DIRECTORY = "/mnt/d/Images_/SIBV/A27/img_train/260622/test"  # Thư mục chứa ảnh gốc .bmp
    DEST_DIRECTORY = "/mnt/d/Images_/SIBV/A27/test"  # Thư mục lưu kết quả cắt

    # Khởi chạy extractor
    extractor = YoloObbExtractor(model_path=MODEL_BEST_PT)

    print("[INFO] Bắt đầu chạy luồng trích xuất dữ liệu trên WSL2...")
    extractor.process_and_crop(
        source_dir=SOURCE_DIRECTORY, 
        dest_dir=DEST_DIRECTORY, 
        conf_threshold=0.6  # Ngưỡng Confidence
    )
    print("[INFO] Hoàn thành toàn bộ tiến trình!")