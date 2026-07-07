
from ultralytics import YOLO
import os
import cv2
import numpy as np

# =========================
# Model
# =========================
model_path = "/mnt/d/python/mlcc/VisionProject/yolo11n_20260602_part1_images_960_vung_tu/weights/best.pt"

# =========================
# Thư mục ảnh đầu vào / đầu ra
# =========================
image_dir = "/mnt/d/Images/JeaYoung/MLCC/image_"
save_dir  = "/mnt/d/Images/JeaYoung/MLCC/test_result"
os.makedirs(save_dir, exist_ok=True)

# =========================
# Cấu hình ROI — vị trí X sẽ được tính TỰ ĐỘNG từ đường ranh giới
#   ROI_X_OFFSET : khoảng cách từ đường ranh giới đến cạnh TRÁI của ROI
#                  (âm = ROI nằm BÊN TRÁI đường ranh giới)
#   ROI_W, ROI_H : kích thước chung của mỗi ROI (pixel)
#   ROI_Y_LIST   : danh sách tọa độ Y (cố định) cho từng ROI
# =========================
ROI_X_OFFSET = -65   # ROI bắt đầu cách đường ranh giới 65px về bên trái
ROI_W        = 60    # chiều rộng ROI
ROI_H        = 50    # chiều cao ROI
ROI_Y_LIST   = [60, 220]   # Y của ROI_1 và ROI_2

# =========================
# Ngưỡng Value (HSV) để tách blob sáng khỏi nền đen
# Tăng nếu bắt nhầm nền, giảm nếu sót blob mờ
# =========================
VALUE_THRESHOLD = 0.60

# =========================
# Load model
# =========================
model = YOLO(model_path)

exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ------------------------------------------------------------------
def find_boundary_line(crop):
    """
    Tìm đường dọc phân ranh giữa vùng TỐI (thân linh kiện) và vùng SÁNG (nền xám/xanh).

    Cách làm:
      1. Chuyển sang grayscale, tính độ sáng trung bình theo từng CỘT.
      2. Smooth nhẹ để loại nhiễu.
      3. Tính gradient (độ thay đổi giữa 2 cột liền kề).
      4. Cột có gradient lớn nhất (trong nửa phải ảnh) = ranh giới tối→sáng.

    Trả về:  tọa độ X của đường ranh giới, hoặc None nếu không tìm thấy.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Tính trung bình độ sáng của mỗi cột (shape = [w])
    col_means = np.mean(gray, axis=0).astype(float)

    # Smooth 5 cột để giảm nhiễu cạnh
    col_means = np.convolve(col_means, np.ones(5) / 5, mode='same')

    # Gradient: sự thay đổi độ sáng giữa 2 cột kề nhau
    gradient = np.diff(col_means)   # shape = [w-1]

    # Chỉ tìm trong nửa phải của ảnh (ranh giới nằm bên phải thân linh kiện)
    search_start = w // 3
    search_end   = w - 3   # bỏ qua vài cột sát cạnh ảnh

    if search_start >= search_end:
        return None

    # Cột có gradient (tối → sáng) lớn nhất = ranh giới
    sub_grad = gradient[search_start:search_end]
    peak_col = int(np.argmax(sub_grad)) + search_start

    return peak_col


# ------------------------------------------------------------------
def find_blobs_in_roi(crop, rx, ry, rw, rh, value_threshold=0.30):
    """
    Tìm blob sáng (gray/yellow) trong vùng ROI của ảnh crop.
    Trả về: [(cx, cy, area, contour_đã_shift_về_crop), ...]
    """
    h_crop, w_crop = crop.shape[:2]

    roi_x1 = max(0, rx)
    roi_y1 = max(0, ry)
    roi_x2 = min(w_crop, rx + rw)
    roi_y2 = min(h_crop, ry + rh)

    if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
        return []

    roi_patch = crop[roi_y1:roi_y2, roi_x1:roi_x2]

    # Chuyển sang HSV, lấy kênh Value để tách vùng sáng
    hsv      = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2HSV)
    v_ch     = hsv[:, :, 2]
    thresh_v = int(value_threshold * 255)
    _, mask  = cv2.threshold(v_ch, thresh_v, 255, cv2.THRESH_BINARY)

    # Morphology: loại hạt nhiễu nhỏ rồi lấp lỗ hổng
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue

        # Chuyển contour từ tọa độ ROI-patch → tọa độ crop
        cnt_shifted = cnt + np.array([roi_x1, roi_y1])

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        blobs.append((cx, cy, area, cnt_shifted))

    return blobs


# ------------------------------------------------------------------
# Vòng lặp chính
# ------------------------------------------------------------------
for file_name in os.listdir(image_dir):

    if not file_name.lower().endswith(exts):
        continue

    img_path = os.path.join(image_dir, file_name)
    img      = cv2.imread(img_path)

    if img is None:
        print(f"Cannot read: {file_name}")
        continue

    results  = model(img)
    crop_idx = 0

    for box in results[0].boxes:

        conf = float(box.conf[0])
        if conf < 0.6:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Cắt crop với padding nhỏ
        pad     = 15
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(img.shape[1], x2 + pad)
        crop_y2 = min(img.shape[0], y2 + pad)

        crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if crop.size == 0:
            continue

        h, w = crop.shape[:2]

        # =========================
        # Tìm đường ranh giới tối → sáng
        # =========================
        boundary_x = find_boundary_line(crop)

        if boundary_x is None:
            # Fallback: dùng vị trí X mặc định nếu không tìm thấy đường
            roi_anchor_x = 70
            print(f"  [{file_name}] crop_{crop_idx}: Không tìm thấy ranh giới, dùng X mặc định={roi_anchor_x}")
        else:
            # ROI bắt đầu tại: đường ranh giới + offset (thường là giá trị âm → bên trái)
            roi_anchor_x = boundary_x + ROI_X_OFFSET
            print(f"  [{file_name}] crop_{crop_idx}: Ranh giới tại x={boundary_x}, ROI anchor x={roi_anchor_x}")

            # Vẽ đường ranh giới (trắng) để kiểm tra
            cv2.line(crop, (boundary_x, 0), (boundary_x, h), (255, 255, 255), 1)

        # =========================
        # Vẽ ROI + tìm blob trong từng ROI
        # =========================
        for roi_idx, roi_y in enumerate(ROI_Y_LIST, start=1):

            # Vị trí ROI = anchor_x (từ đường ranh giới) + Y cố định
            rx = roi_anchor_x
            ry = roi_y
            rw = ROI_W
            rh = ROI_H

            rx1 = max(0, rx)
            ry1 = max(0, ry)
            rx2 = min(w, rx + rw)
            ry2 = min(h, ry + rh)

            if rx1 >= w or ry1 >= h or rx1 >= rx2 or ry1 >= ry2:
                continue

            # ROI 1 màu xanh lá, ROI 2 màu đỏ
            roi_color = (0, 255, 0) if roi_idx == 1 else (0, 0, 255)

            cv2.rectangle(crop, (rx1, ry1), (rx2, ry2), roi_color, 2)
            cv2.putText(
                crop, f"ROI_{roi_idx}",
                (rx1, max(20, ry1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2,
            )

            # ---------- Tìm blob ----------
            blobs = find_blobs_in_roi(crop, rx, ry, rw, rh, VALUE_THRESHOLD)

            for b_idx, (cx, cy, area, cnt_shifted) in enumerate(blobs, start=1):

                # Fill bán trong suốt lên vùng blob
                overlay = crop.copy()
                cv2.drawContours(overlay, [cnt_shifted], -1, (0, 255, 255), cv2.FILLED)
                cv2.addWeighted(overlay, 0.25, crop, 0.75, 0, crop)

                # Viền contour màu vàng
                cv2.drawContours(crop, [cnt_shifted], -1, (0, 255, 255), 2)

                # Crosshair tại tâm blob
                cs = 6
                cv2.line(crop, (cx - cs, cy), (cx + cs, cy), (0, 200, 255), 2)
                cv2.line(crop, (cx, cy - cs), (cx, cy + cs), (0, 200, 255), 2)

                # Label diện tích
                cv2.putText(
                    crop, f"B{b_idx} {area:.0f}px",
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
                )

            status = f"{len(blobs)} blob(s)" if blobs else "NO BLOB"
            print(f"  [{file_name}] crop_{crop_idx} ROI_{roi_idx}: {status}")

        # =========================
        # Lưu ảnh
        # =========================
        save_name = (
            f"{os.path.splitext(file_name)[0]}"
            f"_crop_{crop_idx}"
            f"_conf_{conf:.2f}.jpg"
        )
        save_path = os.path.join(save_dir, save_name)

        if cv2.imwrite(save_path, crop):
            print(f"  Saved: {save_path}")
        else:
            print(f"  Failed: {save_path}")

        crop_idx += 1

    print(f"Processed: {file_name}")

print("Done!")