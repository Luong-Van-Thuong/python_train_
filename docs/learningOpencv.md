# Tư duy xây pipeline OpenCV cho AOI

> Ghi chú học tập — mục tiêu: tự hình thành pipeline mà không cần dò dẫm.
> Ví dụ tham chiếu xuyên suốt: `D:\Projects_\Cong_Ty\Python_\train\src\measure\probe_vungtu_cv.py`
> (script sinh ra các panel debug trong `D:\Images_\JeaYoung\MLCC\Data\vung_tu_debug`).

---

## 1. OpenCV thật ra chỉ có **6 loại phép**

Không cần nhớ 2500 hàm. Phân loại theo **dữ liệu vào → dữ liệu ra**, chỉ có 6 nhóm:

| # | Loại | Vào → Ra | Hàm tiêu biểu |
|---|---|---|---|
| 1 | **Điểm** | 1 pixel → 1 pixel | `threshold`, `inRange`, `LUT`, `convertScaleAbs`, `equalizeHist`, `cvtColor` |
| 2 | **Lân cận (lọc)** | vùng quanh pixel → 1 pixel | `GaussianBlur`, `medianBlur`, `Sobel`, `Laplacian`, `Canny`, `boxFilter`, `morphologyEx`, `bilateralFilter` |
| 3 | **Hình học** | ảnh → ảnh biến dạng | `resize`, `warpAffine`, `warpPerspective`, `remap`, `rotate` |
| 4 | **Trích cấu trúc** | ảnh → danh sách đối tượng | `findContours`, `connectedComponentsWithStats`, `HoughLines/Circles`, `goodFeaturesToTrack`, `ORB/SIFT` |
| 5 | **Đo** | đối tượng → con số | `boundingRect`, `minAreaRect`, `contourArea`, `moments`, `fitEllipse`, `arcLength`, `matchShapes` |
| 6 | **Kết hợp** | 2 ảnh → 1 ảnh | `absdiff`, `bitwise_and/or/xor`, `addWeighted`, `matchTemplate`, `phaseCorrelate` |

Hết. Mọi pipeline AOI trên đời chỉ là xâu chuỗi 6 nhóm này.

---

## 2. Mọi pipeline đều là **cái phễu giảm chiều**

Đây là điều quan trọng nhất. Mục tiêu luôn là ép dữ liệu từ triệu con số xuống 1 quyết định:

```
Ảnh (H×W×3, ~3 triệu số)
  ↓ ① CHUẨN HOÁ      — bỏ biến thiên không liên quan
  ↓ ② TĂNG TƯƠNG PHẢN — biến "thuộc tính tôi quan tâm" thành ĐỘ SÁNG   ← bước sáng tạo
  ↓ ③ NHỊ PHÂN HOÁ    — mask 0/1
  ↓ ④ MORPHOLOGY      — vá lỗ, diệt nhiễu, tách dính
  ↓ ⑤ ĐO              — contour → (x,y,w,h,score) → gate → OK/NG
```

**Người mới hầu như luôn hỏng ở chỗ nhảy thẳng từ ① sang ③** — threshold ngay ảnh gray rồi
ngồi tuning ngưỡng cả ngày không ra. Bước ② mới là nơi quyết định thắng thua: nếu ② làm đúng,
vật thể **tự sáng bật lên** và threshold nào cũng chạy được.

---

## 3. Bảng tra bước ②: "thuộc tính vật lý → toán tử"

Đọc theo cột trái ("mắt tôi phân biệt vật này nhờ cái gì?"), lấy cột phải:

| Vật thể khác nền nhờ… | Toán tử biến nó thành độ sáng |
|---|---|
| **Nét hơn / kém nhoè** (bài out-focus) | `Sobel`→`magnitude`→`boxFilter`; hoặc `Laplacian().var()` cục bộ |
| Có **cạnh sắc** | `Canny`, `Scharr`, `morphologyEx(GRADIENT)` |
| **Sáng/tối tuyệt đối** | `threshold` + ngưỡng percentile |
| **Sáng/tối so với xung quanh** (nền không đều) | `adaptiveThreshold`, hoặc `gray - GaussianBlur(gray, 51)` |
| **Đốm sáng nhỏ** (specular, bụi, hàn thiếc) | `morphologyEx(TOPHAT)` |
| **Vết lõm tối nhỏ** (nứt, void, xước) | `morphologyEx(BLACKHAT)` |
| **Nhám / có kết cấu** vs nền mịn | local std = `sqrt(boxFilter(I²) − boxFilter(I)²)` |
| **Màu riêng** | `cvtColor(HSV/Lab)` → `inRange` kênh H hoặc a/b |
| **Khác với mẫu chuẩn** | căn ảnh rồi `absdiff` với golden template |
| **Vân/chu kỳ đều** | `Gabor`, `dft` |
| **Hình dạng đã biết** | `matchTemplate`, `HoughCircles/Lines`, `matchShapes` |
| **Ở vị trí cố định giữa các ảnh** | `phaseCorrelate` hoặc ORB + `findHomography` để căn |

### Hai dòng đáng giá nhất cho bài MLCC trên tray (chưa dùng)

- **Căn ảnh + `absdiff` golden template** — tray lặp lại đều, đây là vũ khí kinh điển của AOI,
  thường mạnh hơn hẳn "tối ∩ nét".
- **TOPHAT / BLACKHAT** — bắt lỗi bề mặt mà không cần biết nền sáng bao nhiêu.

---

## 4. Quy trình tự sinh pipeline (làm đúng thứ tự này)

1. **Mở ảnh, viết ra MỘT CÂU tiếng Việt**:
   "mắt tôi nhận ra con tụ vì nó ______ hơn xung quanh."
2. Tra bảng mục 3 → ra 1-2 toán tử.
3. **Chạy toán tử đó và NHÌN ẢNH OUTPUT.** Vật thể có sáng bật lên không?
   - Không → **giả thuyết sai**, quay lại bước 1 viết câu khác. Đừng tuning tham số.
   - Có → sang bước 4.
4. Giờ mới threshold — dùng **percentile**, đừng dùng số cứng, để tự thích nghi ánh sáng.
5. Morphology dọn: `CLOSE` vá lỗ trước, `OPEN` diệt nhiễu sau.
6. `findContours` → đo → **gate cứng bằng shape prior** (tỉ lệ w/h, diện tích)
   + **score mềm** để xếp hạng.

> **Nguyên tắc:** mỗi bước phải xem được bằng mắt. Không nhìn thì không phải làm vision, là đoán mò.

### Vài quy ước nhỏ nên theo

- **Chuẩn hoá kích thước xử lý** (vd `WORK_H = 800`) trước khi lọc, để kernel `25×25`, `9×9`…
  có ý nghĩa cố định bất kể resolution ảnh gốc. Nhớ chia lại tỉ lệ khi trả toạ độ về ảnh gốc.
- **Ngưỡng theo percentile** thay vì giá trị tuyệt đối → chống drift ánh sáng giữa các tray.
- **Chuẩn hoá điểm số trong phạm vi TỪNG ẢNH** (`norm01`), vì ta so sánh các ứng viên cùng ảnh,
  không so tuyệt đối liên ảnh.
- **Đo năng lượng trên ĐƯỜNG BAO, không phải trong lòng**: vẽ vành khung dày ~9px quanh box rồi
  lấy mean gradient trên vành đó — mẹo tốt để chấm "viền có sắc nét không".
- Nếu bài toán biết trước "mỗi ảnh đúng 1 vật", khi không ứng viên nào qua gate thì **vẫn chọn
  thằng điểm cao nhất** thay vì trả rỗng — không thì thống kê bị méo.

---

## 5. Soi lại `probe_vungtu_cv.py` — nó chính là khung trên

| Tầng | Trong file |
|---|---|
| ① Chuẩn hoá | `WORK_H=800` resize (dòng 91-92) |
| ② Tăng tương phản | `Sobel`→`magnitude`→`boxFilter` = `E` (98-101) — giả thuyết: *"con tụ NÉT hơn nền"* |
| ③ Nhị phân | 3 mask percentile p82 / p22 / p99 (102-108) |
| ④ Morphology | `dilate` + `AND` + `CLOSE` + `OPEN` (111-114) |
| ⑤ Đo | `findContours` → `boundingRect` → gate + score (119-151) |

Khung đã có sẵn. Cái thiếu chỉ là **kho giả thuyết ở bước ②** — tức bảng mục 3.

### Bài học từ kết quả IoU = 0.18

Nhìn panel `Cam0_0000_..._steps.png`: ô 5 (`dark_mask`) nuốt luôn cả mảng nền tối bên trái,
ô 8 cho thấy box đỏ chỉ ôm được nửa trái con tụ.

→ Nghĩa là **câu giả thuyết sai**, không phải tham số sai. "Tối ∩ nét" không tách được thân tụ
khỏi bóng nền. Theo quy trình mục 4, đúng ra phải quay lại bước 1 viết câu khác, chứ không phải
chỉnh p22 thành p18.

---

## 6. Tách logic khỏi debug — chìa khoá làm nhanh

Trong `probe_vungtu_cv.py`, hàm `detect()` **không in gì, không vẽ gì** — nó chỉ trả về
`(box, dbg)` với `dbg` là dict thu gom toàn bộ ảnh trung gian (dòng 116-117). Việc vẽ tách hẳn
sang `render_debug()`. Nhờ vậy bật/tắt debug bằng `--no-debug` không đụng tới logic.

Kỹ thuật ghép panel (dòng 160-222), dùng lại được cho mọi bài:

- `_bgr()` — đưa mọi thứ về BGR 3 kênh để `hstack` được. 3 trường hợp: float thì
  `normalize` về 0-255, mask 0/1 thì `×255`, gray thì convert thẳng.
  Bản đồ liên tục (như `E`) nên `applyColorMap(JET)` cho dễ đọc.
- `_title()` — dải đen cao 26px + `putText` vàng, `vstack` lên trên tile. Tự dán nhãn để khỏi
  phải nhớ ô nào là ô nào.
- Ghép lưới — resize **mọi tile về đúng cùng bề rộng** (giữ tỉ lệ) → `copyMakeBorder` pad đen
  cho bằng chiều cao lớn nhất → `hstack` n ô/hàng → `vstack` các hàng.
  Đây là pattern chuẩn để lưới không vỡ khi ảnh nguồn khác kích thước.

Ô áp chót và ô cuối là nơi đọc kết luận:
- **Ô ứng viên**: vàng = rớt gate, cyan = qua gate, **đỏ = được chọn**
  → phân biệt được "sai vì không sinh ra ứng viên đúng" hay "sinh ra rồi nhưng chấm điểm chọn nhầm".
- **Ô cuối**: **xanh lá = GT, đỏ = pred**, kèm IoU trên tiêu đề.

---

## 7. Giảm thời gian

**Thời gian phát triển** — thứ tiết kiệm nhiều nhất là *một khung debug panel tái sử dụng*.
`render_debug()` hiện đang hard-code 9 ô cho đúng bài này. Tách thành helper dùng chung:

```python
dbg = StepRecorder()
dbg.add("gray", gray)
dbg.add("E", E, cmap=cv2.COLORMAP_JET)
...
dbg.save(path)     # tự lo resize / pad / ghép lưới n ô
```

Mỗi giả thuyết mới khi đó chỉ tốn ~5 phút thay vì 1 giờ.
Thêm `cv2.createTrackbar` để kéo ngưỡng realtime thì tốc độ thử nghiệm tăng thêm mấy lần nữa.

**Thời gian chạy (cycle time)** — dùng OpenCV để *thu hẹp việc cho AI*:
OpenCV khoanh ROI thô rẻ tiền → chỉ chạy YOLO trên crop nhỏ.
Cộng thêm cổng đo độ nét `Laplacian().var()` per-crop để loại ảnh out-focus **trước khi** tốn
inference (đã nằm trong kế hoạch 2-stage).

---

## 8. Việc tiếp theo có thể làm

- **(a)** Viết `StepRecorder` — khung debug panel dùng chung + tuner trackbar.
- **(b)** Thử lại vùng tụ bằng giả thuyết khác (golden-template + `absdiff`, hoặc `BLACKHAT`),
  xem IoU có nhảy khỏi 0.18 không.
- **(c)** Đi sâu một mục trong bảng mục 3.
