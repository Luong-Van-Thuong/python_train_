# 🗺️ Bản đồ học Machine Vision + Playbook làm dự án nhanh

> File tra cứu riêng, tách khỏi `HOC_unet_loi_nho_6px.md` (file đó học sâu về UNet lỗi nhỏ).
> Mục tiêu file này: có **bản đồ liền mạch** để hết ngợp, và **quy trình 3 ngày** để không phải nghĩ lại từ đầu mỗi dự án.

---

## 1. Vision chia làm 2 NHÁNH — biết chọn nhánh là kỹ năng số 1

```
BÀI TOÁN VISION
│
├── NHÁNH 1: OpenCV "thuần" (viết LUẬT, không AI)
│     ⏱️ nhanh, nhẹ, chạy được trên máy yếu, GIẢI THÍCH ĐƯỢC với khách
│     ✅ dùng khi lỗi/việc CÓ QUY LUẬT rõ:
│        - đo kích thước (mm), đếm số lượng, canh vị trí/tâm
│        - dò cạnh, dò tròn, so màu, ngưỡng sáng-tối
│        - căn (align) ảnh về chuẩn trước khi so
│     📌 VD trong repo: cropa27.py (dò tâm con hàng round/square)
│     👎 chịu thua khi: lỗi không có hình dạng cố định (xước, lem, rỗ mờ)
│
└── NHÁNH 2: Deep Learning (UNet / detection…)
      💪 mạnh, học được lỗi "mắt thấy mà không viết luật nổi"
      ✅ dùng khi lỗi KHÔNG có quy luật, ĂN VÀO NỀN, đa dạng
      👎 tốn: cần DATA + NHÃN (gán mask), cần GPU, chậm hơn, khó giải thích
      📌 VD: train_unet.py (segmentation lỗi nhỏ 6px)
```

**Quy tắc vàng chọn nhánh:**
> Viết được luật rõ ràng bằng lời trong 5 phút? → **OpenCV thuần trước** (rẻ, nhanh, dễ cãi với khách).
> Không viết nổi luật, nhưng có nhiều ảnh mẫu + nhãn? → **Deep Learning**.
> **Thực tế hay dùng CẢ HAI:** OpenCV lo cắt/căn/khoanh vùng con hàng → DL chỉ soi lỗi trong vùng đó. (Đúng pipeline A27: cropa27 cắt → UNet soi.)

---

## 2. Lộ trình học OpenCV thuần (Nhánh 1) — học theo thứ tự này

| Bước | Chủ đề | Làm được gì | Hàm cv2 lõi |
|---|---|---|---|
| 1 | Ảnh là mảng số | hiểu ảnh = ma trận HxWxC (BGR) | `imread`, `shape`, slicing |
| 2 | Màu & kênh | tách nền/vật theo màu | `cvtColor`, HSV, `inRange` |
| 3 | Ngưỡng (threshold) | biến ảnh xám → đen/trắng | `threshold`, `adaptiveThreshold`, Otsu |
| 4 | Hình thái học | dọn nhiễu, nối/tách vùng | `erode`, `dilate`, `morphologyEx` |
| 5 | Contour (đường bao) | đếm, đo diện tích/kích thước lỗi | `findContours`, `contourArea`, `boundingRect` |
| 6 | Cạnh & hình học | dò cạnh, đường, tròn | `Canny`, `HoughLines`, `HoughCircles` |
| 7 | Căn ảnh (align) | đưa ảnh về chuẩn để so sánh | `matchTemplate`, ORB/`findHomography`, `warpAffine` |
| 8 | So sánh / trừ nền | tìm điểm KHÁC so với mẫu OK | `absdiff`, so ảnh chuẩn |

> 👉 Với báo cáo "bao nhiêu mm phát hiện được": bước 5 (contour → đo px) + biết **1 px = bao nhiêu mm** (hiệu chuẩn/calibration) là ra con số. mm/px lấy từ 1 vật đã biết kích thước trong ảnh.

---

## 3. Lộ trình học Deep Learning vision (Nhánh 2)

| Bước | Chủ đề | Ghi chú |
|---|---|---|
| 1 | 3 mức bài toán | Classification / Detection / **Segmentation** (xem HOC_unet Bài 1) |
| 2 | Chuẩn bị data | cắt tile, gán nhãn (mask/box), chia train/val |
| 3 | Class imbalance | lỗi nhỏ = kẻ thù số 1 (HOC_unet Bài 3) |
| 4 | Loss | CE→Dice→Tversky→Focal (HOC_unet Bài 5) |
| 5 | Metric | IoU vs Recall/F1 object-level (HOC_unet Bài 6) |
| 6 | Kiến trúc | Unet / Unet++ / MAnet (HOC_unet Bài 7) |
| 7 | Train & đánh giá | đọc code, chạy ablation (HOC_unet Bài 8-9) |
| 8 | Suy luận (inference) | predict trên ảnh mới, ghép kết quả về ảnh gốc |

> Phần này bạn ĐANG học chi tiết trong `HOC_unet_loi_nho_6px.md`. File đó là bản phóng to của bước 3-9 ở đây.

---

## 4. 🏃 PLAYBOOK 3 NGÀY — quy trình khi có dự án mới

Đừng nghĩ lại từ đầu mỗi lần. Chạy theo checklist:

### NGÀY 0 (nửa ngày đầu) — HIỂU BÀI, đừng code vội
- [ ] Xem tận mắt **10-20 ảnh OK và 10-20 ảnh NG**. Lỗi trông thế nào? To/nhỏ? Có hình dạng cố định không? Ăn vào nền không?
- [ ] Hỏi khách/sếp: lỗi **nhỏ nhất bao nhiêu mm** cần bắt? Tỉ lệ bỏ sót cho phép? Tốc độ yêu cầu (ms/ảnh)?
- [ ] **Chọn nhánh** (mục 1): luật rõ → OpenCV; ăn nền/không luật → DL.
- [ ] Chốt **1 chỉ số nghiệm thu** duy nhất (VD: recall ≥ 95% ở lỗi ≥ 2mm).

### NGÀY 1 — DỰNG KHUNG chạy được đầu-cuối (dù xấu)
- [ ] Pipeline tối thiểu: đọc ảnh → xử lý → ra kết quả (đừng cầu toàn).
- [ ] OpenCV: căn ảnh → khoanh vùng → threshold/contour → đo.
- [ ] DL: cắt tile → gán nhãn 1 ít → train thử 1 model nhanh (baseline).
- [ ] Mục tiêu: **có số đo đầu tiên**, dù kém. Có số mới biết đường sửa.

### NGÀY 2 — VẶN cho đạt chỉ số nghiệm thu
- [ ] Nhìn **ca SAI** (bỏ sót + báo nhầm), không nhìn ca đúng.
- [ ] OpenCV: chỉnh ngưỡng, kernel, lọc theo diện tích/hình dạng.
- [ ] DL: đổi loss (β↑ recall), thêm data ca khó, thử Unet++ (ablation — đổi 1 thứ 1 lúc).
- [ ] Ghi lại **bảng số**: cỡ mm nào bắt được, cỡ nào rớt → chính là báo cáo.

### NGÀY 3 — CHỐT + BÁO CÁO
- [ ] Test trên bộ ảnh CHƯA từng thấy (không phải data đã chỉnh trên đó).
- [ ] Lập bảng: điều kiện (mm/ánh sáng/loại lỗi) → làm được / không.
- [ ] Viết rõ **giới hạn**: "bắt được lỗi ≥ X mm, dưới X mm không đảm bảo vì…".
- [ ] Đóng gói: script chạy + hướng dẫn + model/tham số.

> ⚠️ Bẫy hay chết: cầu toàn Ngày 1 (chưa có số đã lo tối ưu), và test trên chính data đã chỉnh (ảo tưởng giỏi). Luôn để dành ảnh "chưa thấy" để nghiệm thu.

---

## 5. Câu thần chú khi ngợp

> **"Chọn đúng nhánh > thuật toán xịn. Có số đo > cầu toàn. Nhìn ca sai > ca đúng. Test ảnh chưa thấy > ảnh đã chỉnh."**

Thiếu chỗ nào thì tra đúng mục trên, không cần nhớ hết. Bản đồ để **tra**, không để **thuộc lòng**.
