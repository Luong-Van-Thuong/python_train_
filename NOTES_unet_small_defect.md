# Ghi chú: Tối ưu UNet cho nhận diện LỖI NHỎ (6×6 px) vs object thường

> Bối cảnh dữ liệu hiện tại:
> - Tile: **512×512** (chia_data_unet.py --tile 512, overlap 0.20)
> - Encoder: **resnet34** (giảm mẫu 32×)
> - Lỗi mục tiêu: **6×6 px = 36 px ≈ 0.014%** diện tích tile
> - min_defect_px = 10 (giữ được lỗi 6×6), bg_ratio = 2.0, ok_tiles_per_img = 10
> - File train: `train_unet.py` | File chia data: `chia_data_unet.py`
> - Nhánh git: `feature/seg-tune-recall-fp` (đang tune recall & giảm false-positive)

## Kết luận: train_unet.py CHƯA tối ưu cho lỗi 6×6 px

Có 3 điểm yếu cốt lõi khiến lỗi nhỏ dễ bị bỏ sót, cộng vài điểm phụ.

---

## 1. Encoder nuốt mất vật thể nhỏ  (VẤN ĐỀ NẶNG NHẤT)

- `resnet34` giảm mẫu **32×**: ở đáy bottleneck 512 → 16.
- Lỗi 6 px → chỉ còn **6/32 ≈ 0.19 px → biến mất hoàn toàn** ở tầng sâu.
- UNet vẫn tái tạo được nhờ **skip-connection tầng nông** (stride 2–4) giữ lại đặc trưng 6 px,
  nhưng toàn bộ ngữ cảnh sâu (nơi mạng "hiểu" vật thể) gần như vô dụng với lỗi cỡ này.
- Đây là lý do object lớn train dễ, còn lỗi 6×6 px rất khó.

**Nâng cấp đề xuất:**
- Đổi kiến trúc sang **`UnetPlusPlus`** (dense skip → giữ vật thể nhỏ tốt hơn Unet thuần), hoặc **`MAnet`**.
- Cân nhắc encoder có dilation, hoặc `encoder_depth=4` + `decoder_channels` tương ứng để bớt mất phân giải.
- **Giữ train ở phân giải gốc** (đang tile 512 native — TỐT). Tuyệt đối **không resize nhỏ** ảnh đầu vào.

---

## 2. Loss chưa chống được mất cân bằng cực đoan

Hiện tại: `Dice + CE` (CE có class_weights = [0.2, 2.0, 2.0, 1.5, 2.0]).

Vấn đề với vật thể 36 px:
- **Dice cực kỳ bất ổn**: lệch 1–2 px là điểm số nhảy loạn → gradient nhiễu.
- **CE weight chỉ làm mềm**, không tập trung vào pixel khó/hiếm.

**Nâng cấp đề xuất (khớp mục tiêu nhánh `seg-tune-recall-fp`):**
- Thêm **Focal Loss** (thay hoặc kèm CE) để tập trung pixel khó, hiếm.
- Đổi Dice → **Focal Tversky Loss** với `alpha < beta` (vd alpha=0.3, beta=0.7)
  để **phạt nặng False Negative → tăng RECALL** đúng mục tiêu.
  - alpha phạt FP, beta phạt FN. Muốn tăng recall thì beta cao.
  - Nếu FP nhiều quá thì kéo alpha lên dần để cân bằng.

---

## 3. Chọn checkpoint sai tiêu chí

- `best.pt` đang chọn theo **defect IoU per-pixel**.
- Với lỗi 6 px, lệch 1 px làm IoU rớt thảm → dễ **loại nhầm model có recall tốt**.

**Nâng cấp đề xuất:**
- Chọn `best.pt` theo **F1 / Recall ở mức blob (object-level)** thay vì IoU per-pixel.
- Object-level = đếm lỗi được phát hiện đúng (theo cụm/blob), không đếm từng pixel.
- Đúng tinh thần tune recall–FP của nhánh này.

---

## Điểm phụ (ưu tiên thấp hơn)

- `min_defect_px=10` < 36 → OK, không lọc nhầm lỗi 6×6.
- Augmentation hiện tại (flip / rot90 / brightness) **an toàn cho lỗi nhỏ**.
  - rot90 là lossless (không nội suy) → không phá lỗi.
  - **KHÔNG thêm** scale-down / elastic / rotate góc lẻ → sẽ phá lỗi 6 px.
- `bg_ratio=2.0` → 2× tile nền, làm **loãng tín hiệu dương**.
  - Cân nhắc `WeightedRandomSampler` ưu tiên tile có lỗi khi train.
- Thiếu **gradient clipping** và **early-stopping** (nhỏ, không cấp thiết).
- AMP (mixed precision) đang bật — OK, không ảnh hưởng lỗi nhỏ.

---

## Gói triển khai đề xuất (khi tiếp tục)

Ưu tiên cao, ít rủi ro:
1. `arch = UnetPlusPlus` (đổi 1 tham số, giữ nguyên phần còn lại để so sánh).
2. Thay loss → **FocalTversky (beta cao) + Focal**, tham số hoá qua argparse.
3. Thêm metric **Recall / F1 object-level**, chọn `best.pt` theo đó.

Lựa chọn khi quay lại:
- Làm cả 3 gói, HOẶC
- Chỉ làm phần loss + metric (giữ nguyên kiến trúc resnet34/Unet để có baseline so sánh).

---

## Nền tảng lý thuyết để tìm hiểu thêm

- **Vì sao object nhỏ khó**: downsampling stride làm mất phân giải không gian;
  vật thể < stride của tầng sâu thì mất tín hiệu. Tra cứu: "small object segmentation",
  "receptive field vs object size", "output stride / dilated convolution".
- **Loss cho imbalance**: Tversky Loss (2017), Focal Tversky Loss (2018), Focal Loss (RetinaNet 2017).
- **Metric object-level**: connected-components / blob matching, recall–precision theo object,
  thay vì IoU per-pixel (IoU per-pixel phạt vật thể nhỏ rất nặng khi lệch biên).
- **Kiến trúc giữ vật thể nhỏ**: UNet++ (dense skip), HRNet (giữ high-resolution suốt mạng),
  feature pyramid, giảm encoder_depth.
