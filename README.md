# Vision AI — SIBV / A27 (train)

Repo huấn luyện & kiểm tra lỗi bề mặt con hàng (segmentation UNet + YOLO detect/seg + đo đạc OpenCV).

## 🗂️ Cấu trúc thư mục

```
src/
├── data/      Chuẩn bị & chia dữ liệu
│              chia_data*.py (chia train/val), synth_defect.py (sinh lỗi giả),
│              createJsonAndFolder.py (LabelMe→folder), count_classes.py (đếm phân bố lớp)
├── train/     Huấn luyện
│              train_unet.py (segmentation lỗi nhỏ), train_seg.py, train_obd.py, train-cls.py
├── predict/   Dự đoán / suy luận
│              predict_unet.py, predict_seg.py, predict_obd.py
├── crop/      Cắt & định vị con hàng (A26 · A27)
│              cropa27*.py, crop_a26.py, cropa26opencv.py
├── measure/   Đo đạc OpenCV (caliper, vùng)
│              test_a26.py, detect_zones.py
└── debug/     Soi / xem kết quả UNet
               _diag_unet.py, _view_unet.py, _zoom_unet.py

scripts/   File .sh chạy thí nghiệm (run_exp_A/B, run_smoke, _run_diag)
docs/      Ghi chú học tập (HOC_*.md, NOTES_*.md) — ĐỌC ĐẦU BUỔI
configs/   File cấu hình yaml (data.yaml)
weights/   Model pretrained (.pt) — không commit lên git
tools/     Tiện ích lặt vặt (test.py: validate nhãn; test_crop.py)

SIBV/                         Dữ liệu thật (ảnh + mask). data_imgs_unet_1 = bộ đầy đủ.
runs/ logs/ predict_out/      Output khi chạy — không commit (đã gitignore)
```

> Máy chạy Python/OpenCV qua **WSL conda env `vision_ai`** (không có python trên Windows).
> Gọi script: viết `.sh` rồi `wsl -e bash /mnt/d/.../scripts/xxx.sh` **qua PowerShell**.

## 🚀 Lệnh hay dùng (chạy từ gốc repo trong WSL)

```bash
# Đếm phân bố lớp trong dataset (soi mất cân bằng / data thủng)
python src/data/count_classes.py SIBV/A27/data_imgs_unet_1

# Train UNet lỗi nhỏ (đề mới: FocalTversky+Focal, chọn best.pt theo F1 object-level)
python src/train/train_unet.py \
  --data SIBV/A27/data_imgs_unet_1/dataset.yaml \
  --loss ftl_focal --best-metric f1_object --name exp

# Ablation: đổi ĐÚNG 1 biến mỗi lần
#   A: --loss dice_ce  --arch Unet          (baseline)
#   B: --loss ftl_focal --arch Unet          (đổi loss)
#   C: --loss ftl_focal --arch UnetPlusPlus  (đổi arch)
```

### Công tắc quan trọng của `train_unet.py`
| Cờ | Ý nghĩa |
|---|---|
| `--loss` | `dice_ce` (baseline) hoặc `ftl_focal` (đề mới cho lỗi nhỏ) |
| `--best-metric` | `iou_pixel` \| `recall_object` \| `f1_object` — tiêu chí lưu best.pt |
| `--tv-alpha/-beta/-gamma` | núm Tversky (β cao → ít bỏ sót → recall ↑) |
| `--focal-gamma` | núm Focal (dồn sức vào điểm ảnh khó) |
```
