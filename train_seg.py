# -*- coding: utf-8 -*-
"""
train_seg.py
============
Train YOLO11-seg trên dataset tile đã tạo bởi chia_data.py.

Cài đặt (trong WSL2, nên dùng venv):
    pip install ultralytics

Chạy:
    python3 train_seg.py
    # hoặc tuỳ chỉnh:
    python3 train_seg.py --model yolo11s-seg.pt --epochs 200 --batch 16

Lưu ý về defect NHỎ:
    - imgsz để ĐÚNG bằng kích thước tile (640) -> không bị thu nhỏ thêm.
    - copy_paste / mosaic giúp tăng đa dạng khi dữ liệu ít.
    - yolo11n-seg = nhẹ/nhanh; yolo11s-seg = chính xác hơn (khuyến nghị nếu có GPU).
"""

import argparse
from ultralytics import YOLO

DEFAULT_DATA = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A26/data_imgs/seg/data.yaml"
DEFAULT_PROJECT = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A26/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA, help="Đường dẫn data.yaml")
    ap.add_argument("--model", default="yolo11s-seg.pt",
                    help="yolo11n-seg.pt (nhẹ) | yolo11s-seg.pt | yolo11m-seg.pt")
    ap.add_argument("--imgsz", type=int, default=640, help="Bằng kích thước tile")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8,
                    help="VRAM 8GB + seg -> để 8. Giảm còn 4 nếu CUDA out of memory")
    ap.add_argument("--workers", type=int, default=4,
                    help="Số luồng nạp dữ liệu. Để 2-4 vì WSL2 ít RAM (tránh OOM sập VM)")
    ap.add_argument("--device", default="0", help="'0' = GPU0, 'cpu' nếu không có GPU")
    ap.add_argument("--project", default=DEFAULT_PROJECT,
                    help="Thư mục gốc lưu kết quả train")
    ap.add_argument("--name", default="defect_seg",
                    help="Tên run; kết quả nằm ở <project>/<name>")
    args = ap.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        cache=False,         # KHÔNG cache vào RAM -> tránh tràn RAM sập WSL2

        # --- Augmentation: mạnh tay vì dữ liệu ít + defect nhỏ ---
        mosaic=1.0,          # ghép 4 ảnh -> đa dạng bối cảnh
        close_mosaic=10,     # tắt mosaic 10 epoch cuối để ổn định
        copy_paste=0.3,      # copy-paste defect (rất hợp instance-seg, ít data)
        fliplr=0.5,
        flipud=0.5,          # defect không có chiều "đúng" -> lật dọc OK
        degrees=10.0,        # xoay nhẹ
        scale=0.5,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,  # đổi sáng/màu cho robust ánh sáng

        # --- Khác ---
        patience=50,         # early stopping nếu 50 epoch không cải thiện
        plots=True,
        # rect=False, cache=True,  # bật cache nếu RAM dư để train nhanh hơn
    )

    # Đánh giá trên tập val sau khi train
    metrics = model.val()
    print("mAP50-95 (mask):", metrics.seg.map)
    print("mAP50 (mask):   ", metrics.seg.map50)


if __name__ == "__main__":
    main()
