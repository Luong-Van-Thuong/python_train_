# -*- coding: utf-8 -*-
"""
predict_seg.py
==============
Test / inference trên ẢNH GỐC độ phân giải cao trong môi trường WSL2.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ==============================================================================
# CONFIGURATION ZONE - ÔNG THAY ĐỔI ĐƯỜNG DẪN TEST TẠI ĐÂY
# ==============================================================================
# Biến chứa đường dẫn ảnh test riêng, dễ dàng thay đổi khi debug trong WSL2
DEBUG_SOURCE = "/mnt/d/Images_/SIBV/A26/260615_0/tesst/Image__2026-06-16__11-50-50_obj_0.bmp"

DEFAULT_WEIGHTS = "sibv/a26/result/defect_seg/weights/best.pt"
DEFAULT_OUT = "/mnt/d/Projects_/Cong_Ty/Python_/predict_out/folder_data_AI_tra_ve"
IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
# ==============================================================================


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img, ext=".png"):
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok


def gather_sources(source):
    p = Path(source)
    if p.is_dir():
        return [f for f in sorted(p.iterdir()) if f.suffix.lower() in IMG_EXTS]
    elif p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [p]
    return []


def main():
    ap = argparse.ArgumentParser()
    # Bỏ required=True để không ép buộc gõ từ Terminal nếu đã cấu hình biến DEBUG_SOURCE
    ap.add_argument("--source", default=None, help="Đường dẫn ảnh hoặc thư mục ảnh test")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Đường dẫn file best.pt")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Thư mục xuất kết quả")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # BIỆN PHÁP DỰ PHÒNG: Nếu Terminal không truyền --source, bốc ngay biến DEBUG_SOURCE ra dùng
    source_target = args.source if args.source is not None else DEBUG_SOURCE
    
    if not source_target:
        print("[LỖI CHÍ MẠNG] Nguồn ảnh trống! Hãy cấu hình DEBUG_SOURCE hoặc truyền --source.")
        return

    weight_path = Path(args.weights)
    if not weight_path.exists():
        print(f"[LỖI CHÍ MẠNG] Không tìm thấy file Weights tại: {weight_path.resolve()}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Khởi tạo model SAHI trên cấu hình target: {source_target}")
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weight_path),
        confidence_threshold=args.conf,
        image_size=args.tile,
        device=args.device,
    )

    files = gather_sources(source_target)
    if not files:
        print(f"[LỖI] Không tìm thấy ảnh hợp lệ tại: {source_target}")
        return
    print(f"[INFO] Tìm thấy {len(files)} ảnh để xử lý.")

    for f in files:
        img = imread_unicode(f)
        if img is None:
            print(f"[BỎ QUA] Lỗi đọc dữ liệu ảnh: {f.name}")
            continue

        result = get_sliced_prediction(
            image=img[:, :, ::-1],
            detection_model=model,
            slice_height=args.tile,
            slice_width=args.tile,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            postprocess_type="NMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
            verbose=0,
        )

        preds = result.object_prediction_list
        print(f"-> {f.name}: Phát hiện {len(preds)} defect.")

        vis = img.copy()
        lines = []
        for o in preds:
            x1, y1, x2, y2 = map(int, o.bbox.to_xyxy())
            name = o.category.name
            conf = o.score.value
            
            color = (0, 0, 255) if name == "thieu_nhua" else (0, 165, 255)
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{name} {conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if o.mask is not None:
                seg = o.mask.segmentation
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(vis, [pts], True, color, 1)
                    
            lines.append(f"{name} {conf:.4f} {x1} {y1} {x2} {y2}")

        imwrite_unicode(out_dir / f"{f.stem}_pred.png", vis, ".png")
        with open(out_dir / f"{f.stem}.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    print(f"\n[THÀNH CÔNG] Kết quả xuất tại: {out_dir}")


if __name__ == "__main__":
    main()