# -*- coding: utf-8 -*-
"""
chia_data_kolektorsdd2.py
==========================
Adapter: KolektorSDD2 (ảnh + mask _GT.png, train/test đã chia sẵn) -> format mà
train_kolektorsdd2.py đọc (giống train_unet.py: images/, masks/, dataset.yaml).

KHÔNG dùng chia_data_unet.py của repo chính vì script đó đọc nhãn labelme .json
(vẽ tay polygon) — KolektorSDD2 đã cho sẵn mask pixel rồi, đi qua .json là vòng thừa.

Việc script này làm, cho mỗi ảnh:
    1. Tìm mask cùng id (KolektorSDD2 đặt tên "{id}_GT.png", cần đổi thành "{id}.png"
       để khớp quy ước img_dir/mask_dir cùng stem của SegDataset).
    2. Resize CẢ ảnh và mask về 1 kích thước cố định, CHIA HẾT 32 (bắt buộc cho UNet
       5 tầng downsample /2 — xem HOC_unet_loi_nho_6px.md Bài 2). Ảnh KolektorSDD2
       không đồng nhất kích thước (đã đo thật: ~228-231 x 633-645), không chia hết 32.
    3. Remap mask 0/255 -> 0/1 (train_unet.py đọc mask như MÃ LỚP trực tiếp, không
       phải ảnh xám 0-255).

CHIA SPLIT (sửa 2026-07-21 — bản cũ SAI):
    Bản cũ map thẳng test/ gốc (bộ test CHÍNH THỨC của KolektorSDD2, dùng để đánh giá
    công bằng) thành "val" của train_kolektorsdd2.py -- mà "val" lại được dùng để CHỌN
    best.pt trong lúc train. Nghĩa là bộ test "sạch" đã bị dùng để chọn model, rồi nếu
    sau đó lại lấy ảnh trong train/val ra chạy inference thử thì không còn ý nghĩa đánh
    giá tổng quát hóa nữa (model đã thấy/được chọn dựa trên đúng ảnh đó).

    Bản sửa: CHỈ tách train/ gốc (2332 ảnh) thành 2 phần train + val (val dùng để chọn
    best.pt, tỉ lệ mặc định 90/10, --seed cố định để tách lại y hệt nếu chạy lại). Còn
    test/ gốc (1004 ảnh) giữ nguyên thành 1 split "test" riêng -- train_kolektorsdd2.py
    KHÔNG đọc key "test" trong dataset.yaml nên sẽ không đụng tới trong lúc train. Chỉ
    dùng test/ để đánh giá CUỐI CÙNG, 1 lần, sau khi đã chọn xong best.pt bằng val.

Chạy:
    python chia_data_kolektorsdd2.py
    python chia_data_kolektorsdd2.py --src "C:/THUONG/Images/KolektorSDD2" --size 224x640 --val-ratio 0.1
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

import sys as _sys, pathlib as _pl
_sys.path.insert(0, next(str(_p) for _p in _pl.Path(__file__).resolve().parents if (_p / "pathfix.py").exists()))
from pathfix import P

DEFAULT_SRC = P("C:/THUONG/Images/KolektorSDD2")
DEFAULT_OUT = Path(__file__).parent / "data_kolektorsdd2"
# KolektorSDD2 ảnh thật ~229x645 (width x height), không đồng nhất, không chia hết 32.
# 224x640 = gần nhất chia hết 32 (224/32=7, 640/32=20), méo ảnh không đáng kể.
DEFAULT_SIZE = "224x640"  # width x height
DEFAULT_VAL_RATIO = 0.2  # phần trăm của train/ gốc dành ra làm val (chọn best.pt)
DEFAULT_SEED = 42  # cố định để tách train/val lại y hệt nếu chạy lại script


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, img, ext=".png"):
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok


def list_images(src_dir):
    return sorted(p for p in src_dir.glob("*.png") if "_GT" not in p.stem)


def process_list(img_paths, src_dir, img_out_dir, mask_out_dir, w, h, desc):
    """Trả về (n_defect, n_bg) đã xử lý cho đúng danh sách img_paths truyền vào."""
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    n_defect = n_bg = 0

    for img_path in tqdm(img_paths, desc=f"Xử lý {desc}"):
        gt_path = src_dir / f"{img_path.stem}_GT.png"

        # --- THÊM ĐOẠN NÀY ĐỂ BỎ QUA FILE NẾU THIẾU MASK ---
        if not gt_path.exists():
            print(f"\n[CẢNH BÁO] Không tìm thấy file mask tương ứng: {gt_path.name} -> Bỏ qua cặp ảnh này.")
            continue
        # --------------------------------------------------

        img = imread_unicode(img_path, cv2.IMREAD_COLOR)
        mask_raw = imread_unicode(gt_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask_raw is None:
            print(f"[CẢNH BÁO] Thiếu cặp ảnh/mask cho {img_path.name}, bỏ qua.")
            continue

        img_r = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        # INTER_NEAREST bắt buộc cho mask: nội suy tuyến tính sẽ sinh giá trị lớp
        # không nguyên (vd 0.5), phá mã lớp.
        mask_r = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_cls = (mask_r > 127).astype(np.uint8)  # 0/255 -> 0/1 (mã lớp)

        imwrite_unicode(img_out_dir / f"{img_path.stem}.png", img_r)
        imwrite_unicode(mask_out_dir / f"{img_path.stem}.png", mask_cls)

        if mask_cls.any():
            n_defect += 1
        else:
            n_bg += 1

    return n_defect, n_bg


def main():
    ap = argparse.ArgumentParser(description="KolektorSDD2 -> format train_kolektorsdd2.py")
    ap.add_argument("--src", default=DEFAULT_SRC, help="Thư mục gốc KolektorSDD2 (có train/ và test/)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Thư mục dataset đầu ra")
    ap.add_argument("--size", default=DEFAULT_SIZE, help="WxH đích, chia hết 32. Vd 224x640")
    ap.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
                     help="Tỉ lệ tách ra từ train/ gốc làm val (chọn best.pt). Vd 0.1 = 10%%")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                     help="Seed tách train/val, cố định để tách lại y hệt nếu chạy lại script")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    assert w % 32 == 0 and h % 32 == 0, f"--size {args.size} phải chia hết 32 (UNet 5 tầng downsample)"
    assert 0.0 < args.val_ratio < 1.0, f"--val-ratio {args.val_ratio} phải nằm trong (0, 1)"

    src = Path(P(args.src))
    out = Path(args.out)
    if not src.exists():
        print(f"[LỖI] Không tìm thấy {src}")
        return

    train_src_dir = src / "train"
    test_src_dir = src / "test"
    if not train_src_dir.exists():
        print(f"[LỖI] Không tìm thấy {train_src_dir}")
        return

    # --- Tách train/ gốc thành train thật + val (val dùng để chọn best.pt) ---
    train_imgs = list_images(train_src_dir)
    rng = random.Random(args.seed)
    rng.shuffle(train_imgs)
    n_val = max(1, round(len(train_imgs) * args.val_ratio))
    val_imgs = train_imgs[:n_val]
    train_imgs = train_imgs[n_val:]
    print(f"[INFO] train/ gốc {n_val + len(train_imgs)} ảnh -> tách seed={args.seed}: "
          f"train={len(train_imgs)}, val={len(val_imgs)} (val-ratio={args.val_ratio})")

    totals = {}
    n_defect, n_bg = process_list(train_imgs, train_src_dir,
                                   out / "images" / "train", out / "masks" / "train",
                                   w, h, desc="train")
    totals["train"] = (n_defect, n_bg)
    print(f"train: {n_defect} ảnh lỗi, {n_bg} ảnh nền (tổng {n_defect + n_bg})")

    n_defect, n_bg = process_list(val_imgs, train_src_dir,
                                   out / "images" / "val", out / "masks" / "val",
                                   w, h, desc="val")
    totals["val"] = (n_defect, n_bg)
    print(f"val  : {n_defect} ảnh lỗi, {n_bg} ảnh nền (tổng {n_defect + n_bg})")

    # --- test/ gốc: giữ nguyên, TÁCH RIÊNG, không đưa vào key "val" ---
    # train_kolektorsdd2.py chỉ đọc ds["images"]["train"] / ["val"], không đọc "test",
    # nên bộ này sẽ không bị đụng tới trong lúc train/chọn best.pt.
    if test_src_dir.exists():
        test_imgs = list_images(test_src_dir)
        n_defect, n_bg = process_list(test_imgs, test_src_dir,
                                       out / "images" / "test", out / "masks" / "test",
                                       w, h, desc="test (holdout, KHÔNG dùng lúc train)")
        totals["test"] = (n_defect, n_bg)
        print(f"test : {n_defect} ảnh lỗi, {n_bg} ảnh nền (tổng {n_defect + n_bg}) "
              f"-- HOLDOUT, chỉ dùng để đánh giá cuối cùng")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy {test_src_dir}, bỏ qua bộ test holdout.")

    dataset = {
        "path": str(out),
        "images": {"train": "images/train", "val": "images/val", "test": "images/test"},
        "masks": {"train": "masks/train", "val": "masks/val", "test": "masks/test"},
        "num_classes": 2,
        "names": {0: "background", 1: "defect"},
        "tile": [h, w],
    }
    with open(out / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset, f, allow_unicode=True, sort_keys=False)

    print("\n===== HOÀN TẤT =====")
    print(f"dataset.yaml: {out / 'dataset.yaml'}")
    print(f'Chạy train:   python train_kolektorsdd2.py --data "{out / "dataset.yaml"}"')
    print('LƯU Ý: dataset.yaml có thêm key "test" nhưng train_kolektorsdd2.py không đọc key này -- '
          'ảnh trong images/test, masks/test chỉ dùng khi bạn tự viết/chạy script đánh giá cuối cùng.')


if __name__ == "__main__":
    main()
