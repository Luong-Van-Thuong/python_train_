# -*- coding: utf-8 -*-
"""
test_kolektorsdd2.py
======================
Interface đánh giá CUỐI CÙNG trên bộ test holdout của KolektorSDD2 (mặc định:
C:/THUONG/Images/KolektorSDD2/test -- 1004 ảnh gốc, KHÔNG hề được đụng tới lúc
train/chọn best.pt, xem lời dặn trong chia_data_kolektorsdd2.py).

Khác predict_unet.py của công ty (chỉ suy luận + vẽ, dùng cho ảnh THẬT SỰ không có
nhãn): script này ĐỌC LUÔN mask "{id}_GT.png" có sẵn trong thư mục test/ để so với
dự đoán, tính đúng những metric mà evaluate() trong train_kolektorsdd2.py dùng để
chọn best.pt lúc train (IoU pixel lớp lỗi, Precision/Recall/F1 mức CỤC LỖI -- Bài 6),
rồi lưu ảnh overlay (đỏ=dự đoán, vàng=viền GT) để soi bằng mắt từng ảnh sai.

Ảnh KolektorSDD2 gốc không chia hết 32 nên model chỉ nhận đúng size cfg["tile"]
(giống lúc train, KHÔNG sliding-window như predict_unet.py) -- resize cả ảnh về
size đó, suy luận, rồi resize NGƯỢC pred về size gốc để so với GT gốc (không resize
GT xuống, tránh mất chi tiết lỗi nhỏ).

Chạy:
    python test_kolektorsdd2.py
    python test_kolektorsdd2.py --source "C:/THUONG/Images/KolektorSDD2/test" --conf 0.5
    python test_kolektorsdd2.py --limit 20          # thử nhanh 20 ảnh đầu
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
import torch
from tqdm import tqdm

import segmentation_models_pytorch as smp

import sys as _sys, pathlib as _pl
_sys.path.insert(0, next(str(_p) for _p in _pl.Path(__file__).resolve().parents if (_p / "pathfix.py").exists()))
from pathfix import P

DEFAULT_SRC = P("C:/THUONG/Images/KolektorSDD2/test")
DEFAULT_CFG = Path(__file__).parent / "results_260723_2" / "kolektorsdd2_unet" / "model_cfg.yaml"
DEFAULT_WEIGHTS = Path(__file__).parent / "results_260723_2" / "kolektorsdd2_unet" / "weights" / "best.pt"
DEFAULT_OUT = Path(__file__).parent / "results_260723_2" / "kolektorsdd2_unet" / "test_predictions_2"

COLOR_PRED = (0, 0, 255)   # đỏ: vùng model dự đoán là lỗi
COLOR_GT = (0, 255, 255)   # vàng: viền mask thật (ground truth)


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


def gather_sources(source):
    p = Path(source)
    if p.is_dir():
        return sorted(f for f in p.glob("*.png") if "_GT" not in f.stem)
    if p.is_file() and p.suffix.lower() == ".png" and "_GT" not in p.stem:
        return [p]
    return []


def object_stats(pred, gt, num_classes):
    """Đếm mức CỤC LỖI (blob), y hệt _object_stats() trong train_kolektorsdd2.py."""
    tp = fp = fn = 0
    for c in range(1, num_classes):
        gt_c = (gt == c).astype(np.uint8)
        pr_c = (pred == c).astype(np.uint8)
        if gt_c.sum() == 0 and pr_c.sum() == 0:
            continue
        n_gt, lbl_gt = cv2.connectedComponents(gt_c, connectivity=8)
        n_pr, lbl_pr = cv2.connectedComponents(pr_c, connectivity=8)
        for g in range(1, n_gt):
            if pr_c[lbl_gt == g].any():
                tp += 1
            else:
                fn += 1
        for pi in range(1, n_pr):
            if not gt_c[lbl_pr == pi].any():
                fp += 1
    return tp, fp, fn


@torch.no_grad()
def predict_one(model, img_bgr, mean, std, w, h, device, conf = 0.5):
    """Resize cả ảnh về đúng size lúc train (không sliding-window), suy luận,
    rồi resize NGƯỢC pred/prob về size gốc để so khớp với GT gốc."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H0, W0 = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    x = (resized.astype(np.float32) - mean) / std
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        logits = model(x)
    prob = torch.softmax(logits, dim=1)[0].float().cpu().numpy()  # (C, h, w)
    prob_full = np.stack(
        [cv2.resize(prob[c], (W0, H0), interpolation=cv2.INTER_LINEAR) for c in range(prob.shape[0])],
        axis=0,
    )

    pred = prob_full.argmax(0).astype(np.int32)
    prob_max = prob_full.max(0)
    if conf > 0:
        pred[prob_max < conf] = 0  
    else:
        pred[prob_max >= conf] = 1 
    return pred, prob_max


def render(img_bgr, pred, gt, prob_max, conf, min_area):
    vis = img_bgr.copy()
    overlay = vis.copy()

    cls_mask = ((pred == 1) & (prob_max >= conf)).astype(np.uint8)
    if min_area > 0 and cls_mask.sum() > 0:
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(cls_mask, 8)
        keep = np.zeros_like(cls_mask)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] >= min_area:
                keep[lbl == k] = 1
        cls_mask = keep

    if cls_mask.sum() > 0:
        overlay[cls_mask == 1] = COLOR_PRED
        cnts, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, COLOR_PRED, 1)

    if gt is not None and gt.any():
        gt_cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, gt_cnts, -1, COLOR_GT, 1)

    return cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)


def main():
    ap = argparse.ArgumentParser(description="Đánh giá cuối cùng trên bộ test holdout KolektorSDD2")
    ap.add_argument("--source", default=str(DEFAULT_SRC), help="Thư mục test (ảnh + *_GT.png)")
    ap.add_argument("--cfg", default=str(DEFAULT_CFG), help="model_cfg.yaml")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="best.pt")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Thư mục xuất ảnh overlay + log lỗi")
    ap.add_argument("--conf", type=float, default=0.4, help="Ngưỡng xác suất để VẼ overlay (không ảnh hưởng metric)")
    ap.add_argument("--min-area", type=int, default=0, help="Lọc blob nhỏ khi VẼ (không ảnh hưởng metric)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="Chỉ chạy N ảnh đầu (0 = chạy hết), để thử nhanh")
    ap.add_argument("--no-save-vis", dest="save_vis", action="store_false", default=True,
                     help="Không lưu ảnh overlay, chỉ tính metric (chạy nhanh hơn)")
    args = ap.parse_args()

    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        print(f"[LỖI] Không thấy model_cfg.yaml: {cfg_path}")
        return
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    w_path = Path(args.weights)
    if not w_path.exists():
        print(f"[LỖI] Không thấy weights: {w_path}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = int(cfg["num_classes"])
    mean = np.array(cfg["mean"], dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)
    tile_h, tile_w = cfg["tile"]  # lưu dạng [h, w] -- xem chia_data_kolektorsdd2.py

    print(f"[INFO] {cfg['arch']} + {cfg['encoder']} | classes={num_classes} | "
          f"model input = {tile_w}x{tile_h} (WxH) | device={device}")
    model = smp.create_model(cfg["arch"], encoder_name=cfg["encoder"], encoder_weights=None,
                              in_channels=3, classes=num_classes).to(device)
    model.load_state_dict(torch.load(w_path, map_location=device))
    model.eval()

    files = gather_sources(args.source)
    if not files:
        print(f"[LỖI] Không tìm thấy ảnh tại: {args.source}")
        return
    if args.limit > 0:
        files = files[: args.limit]
    print(f"[INFO] Xử lý {len(files)} ảnh test...")

    out_dir = Path(args.out)
    if args.save_vis:
        out_dir.mkdir(parents=True, exist_ok=True)

    tp_pix = fp_pix = fn_pix = 0
    o_tp = o_fp = o_fn = 0
    img_tp = img_fp = img_fn = img_tn = 0  # cấp CON HÀNG -- đúng câu hỏi khách hỏi, khác cục lỗi
    n_no_gt = 0
    loi_can_kiem_tra = []  # (tên ảnh, tp, fp, fn) -- chỉ ghi khi có sai (fp>0 hoặc fn>0)
    hang_sot = []   # tên ảnh: hàng lỗi thật nhưng model báo OK (bỏ sót CẢ con hàng)
    hang_nham = []  # tên ảnh: hàng tốt thật nhưng model báo NG (báo nhầm CẢ con hàng)

    for f in tqdm(files, desc="Kiểm tra"):
        img = imread_unicode(f, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[BỎ QUA] Lỗi đọc: {f.name}")
            continue

        gt_path = f.parent / f"{f.stem}_GT.png"
        gt = None
        if gt_path.exists():
            gt_raw = imread_unicode(gt_path, cv2.IMREAD_GRAYSCALE)
            gt = (gt_raw > 127).astype(np.uint8)
        else:
            n_no_gt += 1

        pred, prob_max = predict_one(model, img, mean, std, tile_w, tile_h, device, conf=args.conf)

        if gt is not None:
            pred_bin = (pred == 1).astype(np.uint8)
            tp_pix += int(((pred_bin == 1) & (gt == 1)).sum())
            fp_pix += int(((pred_bin == 1) & (gt == 0)).sum())
            fn_pix += int(((pred_bin == 0) & (gt == 1)).sum())
            a, b, d = object_stats(pred, gt, num_classes)
            o_tp += a; o_fp += b; o_fn += d
            if b > 0 or d > 0:
                loi_can_kiem_tra.append((f.name, a, b, d))

            # Cấp CON HÀNG (đúng câu hỏi khách hỏi): hàng này thật sự có lỗi/không, model báo NG/OK
            # không quan tâm bắt đúng bao nhiêu cục -- chỉ cần model báo ĐÚNG hướng cho CẢ con hàng.
            gt_hang_loi = bool(gt.any())
            pred_hang_loi = bool(pred_bin.any())
            if gt_hang_loi and pred_hang_loi:
                img_tp += 1
            elif gt_hang_loi and not pred_hang_loi:
                img_fn += 1
                hang_sot.append(f.name)
            elif not gt_hang_loi and pred_hang_loi:
                img_fp += 1
                hang_nham.append(f.name)
            else:
                img_tn += 1

        if args.save_vis:
            vis = render(img, pred, gt, prob_max, args.conf, args.min_area)
            imwrite_unicode(out_dir / f"{f.stem}_pred.png", vis, ".png")

    if n_no_gt:
        print(f"[CẢNH BÁO] {n_no_gt} ảnh không có *_GT.png -> không tính vào metric.")

    n_scored = len(files) - n_no_gt
    if n_scored > 0:
        pix_denom = tp_pix + fp_pix + fn_pix
        iou_defect = (tp_pix / pix_denom) if pix_denom > 0 else 1.0  # 0/0 = toàn nền, đoán đúng hết
        obj_denom = o_tp + o_fn
        recall = (o_tp / obj_denom) if obj_denom > 0 else 1.0
        prec_denom = o_tp + o_fp
        precision = (o_tp / prec_denom) if prec_denom > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 1.0
        n_hang_loi = img_tp + img_fn   # số con hàng lỗi thật
        n_hang_tot = img_fp + img_tn   # số con hàng tốt thật
        recall_hang = (img_tp / n_hang_loi) if n_hang_loi > 0 else 1.0
        fn_rate_hang = 1.0 - recall_hang
        fp_rate_hang = (img_fp / n_hang_tot) if n_hang_tot > 0 else 0.0

        print("\n===== KẾT QUẢ TRÊN BỘ TEST HOLDOUT =====")
        print(f"Số ảnh chấm điểm            = {n_scored}")
        print(f"IoU pixel (lớp lỗi)         = {iou_defect:.4f}")
        print(f"CỤC LỖI: Recall={recall:.3f} Precision={precision:.3f} F1={f1:.3f} "
              f"(bắt={o_tp} nhầm={o_fp} sót={o_fn})")
        print(f"CON HÀNG (đúng câu hỏi khách hỏi -- pass/fail cả sản phẩm, không phải theo cục lỗi):")
        print(f"  Hàng lỗi thật = {n_hang_loi}, hàng tốt thật = {n_hang_tot}")
        print(f"  Recall={recall_hang:.4f} (FN rate={fn_rate_hang:.4f})  FP rate={fp_rate_hang:.4f}")
        print(f"  (bắt đúng={img_tp} bỏ sót={img_fn} báo nhầm={img_fp} đúng OK={img_tn})")
    else:
        print("[CẢNH BÁO] Không có ảnh nào kèm _GT.png để tính metric.")

    if loi_can_kiem_tra:
        log_path = (out_dir if args.save_vis else Path(args.out))
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / "loi_can_kiem_tra.txt"
        with open(log_file, "w", encoding="utf-8") as fh:
            fh.write(f"{len(loi_can_kiem_tra)}/{n_scored} ảnh có sai lệch (nhầm và/hoặc sót cục lỗi)\n")
            fh.write("ten_anh  tp  fp(nhầm)  fn(sót)\n")
            for name, tp, fp, fn in loi_can_kiem_tra:
                fh.write(f"{name}  {tp}  {fp}  {fn}\n")
        print(f"[INFO] Danh sách {len(loi_can_kiem_tra)} ảnh có sai -> {log_file}")

    if hang_sot or hang_nham:
        log_path = (out_dir if args.save_vis else Path(args.out))
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / "hang_sai_capdoconhang.txt"
        with open(log_file, "w", encoding="utf-8") as fh:
            fh.write(f"BỎ SÓT CẢ CON HÀNG (hàng lỗi thật, model báo OK) -- {len(hang_sot)} hàng\n")
            for name in hang_sot:
                fh.write(f"  {name}\n")
            fh.write(f"\nBÁO NHẦM CẢ CON HÀNG (hàng tốt thật, model báo NG) -- {len(hang_nham)} hàng\n")
            for name in hang_nham:
                fh.write(f"  {name}\n")
        print(f"[INFO] Danh sách hàng sai cấp CON HÀNG -> {log_file}")

    if args.save_vis:
        print(f"\n[HOÀN TẤT] Ảnh overlay (đỏ=dự đoán, vàng=GT) lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
