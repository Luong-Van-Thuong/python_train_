# -*- coding: utf-8 -*-
"""
chia_data_seg_obd.py
====================
Chuẩn bị dữ liệu YOLO11 từ nhãn labelme (polygon) DÙNG CHUNG cho 2 bài toán:
    - seg : YOLO11-seg  (nhãn polygon)               -> dùng train_seg.py
    - obd : YOLO11-det  (nhãn bounding box)          -> dùng train_obd.py
    - both: sinh cả hai dataset cùng lúc (nhãn nhất quán vì cùng nguồn polygon)

Vì sao 1 script cho cả hai:
    - Cùng cách cắt tile, cùng cách clip polygon theo biên tile, cùng cách
      chia train/val và cân bằng tile nền -> seg và obd LUÔN khớp nhau, chỉ khác
      ĐỊNH DẠNG nhãn ghi ra. Sau này đổi qua lại giữa seg/obd không lệch dữ liệu.

Khác biệt định dạng nhãn (toạ độ chuẩn hoá theo tile 0..1):
    seg:  <cid> x1 y1 x2 y2 x3 y3 ...        (đa giác)
    obd:  <cid> cx cy w h                    (tâm + rộng/cao của bbox)

Pipeline (giữ nguyên logic của chia_data.py gốc):
    1. Quét cả thư mục NG và OK. Phân loại theo SỰ TỒN TẠI của .json (không theo
       tên thư mục): ảnh CÓ .json -> ảnh lỗi (sinh nhãn); ảnh KHÔNG .json -> nền.
       Nhờ vậy NG/OK lẫn lộn vẫn đúng (OK lọt vào NG, hay NG lọt vào OK).
    2. Tự gán class id ổn định theo alphabet.
    3. Chia train/val theo ẢNH GỐC (tránh rò rỉ tile cùng ảnh sang 2 phía).
       Riêng ảnh có TÊN bắt đầu bằng 'pass' (đổi qua --val-prefix) thì LUÔN
       vào val để test/đánh giá, không bao giờ vào train.
    4. Cắt tile (vd 640x640, overlap 20%), clip polygon bằng Shapely.
    5. Giữ tile có lỗi; cân bằng tile nền theo BG_RATIO.
    6. Ghi nhãn theo (các) task yêu cầu + sinh data.yaml riêng cho từng task.

Cấu trúc thư mục đầu ra (mỗi task là 1 dataset độc lập, train thẳng được):
    <out>/seg/{images,labels}/{train,val} + data.yaml
    <out>/obd/{images,labels}/{train,val} + data.yaml

Cài thư viện:
    pip install opencv-python numpy shapely pyyaml tqdm

Chạy:
    python chia_data_seg_obd.py --task both
    python chia_data_seg_obd.py --task obd
    python chia_data_seg_obd.py --task seg
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Cấu hình mặc định (sửa trực tiếp hoặc truyền qua dòng lệnh)
# Chạy WSL2: ổ D:\ map sang /mnt/d/
# --------------------------------------------------------------------------- #
DEFAULT_SRC = "/mnt/d/Images_/SIBV/A26/img_train/ng_"
DEFAULT_OK = "/mnt/d/Images_/SIBV/A26/img_train/ok_"   # thường là ảnh OK (nền)
DEFAULT_OUT = "/mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A26/data_imgs"

# Ảnh có TÊN bắt đầu bằng tiền tố này -> LUÔN đưa vào val (để test/đánh giá mô
# hình), KHÔNG bao giờ vào train. Đặt "" để tắt tính năng này.
DEFAULT_VAL_PREFIX = "pass"

IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
TASK_CHOICES = ("seg", "obd", "both")


# --------------------------------------------------------------------------- #
# Đọc/ghi ảnh an toàn với đường dẫn unicode (Windows)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Labelme
# --------------------------------------------------------------------------- #
def shape_to_polygon(shape):
    """1 shape labelme -> list điểm [(x,y)...]. Hỗ trợ polygon & rectangle."""
    pts = shape.get("points", [])
    st = shape.get("shape_type", "polygon")
    if st == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if st == "polygon" and len(pts) >= 3:
        return [(float(x), float(y)) for x, y in pts]
    return None  # circle/line/point -> bỏ qua


def collect_dir(dir_path, label=""):
    """Quét 1 thư mục -> list (img_path, json_path | None).

    PHÂN LOẠI THEO SỰ TỒN TẠI CỦA FILE .json, KHÔNG theo tên thư mục:
      - ảnh CÓ file .json trùng tên cạnh nó  -> coi là ảnh LỖI (sinh nhãn).
      - ảnh KHÔNG có .json                    -> coi là ảnh NỀN/OK (nhãn rỗng).
    Nhờ vậy thư mục NG và OK có lẫn lộn cũng xử lý đúng:
      ảnh OK lọt vào NG (thiếu .json) -> xử lý như OK;
      ảnh NG lọt vào OK (có .json)    -> xử lý như NG.
    """
    if not dir_path:
        return []
    src = Path(dir_path)
    if not src.exists():
        print(f"[CẢNH BÁO] Không tìm thấy thư mục {label}: {dir_path}, bỏ qua.")
        return []
    out = []
    for img_path in sorted(src.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        json_path = img_path.with_suffix(".json")
        out.append((img_path, json_path if json_path.exists() else None))
    return out


def group_key(img_path, sep):
    """Khoá gom các ảnh CÙNG MỘT CON HÀNG vật lý.

    Lấy phần tên file đứng TRƯỚC dấu phân tách `sep`.
        sep='@', 'partA@goc1.bmp' -> 'partA'  (partA@goc1, partA@goc2 cùng nhóm)
        sep=''  -> trả nguyên tên file  (mỗi ảnh là 1 nhóm, chia theo từng ảnh)
    Dùng để mọi ảnh chụp cùng 1 con hàng luôn nằm trọn 1 phía train/val,
    tránh rò rỉ dữ liệu (cùng vết lỗi xuất hiện ở cả train lẫn val).
    """
    stem = img_path.stem
    if sep and sep in stem:
        return stem.split(sep)[0]
    return stem


def is_forced_val(img_path, prefix):
    """Ảnh có tên bắt đầu bằng `prefix` (vd 'pass') -> luôn vào val để đánh giá.

    So khớp KHÔNG phân biệt hoa/thường. prefix rỗng -> tắt (luôn trả False).
    """
    if not prefix:
        return False
    return img_path.stem.lower().startswith(prefix.lower())


def build_class_map(pairs):
    """Quét json -> danh sách class sắp xếp alphabet -> {label: id}."""
    labels = set()
    for _, json_path in pairs:
        if json_path is None:
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sh in data.get("shapes", []):
            labels.add(sh["label"])
    classes = sorted(labels)
    return {name: i for i, name in enumerate(classes)}, classes


# --------------------------------------------------------------------------- #
# Cắt tile + clip nhãn
# --------------------------------------------------------------------------- #
def tile_origins(length, tile, overlap):
    """Toạ độ gốc tile dọc 1 chiều, đảm bảo phủ tới mép."""
    if length <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    xs = list(range(0, length - tile + 1, step))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def clip_polygon_to_tile(poly, tile_box):
    """Cắt polygon theo ô tile -> list polygon hợp lệ (đã xử lý mảnh vỡ)."""
    if not poly.is_valid:
        poly = make_valid(poly)
    inter = poly.intersection(tile_box)
    if inter.is_empty:
        return []
    gtype = inter.geom_type
    if gtype == "Polygon":
        geoms = [inter]
    elif gtype in ("MultiPolygon", "GeometryCollection"):
        geoms = [g for g in inter.geoms if g.geom_type == "Polygon"]
    else:
        geoms = []
    return [g for g in geoms if g.area > 0]


def _clamp01(v):
    return min(max(v, 0.0), 1.0)


def seg_line_from_geom(cid, g, x0, y0, tile):
    """Dòng nhãn SEG: cid + polygon chuẩn hoá theo tile."""
    coords = list(g.exterior.coords)[:-1]  # bỏ điểm lặp cuối
    if len(coords) < 3:
        return None
    parts = []
    for px, py in coords:
        nx = _clamp01((px - x0) / tile)
        ny = _clamp01((py - y0) / tile)
        parts.append(f"{nx:.6f} {ny:.6f}")
    return f"{cid} " + " ".join(parts)


def obd_line_from_geom(cid, g, x0, y0, tile):
    """Dòng nhãn OBD: cid cx cy w h (chuẩn hoá theo tile) từ bbox của mảnh clip."""
    minx, miny, maxx, maxy = g.bounds
    bx1 = _clamp01((minx - x0) / tile)
    by1 = _clamp01((miny - y0) / tile)
    bx2 = _clamp01((maxx - x0) / tile)
    by2 = _clamp01((maxy - y0) / tile)
    w = bx2 - bx1
    h = by2 - by1
    if w <= 0 or h <= 0:
        return None
    cx = bx1 + w / 2
    cy = by1 + h / 2
    return f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def process_image(img_path, json_path, class_map, task_dirs, tasks,
                  tile, overlap, min_area_frac, min_area_px, bg_ratio,
                  img_ext, rng, ok_tiles_per_img=-1):
    """Cắt 1 ảnh thành tile, ghi tile + nhãn cho TỪNG task. Trả về thống kê.

    task_dirs: dict {task: (img_dir, lbl_dir)} cho các task cần sinh.
    json_path = None -> ảnh OK: mọi tile là nền, nhãn rỗng.
    """
    img = imread_unicode(img_path)
    if img is None:
        print(f"[LỖI] Không đọc được ảnh {img_path.name}, bỏ qua.")
        return {"pos": 0, "neg": 0, "inst": 0}

    H, W = img.shape[:2]

    if json_path is None:
        data = {"shapes": []}
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # (class_id, polygon shapely, diện tích gốc)
    anns = []
    for sh in data.get("shapes", []):
        ring = shape_to_polygon(sh)
        if ring is None:
            continue
        cid = class_map[sh["label"]]
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.is_empty or poly.area <= 0:
            continue
        anns.append((cid, poly, poly.area))

    xs = tile_origins(W, tile, overlap)
    ys = tile_origins(H, tile, overlap)

    # mỗi phần tử: (x0, y0, {task: [dòng nhãn]})
    pos_tiles, neg_tiles = [], []

    for y0 in ys:
        for x0 in xs:
            tbox = box(x0, y0, x0 + tile, y0 + tile)
            lines = {t: [] for t in tasks}
            for cid, poly, area0 in anns:
                if not poly.intersects(tbox):
                    continue
                for g in clip_polygon_to_tile(poly, tbox):
                    # Lọc mảnh quá nhỏ (sliver) do bị cắt
                    if g.area < min_area_px:
                        continue
                    if g.area < min_area_frac * area0:
                        continue
                    if "seg" in tasks:
                        s = seg_line_from_geom(cid, g, x0, y0, tile)
                        if s:
                            lines["seg"].append(s)
                    if "obd" in tasks:
                        o = obd_line_from_geom(cid, g, x0, y0, tile)
                        if o:
                            lines["obd"].append(o)

            # Tile "có lỗi" nếu BẤT KỲ task nào có nhãn (các task khớp nhau về hình học)
            has_defect = any(lines[t] for t in tasks)
            (pos_tiles if has_defect else neg_tiles).append((x0, y0, lines))

    # Cân bằng tile nền
    if json_path is None:
        if ok_tiles_per_img >= 0:
            rng.shuffle(neg_tiles)
            neg_tiles = neg_tiles[:ok_tiles_per_img]
    elif bg_ratio >= 0:
        keep_neg = int(round(len(pos_tiles) * bg_ratio))
        rng.shuffle(neg_tiles)
        neg_tiles = neg_tiles[:keep_neg]

    stem = img_path.stem
    n_inst = 0

    def save_tile(x0, y0, lines):
        nonlocal n_inst
        crop = img[y0:y0 + tile, x0:x0 + tile]
        ch, cw = crop.shape[:2]
        if (ch, cw) != (tile, tile):  # pad nếu tràn mép
            padded = np.zeros((tile, tile, 3), dtype=img.dtype)
            padded[:ch, :cw] = crop
            crop = padded
        name = f"{stem}__x{x0}_y{y0}"
        for t in tasks:
            img_dir, lbl_dir = task_dirs[t] 
            imwrite_unicode(img_dir / f"{name}{img_ext}", crop, img_ext)
            with open(lbl_dir / f"{name}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(lines[t]))
        # đếm instance theo task đầu tiên (các task khớp số lượng)
        n_inst += len(lines[tasks[0]])

    for x0, y0, lines in pos_tiles:
        save_tile(x0, y0, lines)
    for x0, y0, lines in neg_tiles:
        save_tile(x0, y0, lines)

    return {"pos": len(pos_tiles), "neg": len(neg_tiles), "inst": n_inst}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Cắt tile + convert labelme -> YOLO seg/obd (dùng chung)")
    ap.add_argument("--task", default="both", choices=TASK_CHOICES,
                    help="seg | obd | both (mặc định both)")
    ap.add_argument("--src", default=DEFAULT_SRC, help="Thư mục ảnh + json (NG)")
    ap.add_argument("--ok-src", default=DEFAULT_OK,
                    help="Thư mục ảnh OK (không json). '' để bỏ qua")
    ap.add_argument("--ok-tiles-per-img", type=int, default=-1,
                    help="Số tile nền lấy ngẫu nhiên mỗi ảnh OK. -1 = giữ tất cả")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Thư mục dataset đầu ra")
    ap.add_argument("--tile", type=int, default=640, help="Kích thước tile (px)")
    ap.add_argument("--overlap", type=float, default=0.20, help="Tỉ lệ chồng lấn 0..1")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Tỉ lệ ảnh val")
    ap.add_argument("--val-prefix", default=DEFAULT_VAL_PREFIX,
                    help="Ảnh có tên bắt đầu bằng tiền tố này LUÔN vào val (không "
                         "vào train). Vd 'pass'. '' để tắt")
    ap.add_argument("--group-sep", default="",
                    help="Dấu phân tách để gom ảnh CÙNG MỘT CON HÀNG (lấy phần tên "
                         "trước dấu này). Vd '@': partA@goc1, partA@goc2 -> nhóm "
                         "'partA', luôn cùng 1 phía train/val. '' = chia theo từng ảnh")
    ap.add_argument("--bg-ratio", type=float, default=-1.0,
                    help="Số tile nền / số tile lỗi (mỗi ảnh NG). -1 = giữ tất cả")
    ap.add_argument("--min-area-frac", type=float, default=0.10,
                    help="Giữ mảnh clip nếu diện tích >= frac * diện tích gốc")
    ap.add_argument("--min-area-px", type=float, default=8.0,
                    help="Diện tích tối thiểu (px^2) sau khi clip")
    ap.add_argument("--img-ext", default=".png", choices=[".png", ".jpg"],
                    help="Định dạng tile xuất ra")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tasks = ["seg", "obd"] if args.task == "both" else [args.task]
    rng = random.Random(args.seed)

    # Quét cả 2 thư mục như nhau; phân loại lỗi/nền theo SỰ TỒN TẠI của .json.
    pairs = collect_dir(args.src, "NG")
    ok_pairs = collect_dir(args.ok_src, "OK")
    if not pairs and not ok_pairs:
        print(f"[LỖI] Không tìm thấy ảnh nào trong {args.src} hoặc {args.ok_src}")
        return

    all_pairs = pairs + ok_pairs
    n_defect = sum(1 for _, j in all_pairs if j is not None)
    n_bg = len(all_pairs) - n_defect
    print(f"Tìm thấy {len(all_pairs)} ảnh: {n_defect} ảnh LỖI (có .json), "
          f"{n_bg} ảnh NỀN (không .json).")

    class_map, classes = build_class_map(all_pairs)
    print("Class map (id ổn định theo alphabet):")
    for name, i in class_map.items():
        print(f"   {i}: {name}")

    # Chia train/val GOM theo con hàng (group_key), tách riêng NG và OK rồi gộp.
    # Xáo trộn & chia theo NHÓM con hàng -> mọi ảnh cùng con hàng đi cùng 1 phía.
    def split_pairs(plist):
        if not plist:
            return [], []
        # 1) Tách riêng ảnh BUỘC vào val (tên bắt đầu bằng val_prefix, vd 'pass').
        #    Những ảnh này luôn ở val, không tham gia chia ngẫu nhiên, không vào train.
        forced_val = [p for p in plist if is_forced_val(p[0], args.val_prefix)]
        rest = [p for p in plist if not is_forced_val(p[0], args.val_prefix)]
        # 2) Phần còn lại: gom theo khoá con hàng rồi chia train/val như cũ.
        groups = {}
        for pair in rest:
            key = group_key(pair[0], args.group_sep)
            groups.setdefault(key, []).append(pair)
        keys = list(groups.keys())
        rng.shuffle(keys)
        if keys:
            n_val = max(1, int(round(len(keys) * args.val_ratio)))
        else:
            n_val = 0
        val_keys, train_keys = keys[:n_val], keys[n_val:]
        train = [p for k in train_keys for p in groups[k]]
        val = [p for k in val_keys for p in groups[k]] + forced_val
        return train, val

    if args.group_sep:
        n_parts = len({group_key(p[0], args.group_sep) for p in pairs})
        print(f"Gom ảnh NG theo dấu '{args.group_sep}': {n_parts} con hàng "
              f"từ {len(pairs)} ảnh (mỗi con hàng nằm trọn 1 phía train/val).")

    train_ng, val_ng = split_pairs(pairs)
    train_ok, val_ok = split_pairs(ok_pairs)
    train_pairs = train_ng + train_ok
    val_pairs = val_ng + val_ok
    if args.val_prefix:
        n_forced = sum(is_forced_val(p[0], args.val_prefix)
                       for p in pairs + ok_pairs)
        print(f"Ảnh tên bắt đầu '{args.val_prefix}' -> ép vào val: {n_forced} ảnh "
              f"(không vào train).")
    def dem_loi(plist):
        d = sum(1 for _, j in plist if j is not None)
        return d, len(plist) - d
    tr_d, tr_b = dem_loi(train_pairs)
    va_d, va_b = dem_loi(val_pairs)
    print(f"Chia: train={len(train_pairs)} (lỗi={tr_d}, nền={tr_b}), "
          f"val={len(val_pairs)} (lỗi={va_d}, nền={va_b})")

    # Tạo cây thư mục cho từng task: <out>/<task>/{images,labels}/{train,val}
    out_base = Path(args.out)
    task_split_dirs = {}  # {(task, split): (img_dir, lbl_dir)}
    for t in tasks:
        for split in ("train", "val"):
            img_dir = out_base / t / "images" / split
            lbl_dir = out_base / t / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            task_split_dirs[(t, split)] = (img_dir, lbl_dir)

    totals = {"train": {"pos": 0, "neg": 0, "inst": 0},
              "val": {"pos": 0, "neg": 0, "inst": 0}}

    for split, plist in (("train", train_pairs), ("val", val_pairs)):
        task_dirs = {t: task_split_dirs[(t, split)] for t in tasks}
        for img_path, json_path in tqdm(plist, desc=f"Cắt tile [{split}]"):
            st = process_image(
                img_path, json_path, class_map, task_dirs, tasks,
                tile=args.tile, overlap=args.overlap,
                min_area_frac=args.min_area_frac, min_area_px=args.min_area_px,
                bg_ratio=args.bg_ratio, img_ext=args.img_ext, rng=rng,
                ok_tiles_per_img=args.ok_tiles_per_img,
            )
            for k in totals[split]:
                totals[split][k] += st[k]

    # Sinh data.yaml cho từng task (định dạng giống nhau, chỉ khác nhãn trên đĩa)
    names = {i: name for name, i in class_map.items()}
    for t in tasks:
        data_yaml = {
            "path": str(out_base / t),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }
        with open(out_base / t / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    # Báo cáo
    print("\n===== HOÀN TẤT =====")
    print(f"Task sinh ra: {', '.join(tasks)}")
    for split in ("train", "val"):
        tt = totals[split]
        print(f"{split:5s}: {tt['pos']} tile lỗi, {tt['neg']} tile nền, "
              f"{tt['inst']} instance (mỗi task)")

    print("\nĐường dẫn data.yaml & lệnh train gợi ý:")
    for t in tasks:
        yml = out_base / t / "data.yaml"
        if t == "seg":
            print(f"  [seg] {yml}")
            print(f'        python train_seg.py --data "{yml}" --imgsz {args.tile}')
        else:
            print(f"  [obd] {yml}")
            print(f'        yolo detect train model=yolo11s.pt data="{yml}" '
                  f"imgsz={args.tile} epochs=200 batch=8")


if __name__ == "__main__":
    main()
