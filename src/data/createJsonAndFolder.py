from pathlib import Path
from PIL import Image
import json
import random
import shutil
import yaml


# ==========================
# CONFIG
# ==========================
SOURCE_DIR = r"D:\Images\JeaYoung\MLCC\Data\tu"      # folder chứa cả ảnh + json LabelMe
OUTPUT_DIR = r"D:\Python\mlcc\data"          # folder YOLO output

VAL_SIZE = 0.2                                  # 20% ảnh làm validation
RANDOM_SEED = 42                                # đổi số này nếu muốn shuffle khác

# Label trong file JSON của bạn là "1"
CLASS_NAMES = [
    "d"
]

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


# ==========================
# CREATE EMPTY LABELME JSON
# ==========================
def create_empty_labelme_json(folder_path):
    folder = Path(folder_path)

    count_created = 0
    count_skipped = 0

    for image_path in folder.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        json_path = image_path.with_suffix(".json")

        # Nếu đã có json rồi thì bỏ qua
        if json_path.exists():
            count_skipped += 1
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        # JSON rỗng đúng format LabelMe 6.x
        labelme_data = {
            "version": "6.1.1",
            "flags": {},
            "shapes": [],
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

        print(f"Created empty json: {json_path.name}")
        count_created += 1

    print("-" * 50)
    print(f"Empty json created: {count_created}")
    print(f"Existing json skipped: {count_skipped}")


# ==========================
# FIND IMAGE BY JSON imagePath
# ==========================
def find_image_path(source_dir, json_path, image_name_from_json):
    source_dir = Path(source_dir)

    # Ưu tiên imagePath trong json
    if image_name_from_json:
        candidate = source_dir / image_name_from_json
        if candidate.exists():
            return candidate

    # Nếu không tìm được thì tìm theo cùng stem
    for ext in IMAGE_EXTS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    return None


# ==========================
# LABELME JSON TO YOLO TXT
# ==========================
def labelme_json_to_yolo_txt(json_path, txt_path, class_names, source_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_width = data.get("imageWidth")
    image_height = data.get("imageHeight")
    image_name = data.get("imagePath")

    # Nếu JSON thiếu width/height thì mở ảnh để lấy kích thước
    if image_width is None or image_height is None:
        image_path = find_image_path(source_dir, json_path, image_name)

        if image_path is None:
            print(f"Warning: cannot find image for {json_path.name}")
            txt_path.write_text("", encoding="utf-8")
            return

        with Image.open(image_path) as img:
            image_width, image_height = img.size

    lines = []

    shapes = data.get("shapes", [])

    for shape in shapes:
        label = str(shape.get("label"))
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "")

        if label not in class_names:
            print(f"Warning: label '{label}' not in CLASS_NAMES, skipped: {json_path.name}")
            continue

        if len(points) < 2:
            continue

        class_id = class_names.index(label)

        # LabelMe rectangle sẽ có 2 điểm: góc 1 và góc đối diện
        if shape_type == "rectangle":
            x1, y1 = points[0]
            x2, y2 = points[1]

            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)

        else:
            # Nếu sau này bạn dùng polygon thì vẫn convert được bằng bounding box bao quanh polygon
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            x_min = min(xs)
            y_min = min(ys)
            x_max = max(xs)
            y_max = max(ys)

        # Giới hạn bbox không vượt ra ngoài ảnh
        x_min = max(0, min(x_min, image_width))
        y_min = max(0, min(y_min, image_height))
        x_max = max(0, min(x_max, image_width))
        y_max = max(0, min(y_max, image_height))

        box_width = x_max - x_min
        box_height = y_max - y_min

        if box_width <= 0 or box_height <= 0:
            continue

        x_center = x_min + box_width / 2
        y_center = y_min + box_height / 2

        # Normalize về YOLO format từ 0 đến 1
        x_center = x_center / image_width
        y_center = y_center / image_height
        box_width = box_width / image_width
        box_height = box_height / image_height

        line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        lines.append(line)

    # Nếu không có object thì txt rỗng
    with open(txt_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))
        else:
            f.write("")


# ==========================
# BUILD YOLO DATASET
# ==========================
def build_yolo_dataset(source_dir, output_dir, val_size, random_seed, class_names):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    images_train_dir = output_dir / "images" / "train"
    images_val_dir = output_dir / "images" / "val"
    labels_train_dir = output_dir / "labels" / "train"
    labels_val_dir = output_dir / "labels" / "val"

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in source_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ]

    if len(image_paths) == 0:
        print("No images found.")
        return

    # Xáo trộn vị trí ảnh
    random.seed(random_seed)
    random.shuffle(image_paths)

    val_count = int(len(image_paths) * val_size)

    val_images = image_paths[:val_count]
    train_images = image_paths[val_count:]

    print("-" * 50)
    print(f"Total images: {len(image_paths)}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")

    def process_split(images, img_out_dir, label_out_dir):
        for image_path in images:
            json_path = image_path.with_suffix(".json")

            if not json_path.exists():
                print(f"Missing json, skipped: {image_path.name}")
                continue

            out_image_path = img_out_dir / image_path.name
            out_txt_path = label_out_dir / f"{image_path.stem}.txt"

            shutil.copy2(image_path, out_image_path)

            labelme_json_to_yolo_txt(
                json_path=json_path,
                txt_path=out_txt_path,
                class_names=class_names,
                source_dir=source_dir
            )

    process_split(train_images, images_train_dir, labels_train_dir)
    process_split(val_images, images_val_dir, labels_val_dir)

    data_yaml = {
        "path": str(output_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            i: name for i, name in enumerate(class_names)
        }
    }

    yaml_path = output_dir / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print("-" * 50)
    print(f"YOLO dataset created at: {output_dir}")
    print(f"data.yaml created at: {yaml_path}")
    print("Done.")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    create_empty_labelme_json(SOURCE_DIR)

    build_yolo_dataset(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        val_size=VAL_SIZE,
        random_seed=RANDOM_SEED,
        class_names=CLASS_NAMES
    )