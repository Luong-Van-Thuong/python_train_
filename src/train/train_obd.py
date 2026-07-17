from ultralytics import YOLO

# --- cầu nối đường dẫn WSL2 <-> Windows Anaconda (xem pathfix.py ở gốc repo) ---
import sys as _sys, pathlib as _pl
_sys.path.insert(0, next(str(_p) for _p in _pl.Path(__file__).resolve().parents if (_p / "pathfix.py").exists()))
from pathfix import P, Pyaml


def main():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=Pyaml("/mnt/d/projects_/cong_ty/python_/train/jeayoung/mlcc/data_tu/data.yaml"),

        epochs=150,
        # imgsz=960,
        imgsz=200,
        device=0,

        batch=8,
        workers=4,
        patience=30,
        save=True,
        deterministic=True,

        amp=True,

        weight_decay=0.0005,
        label_smoothing=0.03,

        hsv_h=0.01,
        hsv_s=0.20,
        hsv_v=0.30,

        translate=0.04,
        shear=0.5,
        perspective=0.0003,

        fliplr=0.0,
        flipud=0.0,

        mosaic=0.4,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=30,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,

        project=P("/mnt/d/projects_/cong_ty/python_/train/JeaYoung/MLCC/results"),
        name="yo_n_260715_1_tu",
        exist_ok=True,
    )

    best_model_path = P("/mnt/d/projects_/cong_ty/python_/train/JeaYoung/MLCC/yo_n_260715_1_tu/weights/best.pt")

    export_model = YOLO(best_model_path)

    export_model.export(
        format="openvino",
        imgsz=200,
        half=False,
        int8=False,
        dynamic=False,
        nms=False,
    )


if __name__ == "__main__":
    main()

