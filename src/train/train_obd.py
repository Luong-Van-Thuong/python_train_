# from ultralytics import YOLO
# if __name__ == '__main__':
#     model = YOLO("yolo11n.pt")
#     model.train(data=r"D:\3. Python\Yolo_trainning\images\data.yaml", imgsz= 640, epochs = 120, batch = 16)



# from ultralytics import YOLO


# def main():
#     model = YOLO("yolo11s.pt")

#     results = model.train(
#         data="/mnt/d/python/mlcc/data.yaml",

#         epochs=250,
#         imgsz=960,        # quan trọng cho vật thể nhỏ
#         device=0,

#         batch=8,         # giảm batch vì ảnh lớn hơn
#         workers=2,
#         patience=50,
#         save=True,
#         deterministic=True,

#         amp=True,

#         # chống overfit
#         weight_decay=0.0005,
#         label_smoothing=0.03,

#         # augmentation vừa phải cho công nghiệp
#         hsv_h=0.01,
#         hsv_s=0.20,
#         hsv_v=0.30,

#         translate=0.04,
#         shear=0.5,
#         perspective=0.0003,

#         fliplr=0.0,
#         flipud=0.0,

#         mosaic=0.4,
#         mixup=0.0,
#         copy_paste=0.0,
#         close_mosaic=30,

#         optimizer="AdamW",
#         lr0=0.001,
#         lrf=0.01,

#         project="/mnt/d/python/mlcc/VisionProject",
#         name="yolo11s_20260522_part1_images_960",
#         exist_ok=True,
#     )

#     model.export(format="onnx", half=False)


# if __name__ == "__main__":
#     main()

# yolo export model="/mnt/d/python/mlcc/VisionProject/Industrial_Model_yolo8n_20260522_part1/weights/best.pt" format=openvino
# yolo export model="/mnt/d/python/mlcc/VisionProject/yolo11n_20260522_part2_images_1024/weights/best.pt" format=openvino
# yolo export model="/mnt/d/python/mlcc/VisionProject/yolo11n_20260522_part1_images_960/weights/best.pt" format=openvino


from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="/mnt/d/projects_/cong_ty/python_/train/data.yaml",

        epochs=150,
        # imgsz=960,
         imgsz=960,
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

        project="/mnt/d/projects_/cong_ty/python_/train/SIBV/A26/",
        name="yolo11n_960_260614_part1",
        exist_ok=True,
    )

    best_model_path = "/mnt/d/projects_/cong_ty/python_/train/SIBV/A26/yolo11n_960_260614_part1/weights/best.pt"

    export_model = YOLO(best_model_path)

    export_model.export(
        format="openvino",
        imgsz=960,
        half=False,
        int8=False,
        dynamic=False,
        nms=False,
    )


if __name__ == "__main__":
    main()



# from ultralytics import YOLO


# def main():
#     # Thay vì load model gốc, load last.pt để resume
#     model = YOLO("/mnt/d/python/mlcc/VisionProject/yolo11n_20260602_part1_images_960/weights/last.pt")

#     results = model.train(
#         resume=True,   # <-- thêm dòng này, các tham số khác tự đọc từ checkpoint
#     )

#     best_model_path = "/mnt/d/python/mlcc/VisionProject/yolo11n_20260602_part1_images_960/weights/best.pt"

#     export_model = YOLO(best_model_path)

#     export_model.export(
#         format="openvino",
#         imgsz=960,
#         half=False,
#         int8=False,
#         dynamic=False,
#         nms=False,
#     )


# if __name__ == "__main__":
#     main()