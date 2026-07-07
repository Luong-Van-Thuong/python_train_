#!/usr/bin/env bash
# Smoke test: 2 epoch, kiểm tra metric object-level + best.pt theo f1_object chạy được.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
cd /mnt/d/Projects_/Cong_Ty/Python_/train
echo "=== syntax check ==="
conda run -n vision_ai --no-capture-output python -m py_compile src/train/train_unet.py && echo "OK compile"
echo "=== 2-epoch smoke (f1_object) ==="
conda run -n vision_ai --no-capture-output \
  python -u src/train/train_unet.py \
    --data /mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/data_imgs_unet_1/dataset.yaml \
    --loss ftl_focal --arch Unet \
    --name exp_smoke_metric \
    --best-metric f1_object \
    --epochs 2 --batch 8 \
    2>&1 | tee logs/smoke_metric.log
