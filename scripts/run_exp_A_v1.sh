#!/usr/bin/env bash
# Bài 9 - ablation trên DATA MỚI data_imgs_unet_1 (8374 train / 2067 val).
# Chạy rút gọn 12 epoch để test đường dẫn + xem per-class IoU khác 0.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
cd /mnt/d/Projects_/Cong_Ty/Python_/train
mkdir -p logs
conda run -n vision_ai --no-capture-output \
  python -u src/train/train_unet.py \
    --data /mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/data_imgs_unet_1/dataset.yaml \
    --loss dice_ce --arch Unet \
    --name exp_A_baseline_v1 \
    --epochs 12 --batch 8 \
    2>&1 | tee logs/exp_A_baseline_v1.log
