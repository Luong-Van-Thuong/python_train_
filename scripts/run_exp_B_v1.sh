#!/usr/bin/env bash
# Bài 9 - ablation exp_B: CHỈ đổi loss dice_ce -> ftl_focal (FocalTversky + Focal).
# Giữ y nguyên data_imgs_unet_1, arch Unet, 12 epoch, batch 8 để so đúng A vs B.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
cd /mnt/d/Projects_/Cong_Ty/Python_/train
mkdir -p logs
conda run -n vision_ai --no-capture-output \
  python -u src/train/train_unet.py \
    --data /mnt/d/Projects_/Cong_Ty/Python_/train/SIBV/A27/data_imgs_unet_1/dataset.yaml \
    --loss ftl_focal --arch Unet \
    --name exp_B_ftl_v1 \
    --epochs 12 --batch 8 \
    2>&1 | tee logs/exp_B_ftl_v1.log
