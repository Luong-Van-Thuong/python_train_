#!/usr/bin/env bash
# Bài 9 - ablation: exp_A_baseline (đề CŨ dice_ce + Unet). Log ra file để đọc cùng nhau.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
cd /mnt/d/Projects_/Cong_Ty/Python_/train
mkdir -p logs
conda run -n vision_ai --no-capture-output \
  python -u src/train/train_unet.py \
    --loss dice_ce --arch Unet \
    --name exp_A_baseline \
    --epochs 150 --batch 8 \
    2>&1 | tee logs/exp_A_baseline.log
