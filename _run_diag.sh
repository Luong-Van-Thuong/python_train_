#!/usr/bin/env bash
PY=$HOME/miniconda3/envs/vision_ai/bin/python
cd /mnt/d/Projects_/Cong_Ty/Python_/train
"$PY" -c "import torch, segmentation_models_pytorch as smp, cv2, yaml; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('smp', smp.__version__)"
echo "=================== DIAG ==================="
"$PY" _diag_unet.py
