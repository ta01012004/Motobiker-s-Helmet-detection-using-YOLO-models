#!/usr/bin/env bash
#SBATCH --job-name=yolo-compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

cd "/home/22011107/TA/Deep/Deep"
mkdir -p slurm_logs

MODEL_SIZE="${1:?usage: sbatch scripts/slurm_train_compare.sh n|s}"
case "${MODEL_SIZE}" in
  n)
    WEIGHTS="models/yolov8n.pt"
    NAME="yolov8n_120ep"
    ;;
  s)
    WEIGHTS="models/yolov8s.pt"
    NAME="yolov8s_120ep"
    ;;
  *)
    echo "Unknown model size: ${MODEL_SIZE}. Use n or s." >&2
    exit 2
    ;;
esac

source .venv_yolo/bin/activate
python - <<'PY'
import torch, ultralytics, numpy as np
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
print("numpy", np.__version__, "trapz", hasattr(np, "trapz"))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this node; refusing to start full training.")
PY

python scripts/train_compare_yolov8.py \
  --weights "${WEIGHTS}" \
  --data data.yaml \
  --name "${NAME}" \
  --epochs 120 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --workers 8
