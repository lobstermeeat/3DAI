#!/bin/bash
# Launch Hunyuan3D-Paint PBR fine-tuning on RunPod
#
# Prerequisites:
#   - Run setup_pod.sh --hunyuan first
#   - PBR render data prepared at /workspace/data/hunyuan
#   - Pretrained Paint checkpoint at /workspace/checkpoints/hunyuan/pretrained/hy3dpaint
#
# Usage: bash train_hunyuan_paint.sh

set -euo pipefail

HUNYUAN_DIR="/workspace/repos/Hunyuan3D-2.1"
CONFIG="/workspace/repos/3DAI/configs/hunyuan/hunyuan-paint-pbr-finetune.yaml"
OUTPUT_DIR="/workspace/checkpoints/hunyuan/finetuned/paint"
LOG_DIR="/workspace/logs/hunyuan_paint"

echo "========================================="
echo "Hunyuan3D-Paint PBR Fine-tuning"
echo "========================================="
echo "Config:     $CONFIG"
echo "Output:     $OUTPUT_DIR"
echo "Logs:       $LOG_DIR"
echo "========================================="

# Pre-flight checks
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }
[ -d "$HUNYUAN_DIR" ] || { echo "ERROR: Hunyuan3D-2.1 not found at $HUNYUAN_DIR"; exit 1; }
[ -f "/workspace/data/hunyuan/examples.json" ] || { echo "ERROR: Training data not found. Run prep_hunyuan_data.py first."; exit 1; }
[ -d "/workspace/checkpoints/hunyuan/pretrained/hy3dpaint" ] || { echo "ERROR: Pretrained checkpoint not found. Run setup_pod.sh first."; exit 1; }

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Start TensorBoard in background
tensorboard --logdir "$LOG_DIR" --port 6006 --bind_all &
echo "TensorBoard running on port 6006"

# Run training
cd "$HUNYUAN_DIR/hy3dpaint"
echo ""
echo "Starting training..."

python3 train.py \
    --base "$CONFIG" \
    --name forge3d_paint_ft \
    --logdir "$LOG_DIR" \
    --resume /workspace/checkpoints/hunyuan/pretrained/hy3dpaint \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "========================================="
echo "Training complete!"
echo "Checkpoints: $LOG_DIR"
echo "Log: $OUTPUT_DIR/training.log"
echo "========================================="
