#!/bin/bash
# Launch TRELLIS.2 Shape Flow (img2shape) fine-tuning on RunPod
#
# Prerequisites:
#   - Run setup_pod.sh --trellis first
#   - Data preprocessed to /workspace/data/trellis
#   - ss_flow fine-tuning should be done first (not strictly required but recommended)
#
# Usage: bash train_trellis_shape_flow.sh [--resume STEP]

set -euo pipefail

TRELLIS_DIR="/workspace/repos/TRELLIS.2"
CONFIG="/workspace/repos/3DAI/configs/trellis/slat_flow_img2shape_finetune.json"
OUTPUT_DIR="/workspace/checkpoints/trellis/finetuned/shape_flow"
DATA_DIR='{"Forge3D_Custom": {"base": "/workspace/data/trellis", "shape_latent": "shape_latent", "ss_latent": "ss_latent", "render_cond": "render_cond", "asset_stats": "asset_stats"}}'
NUM_GPUS="${NUM_GPUS:-1}"
RESUME_STEP="${2:-}"

echo "========================================="
echo "TRELLIS.2 Shape Flow (img2shape) Fine-tuning"
echo "========================================="
echo "Config:     $CONFIG"
echo "Output:     $OUTPUT_DIR"
echo "GPUs:       $NUM_GPUS"
echo "========================================="

# Pre-flight checks
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }
[ -d "$TRELLIS_DIR" ] || { echo "ERROR: TRELLIS.2 not found at $TRELLIS_DIR"; exit 1; }
[ -d "/workspace/data/trellis/shape_latent" ] || { echo "ERROR: Training data not found. Run prep_trellis_data.sh first."; exit 1; }

mkdir -p "$OUTPUT_DIR"

# Start TensorBoard in background
tensorboard --logdir "$OUTPUT_DIR" --port 6006 --bind_all &
echo "TensorBoard running on port 6006"

# Build training command
CMD="python train.py --config $CONFIG --output_dir $OUTPUT_DIR --data_dir '$DATA_DIR' --num_gpus $NUM_GPUS"

if [ -n "$RESUME_STEP" ]; then
    CMD="$CMD --ckpt $RESUME_STEP"
    echo "Resuming from step $RESUME_STEP"
fi

# Run training
cd "$TRELLIS_DIR"
echo ""
echo "Starting training..."
echo "Command: $CMD"
echo ""

eval $CMD 2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "========================================="
echo "Training complete!"
echo "Checkpoints: $OUTPUT_DIR"
echo "========================================="
