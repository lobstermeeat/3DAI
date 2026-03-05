#!/bin/bash
# Prepare training data for TRELLIS.2 fine-tuning
# Converts filtered mesh assets to O-Voxel format using TRELLIS.2's data_toolkit
#
# Usage:
#   bash prep_trellis_data.sh /workspace/data/filtered /workspace/data/trellis
#
# Prerequisites:
#   - TRELLIS.2 repo installed at /opt/TRELLIS.2 (or /workspace/repos/TRELLIS.2)
#   - O-Voxel extensions built (setup.sh --o-voxel)

set -euo pipefail

INPUT_DIR="${1:-/workspace/data/filtered}"
OUTPUT_DIR="${2:-/workspace/data/trellis}"
TRELLIS_DIR="${TRELLIS_DIR:-/workspace/repos/TRELLIS.2}"
RESOLUTION="${RESOLUTION:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "========================================="
echo "TRELLIS.2 Data Preprocessing Pipeline"
echo "========================================="
echo "Input:      $INPUT_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Resolution: $RESOLUTION"
echo "Workers:    $NUM_WORKERS"
echo "========================================="

# Create output structure
mkdir -p "$OUTPUT_DIR"/{mesh_dump,dual_grid,pbr_voxel,ss_latent,shape_latent,render_cond,asset_stats}

# Step 1: Convert meshes to dual-grid O-Voxel representation
echo ""
echo "[Step 1/4] Converting meshes to O-Voxel dual-grid..."
cd "$TRELLIS_DIR"

python -m data_toolkit.mesh_to_ovoxel \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR/dual_grid" \
    --resolution "$RESOLUTION" \
    --num_workers "$NUM_WORKERS" \
    2>&1 | tee "$OUTPUT_DIR/log_mesh_to_ovoxel.txt"

echo "[Step 1/4] Done."

# Step 2: Extract PBR voxel data
echo ""
echo "[Step 2/4] Extracting PBR voxel data..."
python -m data_toolkit.extract_pbr_voxel \
    --input_dir "$INPUT_DIR" \
    --dual_grid_dir "$OUTPUT_DIR/dual_grid" \
    --output_dir "$OUTPUT_DIR/pbr_voxel" \
    --num_workers "$NUM_WORKERS" \
    2>&1 | tee "$OUTPUT_DIR/log_pbr_voxel.txt"

echo "[Step 2/4] Done."

# Step 3: Encode latents using pretrained SC-VAE
echo ""
echo "[Step 3/4] Encoding latents with pretrained SC-VAE..."

# Sparse structure latents
python -m data_toolkit.encode_latents \
    --model_name "ss_enc_conv3d_16l8_fp16" \
    --input_dir "$OUTPUT_DIR/dual_grid" \
    --output_dir "$OUTPUT_DIR/ss_latent" \
    --checkpoint "/workspace/checkpoints/trellis/pretrained" \
    2>&1 | tee "$OUTPUT_DIR/log_ss_latent.txt"

# Shape latents
python -m data_toolkit.encode_latents \
    --model_name "shape_vae_next_dc_f16c32_fp16" \
    --input_dir "$OUTPUT_DIR/dual_grid" \
    --output_dir "$OUTPUT_DIR/shape_latent" \
    --checkpoint "/workspace/checkpoints/trellis/pretrained" \
    2>&1 | tee "$OUTPUT_DIR/log_shape_latent.txt"

echo "[Step 3/4] Done."

# Step 4: Render conditioning images
echo ""
echo "[Step 4/4] Rendering conditioning images..."
python -m data_toolkit.render_conditions \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR/render_cond" \
    --num_views 12 \
    --resolution 512 \
    --num_workers "$NUM_WORKERS" \
    2>&1 | tee "$OUTPUT_DIR/log_render_cond.txt"

echo "[Step 4/4] Done."

# Compute asset statistics
echo ""
echo "Computing asset statistics..."
python -m data_toolkit.compute_stats \
    --data_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR/asset_stats" \
    2>&1 | tee "$OUTPUT_DIR/log_stats.txt"

echo ""
echo "========================================="
echo "Data preprocessing complete!"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Directory sizes:"
du -sh "$OUTPUT_DIR"/*
echo "========================================="
