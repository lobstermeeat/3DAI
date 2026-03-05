#!/bin/bash
# Prepare training data for TRELLIS.2 fine-tuning
# Uses TRELLIS.2's actual data_toolkit pipeline.
#
# Usage:
#   bash prep_trellis_data.sh /workspace/data/filtered /workspace/data/trellis_root
#
# Prerequisites:
#   - TRELLIS.2 repo at $TRELLIS_DIR (default: /workspace/repos/TRELLIS.2)
#   - O-Voxel extensions built (setup.sh --o-voxel)
#   - Pretrained checkpoint at /workspace/checkpoints/trellis/pretrained

set -euo pipefail

INPUT_DIR="${1:-/workspace/data/filtered}"
ROOT="${2:-/workspace/data/trellis_root}"
TRELLIS_DIR="${TRELLIS_DIR:-/workspace/repos/TRELLIS.2}"
CKPT_DIR="${CKPT_DIR:-/workspace/checkpoints/trellis/pretrained}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
RESOLUTION="${RESOLUTION:-512}"

echo "========================================="
echo "TRELLIS.2 Data Preprocessing Pipeline"
echo "========================================="
echo "Input:      $INPUT_DIR"
echo "Root:       $ROOT"
echo "TRELLIS:    $TRELLIS_DIR"
echo "Checkpoint: $CKPT_DIR"
echo "Resolution: $RESOLUTION"
echo "========================================="

# Step 0: Set up dataset structure and Forge3D module
echo ""
echo "[Step 0/7] Setting up dataset structure..."
python "$SCRIPTS_DIR/setup_trellis_dataset.py" \
    --input_dir "$INPUT_DIR" \
    --trellis_root "$ROOT" \
    --trellis_dir "$TRELLIS_DIR"

# From here, all commands run from TRELLIS.2 root
cd "$TRELLIS_DIR"

# Install data_toolkit deps if needed
pip install -q easydict open3d open_clip_torch 2>/dev/null || true

# Step 1: Dump meshes (uses Blender)
echo ""
echo "[Step 1/7] Dumping meshes via Blender..."
python data_toolkit/dump_mesh.py Forge3D --root "$ROOT" --max_workers 4 \
    2>&1 | tee "$ROOT/log_dump_mesh.txt"

# Update metadata
python data_toolkit/build_metadata.py Forge3D --root "$ROOT"

# Step 2: Dump PBR textures
echo ""
echo "[Step 2/7] Dumping PBR textures..."
python data_toolkit/dump_pbr.py Forge3D --root "$ROOT" --max_workers 4 \
    2>&1 | tee "$ROOT/log_dump_pbr.txt"

# Update metadata
python data_toolkit/build_metadata.py Forge3D --root "$ROOT"

# Step 3: Asset statistics
echo ""
echo "[Step 3/7] Computing asset statistics..."
python data_toolkit/asset_stats.py --root "$ROOT" \
    2>&1 | tee "$ROOT/log_asset_stats.txt"

# Update metadata
python data_toolkit/build_metadata.py Forge3D --root "$ROOT"

# Step 4: Convert to O-Voxels (dual grid)
echo ""
echo "[Step 4/7] Converting to O-Voxel dual grid (resolution=$RESOLUTION)..."
python data_toolkit/dual_grid.py Forge3D --root "$ROOT" --resolution "$RESOLUTION" \
    2>&1 | tee "$ROOT/log_dual_grid.txt"

# Step 5: Voxelize PBR
echo ""
echo "[Step 5/7] Voxelizing PBR textures..."
python data_toolkit/voxelize_pbr.py Forge3D --root "$ROOT" --resolution "$RESOLUTION" \
    2>&1 | tee "$ROOT/log_voxelize_pbr.txt"

# Update metadata
python data_toolkit/build_metadata.py Forge3D --root "$ROOT"

# Step 6: Encode latents
echo ""
echo "[Step 6/7] Encoding latents..."

# Shape latents
echo "  Encoding shape latents..."
python data_toolkit/encode_shape_latent.py --root "$ROOT" --resolution "$RESOLUTION" \
    2>&1 | tee "$ROOT/log_encode_shape.txt"

# PBR latents
echo "  Encoding PBR latents..."
python data_toolkit/encode_pbr_latent.py --root "$ROOT" --resolution "$RESOLUTION" \
    2>&1 | tee "$ROOT/log_encode_pbr.txt"

# Update metadata before SS latent encoding
python data_toolkit/build_metadata.py Forge3D --root "$ROOT"

# SS latents (needs shape latent name from previous step)
SHAPE_LATENT_NAME=$(ls "$ROOT" | grep "shape_enc_" | head -1 || echo "shape_enc_next_dc_f16c32_fp16_${RESOLUTION}")
echo "  Encoding SS latents (shape_latent_name=$SHAPE_LATENT_NAME)..."
python data_toolkit/encode_ss_latent.py --root "$ROOT" \
    --shape_latent_name "$SHAPE_LATENT_NAME" --resolution 64 \
    2>&1 | tee "$ROOT/log_encode_ss.txt"

# Update metadata
python data_toolkit/build_metadata.py Forge3D --root "$ROOT"

# Step 7: Render conditioning images
echo ""
echo "[Step 7/7] Rendering conditioning images..."
python data_toolkit/render_cond.py Forge3D --root "$ROOT" --num_views 16 \
    2>&1 | tee "$ROOT/log_render_cond.txt"

# Final metadata update
python data_toolkit/build_metadata.py Forge3D --root "$ROOT"

echo ""
echo "========================================="
echo "Data preprocessing complete!"
echo "Root: $ROOT"
echo ""
echo "Directory sizes:"
du -sh "$ROOT"/* 2>/dev/null || true
echo "========================================="
