#!/bin/bash
# RunPod Pod Initialization Script
# Run this once when a pod starts. Clones repos, installs deps, downloads checkpoints.
# Assumes network volume mounted at $WORKSPACE
#
# Usage: bash setup_pod.sh [--trellis|--hunyuan|--both]

set -euo pipefail

MODE="${1:---both}"
WORKSPACE="${WORKSPACE_DIR:-$HOME/workspace}"

echo "========================================="
echo "RunPod Pod Setup"
echo "Mode: $MODE"
echo "Workspace: $WORKSPACE"
echo "========================================="

# Ensure workspace structure
mkdir -p "$WORKSPACE"/{repos,data/{raw,filtered,trellis,hunyuan},checkpoints/{trellis/{pretrained,finetuned},hunyuan/{pretrained,finetuned}},results/{eval,samples},logs}

# Clone repos if not present
cd $WORKSPACE/repos

if [ ! -d "3DAI" ]; then
    echo "[1/5] Cloning 3DAI project..."
    # Replace with your actual repo URL
    git clone https://github.com/lobstermeeat/3DAI.git || echo "3DAI repo not available, skipping..."
fi

if [[ "$MODE" == "--trellis" || "$MODE" == "--both" ]]; then
    if [ ! -d "TRELLIS.2" ]; then
        echo "[2/5] Cloning TRELLIS.2..."
        git clone https://github.com/microsoft/TRELLIS.2.git
    fi

    echo "[2/5] Installing TRELLIS.2 dependencies..."
    cd $WORKSPACE/repos/TRELLIS.2
    pip install -r requirements.txt 2>&1 | tail -5
    bash setup.sh --basic --flash-attn --o-voxel --flexgemm 2>&1 | tail -5
    cd $WORKSPACE/repos
fi

if [[ "$MODE" == "--hunyuan" || "$MODE" == "--both" ]]; then
    if [ ! -d "Hunyuan3D-2.1" ]; then
        echo "[3/5] Cloning Hunyuan3D-2.1..."
        git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
    fi

    echo "[3/5] Installing Hunyuan3D-2.1 dependencies..."
    cd $WORKSPACE/repos/Hunyuan3D-2.1
    pip install -r requirements.txt 2>&1 | tail -5
    cd hy3dpaint && pip install -e . 2>&1 | tail -5
    cd $WORKSPACE/repos
fi

# Install common utilities
echo "[4/5] Installing common utilities..."
pip install tensorboard wandb tqdm pyyaml trimesh pymeshlab open3d 2>&1 | tail -3

# Download pretrained checkpoints
echo "[5/5] Downloading pretrained checkpoints..."

if [[ "$MODE" == "--trellis" || "$MODE" == "--both" ]]; then
    if [ ! -f "$WORKSPACE/checkpoints/trellis/pretrained/config.json" ]; then
        echo "  Downloading TRELLIS.2-4B pretrained weights..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'microsoft/TRELLIS.2-4B',
    local_dir='$WORKSPACE/checkpoints/trellis/pretrained',
    resume_download=True,
)
print('TRELLIS.2-4B download complete')
"
    else
        echo "  TRELLIS.2-4B weights already present, skipping."
    fi
fi

if [[ "$MODE" == "--hunyuan" || "$MODE" == "--both" ]]; then
    if [ ! -d "$WORKSPACE/checkpoints/hunyuan/pretrained/hy3dpaint" ]; then
        echo "  Downloading Hunyuan3D-Paint pretrained weights..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'tencent/Hunyuan3D-2.1',
    local_dir='$WORKSPACE/checkpoints/hunyuan/pretrained',
    allow_patterns=['hy3dpaint/**'],
    resume_download=True,
)
print('Hunyuan3D-Paint download complete')
"
    else
        echo "  Hunyuan3D-Paint weights already present, skipping."
    fi
fi

echo ""
echo "========================================="
echo "Pod setup complete!"
echo ""
echo "Workspace structure:"
du -sh $WORKSPACE/* 2>/dev/null || true
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo "========================================="
