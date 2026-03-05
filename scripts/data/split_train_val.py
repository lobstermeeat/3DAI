#!/usr/bin/env python3
"""
Split filtered dataset into train/val sets (90/10 default).

Usage:
    python split_train_val.py \
        --input_dir /workspace/data/filtered \
        --output_dir /workspace/data \
        --val_ratio 0.1
"""

import argparse
import json
import os
import random
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Find all mesh files
    extensions = {".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl"}
    files = []
    for f in os.listdir(args.input_dir):
        if Path(f).suffix.lower() in extensions:
            files.append(f)

    random.shuffle(files)

    val_count = max(1, int(len(files) * args.val_ratio))
    val_files = files[:val_count]
    train_files = files[val_count:]

    # Create output dirs
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy files
    for f in train_files:
        shutil.copy2(os.path.join(args.input_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy2(os.path.join(args.input_dir, f), os.path.join(val_dir, f))

    # Save split info
    split_info = {
        "total": len(files),
        "train": len(train_files),
        "val": len(val_files),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "train_files": train_files,
        "val_files": val_files,
    }
    with open(os.path.join(args.output_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    logger.info(f"Split complete: {len(train_files)} train, {len(val_files)} val")
    logger.info(f"  Train dir: {train_dir}")
    logger.info(f"  Val dir: {val_dir}")


if __name__ == "__main__":
    main()
