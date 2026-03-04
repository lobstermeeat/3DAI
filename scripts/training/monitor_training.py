#!/usr/bin/env python3
"""
Training monitor — tracks loss, checkpoints, and provides early stopping alerts.

Usage:
    python monitor_training.py \
        --log_dir /workspace/checkpoints/trellis/finetuned/ss_flow \
        --patience 3 \
        --check_interval 60
"""

import argparse
import json
import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_training_log(log_path: str) -> list:
    """Parse training log for loss values."""
    entries = []
    if not os.path.exists(log_path):
        return entries

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            # Look for loss entries (format varies by framework)
            if "loss" in line.lower() and "step" in line.lower():
                try:
                    # Try to extract step and loss from common log formats
                    parts = line.split()
                    step = None
                    loss = None
                    for i, p in enumerate(parts):
                        if "step" in p.lower() and i + 1 < len(parts):
                            try:
                                step = int(parts[i + 1].strip(",.:"))
                            except ValueError:
                                pass
                        if "loss" in p.lower() and i + 1 < len(parts):
                            try:
                                loss = float(parts[i + 1].strip(",.:"))
                            except ValueError:
                                pass
                    if step is not None and loss is not None:
                        entries.append({"step": step, "loss": loss})
                except Exception:
                    pass
    return entries


def find_checkpoints(checkpoint_dir: str) -> list:
    """Find saved checkpoints sorted by step."""
    checkpoints = []
    if not os.path.exists(checkpoint_dir):
        return checkpoints

    for item in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(path) or item.endswith((".pt", ".safetensors", ".ckpt")):
            # Try to extract step number from name
            try:
                step = int("".join(c for c in item if c.isdigit()))
                size_mb = sum(
                    os.path.getsize(os.path.join(path, f))
                    for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
                ) / (1024 * 1024) if os.path.isdir(path) else os.path.getsize(path) / (1024 * 1024)
                checkpoints.append({
                    "path": path,
                    "step": step,
                    "size_mb": round(size_mb, 1),
                    "time": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
                })
            except (ValueError, OSError):
                pass

    checkpoints.sort(key=lambda c: c["step"])
    return checkpoints


def check_early_stopping(losses: list, patience: int) -> dict:
    """Check if training should be stopped early."""
    if len(losses) < patience + 1:
        return {"should_stop": False, "reason": "not_enough_data"}

    # Check if validation loss has been increasing for `patience` consecutive checks
    recent = losses[-patience:]
    increasing = all(recent[i]["loss"] > recent[i - 1]["loss"] for i in range(1, len(recent)))

    if increasing:
        return {
            "should_stop": True,
            "reason": f"validation loss increased for {patience} consecutive checkpoints",
            "best_step": min(losses, key=lambda x: x["loss"])["step"],
            "best_loss": min(losses, key=lambda x: x["loss"])["loss"],
        }

    return {"should_stop": False, "best_step": min(losses, key=lambda x: x["loss"])["step"]}


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Directory containing training log and checkpoints")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (consecutive worsening checks)")
    parser.add_argument("--check_interval", type=int, default=60,
                        help="Check interval in seconds")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (don't loop)")
    args = parser.parse_args()

    while True:
        print("\n" + "=" * 60)
        print(f"Training Monitor — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Parse logs
        log_path = os.path.join(args.log_dir, "training.log")
        losses = parse_training_log(log_path)
        print(f"\nLoss entries found: {len(losses)}")
        if losses:
            latest = losses[-1]
            best = min(losses, key=lambda x: x["loss"])
            print(f"  Latest: step {latest['step']}, loss {latest['loss']:.6f}")
            print(f"  Best:   step {best['step']}, loss {best['loss']:.6f}")

            # Show recent trend
            if len(losses) >= 5:
                print(f"\n  Recent trend (last 5):")
                for entry in losses[-5:]:
                    print(f"    step {entry['step']:>8}: loss {entry['loss']:.6f}")

        # Find checkpoints
        checkpoints = find_checkpoints(args.log_dir)
        print(f"\nCheckpoints: {len(checkpoints)}")
        for ckpt in checkpoints[-5:]:
            print(f"  Step {ckpt['step']:>8}: {ckpt['size_mb']:>8.1f} MB — {ckpt['time']}")

        # Early stopping check
        if len(losses) >= args.patience + 1:
            es = check_early_stopping(losses, args.patience)
            if es["should_stop"]:
                print(f"\n⚠️  EARLY STOPPING ALERT!")
                print(f"  Reason: {es['reason']}")
                print(f"  Best checkpoint: step {es['best_step']} (loss {es['best_loss']:.6f})")
                print(f"  Consider stopping training and using the best checkpoint.")
            else:
                print(f"\n✅ Training looks healthy. Best step: {es.get('best_step', 'N/A')}")

        # GPU status
        try:
            import subprocess
            gpu_info = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if gpu_info.returncode == 0:
                parts = gpu_info.stdout.strip().split(",")
                print(f"\nGPU: {parts[0].strip()}% util | {parts[1].strip()}/{parts[2].strip()} MB | {parts[3].strip()}°C")
        except Exception:
            pass

        if args.once:
            break

        print(f"\nNext check in {args.check_interval}s...")
        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
