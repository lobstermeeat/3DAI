#!/usr/bin/env python3
"""
Evaluate texture quality: FID, LPIPS, PSNR, SSIM.

Compares rendered views of generated textured meshes against ground truth renders.

Usage:
    python eval_texture.py \
        --generated_dir /workspace/results/samples/textured \
        --ground_truth_dir /workspace/data/val_renders \
        --output /workspace/results/eval/texture_metrics.json
"""

import argparse
import json
import os
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((img_a.astype(float) - img_b.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute Structural Similarity Index."""
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(img_a, img_b, channel_axis=2, data_range=255))
    except ImportError:
        # Simplified SSIM
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        mu_a = np.mean(img_a.astype(float), axis=(0, 1))
        mu_b = np.mean(img_b.astype(float), axis=(0, 1))
        sigma_a = np.std(img_a.astype(float), axis=(0, 1))
        sigma_b = np.std(img_b.astype(float), axis=(0, 1))
        sigma_ab = np.mean((img_a.astype(float) - mu_a) * (img_b.astype(float) - mu_b), axis=(0, 1))

        ssim = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / \
               ((mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a ** 2 + sigma_b ** 2 + C2))
        return float(np.mean(ssim))


def compute_lpips_batch(gen_images: list, gt_images: list) -> list:
    """Compute LPIPS perceptual similarity for a batch of image pairs."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="alex").cuda()

        scores = []
        for gen_img, gt_img in zip(gen_images, gt_images):
            # Convert to tensor [-1, 1]
            gen_t = torch.from_numpy(gen_img).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1
            gt_t = torch.from_numpy(gt_img).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1

            with torch.no_grad():
                score = loss_fn(gen_t.cuda(), gt_t.cuda())
            scores.append(float(score.item()))

        return scores
    except ImportError:
        logger.warning("LPIPS not available, skipping")
        return [0.0] * len(gen_images)


def compute_fid(gen_dir: str, gt_dir: str) -> float:
    """Compute Frechet Inception Distance between two directories of images."""
    try:
        from pytorch_fid import fid_score
        score = fid_score.calculate_fid_given_paths(
            [gen_dir, gt_dir],
            batch_size=50,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dims=2048,
        )
        return float(score)
    except ImportError:
        logger.warning("pytorch-fid not available, skipping FID")
        return -1.0
    except Exception as e:
        logger.warning(f"FID computation failed: {e}")
        return -1.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate texture quality")
    parser.add_argument("--generated_dir", type=str, required=True,
                        help="Directory with rendered views of generated meshes")
    parser.add_argument("--ground_truth_dir", type=str, required=True,
                        help="Directory with rendered views of ground truth meshes")
    parser.add_argument("--output", type=str, default="texture_metrics.json")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Expected image resolution")
    args = parser.parse_args()

    # Find matching image pairs
    gen_images = sorted([f for f in os.listdir(args.generated_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    gt_images = sorted([f for f in os.listdir(args.ground_truth_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    # Match by name
    gen_set = {Path(f).stem: f for f in gen_images}
    gt_set = {Path(f).stem: f for f in gt_images}
    common = sorted(set(gen_set.keys()) & set(gt_set.keys()))

    logger.info(f"Found {len(common)} matching image pairs")

    # Load images
    gen_imgs = []
    gt_imgs = []
    per_image_metrics = []

    for name in tqdm(common, desc="Computing per-image metrics"):
        gen_path = os.path.join(args.generated_dir, gen_set[name])
        gt_path = os.path.join(args.ground_truth_dir, gt_set[name])

        gen_img = np.array(Image.open(gen_path).convert("RGB").resize(
            (args.resolution, args.resolution)))
        gt_img = np.array(Image.open(gt_path).convert("RGB").resize(
            (args.resolution, args.resolution)))

        gen_imgs.append(gen_img)
        gt_imgs.append(gt_img)

        psnr = compute_psnr(gen_img, gt_img)
        ssim = compute_ssim(gen_img, gt_img)

        per_image_metrics.append({
            "name": name,
            "psnr": psnr,
            "ssim": ssim,
        })

    # Batch LPIPS
    logger.info("Computing LPIPS...")
    lpips_scores = compute_lpips_batch(gen_imgs, gt_imgs)
    for i, score in enumerate(lpips_scores):
        per_image_metrics[i]["lpips"] = score

    # FID (directory-level metric)
    logger.info("Computing FID...")
    fid = compute_fid(args.generated_dir, args.ground_truth_dir)

    # Aggregate
    report = {
        "total_pairs": len(common),
        "fid": fid,
        "aggregate": {
            "psnr": {
                "mean": float(np.mean([m["psnr"] for m in per_image_metrics])),
                "std": float(np.std([m["psnr"] for m in per_image_metrics])),
            },
            "ssim": {
                "mean": float(np.mean([m["ssim"] for m in per_image_metrics])),
                "std": float(np.std([m["ssim"] for m in per_image_metrics])),
            },
            "lpips": {
                "mean": float(np.mean([m["lpips"] for m in per_image_metrics])),
                "std": float(np.std([m["lpips"] for m in per_image_metrics])),
            },
        },
        "per_image": per_image_metrics,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nResults saved to {args.output}")
    logger.info(f"FID: {fid:.2f}")
    logger.info(f"PSNR: {report['aggregate']['psnr']['mean']:.2f} +/- {report['aggregate']['psnr']['std']:.2f}")
    logger.info(f"SSIM: {report['aggregate']['ssim']['mean']:.4f} +/- {report['aggregate']['ssim']['std']:.4f}")
    logger.info(f"LPIPS: {report['aggregate']['lpips']['mean']:.4f} +/- {report['aggregate']['lpips']['std']:.4f}")


if __name__ == "__main__":
    main()
