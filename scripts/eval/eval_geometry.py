#!/usr/bin/env python3
"""
Evaluate geometry quality: Chamfer Distance, F-Score, Normal Consistency, Volume IoU.

Compares generated meshes against ground truth held-out set.

Usage:
    python eval_geometry.py \
        --generated_dir /workspace/results/samples/geometry \
        --ground_truth_dir /workspace/data/val \
        --output /workspace/results/eval/geometry_metrics.json
"""

import argparse
import json
import os
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import trimesh
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sample_points(mesh: trimesh.Trimesh, n_points: int = 10000) -> np.ndarray:
    """Sample points uniformly from mesh surface."""
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Compute bidirectional Chamfer Distance."""
    from scipy.spatial import KDTree

    tree_a = KDTree(points_a)
    tree_b = KDTree(points_b)

    dist_a_to_b, _ = tree_b.query(points_a)
    dist_b_to_a, _ = tree_a.query(points_b)

    cd = np.mean(dist_a_to_b ** 2) + np.mean(dist_b_to_a ** 2)
    return float(cd)


def f_score(points_a: np.ndarray, points_b: np.ndarray, threshold: float = 0.01) -> float:
    """Compute F-Score at given threshold."""
    from scipy.spatial import KDTree

    tree_a = KDTree(points_a)
    tree_b = KDTree(points_b)

    dist_a_to_b, _ = tree_b.query(points_a)
    dist_b_to_a, _ = tree_a.query(points_b)

    precision = np.mean(dist_a_to_b < threshold)
    recall = np.mean(dist_b_to_a < threshold)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


def normal_consistency(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_points: int = 10000) -> float:
    """Compute Normal Consistency between two meshes."""
    from scipy.spatial import KDTree

    points_a, face_idx_a = trimesh.sample.sample_surface(mesh_a, n_points)
    points_b, face_idx_b = trimesh.sample.sample_surface(mesh_b, n_points)

    normals_a = mesh_a.face_normals[face_idx_a]
    normals_b = mesh_b.face_normals[face_idx_b]

    tree_b = KDTree(points_b)
    _, indices = tree_b.query(points_a)

    # Dot product of normals (absolute value since direction may be flipped)
    dots = np.abs(np.sum(normals_a * normals_b[indices], axis=1))
    return float(np.mean(dots))


def volume_iou(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, resolution: int = 64) -> float:
    """Compute Volume IoU by voxelizing both meshes."""
    try:
        # Create voxel grids
        pitch = max(max(mesh_a.extents), max(mesh_b.extents)) / resolution
        voxels_a = mesh_a.voxelized(pitch)
        voxels_b = mesh_b.voxelized(pitch)

        # Get filled voxel sets
        points_a = set(map(tuple, voxels_a.points.round(6)))
        points_b = set(map(tuple, voxels_b.points.round(6)))

        intersection = len(points_a & points_b)
        union = len(points_a | points_b)

        return float(intersection / union) if union > 0 else 0.0
    except Exception as e:
        logger.warning(f"Volume IoU failed: {e}")
        return 0.0


def evaluate_pair(gen_path: str, gt_path: str, n_points: int = 10000) -> dict:
    """Evaluate a single generated mesh against ground truth."""
    try:
        gen_mesh = trimesh.load(gen_path, force="mesh")
        gt_mesh = trimesh.load(gt_path, force="mesh")

        # Normalize both to unit cube
        for mesh in [gen_mesh, gt_mesh]:
            mesh.vertices -= mesh.centroid
            scale = max(mesh.extents) if max(mesh.extents) > 0 else 1.0
            mesh.vertices /= scale

        # Sample points
        gen_points = sample_points(gen_mesh, n_points)
        gt_points = sample_points(gt_mesh, n_points)

        metrics = {
            "name": Path(gen_path).stem,
            "chamfer_distance": chamfer_distance(gen_points, gt_points),
            "f_score_001": f_score(gen_points, gt_points, threshold=0.01),
            "f_score_005": f_score(gen_points, gt_points, threshold=0.05),
            "normal_consistency": normal_consistency(gen_mesh, gt_mesh, n_points),
            "volume_iou": volume_iou(gen_mesh, gt_mesh),
            "gen_vertices": len(gen_mesh.vertices),
            "gen_faces": len(gen_mesh.faces),
            "gt_vertices": len(gt_mesh.vertices),
            "gt_faces": len(gt_mesh.faces),
        }
        return metrics

    except Exception as e:
        return {"name": Path(gen_path).stem, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate geometry quality")
    parser.add_argument("--generated_dir", type=str, required=True)
    parser.add_argument("--ground_truth_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="geometry_metrics.json")
    parser.add_argument("--n_points", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Match generated files to ground truth by name
    gen_files = {Path(f).stem: os.path.join(args.generated_dir, f)
                 for f in os.listdir(args.generated_dir)}
    gt_files = {Path(f).stem: os.path.join(args.ground_truth_dir, f)
                for f in os.listdir(args.ground_truth_dir)}

    pairs = [(gen_files[name], gt_files[name])
             for name in gen_files if name in gt_files]

    logger.info(f"Evaluating {len(pairs)} pairs...")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_pair, g, gt, args.n_points): (g, gt)
                   for g, gt in pairs}
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    # Compute aggregates
    valid = [r for r in results if "error" not in r]
    report = {
        "total_pairs": len(pairs),
        "valid_evaluations": len(valid),
        "errors": len(results) - len(valid),
        "aggregate": {
            metric: {
                "mean": float(np.mean([r[metric] for r in valid])),
                "std": float(np.std([r[metric] for r in valid])),
                "median": float(np.median([r[metric] for r in valid])),
                "min": float(np.min([r[metric] for r in valid])),
                "max": float(np.max([r[metric] for r in valid])),
            }
            for metric in ["chamfer_distance", "f_score_001", "f_score_005",
                          "normal_consistency", "volume_iou"]
            if valid
        },
        "per_asset": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nResults saved to {args.output}")
    if valid:
        logger.info("Aggregate metrics:")
        for metric, stats in report["aggregate"].items():
            logger.info(f"  {metric}: {stats['mean']:.6f} +/- {stats['std']:.6f}")


if __name__ == "__main__":
    main()
