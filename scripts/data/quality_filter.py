#!/usr/bin/env python3
"""
Quality filter for 3D mesh assets. Validates mesh integrity and scores quality.

Usage:
    python quality_filter.py \
        --input_dir /workspace/data/raw \
        --output_dir /workspace/data/filtered \
        --min_score 4.0
"""

import argparse
import json
import os
import logging
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import trimesh
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def analyze_mesh(filepath: str) -> dict:
    """Analyze a single mesh file and return quality metrics."""
    result = {
        "path": filepath,
        "valid": False,
        "score": 0.0,
        "metrics": {},
        "issues": [],
    }

    try:
        scene = trimesh.load(filepath, force="scene")

        # Handle scenes with multiple meshes
        if isinstance(scene, trimesh.Scene):
            meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                result["issues"].append("no_trimesh_geometry")
                return result
            # Merge all meshes
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(scene, trimesh.Trimesh):
            mesh = scene
        else:
            result["issues"].append("unsupported_geometry_type")
            return result

        vertices = len(mesh.vertices)
        faces = len(mesh.faces)

        result["metrics"] = {
            "vertices": vertices,
            "faces": faces,
            "is_watertight": bool(mesh.is_watertight),
            "is_empty": bool(mesh.is_empty),
            "euler_number": int(mesh.euler_number) if hasattr(mesh, "euler_number") else None,
            "bounds": mesh.bounds.tolist() if mesh.bounds is not None else None,
            "extent": mesh.extents.tolist() if mesh.extents is not None else None,
            "has_vertex_normals": mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0,
            "has_face_normals": mesh.face_normals is not None and len(mesh.face_normals) > 0,
        }

        # Check UV coordinates
        has_uvs = False
        if isinstance(scene, trimesh.Scene):
            for g in scene.geometry.values():
                if isinstance(g, trimesh.Trimesh) and hasattr(g.visual, "uv") and g.visual.uv is not None:
                    has_uvs = True
                    break
        elif hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            has_uvs = True
        result["metrics"]["has_uvs"] = has_uvs

        # Check textures
        has_textures = False
        if isinstance(scene, trimesh.Scene):
            for g in scene.geometry.values():
                if isinstance(g, trimesh.Trimesh):
                    visual = g.visual
                    if hasattr(visual, "material") and visual.material is not None:
                        if hasattr(visual.material, "image") and visual.material.image is not None:
                            has_textures = True
                            break
        result["metrics"]["has_textures"] = has_textures

        # --- Quality Scoring ---
        score = 0.0

        # Vertex count (prefer 1K-100K sweet spot)
        if vertices < 100:
            result["issues"].append("too_few_vertices")
            return result
        elif vertices > 500000:
            result["issues"].append("too_many_vertices")
            return result
        elif 1000 <= vertices <= 100000:
            score += 2.0  # Ideal range
        else:
            score += 1.0  # Acceptable range

        # Face count sanity
        if faces < 50:
            result["issues"].append("too_few_faces")
            return result
        score += 1.0

        # Watertight bonus
        if mesh.is_watertight:
            score += 1.0
        else:
            score += 0.3

        # UV and texture bonuses
        if has_uvs:
            score += 0.5
        if has_textures:
            score += 1.0

        # Aspect ratio check (reject very elongated objects)
        if mesh.extents is not None:
            extents = mesh.extents
            max_extent = max(extents)
            min_extent = min(extents) if min(extents) > 0 else 0.001
            aspect_ratio = max_extent / min_extent
            result["metrics"]["aspect_ratio"] = float(aspect_ratio)
            if aspect_ratio > 10:
                result["issues"].append("extreme_aspect_ratio")
                score -= 1.0
            elif aspect_ratio < 5:
                score += 0.5

        # Degenerate face check
        try:
            areas = mesh.area_faces
            degenerate_count = np.sum(areas < 1e-10)
            degenerate_ratio = degenerate_count / len(areas) if len(areas) > 0 else 0
            result["metrics"]["degenerate_face_ratio"] = float(degenerate_ratio)
            if degenerate_ratio > 0.05:
                result["issues"].append("many_degenerate_faces")
                score -= 0.5
        except Exception:
            pass

        result["score"] = round(score, 2)
        result["valid"] = score >= 2.0  # Minimum viable score

    except Exception as e:
        result["issues"].append(f"load_error: {str(e)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Quality filter for 3D mesh assets")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory with raw mesh files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for filtered assets")
    parser.add_argument("--min_score", type=float, default=4.0,
                        help="Minimum quality score to pass filter")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to manifest.json (optional, for metadata)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all mesh files
    extensions = {".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl"}
    mesh_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                mesh_files.append(os.path.join(root, f))

    logger.info(f"Found {len(mesh_files)} mesh files in {args.input_dir}")

    # Analyze all meshes
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_mesh, fp): fp for fp in mesh_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing"):
            results.append(future.result())

    # Filter and copy
    passed = [r for r in results if r["valid"] and r["score"] >= args.min_score]
    failed = [r for r in results if not r["valid"] or r["score"] < args.min_score]

    logger.info(f"Quality filter results: {len(passed)} passed, {len(failed)} failed")
    logger.info(f"  Score distribution: min={min(r['score'] for r in results):.1f}, "
                f"max={max(r['score'] for r in results):.1f}, "
                f"mean={np.mean([r['score'] for r in results]):.1f}")

    # Copy passed files to output
    for r in tqdm(passed, desc="Copying passed assets"):
        src = r["path"]
        dst = os.path.join(args.output_dir, os.path.basename(src))
        if src != dst:
            shutil.copy2(src, dst)

    # Save filter report
    report = {
        "total_input": len(mesh_files),
        "passed": len(passed),
        "failed": len(failed),
        "min_score_threshold": args.min_score,
        "score_distribution": {
            "min": float(min(r["score"] for r in results)),
            "max": float(max(r["score"] for r in results)),
            "mean": float(np.mean([r["score"] for r in results])),
            "median": float(np.median([r["score"] for r in results])),
        },
        "common_issues": {},
        "passed_assets": [{"path": os.path.basename(r["path"]), "score": r["score"]} for r in passed],
    }

    # Count common issues
    for r in failed:
        for issue in r["issues"]:
            report["common_issues"][issue] = report["common_issues"].get(issue, 0) + 1

    report_path = os.path.join(args.output_dir, "filter_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Filter report saved to {report_path}")
    logger.info(f"Passed assets saved to {args.output_dir}")


if __name__ == "__main__":
    main()
