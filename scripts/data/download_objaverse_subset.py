#!/usr/bin/env python3
"""
Download and filter a high-quality subset from Objaverse++ for fine-tuning.

Usage:
    python download_objaverse_subset.py \
        --output_dir /workspace/data/raw \
        --max_objects 3000 \
        --min_aesthetic_score 4.0 \
        --domain furniture
"""

import argparse
import json
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import objaverse
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Domain keyword filters for selecting homogeneous subsets
DOMAIN_KEYWORDS = {
    "furniture": [
        "chair", "table", "desk", "sofa", "couch", "bed", "shelf", "cabinet",
        "dresser", "wardrobe", "bench", "stool", "bookshelf", "nightstand",
        "ottoman", "armchair", "dining", "lamp", "chandelier"
    ],
    "characters": [
        "character", "human", "person", "figure", "figurine", "statue",
        "robot", "avatar", "warrior", "knight", "soldier", "anime"
    ],
    "vehicles": [
        "car", "truck", "vehicle", "bus", "motorcycle", "bicycle", "airplane",
        "helicopter", "boat", "ship", "train", "tank", "spaceship"
    ],
    "architecture": [
        "building", "house", "tower", "castle", "bridge", "temple",
        "church", "mosque", "skyscraper", "cabin", "cottage", "ruins"
    ],
    "props": [
        "weapon", "sword", "shield", "tool", "container", "box", "barrel",
        "crate", "potion", "gem", "crystal", "key", "scroll", "book"
    ],
}


def load_objaverse_annotations():
    """Load Objaverse annotations with metadata."""
    logger.info("Loading Objaverse annotations...")
    annotations = objaverse.load_annotations()
    logger.info(f"Loaded {len(annotations)} annotations")
    return annotations


def filter_by_domain(annotations: dict, domain: str) -> dict:
    """Filter annotations by domain keywords."""
    if domain == "all":
        return annotations

    keywords = DOMAIN_KEYWORDS.get(domain, [])
    if not keywords:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_KEYWORDS.keys())}")

    filtered = {}
    for uid, meta in annotations.items():
        name = (meta.get("name", "") or "").lower()
        tags = " ".join(t.get("name", "") for t in meta.get("tags", [])).lower()
        categories = " ".join(c.get("name", "") for c in meta.get("categories", [])).lower()
        searchable = f"{name} {tags} {categories}"

        if any(kw in searchable for kw in keywords):
            filtered[uid] = meta

    logger.info(f"Domain '{domain}' filter: {len(annotations)} -> {len(filtered)} objects")
    return filtered


def filter_by_quality(annotations: dict, min_aesthetic_score: float) -> dict:
    """Filter by aesthetic/quality score if available."""
    filtered = {}
    for uid, meta in annotations.items():
        # Check if the object has quality indicators
        has_textures = meta.get("textureCount", 0) > 0
        vertex_count = meta.get("vertexCount", 0)
        face_count = meta.get("faceCount", 0)

        # Basic quality heuristics
        if vertex_count < 100 or vertex_count > 500000:
            continue
        if face_count < 50:
            continue

        # Prefer textured objects
        score = 3.0  # base score
        if has_textures:
            score += 1.0
        if 1000 <= vertex_count <= 100000:
            score += 0.5  # sweet spot for vertex count
        if meta.get("name"):
            score += 0.3
        if meta.get("tags"):
            score += 0.2

        if score >= min_aesthetic_score:
            meta["_quality_score"] = score
            filtered[uid] = meta

    logger.info(f"Quality filter (>={min_aesthetic_score}): {len(annotations)} -> {len(filtered)} objects")
    return filtered


def download_objects(uids: list, output_dir: str, max_workers: int = 8) -> dict:
    """Download 3D objects from Objaverse."""
    logger.info(f"Downloading {len(uids)} objects to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    objects = objaverse.load_objects(
        uids=uids,
        download_processes=max_workers,
    )

    # Copy files from objaverse cache to output_dir
    import shutil
    copied = {}
    for uid, cached_path in objects.items():
        if os.path.exists(cached_path):
            ext = os.path.splitext(cached_path)[1]
            dest = os.path.join(output_dir, f"{uid}{ext}")
            shutil.copy2(cached_path, dest)
            copied[uid] = dest
        else:
            logger.warning(f"Cached file not found for {uid}: {cached_path}")

    logger.info(f"Downloaded {len(objects)} objects, copied {len(copied)} to {output_dir}")
    return copied


def save_manifest(objects: dict, annotations: dict, output_dir: str):
    """Save a manifest JSON with metadata for each downloaded object."""
    manifest = []
    for uid, path in objects.items():
        meta = annotations.get(uid, {})
        manifest.append({
            "uid": uid,
            "path": str(path),
            "name": meta.get("name", ""),
            "tags": [t.get("name", "") for t in meta.get("tags", [])],
            "vertex_count": meta.get("vertexCount", 0),
            "face_count": meta.get("faceCount", 0),
            "has_textures": meta.get("textureCount", 0) > 0,
            "quality_score": meta.get("_quality_score", 0.0),
        })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved manifest with {len(manifest)} entries to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Objaverse++ subset for fine-tuning")
    parser.add_argument("--output_dir", type=str, default="/workspace/data/raw",
                        help="Output directory for downloaded objects")
    parser.add_argument("--max_objects", type=int, default=3000,
                        help="Maximum number of objects to download")
    parser.add_argument("--min_aesthetic_score", type=float, default=4.0,
                        help="Minimum quality/aesthetic score threshold")
    parser.add_argument("--domain", type=str, default="all",
                        choices=["all"] + list(DOMAIN_KEYWORDS.keys()),
                        help="Domain to filter for (homogeneous data converges better)")
    parser.add_argument("--download_workers", type=int, default=8,
                        help="Number of parallel download workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible subset selection")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    # Step 1: Load annotations
    annotations = load_objaverse_annotations()

    # Step 2: Filter by domain
    annotations = filter_by_domain(annotations, args.domain)

    # Step 3: Filter by quality
    annotations = filter_by_quality(annotations, args.min_aesthetic_score)

    # Step 4: Select subset
    uids = list(annotations.keys())
    if len(uids) > args.max_objects:
        # Sort by quality score, take top N
        uids.sort(key=lambda u: annotations[u].get("_quality_score", 0), reverse=True)
        uids = uids[:args.max_objects]
        logger.info(f"Selected top {args.max_objects} objects by quality score")

    logger.info(f"Final dataset: {len(uids)} objects")

    # Step 5: Download
    objects = download_objects(uids, args.output_dir, args.download_workers)

    # Step 6: Save manifest
    save_manifest(objects, annotations, args.output_dir)

    logger.info("Done!")
    logger.info(f"  Objects: {len(objects)}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  Manifest: {os.path.join(args.output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
