#!/usr/bin/env python3
"""
Set up custom dataset for TRELLIS.2 data_toolkit pipeline.

Creates:
1. datasets/Forge3D.py module in TRELLIS.2's data_toolkit directory
2. metadata.csv from our downloaded GLB files
3. Proper directory structure expected by TRELLIS.2

Usage:
    python setup_trellis_dataset.py \
        --input_dir /workspace/data/filtered \
        --trellis_root /workspace/data/trellis_root \
        --trellis_dir /workspace/repos/TRELLIS.2
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def compute_sha256(filepath):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def create_dataset_module(trellis_dir):
    """Create datasets/Forge3D.py in TRELLIS.2's data_toolkit directory."""
    datasets_dir = os.path.join(trellis_dir, 'data_toolkit', 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)

    # Create __init__.py if missing
    init_path = os.path.join(datasets_dir, '__init__.py')
    if not os.path.exists(init_path):
        with open(init_path, 'w') as f:
            f.write('')

    module_code = '''"""
Forge3D custom dataset module for TRELLIS.2 data_toolkit.
Handles locally downloaded GLB files from Objaverse.
"""
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def add_args(parser):
    """Add dataset-specific arguments."""
    parser.add_argument('--source', type=str, default=None,
                        help='Data source (unused for Forge3D)')


def foreach_instance(metadata, root, func, max_workers=0, desc='Processing', no_file=False):
    """
    Iterate over dataset instances and apply func.

    Args:
        metadata: DataFrame with sha256, local_path columns
        root: root directory containing raw/ subdirectory
        func: callable(file_path, metadatum) -> dict
        max_workers: number of parallel workers (0 = sequential)
        desc: progress bar description
        no_file: if True, pass None as file_path (used for steps that
                 read from processed dirs, not raw files)

    Returns:
        DataFrame with results
    """
    records = []

    if max_workers <= 0:
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc=desc):
            if no_file:
                file_path = None
            else:
                file_path = os.path.join(root, 'raw', row['local_path'])
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue
            try:
                result = func(file_path, row.to_dict())
                records.append(result)
            except Exception as e:
                print(f"Error processing {row['sha256']}: {e}")
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for _, row in metadata.iterrows():
                if no_file:
                    file_path = None
                else:
                    file_path = os.path.join(root, 'raw', row['local_path'])
                    if not os.path.exists(file_path):
                        print(f"File not found: {file_path}")
                        continue
                future = executor.submit(func, file_path, row.to_dict())
                futures[future] = row['sha256']

            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                try:
                    result = future.result()
                    records.append(result)
                except Exception as e:
                    sha256 = futures[future]
                    print(f"Error processing {sha256}: {e}")

    return pd.DataFrame.from_records(records)
'''

    module_path = os.path.join(datasets_dir, 'Forge3D.py')
    with open(module_path, 'w') as f:
        f.write(module_code)
    print(f"Created dataset module: {module_path}")


def create_metadata_and_structure(input_dir, trellis_root):
    """
    Create metadata.csv and directory structure from filtered GLB files.

    Structure:
        trellis_root/
            metadata.csv
            raw/
                {sha256}.glb -> symlink or copy
    """
    os.makedirs(os.path.join(trellis_root, 'raw'), exist_ok=True)

    # Find all mesh files
    mesh_extensions = {'.glb', '.gltf', '.obj', '.fbx', '.ply'}
    mesh_files = []
    for f in os.listdir(input_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in mesh_extensions:
            mesh_files.append(f)

    print(f"Found {len(mesh_files)} mesh files in {input_dir}")

    # Load manifest if available for aesthetic scores
    manifest_scores = {}
    manifest_path = os.path.join(input_dir, '..', 'raw', 'manifest.json')
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        for entry in manifest:
            uid = entry['uid']
            manifest_scores[uid] = entry.get('quality_score', 5.0)

    # Also check filter_report for scores
    filter_report_path = os.path.join(input_dir, 'filter_report.json')

    records = []
    for filename in tqdm(mesh_files, desc="Building metadata"):
        filepath = os.path.join(input_dir, filename)
        sha256 = compute_sha256(filepath)
        uid = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]

        # Copy/symlink to raw directory with sha256 name
        dest = os.path.join(trellis_root, 'raw', f'{sha256}{ext}')
        if not os.path.exists(dest):
            shutil.copy2(filepath, dest)

        # Get aesthetic score
        score = manifest_scores.get(uid, 5.0)

        records.append({
            'sha256': sha256,
            'local_path': f'{sha256}{ext}',
            'original_uid': uid,
            'aesthetic_score': score,
        })

    # Save metadata
    df = pd.DataFrame.from_records(records)
    metadata_path = os.path.join(trellis_root, 'metadata.csv')
    df.to_csv(metadata_path, index=False)
    print(f"Created metadata.csv with {len(df)} entries at {metadata_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Set up TRELLIS.2 dataset from filtered GLBs")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory with filtered GLB files')
    parser.add_argument('--trellis_root', type=str, required=True,
                        help='Output root directory for TRELLIS.2 dataset')
    parser.add_argument('--trellis_dir', type=str, default='/workspace/repos/TRELLIS.2',
                        help='Path to TRELLIS.2 repo')
    args = parser.parse_args()

    print("=" * 50)
    print("Setting up TRELLIS.2 dataset")
    print("=" * 50)

    # Step 1: Create dataset module
    print("\n[1/2] Creating Forge3D dataset module...")
    create_dataset_module(args.trellis_dir)

    # Step 2: Create metadata and directory structure
    print("\n[2/2] Creating metadata and copying files...")
    df = create_metadata_and_structure(args.input_dir, args.trellis_root)

    print(f"\nSetup complete!")
    print(f"  Dataset root: {args.trellis_root}")
    print(f"  Assets: {len(df)}")
    print(f"  Metadata: {os.path.join(args.trellis_root, 'metadata.csv')}")
    print(f"\nNext: run the TRELLIS.2 data_toolkit pipeline:")
    print(f"  cd {args.trellis_dir}")
    print(f"  python data_toolkit/dump_mesh.py Forge3D --root {args.trellis_root}")


if __name__ == "__main__":
    main()
