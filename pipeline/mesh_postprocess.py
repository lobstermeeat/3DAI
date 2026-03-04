"""
Mesh Post-Processing Pipeline.

Handles UV unwrapping, texture baking, mesh cleanup, and GLB export.
"""

import os
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PostProcessConfig:
    """Configuration for mesh post-processing."""
    # Mesh cleanup
    weld_vertices: bool = True
    weld_threshold: float = 1e-6
    remove_degenerate: bool = True
    remove_duplicate_faces: bool = True

    # UV unwrapping
    uv_method: str = "xatlas"  # "xatlas" or "smart_uv" (blender)
    uv_padding: int = 2

    # Normalization
    center_origin: bool = True
    normalize_scale: bool = True
    target_scale: float = 1.0

    # Export
    draco_compression: bool = True
    draco_quantization_position: int = 14
    draco_quantization_normal: int = 10
    draco_quantization_texcoord: int = 12


class MeshPostProcessor:
    """Post-process generated meshes for production use."""

    def __init__(self, config: PostProcessConfig = None):
        self.config = config or PostProcessConfig()

    def clean_mesh(self, mesh) -> "trimesh.Trimesh":
        """Clean mesh: weld vertices, remove degenerate faces."""
        import trimesh

        logger.info(f"Cleaning mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        start = time.time()

        if self.config.remove_degenerate:
            # Remove zero-area faces
            mask = mesh.area_faces > 1e-10
            if not mask.all():
                removed = (~mask).sum()
                mesh.update_faces(mask)
                logger.info(f"  Removed {removed} degenerate faces")

        if self.config.remove_duplicate_faces:
            # Remove duplicate faces
            unique_faces = np.unique(np.sort(mesh.faces, axis=1), axis=0)
            if len(unique_faces) < len(mesh.faces):
                removed = len(mesh.faces) - len(unique_faces)
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=unique_faces)
                logger.info(f"  Removed {removed} duplicate faces")

        if self.config.weld_vertices:
            # Merge close vertices
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            logger.info(f"  Welded vertices: now {len(mesh.vertices)} verts")

        logger.info(f"Mesh cleaned in {time.time() - start:.2f}s")
        return mesh

    def normalize(self, mesh) -> "trimesh.Trimesh":
        """Center at origin and normalize scale."""
        if self.config.center_origin:
            centroid = mesh.centroid
            mesh.vertices -= centroid

        if self.config.normalize_scale:
            extent = max(mesh.extents) if max(mesh.extents) > 0 else 1.0
            scale = self.config.target_scale / extent
            mesh.vertices *= scale

        return mesh

    def unwrap_uvs(self, mesh) -> "trimesh.Trimesh":
        """Generate UV coordinates using xatlas."""
        import trimesh

        if self.config.uv_method == "xatlas":
            try:
                import xatlas

                logger.info("Generating UV coordinates with xatlas...")
                start = time.time()

                vmapping, indices, uvs = xatlas.parametrize(
                    mesh.vertices.astype(np.float32),
                    mesh.faces.astype(np.uint32),
                )

                # Rebuild mesh with UV coordinates
                new_vertices = mesh.vertices[vmapping]
                new_faces = indices

                # Create textured mesh with UVs
                visual = trimesh.visual.TextureVisuals(uv=uvs)
                mesh = trimesh.Trimesh(
                    vertices=new_vertices,
                    faces=new_faces,
                    visual=visual,
                )

                logger.info(f"UV unwrap complete in {time.time() - start:.2f}s: "
                           f"{len(mesh.vertices)} verts, {len(uvs)} UVs")

            except ImportError:
                logger.warning("xatlas not available, skipping UV unwrap")
        else:
            logger.warning(f"Unknown UV method: {self.config.uv_method}")

        return mesh

    def export_glb(self, mesh, output_path: str) -> str:
        """Export mesh as GLB with optional Draco compression."""
        import trimesh

        logger.info(f"Exporting GLB to {output_path}...")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Export to GLB
        mesh.export(output_path, file_type="glb")

        # Apply Draco compression if requested
        if self.config.draco_compression:
            try:
                from gltf_pipeline import Pipeline

                compressed_path = output_path.replace(".glb", "_draco.glb")
                Pipeline.process_file(
                    output_path,
                    compressed_path,
                    draco_compression=True,
                )
                # Replace original with compressed
                os.replace(compressed_path, output_path)
                logger.info("Applied Draco compression")
            except ImportError:
                logger.warning("gltf-pipeline not available, skipping Draco compression")

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Exported: {output_path} ({size_mb:.1f} MB)")

        return output_path

    def finalize(self, mesh, output_path: str) -> str:
        """
        Run full post-processing pipeline and export.

        Args:
            mesh: trimesh.Trimesh (textured or untextured)
            output_path: Path to save final GLB

        Returns:
            str: Path to exported GLB file
        """
        logger.info("Starting post-processing pipeline...")
        start = time.time()

        # Step 1: Clean
        mesh = self.clean_mesh(mesh)

        # Step 2: Normalize
        mesh = self.normalize(mesh)

        # Step 3: UV unwrap (only if mesh doesn't have UVs)
        has_uvs = (hasattr(mesh.visual, "uv") and mesh.visual.uv is not None
                   and len(mesh.visual.uv) > 0)
        if not has_uvs:
            mesh = self.unwrap_uvs(mesh)

        # Step 4: Export
        result_path = self.export_glb(mesh, output_path)

        duration = time.time() - start
        logger.info(f"Post-processing complete in {duration:.1f}s")

        return result_path
