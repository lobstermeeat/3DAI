"""
TRELLIS.2 Geometry Generation Stage.

Generates untextured 3D mesh from an input image using fine-tuned
TRELLIS.2 sparse structure flow + shape flow models.
"""

import os
import sys
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrellisConfig:
    """Configuration for TRELLIS.2 geometry generation."""
    # Model paths
    pretrained_dir: str = "/workspace/checkpoints/trellis/pretrained"
    finetuned_ss_flow: Optional[str] = None  # If None, uses pretrained
    finetuned_shape_flow: Optional[str] = None  # If None, uses pretrained

    # Generation params
    seed: int = 42
    ss_guidance_scale: float = 7.5
    shape_guidance_scale: float = 7.5
    ss_num_steps: int = 12
    shape_num_steps: int = 12
    resolution: int = 512  # O-Voxel resolution

    # Output
    output_format: str = "trimesh"  # "trimesh" or "glb"
    simplify_faces: Optional[int] = 100000  # Target face count, None to skip


class TrellisGeometryStage:
    """TRELLIS.2 geometry generation wrapper."""

    def __init__(self, config: TrellisConfig):
        self.config = config
        self.model = None
        self._loaded = False

    def load(self):
        """Load TRELLIS.2 model into GPU memory."""
        if self._loaded:
            return

        logger.info("Loading TRELLIS.2 model...")
        start = time.time()

        try:
            # Add TRELLIS.2 to path
            trellis_path = os.environ.get("TRELLIS_PATH", "/opt/TRELLIS.2")
            if trellis_path not in sys.path:
                sys.path.insert(0, trellis_path)

            from trellis.pipelines import TrellisImageTo3DPipeline

            # Load pipeline
            self.model = TrellisImageTo3DPipeline.from_pretrained(
                self.config.pretrained_dir
            )

            # Override with fine-tuned checkpoints if available
            if self.config.finetuned_ss_flow and os.path.exists(self.config.finetuned_ss_flow):
                logger.info(f"Loading fine-tuned ss_flow from {self.config.finetuned_ss_flow}")
                self.model.load_finetuned(
                    "ss_flow", self.config.finetuned_ss_flow
                )

            if self.config.finetuned_shape_flow and os.path.exists(self.config.finetuned_shape_flow):
                logger.info(f"Loading fine-tuned shape_flow from {self.config.finetuned_shape_flow}")
                self.model.load_finetuned(
                    "slat_flow_img2shape", self.config.finetuned_shape_flow
                )

            self.model.cuda()
            self._loaded = True
            logger.info(f"TRELLIS.2 loaded in {time.time() - start:.1f}s")

        except ImportError as e:
            logger.error(f"Failed to import TRELLIS.2: {e}")
            logger.error("Ensure TRELLIS.2 is installed and TRELLIS_PATH is set")
            raise

    def unload(self):
        """Unload model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._loaded = False
            torch.cuda.empty_cache()
            logger.info("TRELLIS.2 unloaded from GPU")

    def generate(self, image, seed: Optional[int] = None) -> "trimesh.Trimesh":
        """
        Generate 3D geometry from an input image.

        Args:
            image: PIL Image or path to image file
            seed: Random seed (overrides config)

        Returns:
            trimesh.Trimesh: Generated mesh (untextured)
        """
        import trimesh
        from PIL import Image

        self.load()

        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGBA")

        seed = seed or self.config.seed
        torch.manual_seed(seed)

        logger.info(f"Generating geometry (seed={seed}, res={self.config.resolution})...")
        start = time.time()

        # Run TRELLIS.2 pipeline
        outputs = self.model.run(
            image,
            seed=seed,
            ss_guidance_strength=self.config.ss_guidance_scale,
            ss_sampling_steps=self.config.ss_num_steps,
            slat_guidance_strength=self.config.shape_guidance_scale,
            slat_sampling_steps=self.config.shape_num_steps,
        )

        # Extract mesh from output
        # TRELLIS.2 outputs can include Gaussians, NeRF, and mesh
        mesh_output = outputs.get("mesh") or outputs.get("trimesh")

        if mesh_output is None:
            # Try to extract mesh from O-Voxel output
            logger.info("Extracting mesh from O-Voxel representation...")
            mesh_output = self.model.extract_mesh(outputs)

        if isinstance(mesh_output, trimesh.Trimesh):
            mesh = mesh_output
        elif hasattr(mesh_output, "vertices") and hasattr(mesh_output, "faces"):
            mesh = trimesh.Trimesh(
                vertices=np.array(mesh_output.vertices),
                faces=np.array(mesh_output.faces),
            )
        else:
            raise ValueError(f"Unexpected mesh output type: {type(mesh_output)}")

        duration = time.time() - start
        logger.info(f"Geometry generated in {duration:.1f}s: "
                     f"{len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Optional simplification
        if self.config.simplify_faces and len(mesh.faces) > self.config.simplify_faces:
            logger.info(f"Simplifying mesh: {len(mesh.faces)} -> {self.config.simplify_faces} faces")
            try:
                import pymeshlab
                ms = pymeshlab.MeshSet()
                ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=self.config.simplify_faces,
                    preservenormal=True,
                )
                simplified = ms.current_mesh()
                mesh = trimesh.Trimesh(
                    vertices=simplified.vertex_matrix(),
                    faces=simplified.face_matrix(),
                )
                logger.info(f"Simplified to {len(mesh.faces)} faces")
            except Exception as e:
                logger.warning(f"Simplification failed: {e}")

        return mesh

    @property
    def vram_usage_mb(self) -> float:
        """Estimate current VRAM usage."""
        if not self._loaded:
            return 0.0
        return torch.cuda.memory_allocated() / (1024 * 1024)
