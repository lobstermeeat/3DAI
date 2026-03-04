"""
Hunyuan3D-Paint 2.1 PBR Texture Generation Stage.

Takes an untextured mesh + reference image and generates
PBR textures (albedo, metallic, roughness).
"""

import os
import sys
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HunyuanPaintConfig:
    """Configuration for Hunyuan3D-Paint texture generation."""
    # Model paths
    pretrained_dir: str = "/workspace/checkpoints/hunyuan/pretrained/hy3dpaint"
    finetuned_dir: Optional[str] = None  # If None, uses pretrained

    # Generation params
    seed: int = 42
    num_views: int = 6
    resolution: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50

    # PBR settings
    generate_pbr: bool = True  # Generate metallic + roughness maps
    use_super_resolution: bool = True  # Apply RealESRGAN upscaling


class HunyuanPaintStage:
    """Hunyuan3D-Paint 2.1 texture generation wrapper."""

    def __init__(self, config: HunyuanPaintConfig):
        self.config = config
        self.pipeline = None
        self._loaded = False

    def load(self):
        """Load Hunyuan3D-Paint model into GPU memory."""
        if self._loaded:
            return

        logger.info("Loading Hunyuan3D-Paint model...")
        start = time.time()

        try:
            hunyuan_path = os.environ.get("HUNYUAN_PATH", "/opt/Hunyuan3D-2.1")
            if hunyuan_path not in sys.path:
                sys.path.insert(0, hunyuan_path)

            # Use fine-tuned or pretrained weights
            model_dir = self.config.finetuned_dir or self.config.pretrained_dir

            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Paint model not found at {model_dir}")

            # Import and load Hunyuan3D Paint pipeline
            from hy3dpaint.pipeline import HunyuanPaintPipeline

            self.pipeline = HunyuanPaintPipeline.from_pretrained(model_dir)
            self.pipeline.to("cuda")

            self._loaded = True
            logger.info(f"Hunyuan3D-Paint loaded in {time.time() - start:.1f}s")

        except ImportError as e:
            logger.error(f"Failed to import Hunyuan3D-Paint: {e}")
            logger.error("Ensure Hunyuan3D-2.1 is installed and HUNYUAN_PATH is set")
            raise

    def unload(self):
        """Unload model from GPU memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._loaded = False
            torch.cuda.empty_cache()
            logger.info("Hunyuan3D-Paint unloaded from GPU")

    def apply_textures(
        self,
        mesh,
        reference_image,
        seed: Optional[int] = None,
    ):
        """
        Generate PBR textures for an untextured mesh.

        Args:
            mesh: trimesh.Trimesh — untextured mesh from TRELLIS.2
            reference_image: PIL Image or path — reference for texture generation
            seed: Random seed (overrides config)

        Returns:
            trimesh.Trimesh: Textured mesh with PBR materials
        """
        import trimesh
        from PIL import Image

        self.load()

        # Load reference image if path
        if isinstance(reference_image, (str, Path)):
            reference_image = Image.open(reference_image).convert("RGBA")

        seed = seed or self.config.seed
        torch.manual_seed(seed)

        logger.info(f"Generating PBR textures (seed={seed}, views={self.config.num_views})...")
        start = time.time()

        # Prepare mesh for Hunyuan3D-Paint
        # The paint module accepts trimesh or a mesh file path
        temp_mesh_path = "/tmp/forge3d_input_mesh.glb"
        mesh.export(temp_mesh_path)

        # Run Paint pipeline
        result = self.pipeline(
            mesh_path=temp_mesh_path,
            image=reference_image,
            num_views=self.config.num_views,
            resolution=self.config.resolution,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            seed=seed,
            pbr=self.config.generate_pbr,
        )

        # Extract textured mesh
        if hasattr(result, "mesh"):
            textured_mesh = result.mesh
        elif hasattr(result, "export"):
            textured_mesh = result
        else:
            # Load from output path
            output_path = getattr(result, "output_path", "/tmp/forge3d_textured.glb")
            textured_mesh = trimesh.load(output_path)

        duration = time.time() - start
        logger.info(f"PBR textures generated in {duration:.1f}s")

        # Cleanup temp file
        if os.path.exists(temp_mesh_path):
            os.remove(temp_mesh_path)

        return textured_mesh

    @property
    def vram_usage_mb(self) -> float:
        """Estimate current VRAM usage."""
        if not self._loaded:
            return 0.0
        return torch.cuda.memory_allocated() / (1024 * 1024)
