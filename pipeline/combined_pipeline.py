"""
Forge3D Combined Pipeline: TRELLIS.2 Geometry + Hunyuan3D-Paint PBR Textures.

This is the main pipeline that orchestrates both models sequentially,
managing VRAM by loading/unloading each model as needed.
"""

import os
import logging
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch

from .trellis_geometry import TrellisConfig, TrellisGeometryStage
from .hunyuan_paint import HunyuanPaintConfig, HunyuanPaintStage
from .mesh_postprocess import PostProcessConfig, MeshPostProcessor

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    trellis: TrellisConfig = field(default_factory=TrellisConfig)
    hunyuan: HunyuanPaintConfig = field(default_factory=HunyuanPaintConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    output_dir: str = "/workspace/results/generated"

    @classmethod
    def from_finetuned(
        cls,
        ss_flow_ckpt: str = "/workspace/checkpoints/trellis/finetuned/ss_flow",
        shape_flow_ckpt: str = "/workspace/checkpoints/trellis/finetuned/shape_flow",
        paint_ckpt: str = "/workspace/checkpoints/hunyuan/finetuned/paint",
        **kwargs,
    ) -> "PipelineConfig":
        """Create config pointing to fine-tuned checkpoints."""
        return cls(
            trellis=TrellisConfig(
                finetuned_ss_flow=ss_flow_ckpt,
                finetuned_shape_flow=shape_flow_ckpt,
                **{k: v for k, v in kwargs.items() if hasattr(TrellisConfig, k)},
            ),
            hunyuan=HunyuanPaintConfig(
                finetuned_dir=paint_ckpt,
                **{k: v for k, v in kwargs.items() if hasattr(HunyuanPaintConfig, k)},
            ),
        )


@dataclass
class GenerationResult:
    """Result of a pipeline generation."""
    job_id: str
    output_path: str
    geometry_time_s: float
    texture_time_s: float
    postprocess_time_s: float
    total_time_s: float
    vertex_count: int
    face_count: int
    file_size_mb: float


class Forge3DCombinedPipeline:
    """
    TRELLIS.2 geometry + Hunyuan3D-Paint textures = production-ready 3D.

    Models are loaded sequentially to manage VRAM:
    1. Load TRELLIS.2 → generate geometry → unload
    2. Load Hunyuan3D-Paint → generate PBR textures → unload
    3. Post-process and export GLB
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.trellis = TrellisGeometryStage(self.config.trellis)
        self.paint = HunyuanPaintStage(self.config.hunyuan)
        self.postprocess = MeshPostProcessor(self.config.postprocess)

    def generate(
        self,
        image,
        seed: int = 42,
        skip_textures: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> GenerationResult:
        """
        Generate a textured 3D model from an input image.

        Args:
            image: PIL Image or path to image file
            seed: Random seed for reproducibility
            skip_textures: If True, only generate geometry (no Hunyuan3D-Paint)
            progress_callback: Optional callback(stage, progress_pct, message)

        Returns:
            GenerationResult with output path and timing info
        """
        job_id = str(uuid.uuid4())[:8]
        total_start = time.time()

        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, f"{job_id}.glb")

        def _progress(stage, pct, msg):
            if progress_callback:
                progress_callback(stage, pct, msg)
            logger.info(f"[{stage}] {pct}% - {msg}")

        try:
            # ═══════════════════════════════════════════
            # Stage 1: TRELLIS.2 Geometry
            # ═══════════════════════════════════════════
            _progress("geometry", 0, "Loading TRELLIS.2...")
            geo_start = time.time()

            mesh = self.trellis.generate(image, seed=seed)
            geometry_time = time.time() - geo_start

            _progress("geometry", 100, f"Geometry done: {len(mesh.faces)} faces in {geometry_time:.1f}s")

            # Unload TRELLIS.2 to free VRAM for Paint
            self.trellis.unload()
            torch.cuda.empty_cache()

            # ═══════════════════════════════════════════
            # Stage 2: Hunyuan3D-Paint Textures
            # ═══════════════════════════════════════════
            texture_time = 0.0
            if not skip_textures:
                _progress("textures", 0, "Loading Hunyuan3D-Paint...")
                tex_start = time.time()

                textured_mesh = self.paint.apply_textures(
                    mesh=mesh,
                    reference_image=image,
                    seed=seed,
                )
                texture_time = time.time() - tex_start

                _progress("textures", 100, f"Textures done in {texture_time:.1f}s")

                # Unload Paint
                self.paint.unload()
                torch.cuda.empty_cache()

                mesh = textured_mesh

            # ═══════════════════════════════════════════
            # Stage 3: Post-Processing
            # ═══════════════════════════════════════════
            _progress("postprocess", 0, "Post-processing mesh...")
            pp_start = time.time()

            final_path = self.postprocess.finalize(mesh, output_path)
            postprocess_time = time.time() - pp_start

            _progress("postprocess", 100, f"Post-processing done in {postprocess_time:.1f}s")

            total_time = time.time() - total_start
            file_size = os.path.getsize(final_path) / (1024 * 1024)

            result = GenerationResult(
                job_id=job_id,
                output_path=final_path,
                geometry_time_s=round(geometry_time, 2),
                texture_time_s=round(texture_time, 2),
                postprocess_time_s=round(postprocess_time, 2),
                total_time_s=round(total_time, 2),
                vertex_count=len(mesh.vertices),
                face_count=len(mesh.faces),
                file_size_mb=round(file_size, 2),
            )

            logger.info(f"Generation complete: {result.total_time_s}s total, "
                       f"{result.file_size_mb}MB output")

            return result

        except Exception as e:
            # Ensure models are unloaded on error
            self.trellis.unload()
            self.paint.unload()
            torch.cuda.empty_cache()
            raise

    def generate_geometry_only(self, image, seed: int = 42) -> GenerationResult:
        """Generate geometry without textures."""
        return self.generate(image, seed=seed, skip_textures=True)
