"""
FastAPI Inference Server for the Forge3D Combined Pipeline.

Endpoints:
    POST /generate         - Full pipeline (geometry + textures)
    POST /generate/geometry - Geometry only (TRELLIS.2)
    POST /generate/texture  - Texture only (Hunyuan3D-Paint on existing mesh)
    GET  /status/{job_id}  - Job status
    GET  /result/{job_id}  - Download result GLB
    GET  /health           - Health check with VRAM info
"""

import os
import uuid
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Optional
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .combined_pipeline import Forge3DCombinedPipeline, PipelineConfig, GenerationResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Forge3D Combined Pipeline API",
    description="TRELLIS.2 Geometry + Hunyuan3D-Paint PBR Textures",
    version="0.1.0",
)

# Pipeline singleton
pipeline: Optional[Forge3DCombinedPipeline] = None

# Job tracking
jobs: dict = {}  # job_id -> {"status": str, "progress": float, "result": dict}

# Single-threaded executor (one GPU job at a time)
executor = ThreadPoolExecutor(max_workers=1)


class GenerateRequest(BaseModel):
    """Request body for generation."""
    seed: int = 42
    skip_textures: bool = False


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 - 1.0
    stage: Optional[str] = None  # "geometry", "textures", "postprocess"
    message: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


@app.on_event("startup")
async def startup():
    """Initialize pipeline on server start."""
    global pipeline

    config = PipelineConfig.from_finetuned(
        ss_flow_ckpt=os.environ.get("SS_FLOW_CKPT", "/workspace/checkpoints/trellis/finetuned/ss_flow"),
        shape_flow_ckpt=os.environ.get("SHAPE_FLOW_CKPT", "/workspace/checkpoints/trellis/finetuned/shape_flow"),
        paint_ckpt=os.environ.get("PAINT_CKPT", "/workspace/checkpoints/hunyuan/finetuned/paint"),
    )
    config.output_dir = os.environ.get("OUTPUT_DIR", "/workspace/results/generated")

    pipeline = Forge3DCombinedPipeline(config)
    logger.info("Pipeline initialized")


def _run_generation(job_id: str, image_path: str, seed: int, skip_textures: bool):
    """Run generation in background thread."""
    global jobs

    def progress_callback(stage, pct, msg):
        # Map stage progress to overall progress
        stage_weights = {"geometry": 0.4, "textures": 0.4, "postprocess": 0.2}
        stage_offsets = {"geometry": 0.0, "textures": 0.4, "postprocess": 0.8}
        overall = stage_offsets.get(stage, 0) + (pct / 100) * stage_weights.get(stage, 0.1)
        jobs[job_id].update({
            "progress": min(overall, 0.99),
            "stage": stage,
            "message": msg,
        })

    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGBA")

        result = pipeline.generate(
            image=image,
            seed=seed,
            skip_textures=skip_textures,
            progress_callback=progress_callback,
        )

        jobs[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "result": asdict(result),
        })

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
        })


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    seed: int = 42,
    skip_textures: bool = False,
):
    """Submit a generation job."""
    job_id = str(uuid.uuid4())[:8]

    # Save uploaded image
    upload_dir = "/tmp/forge3d_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    image_path = os.path.join(upload_dir, f"{job_id}_{image.filename}")

    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)

    # Initialize job
    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "stage": None,
        "message": "Queued",
        "result": None,
        "error": None,
    }

    # Submit to executor
    executor.submit(_run_generation, job_id, image_path, seed, skip_textures)
    jobs[job_id]["status"] = "processing"

    return {"job_id": job_id, "status": "processing"}


@app.post("/generate/geometry")
async def generate_geometry(
    image: UploadFile = File(...),
    seed: int = 42,
):
    """Generate geometry only (no textures)."""
    return await generate(image=image, seed=seed, skip_textures=True)


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Download the result GLB file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not complete. Status: {job['status']}")

    output_path = job["result"]["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        output_path,
        media_type="model/gltf-binary",
        filename=f"{job_id}.glb",
    )


@app.get("/health")
async def health():
    """Health check with GPU/VRAM info."""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)),
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024)),
            "vram_reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024)),
        }

    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "processing"),
        "completed_jobs": sum(1 for j in jobs.values() if j["status"] == "completed"),
        "gpu": gpu_info,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
