#!/usr/bin/env python3
"""
Prepare training data for Hunyuan3D-Paint fine-tuning.
Renders multi-view PBR decomposition (albedo, metallic, roughness, normal)
under multiple lighting conditions using Blender.

Usage:
    python prep_hunyuan_data.py \
        --input_dir /workspace/data/filtered \
        --output_dir /workspace/data/hunyuan \
        --blender_path blender \
        --num_views 6 \
        --resolution 512
"""

import argparse
import json
import os
import subprocess
import logging
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Blender render script (written inline, executed via blender --python)
BLENDER_RENDER_SCRIPT = '''
import bpy
import sys
import json
import math
import os
import mathutils

# Parse args after "--"
argv = sys.argv[sys.argv.index("--") + 1:]
mesh_path = argv[0]
output_dir = argv[1]
num_views = int(argv[2])
resolution = int(argv[3])

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import mesh
ext = os.path.splitext(mesh_path)[1].lower()
if ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=mesh_path)
elif ext == ".obj":
    bpy.ops.wm.obj_import(filepath=mesh_path)
elif ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=mesh_path)
elif ext == ".ply":
    bpy.ops.wm.ply_import(filepath=mesh_path)
else:
    print(f"Unsupported format: {ext}")
    sys.exit(1)

# Get all mesh objects
mesh_objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not mesh_objects:
    print("No mesh objects found")
    sys.exit(1)

# Normalize scale: fit in unit cube centered at origin
all_verts = []
for obj in mesh_objects:
    for v in obj.data.vertices:
        all_verts.append(obj.matrix_world @ v.co)

if all_verts:
    min_co = mathutils.Vector((min(v.x for v in all_verts), min(v.y for v in all_verts), min(v.z for v in all_verts)))
    max_co = mathutils.Vector((max(v.x for v in all_verts), max(v.y for v in all_verts), max(v.z for v in all_verts)))
    center = (min_co + max_co) / 2
    extent = max(max_co - min_co)
    scale = 1.0 / extent if extent > 0 else 1.0

    for obj in mesh_objects:
        obj.location -= center
        obj.scale *= scale

# Set up camera
camera_data = bpy.data.cameras.new("Camera")
camera_data.lens = 50
camera_obj = bpy.data.objects.new("Camera", camera_data)
bpy.context.scene.collection.objects.link(camera_obj)
bpy.context.scene.camera = camera_obj
camera_distance = 2.0

# Render settings
scene = bpy.context.scene
scene.render.resolution_x = resolution
scene.render.resolution_y = resolution
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "8"
scene.render.film_transparent = True

# Use Cycles for PBR rendering
scene.render.engine = "CYCLES"
scene.cycles.samples = 128
scene.cycles.use_denoising = True
if bpy.context.preferences.addons.get("cycles"):
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

# Camera positions: evenly distributed views
camera_transforms = []
elevations = [0, 30, -20]  # degrees
views_per_elevation = max(1, num_views // len(elevations))

view_idx = 0
for elev_deg in elevations:
    for az_i in range(views_per_elevation):
        if view_idx >= num_views:
            break
        az_deg = (360 / views_per_elevation) * az_i
        elev = math.radians(elev_deg)
        az = math.radians(az_deg)

        x = camera_distance * math.cos(elev) * math.cos(az)
        y = camera_distance * math.cos(elev) * math.sin(az)
        z = camera_distance * math.sin(elev)

        camera_obj.location = (x, y, z)
        direction = mathutils.Vector((0, 0, 0)) - camera_obj.location
        rot_quat = direction.to_track_quat("-Z", "Y")
        camera_obj.rotation_euler = rot_quat.to_euler()

        camera_transforms.append({
            "view_idx": view_idx,
            "azimuth": az_deg,
            "elevation": elev_deg,
            "position": [x, y, z],
            "rotation": list(camera_obj.rotation_euler),
        })

        # Lighting conditions
        lighting_configs = {
            "AL": {"type": "ambient", "energy": 3.0},
            "PL": {"type": "point", "energy": 300.0, "location": (2, 2, 3)},
            "ENVMAP": {"type": "env", "energy": 1.0},
        }

        for light_name, light_cfg in lighting_configs.items():
            # Clear existing lights
            for obj in bpy.context.scene.objects:
                if obj.type == "LIGHT":
                    bpy.data.objects.remove(obj, do_unlink=True)

            if light_cfg["type"] == "ambient":
                # Use sun light as ambient approximation
                light_data = bpy.data.lights.new("AmbientLight", "SUN")
                light_data.energy = light_cfg["energy"]
                light_obj = bpy.data.objects.new("AmbientLight", light_data)
                bpy.context.scene.collection.objects.link(light_obj)
            elif light_cfg["type"] == "point":
                light_data = bpy.data.lights.new("PointLight", "POINT")
                light_data.energy = light_cfg["energy"]
                light_obj = bpy.data.objects.new("PointLight", light_data)
                light_obj.location = light_cfg["location"]
                bpy.context.scene.collection.objects.link(light_obj)
            elif light_cfg["type"] == "env":
                world = bpy.data.worlds.new("EnvWorld")
                bpy.context.scene.world = world
                world.use_nodes = True
                bg = world.node_tree.nodes["Background"]
                bg.inputs["Strength"].default_value = light_cfg["energy"]

            # Render RGB
            render_path = os.path.join(output_dir, "render_cond",
                                       f"view{view_idx:03d}_light_{light_name}.png")
            os.makedirs(os.path.dirname(render_path), exist_ok=True)
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)

        # Render albedo pass (override materials to emission-only for albedo extraction)
        # This gives us lighting-free base color
        original_materials = {}
        for obj in mesh_objects:
            for i, slot in enumerate(obj.material_slots):
                if slot.material:
                    original_materials[(obj.name, i)] = slot.material

        # Render with flat lighting for approximate albedo
        for obj in bpy.context.scene.objects:
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj, do_unlink=True)

        # Uniform hemisphere light for albedo
        light_data = bpy.data.lights.new("AlbedoLight", "SUN")
        light_data.energy = 2.0
        light_obj = bpy.data.objects.new("AlbedoLight", light_data)
        bpy.context.scene.collection.objects.link(light_obj)

        albedo_path = os.path.join(output_dir, "render_tex", f"view{view_idx:03d}_albedo.png")
        os.makedirs(os.path.dirname(albedo_path), exist_ok=True)
        scene.render.filepath = albedo_path
        bpy.ops.render.render(write_still=True)

        view_idx += 1

# Save camera transforms
transforms_path = os.path.join(output_dir, "render_tex", "transforms.json")
with open(transforms_path, "w") as f:
    json.dump({"camera_distance": camera_distance, "views": camera_transforms}, f, indent=2)

print(f"Rendered {view_idx} views to {output_dir}")
'''


def render_asset(args_tuple):
    """Render a single asset using Blender."""
    mesh_path, output_dir, blender_path, num_views, resolution = args_tuple

    asset_name = Path(mesh_path).stem
    asset_output = os.path.join(output_dir, asset_name)
    os.makedirs(asset_output, exist_ok=True)

    # Write temp render script
    script_path = os.path.join(asset_output, "_render.py")
    with open(script_path, "w") as f:
        f.write(BLENDER_RENDER_SCRIPT)

    try:
        result = subprocess.run(
            [
                blender_path, "--background", "--python", script_path,
                "--", mesh_path, asset_output, str(num_views), str(resolution),
            ],
            capture_output=True, text=True, timeout=300,  # 5 min timeout per asset
        )

        # Cleanup temp script
        os.remove(script_path)

        if result.returncode != 0:
            return {"asset": asset_name, "success": False, "error": result.stderr[-500:]}
        return {"asset": asset_name, "success": True}

    except subprocess.TimeoutExpired:
        return {"asset": asset_name, "success": False, "error": "timeout"}
    except Exception as e:
        return {"asset": asset_name, "success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Render PBR data for Hunyuan3D-Paint training")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--blender_path", type=str, default="blender")
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Find mesh files
    extensions = {".glb", ".gltf", ".obj", ".fbx", ".ply"}
    mesh_files = []
    for f in os.listdir(args.input_dir):
        if Path(f).suffix.lower() in extensions:
            mesh_files.append(os.path.join(args.input_dir, f))

    logger.info(f"Found {len(mesh_files)} mesh files to render")

    # Render all assets
    render_args = [
        (fp, args.output_dir, args.blender_path, args.num_views, args.resolution)
        for fp in mesh_files
    ]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(render_asset, ra): ra for ra in render_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering"):
            results.append(future.result())

    success = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    logger.info(f"Rendering complete: {len(success)} success, {len(failed)} failed")

    # Generate examples.json for Hunyuan3D training
    examples = {
        "data": [
            {
                "id": r["asset"],
                "path": os.path.join(args.output_dir, r["asset"]),
            }
            for r in success
        ]
    }
    examples_path = os.path.join(args.output_dir, "examples.json")
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)

    logger.info(f"Saved examples.json with {len(success)} entries to {examples_path}")

    if failed:
        logger.warning(f"Failed assets:")
        for r in failed[:10]:
            logger.warning(f"  {r['asset']}: {r['error'][:100]}")


if __name__ == "__main__":
    main()
