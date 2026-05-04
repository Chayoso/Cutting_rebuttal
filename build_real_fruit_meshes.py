"""
Build the 14 fruit meshes for assets/fruits/ from local FBX packs at
~/Desktop/assets/. Each FBX is first converted to OBJ via headless Blender,
then repaired (voxel-remesh if non-watertight), decimated to ~2000 faces, and
scaled/centered to fit the existing config dimensions.

Two slices (pineapple_slice, watermelon_slice) are cut from the corresponding
whole-fruit FBX. Tomato uses a small Rockit apple model in lieu of a real
tomato source in this asset pack.

Run from the repo root (assumes /home/chayo/Desktop/assets exists locally):
    python build_real_fruit_meshes.py
"""

import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
import trimesh

ROOT = Path(__file__).parent
LOCAL_PACKS = Path(os.path.expanduser("~/Desktop/assets"))
WORK_DIR = ROOT / ".fbx_work"
WORK_DIR.mkdir(exist_ok=True)
OUT_DIR = ROOT / "assets" / "fruits"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BLENDER = "/snap/bin/blender"

# fruit -> (pack_dir_name, fbx_relative_path inside the pack[, "horizontal_slice"])
FBX_SOURCES = {
    "apple":            ("uploads_files_3935237_Granny+Smith+5_Apple_Model+FBX",
                         "FBX/5_Apple_Model.fbx"),
    "avocado":          ("uploads_files_3935237_Avocado+FBX",
                         "Avocado FBX/Avocado_Model.fbx"),
    "banana":           ("uploads_files_3935237_Banana+FBX",
                         "Banana FBX/Banana_Model.fbx"),
    "grape":            ("uploads_files_3935237_Cherry+FBX",
                         "FBX/Cherry_Model_2.fbx"),
    "kiwi":             ("uploads_files_3935237_Kiwi+FBX",
                         "Kiwi FBX/Kiwi_Model.fbx"),
    "lemon":            ("LemonModel_FBX",
                         "Lemon_Model.fbx"),
    "mango":            ("uploads_files_3935237_Mango+FBX",
                         "Mango FBX/Mango_Model.fbx"),
    "orange":           ("uploads_files_3935237_Orange+FBX",
                         "Orange FBX/Orange_Model.fbx"),
    "peach":            ("uploads_files_3935237_Red+plum+FBX",
                         "FBX/Red plum_Model_1.fbx"),
    "pear":             ("uploads_files_3935237_Pear+FBX",
                         "Pear_Model.fbx"),
    "persimmon":        ("uploads_files_3935237_Pumpkin+FBX",
                         "Pumpkin FBX/Pumpkin_Model.fbx"),
    "pineapple_slice":  ("uploads_files_3935237_Pineapple+FBX",
                         "Pineapple0.fbx", "horizontal_slice"),
    "strawberry":       ("uploads_files_3935237_Strawberry_FBX",
                         "Strawberry_Model_1.fbx"),
    "tomato":           ("uploads_files_3935237_Rockit+1_Apple_Model+FBX",
                         "FBX/1_Apple_Model.fbx"),
    "watermelon_slice": ("uploads_files_3935237_Watermelon+FBX",
                         "FBX/Watermelon_Model.fbx", "horizontal_slice"),
}

TARGET_XZ = {
    "apple":            0.070,
    "avocado":          0.075,
    "banana":           0.180,
    "grape":            0.020,
    "kiwi":             0.060,
    "lemon":            0.060,
    "mango":            0.090,
    "orange":           0.075,
    "peach":            0.070,
    "pear":             0.070,
    "persimmon":        0.065,
    "pineapple_slice":  0.090,
    "strawberry":       0.040,
    "tomato":           0.065,
    "watermelon_slice": 0.120,
}

BLENDER_CONVERT = ROOT / ".blender_fbx_to_obj.py"


def write_blender_script():
    BLENDER_CONVERT.write_text("""
import sys, os
import bpy
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
in_path, out_path = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=in_path)
mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not mesh_objs:
    print("ERROR: no mesh"); sys.exit(1)
bpy.ops.object.select_all(action="DESELECT")
for o in mesh_objs:
    o.select_set(True)
bpy.context.view_layer.objects.active = mesh_objs[0]
if len(mesh_objs) > 1:
    bpy.ops.object.join()
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
bpy.ops.wm.obj_export(filepath=out_path, export_selected_objects=True,
                      export_materials=False, export_uv=False,
                      export_normals=False, export_triangulated_mesh=True,
                      forward_axis="NEGATIVE_Z", up_axis="Y")
""".lstrip())


def convert_fbx_to_obj(fbx_path: Path, obj_path: Path):
    if obj_path.exists():
        return
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [BLENDER, "--background", "--python", str(BLENDER_CONVERT),
         "--", str(fbx_path), str(obj_path)],
        capture_output=True, timeout=180,
    )
    if res.returncode != 0 or not obj_path.exists():
        raise RuntimeError(f"Blender conversion failed for {fbx_path.name}: "
                           f"{res.stderr.decode()[:300]}")


def repair_mesh(m, target_faces=2000):
    m = m.copy()
    m.process(validate=True)
    m.remove_unreferenced_vertices()
    nd = m.nondegenerate_faces()
    if len(nd) < len(m.faces):
        m.update_faces(nd)
    if not m.is_watertight:
        trimesh.repair.fill_holes(m)
    if not m.is_watertight:
        ext = float(np.max(m.bounding_box.extents))
        pitch = max(ext / 80.0, 1e-4)
        try:
            vox = m.voxelized(pitch=pitch).fill()
            mr = vox.marching_cubes
            if mr.is_watertight and len(mr.faces) > 0:
                m = mr
        except Exception:
            pass
    if len(m.faces) > target_faces * 1.5:
        try:
            m_dec = m.simplify_quadric_decimation(face_count=target_faces)
            if len(m_dec.faces) > 0:
                if not m_dec.is_watertight:
                    trimesh.repair.fill_holes(m_dec)
                if not m_dec.is_watertight:
                    ext = float(np.max(m_dec.bounding_box.extents))
                    pitch = max(ext / 50.0, 1e-4)
                    try:
                        vox = m_dec.voxelized(pitch=pitch).fill()
                        mr = vox.marching_cubes
                        if mr.is_watertight and len(mr.faces) > 0:
                            if len(mr.faces) > target_faces * 1.5:
                                mr = mr.simplify_quadric_decimation(
                                    face_count=target_faces)
                            m_dec = mr
                    except Exception:
                        pass
                m = m_dec
        except Exception:
            pass
    return m


def normalize_pose(m):
    m = m.copy()
    cx, _, cz = m.bounding_box.centroid
    miny = m.bounds[0, 1]
    m.apply_translation([-cx, -miny, -cz])
    return m


def scale_to_xz(m, target_xz):
    m = m.copy()
    bb = m.bounding_box.extents
    longest_xz = max(bb[0], bb[2])
    if longest_xz < 1e-6:
        return m
    s = target_xz / longest_xz
    m.apply_scale(s)
    return m


def horizontal_slab(src, frac_low, frac_high):
    m = src.copy()
    miny, maxy = m.bounds[:, 1]
    h = maxy - miny
    y0 = miny + frac_low * h
    y1 = miny + frac_high * h
    m = trimesh.intersections.slice_mesh_plane(
        m, plane_normal=[0, 1, 0], plane_origin=[0, y0, 0], cap=True)
    m = trimesh.intersections.slice_mesh_plane(
        m, plane_normal=[0, -1, 0], plane_origin=[0, y1, 0], cap=True)
    return m


def build(name):
    if name not in FBX_SOURCES:
        raise ValueError(name)
    spec = FBX_SOURCES[name]
    pack, rel_fbx = spec[0], spec[1]
    mode = spec[2] if len(spec) > 2 else "whole"

    fbx_path = LOCAL_PACKS / pack / rel_fbx
    obj_path = WORK_DIR / f"{name}_raw.obj"
    convert_fbx_to_obj(fbx_path, obj_path)

    m = trimesh.load(obj_path, force="mesh")
    if mode == "horizontal_slice":
        if name == "watermelon_slice":
            m = horizontal_slab(m, 0.45, 0.55)
        elif name == "pineapple_slice":
            m = horizontal_slab(m, 0.30, 0.42)

    m = repair_mesh(m)
    m = scale_to_xz(m, TARGET_XZ[name])
    m = normalize_pose(m)
    return m


def main():
    if not LOCAL_PACKS.exists():
        raise SystemExit(f"FBX packs not found at {LOCAL_PACKS}")

    write_blender_script()

    print(f"\n{'fruit':<20s}  {'faces':>6s}  {'verts':>6s}  "
          f"{'bbox_x':>8s}  {'bbox_y':>8s}  {'bbox_z':>8s}  watertight  source")
    print("-" * 110)
    for name in sorted(FBX_SOURCES.keys()):
        try:
            m = build(name)
            out = OUT_DIR / f"{name}.obj"
            m.export(out)
            bb = m.bounding_box.extents
            src_label = Path(FBX_SOURCES[name][1]).name
            print(f"{name:<20s}  {len(m.faces):>6d}  {len(m.vertices):>6d}  "
                  f"{bb[0]:>8.4f}  {bb[1]:>8.4f}  {bb[2]:>8.4f}  "
                  f"{str(m.is_watertight):>10s}  {src_label}")
        except Exception as e:
            print(f"{name:<20s}  FAIL: {e}")
    print(f"\nWrote 14 fruits to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
