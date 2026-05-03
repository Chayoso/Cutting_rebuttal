"""
Build the 14 fruit meshes for assets/fruits/ using a mix of:
  - Kenney Food Kit (CC0) for 11 fruits (apple, avocado, banana, grape, lemon,
    orange, pear, pineapple, strawberry, tomato, watermelon).
  - Geometric slicing for 2 (pineapple_slice from pineapple, watermelon_slice
    from watermelon — half-cylinder/wedge cut).
  - Procedural fruit-characteristic shapes for 4 (kiwi, mango, peach, persimmon)
    that Kenney does not include.

All meshes are repaired (fill_holes + manifold), centered in XZ, translated so
y_min = 0, and scaled so the longest XZ dimension matches the size used by the
existing configs (cm-level realistic).

Run from the repo root:
    .venv/bin/python build_real_fruit_meshes.py
"""

import os
import shutil
from pathlib import Path
import numpy as np
import trimesh

ROOT = Path(__file__).parent
KENNEY_DIR = ROOT / "assets" / "external_packs" / "Models" / "OBJ format"
OUT_DIR = ROOT / "assets" / "fruits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target longest XZ extent (m). These match the previous procedural assumptions
# (matching the configs' SDF-voxel auto-scale at ~1.5 mm cell).
TARGET_XZ = {
    "apple":           0.070,
    "avocado":         0.075,
    "banana":          0.180,
    "grape":           0.020,
    "kiwi":            0.060,
    "lemon":           0.060,
    "mango":           0.090,
    "orange":          0.075,
    "peach":           0.070,
    "pear":            0.070,
    "persimmon":       0.065,
    "pineapple_slice": 0.090,
    "strawberry":      0.040,
    "tomato":          0.065,
    "watermelon_slice":0.120,
}

# Mapping fruit -> Kenney source filename (None if procedural / sliced)
KENNEY_FILE = {
    "apple":      "apple.obj",
    "avocado":    "avocado.obj",
    "banana":     "banana.obj",
    "grape":      "grapes.obj",
    "lemon":      "lemon.obj",
    "orange":     "orange.obj",
    "pear":       "pear.obj",
    "strawberry": "strawberry.obj",
    "tomato":     "tomato.obj",
}

# Slicing sources (whole fruit in Kenney, we cut a slice/wedge)
SLICE_FROM = {
    "pineapple_slice": ("pineapple.obj", "horizontal"),  # cut a horizontal slab
    "watermelon_slice": ("watermelon.obj", "wedge"),      # cut a wedge (1/6 sphere)
}


def repair_mesh(m, target_faces=2000):
    """Best-effort manifold repair, then voxel-remesh fallback for any
    remaining non-watertight result, then decimation to keep face count
    manageable for SDF voxelization."""
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
            m_remesh = vox.marching_cubes
            if m_remesh.is_watertight and len(m_remesh.faces) > 0:
                m = m_remesh
        except Exception as e:
            print(f"    voxel-remesh failed: {e}")

    # Decimate if too dense (voxel remesh produces 30k+ faces; we want ~2k)
    if len(m.faces) > target_faces * 1.5:
        try:
            m_dec = m.simplify_quadric_decimation(face_count=target_faces)
            if len(m_dec.faces) > 0:
                if not m_dec.is_watertight:
                    trimesh.repair.fill_holes(m_dec)
                # Voxel-remesh at coarse pitch as last resort to restore manifold
                if not m_dec.is_watertight:
                    ext = float(np.max(m_dec.bounding_box.extents))
                    pitch = max(ext / 50.0, 1e-4)
                    try:
                        vox = m_dec.voxelized(pitch=pitch).fill()
                        m_remesh = vox.marching_cubes
                        if m_remesh.is_watertight and len(m_remesh.faces) > 0:
                            # Re-decimate if still too dense after re-remesh
                            if len(m_remesh.faces) > target_faces * 1.5:
                                m_remesh = m_remesh.simplify_quadric_decimation(
                                    face_count=target_faces)
                            m_dec = m_remesh
                    except Exception:
                        pass
                m = m_dec
        except Exception as e:
            print(f"    decimation failed: {e}")
    return m


def normalize_pose(m):
    """Center XZ, place y_min on board (y=0)."""
    m = m.copy()
    cx, cy, cz = m.bounding_box.centroid
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


def make_kiwi():
    """Kiwi: smooth ellipsoid, longer in one axis."""
    m = trimesh.creation.icosphere(subdivisions=3)
    m.apply_scale([0.5, 0.5, 1.0])  # elongated
    return m


def make_mango():
    """Mango: asymmetric ellipsoid, slight bend."""
    m = trimesh.creation.icosphere(subdivisions=3)
    m.apply_scale([0.55, 0.55, 1.0])
    # Slight asymmetry: shift top vertices forward
    v = m.vertices.copy()
    z = v[:, 2]
    z_max = z.max()
    bias = 0.06 * np.maximum(0.0, z / z_max) ** 2  # nonlinear bend toward top
    v[:, 0] += bias
    m.vertices = v
    return m


def make_peach():
    """Peach: nearly sphere, with vertical crease (shallow groove on +X side)."""
    m = trimesh.creation.icosphere(subdivisions=4)
    v = m.vertices.copy()
    # Crease: depress vertices near x>0 axis on plane y close to 0
    x = v[:, 0]
    y = v[:, 1]
    crease = 0.08 * np.exp(-((y / 0.15) ** 2)) * np.maximum(0.0, x)
    v[:, 0] -= crease
    m.vertices = v
    return m


def make_persimmon():
    """Persimmon: oblate sphere (squashed top→bottom) with small calyx bump."""
    m = trimesh.creation.icosphere(subdivisions=4)
    m.apply_scale([1.0, 0.78, 1.0])  # oblate (Y is up here in trimesh default)
    # Small calyx (4-leaf cup) on top: add a small disc/cup at +Y
    calyx = trimesh.creation.cylinder(radius=0.30, height=0.06, sections=8)
    calyx.apply_translation([0.0, 0.78 + 0.03, 0.0])
    m = trimesh.util.concatenate([m, calyx])
    m.merge_vertices()
    return m


PROCEDURAL = {
    "kiwi":       make_kiwi,
    "mango":      make_mango,
    "peach":      make_peach,
    "persimmon":  make_persimmon,
}


def _horizontal_slab(src, frac_low, frac_high):
    """Cut a horizontal slab between two Y planes (frac of full height)."""
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


def slice_pineapple(src):
    """Pineapple slice: thin horizontal slab from the middle of the pineapple
    (where the body is thickest, before the leafy crown). ~12% thick."""
    return _horizontal_slab(src, 0.30, 0.42)


def slice_watermelon_wedge(src):
    """Watermelon slice: thin horizontal slab cut through the middle of the
    watermelon. Looks like a serving slice on a plate. ~10% thick."""
    return _horizontal_slab(src, 0.45, 0.55)


def build(name):
    if name in KENNEY_FILE:
        src_path = KENNEY_DIR / KENNEY_FILE[name]
        m = trimesh.load(src_path, force="mesh")
    elif name in SLICE_FROM:
        src_file, mode = SLICE_FROM[name]
        src = trimesh.load(KENNEY_DIR / src_file, force="mesh")
        if mode == "horizontal":
            m = slice_pineapple(src)
        elif mode == "wedge":
            m = slice_watermelon_wedge(src)
        else:
            raise ValueError(mode)
    elif name in PROCEDURAL:
        m = PROCEDURAL[name]()
    else:
        raise ValueError(f"unknown fruit: {name}")

    m = repair_mesh(m)
    m = scale_to_xz(m, TARGET_XZ[name])
    m = normalize_pose(m)
    return m


def main():
    # Backup existing assets/fruits/
    backup = ROOT / "assets" / "fruits.procedural_backup"
    if OUT_DIR.exists() and not backup.exists():
        shutil.copytree(OUT_DIR, backup)
        print(f"  Backed up old procedural meshes → {backup.relative_to(ROOT)}")

    print(f"\n{'fruit':<20s}  {'faces':>6s}  {'verts':>6s}  "
          f"{'bbox_x':>8s}  {'bbox_y':>8s}  {'bbox_z':>8s}  watertight")
    print("-" * 84)
    for name in sorted(TARGET_XZ.keys()):
        try:
            m = build(name)
            out = OUT_DIR / f"{name}.obj"
            m.export(out)
            bb = m.bounding_box.extents
            print(f"{name:<20s}  {len(m.faces):>6d}  {len(m.vertices):>6d}  "
                  f"{bb[0]:>8.4f}  {bb[1]:>8.4f}  {bb[2]:>8.4f}  "
                  f"{str(m.is_watertight)}")
        except Exception as e:
            print(f"{name:<20s}  FAIL: {e}")

    print(f"\nWrote 14 fruits to {OUT_DIR.relative_to(ROOT)}")
    print("License: Kenney Food Kit (CC0) for non-procedural; procedural meshes "
          "under repo license.")


if __name__ == "__main__":
    main()
