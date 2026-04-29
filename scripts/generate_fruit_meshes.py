"""
Procedural Fruit Mesh Generator for CulinaryCut
================================================
Generates 15 fruit meshes (.obj) using trimesh primitives.
All meshes are oriented with Y as up and centered at origin in XZ plane,
sitting on the Y=0 ground plane (suitable for knife cutting).

Size scale matches real fruits (cm-level). Banana reference: ~31 cm long.
"""
import numpy as np
import trimesh
from pathlib import Path

OUT_DIR = Path("assets/fruits")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(mesh: trimesh.Trimesh, name: str):
    """Center on XZ, sit on Y=0 plane, export to OBJ."""
    # Center in XZ
    c = mesh.centroid
    mesh.apply_translation([-c[0], 0, -c[2]])
    # Shift to sit on Y=0
    ymin = mesh.bounds[0, 1]
    mesh.apply_translation([0, -ymin, 0])
    path = OUT_DIR / f"{name}.obj"
    mesh.export(path)
    print(f"  {name:12s}  verts={len(mesh.vertices):5d}  "
          f"extent=[{mesh.extents[0]*100:.1f},{mesh.extents[1]*100:.1f},"
          f"{mesh.extents[2]*100:.1f}] cm")


# ── Fruit generators ────────────────────────────────────────────────────────

def strawberry(r=0.02, h=0.04):
    """Small cone with rounded top (point down, broad top)."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=r)
    # Taper bottom half into cone by scaling
    v = m.vertices.copy()
    # Y from -r..+r. Scale X,Z based on Y: bottom tapered, top full
    for i, y in enumerate(v[:, 1]):
        # y=-r (bottom): scale 0.1  |  y=+r (top): scale 1.2
        t = (y + r) / (2 * r)  # 0..1
        s = 0.15 + 1.15 * t
        v[i, 0] *= s
        v[i, 2] *= s
    # Stretch Y
    v[:, 1] *= (h / (2 * r))
    m.vertices = v
    return m


def peach(d=0.06):
    """Sphere with slight top dimple."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=d / 2)
    # Slight vertical squish
    m.apply_scale([1.0, 0.95, 1.0])
    return m


def pear(w=0.06, h=0.09):
    """Droplet shape — two ellipsoids stacked."""
    # Top narrower, bottom wider
    top = trimesh.creation.icosphere(subdivisions=2, radius=w * 0.35)
    top.apply_scale([1.0, 1.3, 1.0])
    top.apply_translation([0, h * 0.55, 0])
    bot = trimesh.creation.icosphere(subdivisions=2, radius=w / 2)
    bot.apply_scale([1.0, 1.1, 1.0])
    bot.apply_translation([0, h * 0.25, 0])
    m = trimesh.util.concatenate([top, bot])
    # Union via convex hull approximation (simple, fast)
    m = m.convex_hull
    return m


def apple(d=0.07):
    """Flattened sphere."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=d / 2)
    m.apply_scale([1.0, 0.9, 1.0])
    return m


def kiwi(w=0.04, h=0.06):
    """Ellipsoid along Z (lying down)."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=w / 2)
    m.apply_scale([1.0, 1.0, h / w])
    return m


def tomato(d=0.06):
    """Flattened sphere with tiny dimple on top (ignored in geometry)."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=d / 2)
    m.apply_scale([1.0, 0.75, 1.0])
    return m


def mango(w=0.05, h=0.09):
    """Elongated ellipsoid (lying down), slightly asymmetric."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=w / 2)
    m.apply_scale([1.0, 0.95, h / w])
    return m


def persimmon(d=0.07):
    """Very flat sphere."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=d / 2)
    m.apply_scale([1.0, 0.65, 1.0])
    return m


def avocado(w=0.05, h=0.10):
    """Pear-like, elongated (lying down)."""
    top = trimesh.creation.icosphere(subdivisions=2, radius=w * 0.3)
    top.apply_translation([0, 0, h * 0.4])
    bot = trimesh.creation.icosphere(subdivisions=2, radius=w / 2)
    bot.apply_translation([0, 0, -h * 0.1])
    m = trimesh.util.concatenate([top, bot]).convex_hull
    return m


def grape(d=0.022):
    """Small sphere."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=d / 2)
    return m


def orange(d=0.07):
    """Sphere."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=d / 2)
    return m


def lemon(w=0.04, h=0.06):
    """Ellipsoid with slight bumps at ends."""
    m = trimesh.creation.icosphere(subdivisions=2, radius=w / 2)
    m.apply_scale([1.0, 1.0, h / w])
    # Subtle end bumps — add small spheres at Z extremes
    b1 = trimesh.creation.icosphere(subdivisions=2, radius=w * 0.2)
    b1.apply_translation([0, 0, h / 2])
    b2 = trimesh.creation.icosphere(subdivisions=2, radius=w * 0.2)
    b2.apply_translation([0, 0, -h / 2])
    m = trimesh.util.concatenate([m, b1, b2]).convex_hull
    return m


def pineapple_slice(outer_d=0.09, inner_d=0.02, h=0.02, n_seg=48):
    """Annular prism (ring) built from triangles — no boolean ops needed."""
    ro = outer_d / 2
    ri = inner_d / 2
    theta = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Four vertex rings: outer-top, outer-bot, inner-top, inner-bot
    ot = np.stack([ro * cos_t, np.full_like(cos_t, h / 2), ro * sin_t], axis=-1)
    ob = np.stack([ro * cos_t, np.full_like(cos_t, -h / 2), ro * sin_t], axis=-1)
    it = np.stack([ri * cos_t, np.full_like(cos_t, h / 2), ri * sin_t], axis=-1)
    ib = np.stack([ri * cos_t, np.full_like(cos_t, -h / 2), ri * sin_t], axis=-1)
    verts = np.vstack([ot, ob, it, ib])
    N = n_seg
    faces = []
    for i in range(n_seg):
        j = (i + 1) % n_seg
        # Top annulus (ot -> it)
        faces.append([i, j, N * 2 + j])
        faces.append([i, N * 2 + j, N * 2 + i])
        # Bottom annulus (reverse normal)
        faces.append([N + i, N * 3 + j, N + j])
        faces.append([N + i, N * 3 + i, N * 3 + j])
        # Outer side
        faces.append([i, N + i, N + j])
        faces.append([i, N + j, j])
        # Inner side (reverse normal)
        faces.append([N * 2 + i, N * 2 + j, N * 3 + j])
        faces.append([N * 2 + i, N * 3 + j, N * 3 + i])
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.fix_normals()
    return m


def watermelon_slice(w=0.12, h=0.04, t=0.03):
    """Triangular wedge (half-disk): like a slice of watermelon."""
    # Half-disk: create circle, cut in half, extrude
    angles = np.linspace(0, np.pi, 24)
    ring = np.stack([np.cos(angles) * w / 2,
                     np.zeros_like(angles),
                     np.sin(angles) * w / 2], axis=-1)
    # Add center point
    verts_top = np.vstack([[[0, 0, 0]], ring])
    # Extrude in Y
    verts_bot = verts_top.copy()
    verts_bot[:, 1] = -h
    verts = np.vstack([verts_top, verts_bot])
    n = len(verts_top)
    # Faces: top fan, bottom fan, side
    faces = []
    # Top (facing up)
    for i in range(1, n - 1):
        faces.append([0, i, i + 1])
    # Bottom (facing down, flipped)
    for i in range(1, n - 1):
        faces.append([n, n + i + 1, n + i])
    # Sides
    for i in range(1, n):
        nxt = i + 1 if i < n - 1 else 0
        if nxt == 0:
            # close back to center
            faces.append([i, 0, n])
            faces.append([i, n, n + i])
        else:
            faces.append([i, nxt, n + nxt])
            faces.append([i, n + nxt, n + i])
    m = trimesh.Trimesh(vertices=verts, faces=faces)
    m.fix_normals()
    return m


# ── Build registry ──────────────────────────────────────────────────────────
FRUITS = {
    # Easy
    "strawberry":  strawberry,
    "peach":       peach,
    "pear":        pear,
    "apple":       apple,
    # Medium
    "kiwi":        kiwi,
    "tomato":      tomato,
    "mango":       mango,
    "persimmon":   persimmon,
    "avocado":     avocado,
    # Hard
    "grape":       grape,
    "orange":      orange,
    "lemon":       lemon,
    "pineapple_slice":   pineapple_slice,
    "watermelon_slice":  watermelon_slice,
}

print(f"Generating {len(FRUITS)} fruit meshes into {OUT_DIR}/")
print("-" * 70)
for name, fn in FRUITS.items():
    try:
        m = fn()
        save(m, name)
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

print("-" * 70)
print("Done. Existing banana.obj stays in assets/")
