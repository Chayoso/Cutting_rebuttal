"""
Quick center-cut verification: no simulation needed.
Loads each fruit's SDF cache, projects solid voxels onto XY (Z=0 collapsed),
and overlays the knife blade. Also shows top-down XZ view with the cut line.
"""
import sys, os, yaml, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from run import _build_or_load_sdf

FRUITS = ["grape", "pear", "tomato", "plum", "lemon", "shine_muscat", "cherry", "kiwi"]
CFG_DIR   = "configs/fruits"
CACHE_DIR = "cache_sdf"
OUT_IMG   = "center_cut_verify.png"

FRUIT_COLOR = {
    "grape":        "#8B30BB",
    "pear":         "#BBDD40",
    "tomato":       "#E82010",
    "plum":         "#6A1060",
    "lemon":        "#F8F010",
    "shine_muscat": "#88DD28",
    "cherry":       "#CC0820",
    "kiwi":         "#55A828",
}


def get_solid_world_coords(pack):
    """Return (N,3) world coords of all solid voxels (SDF < 0)."""
    sdf    = np.asarray(pack["sdf"], dtype=np.float32)
    origin = np.asarray(pack["origin"], dtype=np.float32)
    vsize  = float(pack["voxel_size"])
    iz, iy, ix = np.where(sdf < 0.0)
    x = origin[0] + (ix + 0.5) * vsize
    y = origin[1] + (iy + 0.5) * vsize
    z = origin[2] + (iz + 0.5) * vsize
    return np.stack([x, y, z], axis=1)


def get_fruit_z_center(pack):
    sdf    = np.asarray(pack["sdf"], dtype=np.float32)
    origin = np.asarray(pack["origin"], dtype=np.float32)
    vsize  = float(pack["voxel_size"])
    iz, _, _ = np.where(sdf < 0.0)
    zmin = origin[2] + (iz.min() + 0.5) * vsize
    zmax = origin[2] + (iz.max() + 0.5) * vsize
    return zmin, zmax, (zmin + zmax) / 2.0


def get_knife_x_range(pack):
    """Return (xmin, xmax) of the knife solid region."""
    sdf    = np.asarray(pack["sdf"], dtype=np.float32)
    origin = np.asarray(pack["origin"], dtype=np.float32)
    vsize  = float(pack["voxel_size"])
    _, _, ix = np.where(sdf < 0.0)
    xmin = origin[0] + (ix.min() + 0.5) * vsize
    xmax = origin[0] + (ix.max() + 0.5) * vsize
    return xmin, xmax


def main():
    fig, axes = plt.subplots(2, 8, figsize=(24, 6))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle("Center-cut verification  |  top row: front view (X-Y, Z collapsed)"
                 "  |  bottom row: top view (X-Z) with cut line",
                 color="white", fontsize=10, y=1.01)

    # Load knife pack once (shared across fruits)
    first_cfg = yaml.safe_load(open(os.path.join(CFG_DIR, "kiwi.yaml"), "r", encoding="utf-8"))
    knife_pack = _build_or_load_sdf("Knife", first_cfg["knife"],
                                    CACHE_DIR, use_cache=True, force_rebuild=False)
    k_xmin, k_xmax = get_knife_x_range(knife_pack)

    for col, fruit in enumerate(FRUITS):
        cfg_path = os.path.join(CFG_DIR, f"{fruit}.yaml")
        cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

        pack = _build_or_load_sdf("CuttingMesh", cfg["cutting_mesh"],
                                  CACHE_DIR, use_cache=True, force_rebuild=False)
        pts      = get_solid_world_coords(pack)          # (N,3)
        zmin, zmax, z_cut = get_fruit_z_center(pack)
        color    = FRUIT_COLOR.get(fruit, "#aaaaaa")

        xmin_f, xmax_f = pts[:, 0].min(), pts[:, 0].max()
        ymin_f, ymax_f = pts[:, 1].min(), pts[:, 1].max()
        pad_x = (xmax_f - xmin_f) * 0.15
        pad_y = (ymax_f - ymin_f) * 0.15
        pad_z = (zmax - zmin) * 0.20

        # ── TOP ROW: front view XY (Z collapsed) ──────────────────────────
        ax_front = axes[0][col]
        ax_front.set_facecolor("#1a1a1a")
        ax_front.scatter(pts[:, 0], pts[:, 1], s=1.0, c=color, alpha=0.6, linewidths=0)
        # knife blade: thin vertical band clamped to fruit X range
        ax_front.axvspan(xmin_f - pad_x, xmax_f + pad_x, alpha=0.12, color="silver")
        ax_front.set_title(fruit, color="white", fontsize=8)
        ax_front.set_xlim(xmin_f - pad_x, xmax_f + pad_x)
        ax_front.set_ylim(ymin_f - pad_y, ymax_f + pad_y)
        ax_front.set_aspect("equal")
        ax_front.axis("off")

        # ── BOTTOM ROW: top-down XZ with cut line ─────────────────────────
        ax_top = axes[1][col]
        ax_top.set_facecolor("#1a1a1a")
        ax_top.scatter(pts[:, 0], pts[:, 2], s=1.0, c=color, alpha=0.6, linewidths=0)
        # cut plane at z_center
        ax_top.axhline(z_cut, color="white", linewidth=1.5, linestyle="--")
        ax_top.axhline(zmin,  color="#555555", linewidth=0.7, linestyle=":")
        ax_top.axhline(zmax,  color="#555555", linewidth=0.7, linestyle=":")
        ax_top.set_title(f"z=[{zmin:.3f}, {zmax:.3f}]\ncut={z_cut:.3f}",
                         color="#aaaaaa", fontsize=6.5)
        ax_top.set_xlim(xmin_f - pad_x, xmax_f + pad_x)
        ax_top.set_ylim(zmin - pad_z, zmax + pad_z)
        ax_top.set_aspect("equal")
        ax_top.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    print(f"[SAVED] {OUT_IMG}")


if __name__ == "__main__":
    main()
