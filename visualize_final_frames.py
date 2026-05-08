"""
Center-cut verification in ManiSkill orientation.

ManiSkill axis_map: (ms_x, ms_y, ms_z) = (world_x, world_z, world_y)
  -> ms_Z is UP  (= world Y),  knife descends in -ms_Z (= -world_Y)
  -> ms_Y is CUT-DEPTH direction (= world Z, where z_cut lives)

Row 1 -- FRONT VIEW  (world X vs Y): ManiSkill XZ plane, stem up, knife sweeps X
Row 2 -- SIDE VIEW   (world Z vs Y): ManiSkill YZ plane, vertical dashed = cut at z_center
"""
import sys, os, yaml, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from run import _build_or_load_sdf

FRUITS    = ["grape", "pear", "tomato", "plum", "lemon", "shine_muscat", "cherry", "kiwi"]
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


def lim(arr, pad=0.18):
    lo, hi = float(arr.min()), float(arr.max())
    p = (hi - lo) * pad
    return lo - p, hi + p


def main():
    fig, axes = plt.subplots(2, 8, figsize=(24, 7))
    fig.patch.set_facecolor("#161616")
    fig.suptitle(
        "ManiSkill orientation  |  "
        "Row 1: front view (world X-Y, stem up, knife sweeps X, cut goes INTO screen)  |  "
        "Row 2: side view (world Z-Y, white dashed = cut plane at Z-center)",
        color="#cccccc", fontsize=8.5, y=1.01
    )

    for col, fruit in enumerate(FRUITS):
        cfg = yaml.safe_load(open(f"{CFG_DIR}/{fruit}.yaml", "r", encoding="utf-8"))
        pack = _build_or_load_sdf("CuttingMesh", cfg["cutting_mesh"],
                                  CACHE_DIR, use_cache=True, force_rebuild=False)

        pts  = get_solid_world_coords(pack)
        zmin, zmax, z_cut = get_fruit_z_center(pack)
        c = FRUIT_COLOR.get(fruit, "#aaaaaa")

        xlim_x = lim(pts[:, 0])
        ylim_y = lim(pts[:, 1])
        xlim_z = lim(pts[:, 2])

        # Row 1: Front view -- world X (horiz) vs world Y (vert, stem up)
        ax0 = axes[0][col]
        ax0.set_facecolor("#161616")
        ax0.scatter(pts[:, 0], pts[:, 1], s=1.2, c=c, alpha=0.65, linewidths=0)
        ax0.set_xlim(*xlim_x)
        ax0.set_ylim(*ylim_y)
        ax0.set_aspect("equal")
        ax0.axis("off")
        ax0.set_title(fruit, color="white", fontsize=8, pad=3)
        ax0.text(0.5, 0.98, "knife down", transform=ax0.transAxes,
                 ha="center", va="top", color="#666666", fontsize=6)

        # Row 2: Side view -- world Z (horiz, cut depth) vs world Y (vert, stem up)
        ax1 = axes[1][col]
        ax1.set_facecolor("#161616")
        ax1.scatter(pts[:, 2], pts[:, 1], s=1.2, c=c, alpha=0.65, linewidths=0)
        ax1.axvline(z_cut, color="white",   linewidth=1.5, linestyle="--", alpha=0.9)
        ax1.axvline(zmin,  color="#444444", linewidth=0.7, linestyle=":")
        ax1.axvline(zmax,  color="#444444", linewidth=0.7, linestyle=":")
        ax1.set_xlim(*xlim_z)
        ax1.set_ylim(*ylim_y)
        ax1.set_aspect("equal")
        ax1.axis("off")
        ax1.set_title(f"cut={z_cut:.3f}  [{zmin:.3f},{zmax:.3f}]",
                      color="#888888", fontsize=6, pad=3)

    plt.tight_layout(h_pad=0.8)
    plt.savefig(OUT_IMG, dpi=150, bbox_inches="tight", facecolor="#161616")
    print(f"[SAVED] {OUT_IMG}")


if __name__ == "__main__":
    main()
