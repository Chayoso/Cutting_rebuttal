"""
stem_finder.py
Renders each fruit + knife from a ManiSkill-style camera perspective.

ManiSkill frame: ms_X = world_X,  ms_Y = world_Z,  ms_Z = world_Y (UP)
Camera: slightly above, looking from the front-right  (elev=25, azim=-55)
Knife descends from ms_Z+ (world_Y+).

Proposed rotations (euler_deg_xyz = Rz@Ry@Rx, degrees):
  tomato       : [  0,  0, 0]
  plum         : [-90,  0, 0]
  pear         : [-90,  0, 0]
  lemon        : [ 90, 90, 0]
  shine_muscat : [-90,  0, 0]
  kiwi         : [-90,  0, 0]
"""
import sys, os, yaml, numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

sys.path.insert(0, os.path.dirname(__file__))
from sdf_utils.transforms import euler_xyz_to_mat

CFG_DIR   = "configs/fruits"
KNIFE_OBJ = "assets/knife.obj"

FRUITS = [
    # (name, color, euler_deg_xyz, y_offset)
    # y_offset: shift fruit in world-Y so equator lands at y=0
    ("tomato",       "#E82010", [  0,  0,  0],  0.0),
    ("plum",         "#CC44AA", [-90,  0,  0],  0.0),
    ("pear",         "#BBDD40", [  0,  0,  0],  0.0),
    ("lemon",        "#F8F010", [ 90,  0,  0],  0.0043),
    ("shine_muscat", "#88DD28", [-90,  0,  0],  0.0),
    ("kiwi",         "#88CC55", [  0, 90,  0],  0.0),
]

CAM_ELEV = 25
CAM_AZIM = -55


def load_fruit_verts(cfg, euler, y_offset=0.0):
    mesh_path = cfg["cutting_mesh"]["mesh_path"]
    s_raw = cfg["cutting_mesh"]["initial_transform"].get("scale", [1, 1, 1])
    if isinstance(s_raw, (int, float)):
        scale = np.array([s_raw] * 3, dtype=float)
    else:
        scale = np.array(s_raw, dtype=float)
        if len(scale) == 1:
            scale = np.repeat(scale, 3)
    mesh  = trimesh.load(mesh_path, force="mesh", process=False)
    verts = np.array(mesh.vertices, dtype=np.float64) * scale
    R     = np.array(euler_xyz_to_mat(float(euler[0]), float(euler[1]), float(euler[2])))
    verts = verts @ R.T
    # Re-center at origin after rotation (bbox center)
    verts -= (verts.max(axis=0) + verts.min(axis=0)) * 0.5
    # Shift so equator (widest cross-section) is at y=0
    verts[:, 1] -= y_offset
    return verts


def load_knife_verts():
    mesh = trimesh.load(KNIFE_OBJ, force="mesh", process=False)
    return np.array(mesh.vertices, dtype=np.float64)


EEF_VIZ_FRAC = 0.55   # 55% from blade tip = blade-handle boundary (matches sim EEF def)


def position_knife_initial(knife_raw, fruit_verts, start_y):
    """Place knife at its simulation start height (blade tip at start_y).
    Knife is centered in X/Z over the fruit bbox center.
    Returns (positioned_verts, eef_world_xyz).
    """
    kx = knife_raw[:, 0];  ky = knife_raw[:, 1];  kz = knife_raw[:, 2]
    fc = (fruit_verts.max(axis=0) + fruit_verts.min(axis=0)) * 0.5  # fruit bbox center

    tx = fc[0] - (kx.min() + kx.max()) * 0.5   # X center over fruit
    ty = start_y - ky.min()                      # blade tip at start_y
    tz = fc[2] - (kz.min() + kz.max()) * 0.5   # Z center over fruit

    v = knife_raw.copy()
    v[:, 0] += tx;  v[:, 1] += ty;  v[:, 2] += tz

    # EEF world position (35% from blade tip)
    eef_y = start_y + EEF_VIZ_FRAC * (ky.max() - ky.min())
    eef_world = np.array([fc[0], eef_y, fc[2]])
    return v, eef_world


def position_knife(knife_raw, fruit_verts):
    """Align knife EEF to fruit geometric center (mid-cut visualization).

    EEF Y = 35% from blade tip (ky.min()) toward handle.
    EEF X/Z = knife bounding-box center.
    Returns (positioned_verts, eef_world_xyz).
    """
    fx  = fruit_verts[:, 0];  fy = fruit_verts[:, 1];  fz = fruit_verts[:, 2]
    kx  = knife_raw[:, 0];    ky = knife_raw[:, 1];     kz = knife_raw[:, 2]

    # EEF local coords
    eef_x_local = (kx.min() + kx.max()) * 0.5 - 0.02                 # knife X center − 0.02
    eef_y_local = ky.min() + EEF_VIZ_FRAC * (ky.max() - ky.min())   # 35% from tip
    eef_z_local = (kz.min() + kz.max()) * 0.5                        # knife Z center

    # Fruit bbox center — exactly (0,0,0) after load_fruit_verts re-centering
    fc = (fruit_verts.max(axis=0) + fruit_verts.min(axis=0)) * 0.5

    # Translate so EEF aligns to fruit bbox center
    tx = fc[0] - eef_x_local
    ty = fc[1] - eef_y_local
    tz = fc[2] - eef_z_local

    v = knife_raw.copy()
    v[:, 0] += tx;  v[:, 1] += ty;  v[:, 2] += tz

    eef_world = fc.copy()
    return v, eef_world


def depth_colors(pts_mpl, rgb, base_alpha=0.25, bright_alpha=0.85,
                 cam_elev=CAM_ELEV, cam_azim=CAM_AZIM):
    """Compute per-point depth-shaded RGBA colors (mpl frame: Z=up)."""
    cam_az  = np.radians(cam_azim)
    cam_el  = np.radians(cam_elev)
    cam_dir = np.array([np.cos(cam_el)*np.cos(cam_az),
                        np.cos(cam_el)*np.sin(cam_az),
                        np.sin(cam_el)])
    mx, my, mz = pts_mpl[:, 0], pts_mpl[:, 1], pts_mpl[:, 2]
    depth = mx*cam_dir[0] + my*cam_dir[1] + mz*cam_dir[2]
    d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    r, g, b = rgb
    colors = np.stack([r+(1-r)*d, g+(1-g)*d, b+(1-b)*d,
                       base_alpha + (bright_alpha - base_alpha)*d], axis=1)
    return colors, np.argsort(d)


def world_to_mpl(v):
    """world → matplotlib axes  (mpl_x=world_x, mpl_y=world_z, mpl_z=world_y=UP)"""
    return np.stack([v[:, 0], v[:, 2], v[:, 1]], axis=1)


def plot_scene(ax, fruit_verts, fruit_color, knife_verts, eef_world, label,
               elev=CAM_ELEV, azim=CAM_AZIM):
    import matplotlib.colors as mcolors

    # --- fruit ---
    fmpl   = world_to_mpl(fruit_verts)
    fr, fg, fb = mcolors.to_rgb(fruit_color)
    f_col, f_ord = depth_colors(fmpl, (fr, fg, fb), base_alpha=0.25, bright_alpha=0.9,
                                 cam_elev=elev, cam_azim=azim)
    ax.scatter(fmpl[f_ord, 0], fmpl[f_ord, 1], fmpl[f_ord, 2],
               s=0.8, c=f_col[f_ord], linewidths=0, depthshade=False)

    # stem tip (top 3 % in world Y = mpl Z)
    hi = np.percentile(fmpl[:, 2], 97)
    mask = fmpl[:, 2] >= hi
    if mask.sum():
        ax.scatter(fmpl[mask, 0], fmpl[mask, 1], fmpl[mask, 2],
                   s=12, c="#FF8800", alpha=0.95, linewidths=0,
                   zorder=5, depthshade=False)

    # --- knife ---
    kmpl   = world_to_mpl(knife_verts)
    k_col, k_ord = depth_colors(kmpl, (0.85, 0.88, 0.92),
                                 base_alpha=0.35, bright_alpha=0.95,
                                 cam_elev=elev, cam_azim=azim)
    ax.scatter(kmpl[k_ord, 0], kmpl[k_ord, 1], kmpl[k_ord, 2],
               s=0.5, c=k_col[k_ord], linewidths=0, depthshade=False)

    # --- EEF marker (cyan cross + sphere) ---
    ex, ey, ez = eef_world[0], eef_world[2], eef_world[1]   # world→mpl
    ax.scatter([ex], [ey], [ez], s=120, c="#00FFFF", marker="+",
               linewidths=1.5, zorder=10, depthshade=False)
    ax.scatter([ex], [ey], [ez], s=60, c="#00FFFF", alpha=0.7,
               linewidths=0, zorder=10, depthshade=False)

    # --- axes limits: XY symmetric around fruit center, Z covers fruit+knife ---
    half_xy = max(fmpl[:, 0].max() - fmpl[:, 0].min(), fmpl[:, 1].max() - fmpl[:, 1].min()) * 0.75
    all_z   = np.concatenate([fmpl[:, 2], kmpl[:, 2]])
    z_lo    = all_z.min() - half_xy * 0.1
    z_hi    = all_z.max() + half_xy * 0.1
    ax.set_xlim(-half_xy, half_xy)
    ax.set_ylim(-half_xy, half_xy)
    ax.set_zlim(z_lo, z_hi)

    ax.set_xlabel("world_X", color="#666666", fontsize=6, labelpad=1)
    ax.set_ylabel("world_Z", color="#666666", fontsize=6, labelpad=1)
    ax.set_zlabel("world_Y↑", color="#666666", fontsize=6, labelpad=1)
    ax.tick_params(colors="#444444", labelsize=5)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#1e1e1e")
    ax.grid(False)
    ax.set_title(label, color="white", fontsize=7, pad=2)
    ax.view_init(elev=elev, azim=azim)


def render_figure(scenes, elev, azim, filename, title):
    n    = len(scenes)
    cols = 3
    rows = (n + cols - 1) // cols
    fig  = plt.figure(figsize=(cols * 4.2, rows * 4.0))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(title, color="white", fontsize=8, y=1.01)

    for i, (fruit, color, euler, fruit_verts, knife_verts, eef_w) in enumerate(scenes):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_facecolor("#0d0d0d")
        plot_scene(ax, fruit_verts, color, knife_verts, eef_w,
                   f"{fruit}  euler{euler}", elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"[SAVED] {filename}")


def main():
    knife_raw = load_knife_verts()

    scenes_mid   = []   # knife at mid-cut (EEF = fruit center)
    scenes_init  = []   # knife at initial start_y

    for fruit, color, euler, y_off in FRUITS:
        cfg         = yaml.safe_load(open(f"{CFG_DIR}/{fruit}.yaml", "r", encoding="utf-8"))
        fruit_verts = load_fruit_verts(cfg, euler, y_off)

        # mid-cut
        kv_mid, eef_mid = position_knife(knife_raw, fruit_verts)
        scenes_mid.append((fruit, color, euler, fruit_verts, kv_mid, eef_mid))

        # initial position
        start_y = float(cfg["knife"]["motion"]["knife_start_y"])
        kv_init, eef_init = position_knife_initial(knife_raw, fruit_verts, start_y)
        scenes_init.append((fruit, color, euler, fruit_verts, kv_init, eef_init))

        print(f"  {fruit}  start_y={start_y:.3f}")

    # --- initial position renders ---
    render_figure(
        scenes_init, CAM_ELEV, CAM_AZIM, "stem_finder_init.png",
        f"Initial knife position  |  blade tip at knife_start_y  |  cyan=EEF",
    )
    render_figure(
        scenes_init, 0, 180, "stem_finder_init_zview.png",
        "Initial knife position — front view (+Z)  |  X=blade  Y↑=descent  |  cyan=EEF",
    )

    # --- mid-cut renders (existing) ---
    render_figure(
        scenes_mid, CAM_ELEV, CAM_AZIM, "stem_finder.png",
        f"ManiSkill camera  elev={CAM_ELEV}°  azim={CAM_AZIM}°  |  "
        "world_Y=UP  |  cyan=EEF(35%)  |  orange=stem/tip",
    )
    render_figure(
        scenes_mid, 0, 180, "stem_finder_zview.png",
        "Front view from world +Z  |  X=blade dir  Y↑=knife descent  |  cyan=EEF",
    )


if __name__ == "__main__":
    main()
