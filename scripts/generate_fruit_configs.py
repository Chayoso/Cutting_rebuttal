"""
Generate per-fruit YAML configs with literature-based material properties.
Base: configs/example_cutting.yaml (banana). Only cutting_mesh & plasticity
material fields + mesh_path are overridden per fruit.

Material properties are from food-science literature (median of reported ranges).
Units:
    E_Pa           Young's modulus [Pa]
    nu             Poisson ratio
    rho            Density [kg/m^3]
    sigma_y_kPa    Yield stress [kPa]
    v_cut          Cutting speed [m/s] (default 0.3 matches banana)
"""
import copy
import yaml
import trimesh
from pathlib import Path

BASE = "configs/example_cutting.yaml"
OUT_DIR = Path("configs/fruits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target SDF cell size (isotropic, meters). Keep consistent across fruits
# to make comparisons fair. ~1.5mm is fine enough for cm-scale fruits and
# avoids the MemoryError we hit with sdf_voxel=160 on small assets.
SDF_TARGET_VSIZE_M = 0.0015
SDF_VOXEL_MIN = 30
SDF_VOXEL_MAX = 96


def compute_sdf_voxel(mesh_path: str) -> int:
    """Return sdf_voxel so that vsize ≈ SDF_TARGET_VSIZE_M on longest edge."""
    m = trimesh.load(mesh_path, force="mesh")
    maxlen = float(max(m.extents))
    n = int(round(maxlen / SDF_TARGET_VSIZE_M))
    return max(SDF_VOXEL_MIN, min(SDF_VOXEL_MAX, n))

# Literature-based material properties
# (median / representative value chosen from published ranges)
FRUITS = {
    # name:           (E_Pa,     nu,   rho,   sigma_y_kPa, v_cut, mesh)
    # ── Easy ────────────────────────────────────────────────────────────
    "banana":         (7.0e3,   0.35, 960.0,  5.0,  0.30, "assets/Banana.obj"),
    "strawberry":     (2.0e5,   0.35, 900.0, 10.0,  0.30, "assets/fruits/strawberry.obj"),
    "peach":          (1.2e6,   0.33, 980.0, 50.0,  0.30, "assets/fruits/peach.obj"),
    "pear":           (5.5e6,   0.30, 1000.0, 200.0, 0.30, "assets/fruits/pear.obj"),
    "apple":          (6.5e6,   0.30, 850.0, 300.0, 0.30, "assets/fruits/apple.obj"),
    # ── Medium ──────────────────────────────────────────────────────────
    "kiwi":           (3.5e5,   0.33, 1000.0, 20.0, 0.30, "assets/fruits/kiwi.obj"),
    "tomato":         (6.5e5,   0.33, 950.0, 30.0,  0.30, "assets/fruits/tomato.obj"),
    "mango":          (1.8e6,   0.33, 1000.0, 80.0, 0.30, "assets/fruits/mango.obj"),
    "persimmon":      (1.0e6,   0.33, 970.0, 40.0,  0.30, "assets/fruits/persimmon.obj"),
    "avocado":        (1.2e6,   0.33, 980.0, 50.0,  0.30, "assets/fruits/avocado.obj"),
    # ── Hard ────────────────────────────────────────────────────────────
    "grape":          (3.5e5,   0.35, 1050.0, 15.0, 0.30, "assets/fruits/grape.obj"),
    "orange":         (1.2e6,   0.33, 900.0, 80.0,  0.30, "assets/fruits/orange.obj"),
    "lemon":          (1.5e6,   0.33, 900.0, 100.0, 0.30, "assets/fruits/lemon.obj"),
    "pineapple_slice":(3.0e6,   0.30, 950.0, 200.0, 0.30, "assets/fruits/pineapple_slice.obj"),
    "watermelon_slice":(1.2e6,  0.33, 950.0, 60.0,  0.30, "assets/fruits/watermelon_slice.obj"),
}


def make_config(base, name, E, nu, rho, sy, vc, mesh):
    cfg = copy.deepcopy(base)
    cfg["cutting_mesh"]["mesh_path"] = mesh
    cfg["cutting_mesh"]["density"] = float(rho)
    cfg["cutting_mesh"]["elasticity"]["youngs_modulus"] = float(E)
    cfg["cutting_mesh"]["elasticity"]["poisson_ratio"] = float(nu)
    cfg["plasticity"]["yield_stress_kpa"] = float(sy)
    cfg["knife"]["motion"]["cutting_speed_mps"] = float(vc)
    # Per-fruit SDF resolution to keep vsize ~1.5mm and avoid OOM on small items.
    # Banana keeps its original value (160) from the base config.
    if name != "banana":
        cfg["cutting_mesh"]["sdf_voxel"] = compute_sdf_voxel(mesh)
    return cfg


def main():
    with open(BASE, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    print(f"Generating {len(FRUITS)} fruit configs in {OUT_DIR}/")
    print("-" * 78)
    print(f"{'fruit':18s} {'E (Pa)':>10s}  {'nu':>5s}  {'rho':>7s}  "
          f"{'sigma_y':>8s}  {'v_cut':>6s}")
    print("-" * 78)

    for name, (E, nu, rho, sy, vc, mesh) in FRUITS.items():
        cfg = make_config(base, name, E, nu, rho, sy, vc, mesh)
        path = OUT_DIR / f"{name}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        voxel = cfg["cutting_mesh"]["sdf_voxel"]
        print(f"{name:18s} {E:>10.2e}  {nu:>5.2f}  {rho:>7.0f}  "
              f"{sy:>7.1f}k  {vc:>6.2f}  voxel={voxel}")

    print("-" * 78)
    print(f"Done. Configs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
