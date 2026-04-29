"""
Velocity Variation Experiment (per asset)
==========================================
Sweeps cutting speed (v_cut) over a grid and collects EEF force curves
for a single asset. Used to study how cutting force depends on speed
for each material. Output per asset in logs/velocity_variation/<asset>/.

Usage:
  python exp_velocity_variation.py --config configs/fruits/strawberry.yaml
  python exp_velocity_variation.py --config configs/fruits/apple.yaml \
         --velocities 0.15 0.20 0.25 0.30 0.35 0.40 0.45 \
         --trials-per-velocity 3
"""
import os
import sys
import json
import copy
import time
import yaml
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# ── CLI ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True,
                help="Fruit config YAML (e.g. configs/fruits/apple.yaml)")
ap.add_argument("--velocities", type=float, nargs="+",
                default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
                help="Cutting speeds to test (m/s)")
ap.add_argument("--trials-per-velocity", type=int, default=1,
                help="Repeat each velocity N times (stochastic if >1)")
ap.add_argument("--out-root", default="logs/velocity_variation")
ap.add_argument("--skip-run", action="store_true",
                help="Re-plot only from existing logs")
args = ap.parse_args()

asset_name = Path(args.config).stem
OUT_DIR = Path(args.out_root) / asset_name
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DUR = 16.0
N_INTERP = 500

# ── Load base config ─────────────────────────────────────────────────────────
with open(args.config, "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

NOM_V = float(base_cfg["knife"]["motion"]["cutting_speed_mps"])
NOM_E = float(base_cfg["cutting_mesh"]["elasticity"]["youngs_modulus"])
print(f"Velocity Variation for: {asset_name}")
print("=" * 60)
print(f"  Config:       {args.config}")
print(f"  Nominal E:    {NOM_E:.2e} Pa")
print(f"  Nominal v:    {NOM_V:.2f} m/s")
print(f"  Velocities:   {args.velocities}")
print(f"  Trials/v:     {args.trials_per_velocity}")
print(f"  Out dir:      {OUT_DIR}")
print()


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_trial_config(base, v_cut, trial_id):
    cfg = copy.deepcopy(base)
    cfg["knife"]["motion"]["cutting_speed_mps"] = float(v_cut)
    trial_log_dir = str(OUT_DIR / trial_id)
    cfg.setdefault("output", {})["enabled"] = True
    cfg["output"]["fps"] = 60.0
    cfg["output"].setdefault("export", {})["enabled"] = False
    cfg["output"].setdefault("logging", {})["enabled"] = True
    cfg["output"]["logging"]["out_dir"] = trial_log_dir
    return cfg


def run_trial(v_cut, trial_id):
    cfg = make_trial_config(base_cfg, v_cut, trial_id)
    trial_log_dir = cfg["output"]["logging"]["out_dir"]
    os.makedirs(trial_log_dir, exist_ok=True)

    tmp_cfg_path = str(OUT_DIR / f"_tmp_{trial_id}.yaml")
    with open(tmp_cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [sys.executable, "run.py",
           "--config", tmp_cfg_path, "--headless", "--run-sim"]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                            encoding="utf-8", errors="replace", env=env)
    elapsed = time.time() - t0

    try:
        os.remove(tmp_cfg_path)
    except OSError:
        pass

    json_path = os.path.join(trial_log_dir, "ee_force_fps.json")
    if result.returncode != 0 or not os.path.exists(json_path):
        print(f"  [FAIL] {trial_id}: rc={result.returncode} ({elapsed:.0f}s)")
        if result.stderr:
            print(f"         {result.stderr[:300]}")
        return None

    try:
        t, f_raw = load_force_curve(json_path)
        peak = np.max(np.abs(f_raw))
    except Exception:
        peak = -1
    print(f"  [OK] {trial_id:20s} v={v_cut:.3f} m/s  "
          f"{elapsed:5.0f}s  peak_F={peak:.2f}")
    return json_path


def load_force_curve(json_path, target_duration=16.0):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    raw_times = np.array([r["time"] for r in data])
    knife_y = np.array([r["knife"]["y_anim"] for r in data])
    cut_end = len(data)
    for i in range(1, len(knife_y)):
        if knife_y[i] > knife_y[i - 1] + 1e-6:
            cut_end = i
            break
    data = data[:cut_end]
    raw_times = raw_times[:cut_end]
    forces = np.array([-r["force_world_N"]["z"] for r in data])
    if raw_times[-1] > 1e-6:
        times = raw_times * (target_duration / raw_times[-1])
    else:
        times = raw_times
    return times, forces


def collect(paths):
    """Return dict: velocity -> list of (common_t, force_curve)."""
    common_t = np.linspace(0, TARGET_DUR, N_INTERP)
    data = {}
    for v, trial_list in paths.items():
        curves = []
        for jp in trial_list:
            if jp is None or not os.path.exists(jp):
                continue
            try:
                t, f_raw = load_force_curve(jp)
                f_interp = np.interp(common_t, t, f_raw, left=0.0,
                                     right=f_raw[-1])
                curves.append(f_interp)
            except Exception:
                pass
        if curves:
            data[v] = (common_t, np.array(curves))
    return data


def plot_summary(data, tag=""):
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"wspace": 0.28})

    cmap = plt.get_cmap("viridis")
    velocities = sorted(data.keys())
    colors = [cmap(i / max(1, len(velocities) - 1)) for i in range(len(velocities))]

    # (a) Force curves overlaid per velocity
    ax = axes[0]
    for v, c in zip(velocities, colors):
        common_t, arr = data[v]
        mean_f = np.mean(arr, axis=0)
        ax.plot(common_t, mean_f, lw=1.8, color=c, label=f"v={v:.2f} m/s")
        if arr.shape[0] > 1:
            std_f = np.std(arr, axis=0)
            ax.fill_between(common_t, mean_f - std_f, mean_f + std_f,
                            alpha=0.15, color=c)
    ax.set_title(f"(a) Force curves by cutting speed — {asset_name}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Contact Force (N)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, TARGET_DUR)

    # (b) Peak force vs. velocity
    ax = axes[1]
    peaks_mean, peaks_std = [], []
    for v in velocities:
        _, arr = data[v]
        pk = np.max(arr, axis=1)
        peaks_mean.append(np.mean(pk))
        peaks_std.append(np.std(pk) if arr.shape[0] > 1 else 0.0)
    ax.errorbar(velocities, peaks_mean, yerr=peaks_std,
                fmt="o-", lw=2, ms=7, capsize=4, color="#c0392b")
    ax.set_title(f"(b) Peak force vs. cutting speed — {asset_name}")
    ax.set_xlabel("Cutting speed v_cut (m/s)")
    ax.set_ylabel("Peak Contact Force (N)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Velocity Variation — {asset_name}  "
                 f"(E={NOM_E:.1e} Pa)", y=1.02)
    fname = f"velocity_variation_{tag}.png" if tag else "velocity_variation.png"
    plt.savefig(str(OUT_DIR / fname), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  >> Plot saved: {OUT_DIR / fname}")


# ── Run sweeps ───────────────────────────────────────────────────────────────
paths = {v: [] for v in args.velocities}

if not args.skip_run:
    t_start = time.time()
    total = len(args.velocities) * args.trials_per_velocity
    done = 0
    for v in args.velocities:
        for r in range(args.trials_per_velocity):
            done += 1
            trial_id = f"v{v:.3f}_r{r:02d}".replace(".", "p")
            print(f"[{done}/{total}] v={v:.3f} m/s  trial={r}")
            jp = run_trial(v, trial_id)
            paths[v].append(jp)
    print(f"\nAll done in {(time.time() - t_start)/60:.1f} min")

    # Save log
    log = {f"{v:.3f}": [p for p in paths[v] if p] for v in args.velocities}
    with open(OUT_DIR / "velocity_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
else:
    log_path = OUT_DIR / "velocity_log.json"
    if not log_path.exists():
        print(f"[ERROR] No velocity_log.json in {OUT_DIR}")
        sys.exit(1)
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    for vstr, trial_list in log.items():
        v = float(vstr)
        paths[v] = trial_list

# ── Plot ─────────────────────────────────────────────────────────────────────
print("\n--- Summary plot ---")
data = collect(paths)
print(f"  Loaded {sum(len(arr) for _, arr in data.values())} curves "
      f"across {len(data)} velocities")
plot_summary(data, tag="final")

# Print peak-vs-velocity table
print("\n" + "=" * 60)
print(f"{'v_cut (m/s)':>12s}  {'peak mean':>10s}  {'peak std':>10s}  {'n':>4s}")
print("-" * 60)
for v in sorted(data.keys()):
    _, arr = data[v]
    pk = np.max(arr, axis=1)
    print(f"{v:>12.3f}  {np.mean(pk):>10.2f}  "
          f"{np.std(pk):>10.2f}  {arr.shape[0]:>4d}")
print("=" * 60)
