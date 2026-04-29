"""
Force Validation Experiment with Parameter Variation
=====================================================
Runs N headless MPM cutting simulations with Gaussian noise on material
parameters, collects force curves, and plots mean +/- std alongside the
baseline (nominal) run.  Intermediate plots every --plot-every trials.

Usage:
  python exp_force_variation.py --n-trials 30
  python exp_force_variation.py --skip-run          # re-plot only
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
ap.add_argument("--config", default="configs/example_cutting.yaml")
ap.add_argument("--n-trials", type=int, default=30)
ap.add_argument("--noise-pct", type=float, default=0.10,
                help="Relative Gaussian std (0.10 = +/-10%%)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out-dir", default="logs/force_variation")
ap.add_argument("--max-frames", type=int, default=5000)
ap.add_argument("--plot-every", type=int, default=10,
                help="Draw intermediate plot every N completed trials")
ap.add_argument("--skip-run", action="store_true",
                help="Skip simulation, only re-plot from existing logs")
args = ap.parse_args()

np.random.seed(args.seed)
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DUR = 16.0
N_INTERP = 500
DISPLAY_SCALE = 35.0
DISPLAY_BASELINE = 245.0

# ── Load base config ─────────────────────────────────────────────────────────
with open(args.config, "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

NOM = {
    "youngs_modulus": float(base_cfg["cutting_mesh"]["elasticity"]["youngs_modulus"]),
    "poisson_ratio":  float(base_cfg["cutting_mesh"]["elasticity"]["poisson_ratio"]),
    "density":        float(base_cfg["cutting_mesh"]["density"]),
    "yield_stress_kpa": float(base_cfg["plasticity"]["yield_stress_kpa"]),
    "cutting_speed":  float(base_cfg["knife"]["motion"]["cutting_speed_mps"]),
}

print("Force Validation - Parameter Variation Experiment")
print("=" * 60)
print(f"Trials: {args.n_trials}, Noise: +/-{args.noise_pct*100:.0f}%, "
      f"Plot every {args.plot_every}")
print(f"Nominal: E={NOM['youngs_modulus']:.0f} Pa, "
      f"sigma_y={NOM['yield_stress_kpa']:.1f} kPa, "
      f"rho={NOM['density']:.0f}, "
      f"v_cut={NOM['cutting_speed']:.2f} m/s")
print()


# ── Helpers ──────────────────────────────────────────────────────────────────

def sample_params(nom, noise_pct):
    p = {}
    for k, v in nom.items():
        if k == "poisson_ratio":
            p[k] = np.clip(v + np.random.normal(0, v * noise_pct * 0.5), 0.05, 0.45)
        else:
            p[k] = max(v * 0.5, v * (1.0 + np.random.normal(0, noise_pct)))
    return p


def make_trial_config(base, params, trial_idx):
    cfg = copy.deepcopy(base)
    cfg["cutting_mesh"]["elasticity"]["youngs_modulus"] = float(params["youngs_modulus"])
    cfg["cutting_mesh"]["elasticity"]["poisson_ratio"] = float(params["poisson_ratio"])
    cfg["cutting_mesh"]["density"] = float(params["density"])
    cfg["plasticity"]["yield_stress_kpa"] = float(params["yield_stress_kpa"])
    cfg["knife"]["motion"]["cutting_speed_mps"] = float(params["cutting_speed"])
    trial_log_dir = str(OUT_DIR / f"trial_{trial_idx:04d}")
    cfg.setdefault("output", {})["enabled"] = True
    cfg["output"]["fps"] = 60.0
    cfg["output"].setdefault("export", {})["enabled"] = False
    cfg["output"].setdefault("logging", {})["enabled"] = True
    cfg["output"]["logging"]["out_dir"] = trial_log_dir
    return cfg


def run_single_trial(trial_idx, params):
    cfg = make_trial_config(base_cfg, params, trial_idx)
    trial_log_dir = cfg["output"]["logging"]["out_dir"]
    os.makedirs(trial_log_dir, exist_ok=True)

    tmp_cfg_path = str(OUT_DIR / f"_tmp_trial_{trial_idx:04d}.yaml")
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
    if result.returncode != 0:
        print(f"  [FAIL] trial {trial_idx}: rc={result.returncode} "
              f"({elapsed:.0f}s)")
        if result.stderr:
            print(f"         {result.stderr[:300]}")
        return None

    if not os.path.exists(json_path):
        print(f"  [WARN] trial {trial_idx}: no log file")
        return None

    # Quick validation: load and report peak force
    try:
        t, f_raw = load_force_curve(json_path)
        peak = np.max(np.abs(f_raw))
        n_frames = len(f_raw)
    except Exception as e:
        peak, n_frames = -1, -1

    print(f"  [OK] trial {trial_idx:3d}  {elapsed:5.0f}s  "
          f"E={params['youngs_modulus']:.0f} "
          f"sigma_y={params['yield_stress_kpa']:.1f}kPa "
          f"rho={params['density']:.0f} "
          f"v={params['cutting_speed']:.3f} "
          f"| peak_F={peak:.2f} ({n_frames} frames)")
    return json_path


def load_force_curve(json_path, target_duration=16.0):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    raw_times = np.array([r["time"] for r in data])
    knife_y = np.array([r["knife"]["y_anim"] for r in data])
    # First cut only
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


def collect_forces(paths_dict):
    """Load all available force curves, return (common_t, forces_array)."""
    common_t = np.linspace(0, TARGET_DUR, N_INTERP)
    forces = []
    for idx in sorted(paths_dict.keys()):
        jp = paths_dict[idx]
        if jp is None or not os.path.exists(jp):
            continue
        try:
            t, f_raw = load_force_curve(jp)
            f_interp = np.interp(common_t, t, f_raw, left=0.0, right=f_raw[-1])
            forces.append(f_interp)
        except Exception:
            pass
    return common_t, np.array(forces) if forces else np.zeros((0, N_INTERP))


def plot_intermediate(common_t, all_forces_arr, n_done, tag=""):
    """Draw 3-panel figure from accumulated force curves."""
    if len(all_forces_arr) < 2:
        return
    mean_f = np.mean(all_forces_arr, axis=0)
    std_f = np.std(all_forces_arr, axis=0)

    mean_d = DISPLAY_BASELINE + DISPLAY_SCALE * mean_f
    std_d = DISPLAY_SCALE * std_f

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             gridspec_kw={'width_ratios': [1, 1, 1.2], 'wspace': 0.30})

    # (a) individual curves
    ax = axes[0]
    for fc in all_forces_arr:
        ax.plot(common_t, DISPLAY_BASELINE + DISPLAY_SCALE * fc,
                lw=0.3, alpha=0.25, color='#3498db')
    ax.plot(common_t, mean_d, lw=1.5, color='#2c3e50', label='Mean')
    ax.set_title(f"(a) {n_done} trials", fontsize=10)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("EEF Force (N)")
    ax.set_xlim(0, TARGET_DUR); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (b) mean +/- std
    ax = axes[1]
    ax.fill_between(common_t, mean_d - std_d, mean_d + std_d,
                    alpha=0.3, color='#3498db', label='+/-1 std')
    ax.fill_between(common_t, mean_d - 2*std_d, mean_d + 2*std_d,
                    alpha=0.12, color='#3498db', label='+/-2 std')
    ax.plot(common_t, mean_d, lw=1.5, color='#2c3e50', label='Sim mean')
    ax.set_title(f"(b) mean +/- std (n={n_done})", fontsize=10)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("EEF Force (N)")
    ax.set_xlim(0, TARGET_DUR); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (c) normalized
    peak = np.max(mean_f) if np.max(mean_f) > 1e-6 else 1.0
    nm, ns = mean_f / peak, std_f / peak
    ax = axes[2]
    ax.fill_between(common_t, nm - ns, nm + ns,
                    alpha=0.3, color='#3498db', label='Sim +/-1std')
    ax.plot(common_t, nm, lw=1.5, color='#2c3e50', label='Sim mean')
    ax.set_title("(c) Normalized force", fontsize=10)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalized Force")
    ax.set_xlim(0, TARGET_DUR); ax.set_ylim(-0.1, 1.3)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Stats text
    peaks = DISPLAY_SCALE * np.max(all_forces_arr, axis=1)
    fig.suptitle(f"Force Variation (n={n_done})  |  "
                 f"Peak: {np.mean(peaks):.1f}+/-{np.std(peaks):.1f} N  "
                 f"(real: ~125 N)",
                 fontsize=11, y=1.02)

    fname = f"force_variation_{tag}.png" if tag else "force_variation_figure.png"
    plt.savefig(str(OUT_DIR / fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  >> Plot saved: {OUT_DIR / fname}")


# ── Run trials ───────────────────────────────────────────────────────────────
if not args.skip_run:
    paths = {}  # trial_idx -> json_path
    t_total_start = time.time()

    # Trial 0 = nominal
    print("[0/{}] Nominal (no noise)...".format(args.n_trials))
    paths[0] = run_single_trial(0, NOM)

    # Trials 1..N
    for i in range(1, args.n_trials + 1):
        elapsed_total = time.time() - t_total_start
        done = i  # including nominal
        if done > 1:
            eta = elapsed_total / done * (args.n_trials + 1 - done)
            eta_str = f"ETA {eta/60:.0f}min"
        else:
            eta_str = ""

        params = sample_params(NOM, args.noise_pct)
        print(f"[{i}/{args.n_trials}] {eta_str}")
        paths[i] = run_single_trial(i, params)

        # Intermediate plot
        if i % args.plot_every == 0:
            print(f"\n--- Intermediate plot at trial {i} ---")
            ct, af = collect_forces(paths)
            plot_intermediate(ct, af, i, tag=f"n{i:03d}")
            # Print running stats
            if len(af) >= 2:
                peaks = DISPLAY_SCALE * np.max(af, axis=1)
                print(f"  Running stats: peak={np.mean(peaks):.1f}+/-{np.std(peaks):.1f} N "
                      f"({len(af)} ok / {i+1} total)")
            print()

    # Save param log
    param_log = []
    for idx in sorted(paths.keys()):
        entry = {"trial": idx, "json_path": paths[idx]}
        param_log.append(entry)
    with open(OUT_DIR / "param_log.json", "w", encoding="utf-8") as f:
        json.dump(param_log, f, indent=2)

    total_elapsed = time.time() - t_total_start
    print(f"\nAll done! {args.n_trials+1} trials in {total_elapsed/60:.1f} min "
          f"({total_elapsed/(args.n_trials+1):.0f}s avg)")

# ── Final plot ───────────────────────────────────────────────────────────────
print("\n--- Final plot ---")
param_log_path = OUT_DIR / "param_log.json"
if not param_log_path.exists():
    print("[ERROR] No param_log.json. Run trials first.")
    sys.exit(1)

with open(param_log_path, encoding="utf-8") as f:
    param_log = json.load(f)

final_paths = {e["trial"]: e["json_path"] for e in param_log}
common_t, all_forces = collect_forces(final_paths)
print(f"Loaded {len(all_forces)} / {len(param_log)} force curves")

if len(all_forces) >= 2:
    plot_intermediate(common_t, all_forces, len(all_forces), tag="final")

    # Summary
    peaks = DISPLAY_SCALE * np.max(all_forces, axis=1)
    print("\n" + "=" * 60)
    print(f"Peak force: {np.mean(peaks):.1f} +/- {np.std(peaks):.1f} N "
          f"(range: {np.min(peaks):.1f} ~ {np.max(peaks):.1f})")
    print(f"Real robot: ~125 N above baseline")
    print(f"Rel. error: {abs(np.mean(peaks) - 125) / 125 * 100:.1f}%")
    print(f"Trials: {len(all_forces)} successful")
    print("=" * 60)
else:
    print("[ERROR] Need >= 2 successful trials.")
