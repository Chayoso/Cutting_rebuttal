"""
Material Variation Experiment (per asset)
==========================================
Runs N headless MPM cutting simulations with Gaussian noise on **material**
parameters only (E, nu, rho, sigma_y) — kinematic v_cut is held fixed at the
nominal value. Trial 0 = nominal (no noise) is run separately and reported as
a reference, NOT included in the summary statistics.

Force values reported are raw simulator output `-force_world_N.z` in Newtons,
without any post-hoc display scaling.

Usage:
  python exp_force_variation.py --config configs/fruits/strawberry.yaml --n-trials 30
  python exp_force_variation.py --config configs/fruits/strawberry.yaml --skip-run
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
ap.add_argument("--n-trials", type=int, default=30,
                help="Number of *noise* trials (nominal trials are run separately)")
ap.add_argument("--n-nominal", type=int, default=3,
                help="Number of nominal (no-noise) trials. Reported as own mean+/-std")
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

# ── Load base config ─────────────────────────────────────────────────────────
with open(args.config, "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

NOM = {
    "youngs_modulus":   float(base_cfg["cutting_mesh"]["elasticity"]["youngs_modulus"]),
    "poisson_ratio":    float(base_cfg["cutting_mesh"]["elasticity"]["poisson_ratio"]),
    "density":          float(base_cfg["cutting_mesh"]["density"]),
    "yield_stress_kpa": float(base_cfg["plasticity"]["yield_stress_kpa"]),
}
NOM_V = float(base_cfg["knife"]["motion"]["cutting_speed_mps"])

print("Material Variation Experiment (raw simulator forces)")
print("=" * 60)
print(f"Trials: {args.n_trials} (+ 1 nominal),  Noise: +/-{args.noise_pct*100:.0f}%, "
      f"Plot every {args.plot_every}")
print(f"Nominal: E={NOM['youngs_modulus']:.0f} Pa, "
      f"sigma_y={NOM['yield_stress_kpa']:.1f} kPa, "
      f"rho={NOM['density']:.0f} kg/m^3, "
      f"v_cut={NOM_V:.3f} m/s (held fixed)")
print()


# ── Helpers ──────────────────────────────────────────────────────────────────

def sample_params(nom, noise_pct):
    p = {}
    for k, v in nom.items():
        if k == "poisson_ratio":
            p[k] = float(np.clip(v + np.random.normal(0, v * noise_pct * 0.5), 0.05, 0.45))
        else:
            p[k] = float(max(v * 0.5, v * (1.0 + np.random.normal(0, noise_pct))))
    return p


def make_trial_config(base, params, trial_dirname):
    cfg = copy.deepcopy(base)
    cfg["cutting_mesh"]["elasticity"]["youngs_modulus"] = float(params["youngs_modulus"])
    cfg["cutting_mesh"]["elasticity"]["poisson_ratio"] = float(params["poisson_ratio"])
    cfg["cutting_mesh"]["density"] = float(params["density"])
    cfg["plasticity"]["yield_stress_kpa"] = float(params["yield_stress_kpa"])
    # v_cut intentionally NOT perturbed — kept at nominal.
    trial_log_dir = str(OUT_DIR / trial_dirname)
    cfg.setdefault("output", {})["enabled"] = True
    cfg["output"]["fps"] = 60.0
    cfg["output"].setdefault("export", {})["enabled"] = False
    cfg["output"].setdefault("logging", {})["enabled"] = True
    cfg["output"]["logging"]["out_dir"] = trial_log_dir
    return cfg


def run_single_trial(trial_dirname, params, label=""):
    cfg = make_trial_config(base_cfg, params, trial_dirname)
    trial_log_dir = cfg["output"]["logging"]["out_dir"]
    os.makedirs(trial_log_dir, exist_ok=True)

    cfg_snapshot = os.path.join(trial_log_dir, "trial_config.yaml")
    with open(cfg_snapshot, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [sys.executable, "run.py",
           "--config", cfg_snapshot, "--headless", "--run-sim"]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                            encoding="utf-8", errors="replace", env=env)
    elapsed = time.time() - t0

    json_path = os.path.join(trial_log_dir, "ee_force_fps.json")
    if result.returncode != 0:
        print(f"  [FAIL] {label or trial_dirname}: rc={result.returncode} "
              f"({elapsed:.0f}s)")
        if result.stderr:
            print(f"         {result.stderr[:300]}")
        return None, elapsed

    if not os.path.exists(json_path):
        print(f"  [WARN] {label or trial_dirname}: no log file")
        return None, elapsed

    try:
        t_norm, f_raw, dur = load_force_curve(json_path)
        peak = float(np.max(np.abs(f_raw)))
        n_frames = len(f_raw)
    except Exception:
        peak, n_frames, dur = -1.0, -1, -1.0

    print(f"  [OK] {label:15s}  {elapsed:5.0f}s  "
          f"E={params['youngs_modulus']:.0f} "
          f"sigma_y={params['yield_stress_kpa']:.1f}kPa "
          f"rho={params['density']:.0f} "
          f"nu={params['poisson_ratio']:.3f} "
          f"| peak|F|={peak:.3f} N  cut={dur:.2f}s  ({n_frames} frames)")
    return json_path, elapsed


def load_force_curve(json_path, target_duration=TARGET_DUR):
    """Return (rescaled_times, forces, raw_duration_s).

    The time axis is rescaled to [0, target_duration] for cross-trial
    averaging; raw cut duration is returned separately so the caller can
    report the natural time scale.
    """
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
    raw_dur = float(raw_times[-1]) if len(raw_times) > 0 else 0.0
    if raw_dur > 1e-6:
        times = raw_times * (target_duration / raw_dur)
    else:
        times = raw_times
    return times, forces, raw_dur


def collect_forces(paths_dict):
    """Load all available force curves, return (common_t, forces_array, durations)."""
    common_t = np.linspace(0, TARGET_DUR, N_INTERP)
    forces, durs = [], []
    for idx in sorted(paths_dict.keys()):
        jp = paths_dict[idx]
        if jp is None or not os.path.exists(jp):
            continue
        try:
            t, f_raw, dur = load_force_curve(jp)
            f_interp = np.interp(common_t, t, f_raw, left=0.0, right=f_raw[-1])
            forces.append(f_interp)
            durs.append(dur)
        except Exception:
            pass
    if not forces:
        return common_t, np.zeros((0, N_INTERP)), np.zeros(0)
    return common_t, np.array(forces), np.array(durs)


def plot_intermediate(common_t, all_forces_arr, durations, n_done,
                      nominal_curve=None, tag=""):
    """Draw 3-panel figure from accumulated force curves (raw N)."""
    if len(all_forces_arr) < 2:
        return
    mean_f = np.mean(all_forces_arr, axis=0)
    std_f = np.std(all_forces_arr, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             gridspec_kw={'width_ratios': [1, 1, 1.2], 'wspace': 0.30})

    # (a) individual curves
    ax = axes[0]
    for fc in all_forces_arr:
        ax.plot(common_t, fc, lw=0.3, alpha=0.25, color='#3498db')
    ax.plot(common_t, mean_f, lw=1.5, color='#2c3e50', label='Mean (noise trials)')
    if nominal_curve is not None:
        ax.plot(common_t, nominal_curve, lw=1.5, color='#e67e22',
                linestyle='--', label='Nominal')
    ax.set_title(f"(a) {n_done} noise trials", fontsize=10)
    ax.set_xlabel("Rescaled time (cut → 16 s)")
    ax.set_ylabel("Knife cutting force, -F_z (N)")
    ax.set_xlim(0, TARGET_DUR); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (b) mean +/- std
    ax = axes[1]
    ax.fill_between(common_t, mean_f - std_f, mean_f + std_f,
                    alpha=0.3, color='#3498db', label='+/-1 std')
    ax.fill_between(common_t, mean_f - 2*std_f, mean_f + 2*std_f,
                    alpha=0.12, color='#3498db', label='+/-2 std')
    ax.plot(common_t, mean_f, lw=1.5, color='#2c3e50', label='Sim mean')
    if nominal_curve is not None:
        ax.plot(common_t, nominal_curve, lw=1.2, color='#e67e22',
                linestyle='--', label='Nominal')
    ax.set_title(f"(b) mean +/- std (n={n_done})", fontsize=10)
    ax.set_xlabel("Rescaled time (cut → 16 s)")
    ax.set_ylabel("Knife cutting force, -F_z (N)")
    ax.set_xlim(0, TARGET_DUR); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (c) normalized
    peak = np.max(np.abs(mean_f)) if np.max(np.abs(mean_f)) > 1e-6 else 1.0
    nm, ns = mean_f / peak, std_f / peak
    ax = axes[2]
    ax.fill_between(common_t, nm - ns, nm + ns,
                    alpha=0.3, color='#3498db', label='+/-1 std')
    ax.plot(common_t, nm, lw=1.5, color='#2c3e50', label='Sim mean')
    ax.set_title("(c) Normalized force", fontsize=10)
    ax.set_xlabel("Rescaled time")
    ax.set_ylabel("Force / peak(mean)")
    ax.set_xlim(0, TARGET_DUR); ax.set_ylim(-0.5, 1.3)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    peaks = np.max(np.abs(all_forces_arr), axis=1)
    dur_str = (f"  cut dur={np.mean(durations):.2f}+/-{np.std(durations):.2f}s"
               if len(durations) > 0 else "")
    fig.suptitle(f"Material Variation (n={n_done})  |  "
                 f"peak|F|={np.mean(peaks):.3f}+/-{np.std(peaks):.3f} N{dur_str}",
                 fontsize=11, y=1.02)

    fname = f"force_variation_{tag}.png" if tag else "force_variation_figure.png"
    plt.savefig(str(OUT_DIR / fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  >> Plot saved: {OUT_DIR / fname}")


# ── Run trials ───────────────────────────────────────────────────────────────
if not args.skip_run:
    nominal_paths = []      # list of json_paths for nominal repeats
    paths = {}              # trial_idx -> json_path  (noise trials only)
    params_per_trial = {}   # trial_idx -> perturbed param dict
    t_total_start = time.time()

    # Nominal trials (no noise) — repeated --n-nominal times
    for k in range(args.n_nominal):
        print(f"[nominal {k+1}/{args.n_nominal}]")
        jp, _ = run_single_trial(f"nominal_{k:02d}", NOM, label=f"nominal_{k:02d}")
        if jp:
            nominal_paths.append(jp)

    # Noise trials 1..N — material noise only, v_cut held fixed
    for i in range(1, args.n_trials + 1):
        elapsed_total = time.time() - t_total_start
        n_done_so_far = args.n_nominal + i - 1
        n_total = args.n_nominal + args.n_trials
        if n_done_so_far > 0:
            avg = elapsed_total / n_done_so_far
            eta = avg * (n_total - n_done_so_far)
            eta_str = f"ETA {eta/60:.0f}min"
        else:
            eta_str = ""

        params = sample_params(NOM, args.noise_pct)
        params_per_trial[i] = params
        print(f"[noise {i}/{args.n_trials}] {eta_str}")
        jp, _ = run_single_trial(f"trial_{i:04d}", params, label=f"noise_{i:03d}")
        paths[i] = jp

        if i % args.plot_every == 0:
            print(f"\n--- Intermediate plot at trial {i} ---")
            ct, af, durs = collect_forces(paths)
            nom_curve = None
            if nominal_paths and os.path.exists(nominal_paths[0]):
                try:
                    nt, nf, _ = load_force_curve(nominal_paths[0])
                    nom_curve = np.interp(ct, nt, nf, left=0.0, right=nf[-1])
                except Exception:
                    pass
            plot_intermediate(ct, af, durs, len(af), nominal_curve=nom_curve,
                              tag=f"n{i:03d}")
            if len(af) >= 2:
                pk = np.max(np.abs(af), axis=1)
                print(f"  Running stats: peak|F|={np.mean(pk):.3f}"
                      f"+/-{np.std(pk):.3f} N  ({len(af)} ok / {i} noise trials)")
            print()

    # Save param_log including perturbed values + nominal repeats
    param_log = {
        "config": str(args.config),
        "seed": int(args.seed),
        "noise_pct": float(args.noise_pct),
        "n_trials": int(args.n_trials),
        "n_nominal": int(args.n_nominal),
        "nominal_params": {
            "youngs_modulus": NOM["youngs_modulus"],
            "poisson_ratio": NOM["poisson_ratio"],
            "density": NOM["density"],
            "yield_stress_kpa": NOM["yield_stress_kpa"],
            "cutting_speed_mps_fixed": NOM_V,
        },
        "nominal_paths": nominal_paths,
        "trials": [
            {
                "trial": idx,
                "json_path": paths[idx],
                "params": params_per_trial.get(idx),
            }
            for idx in sorted(paths.keys())
        ],
    }
    with open(OUT_DIR / "param_log.json", "w", encoding="utf-8") as f:
        json.dump(param_log, f, indent=2)

    total_elapsed = time.time() - t_total_start
    n_done = args.n_nominal + args.n_trials
    print(f"\nAll done! {n_done} trials in {total_elapsed/60:.1f} min "
          f"({total_elapsed/n_done:.0f}s avg)")

# ── Final plot ───────────────────────────────────────────────────────────────
print("\n--- Final plot ---")
param_log_path = OUT_DIR / "param_log.json"
if not param_log_path.exists():
    print("[ERROR] No param_log.json. Run trials first.")
    sys.exit(1)

with open(param_log_path, encoding="utf-8") as f:
    plog = json.load(f)

# Schema variants from prior runs:
nominal_paths = []
if isinstance(plog, list):
    final_paths = {e["trial"]: e["json_path"] for e in plog if e["trial"] != 0}
else:
    nominal_paths = plog.get("nominal_paths") or [plog.get("nominal", {}).get("json_path")]
    nominal_paths = [p for p in nominal_paths if p]
    final_paths = {e["trial"]: e["json_path"] for e in plog["trials"]}

common_t, all_forces, durations = collect_forces(final_paths)
print(f"Loaded {len(all_forces)} / {len(final_paths)} noise-trial force curves")

# Nominal curves: average for plot reference, also compute peak stats
nominal_curve = None
nominal_peaks = []
nominal_durs = []
if nominal_paths:
    nom_curves = []
    for npath in nominal_paths:
        if not (npath and os.path.exists(npath)):
            continue
        try:
            nt, nf, ndur = load_force_curve(npath)
            nom_curves.append(np.interp(common_t, nt, nf, left=0.0, right=nf[-1]))
            nominal_peaks.append(float(np.max(np.abs(nf))))
            nominal_durs.append(float(ndur))
        except Exception:
            pass
    if nom_curves:
        nominal_curve = np.mean(np.array(nom_curves), axis=0)

if len(all_forces) >= 2:
    plot_intermediate(common_t, all_forces, durations, len(all_forces),
                      nominal_curve=nominal_curve, tag="final")

    peaks = np.max(np.abs(all_forces), axis=1)
    print("\n" + "=" * 60)
    print(f"Noise trials (n={len(all_forces)}):")
    print(f"  peak|F|: {np.mean(peaks):.3f} +/- {np.std(peaks):.3f} N "
          f"(range: {np.min(peaks):.3f} ~ {np.max(peaks):.3f})")
    print(f"  cut duration: {np.mean(durations):.3f} +/- {np.std(durations):.3f} s")
    if nominal_peaks:
        npa = np.array(nominal_peaks)
        nda = np.array(nominal_durs) if nominal_durs else np.zeros(0)
        print(f"Nominal trials (n={len(npa)}):")
        print(f"  peak|F|: {np.mean(npa):.3f} +/- {np.std(npa):.3f} N "
              f"(range: {np.min(npa):.3f} ~ {np.max(npa):.3f})")
        if nda.size > 0:
            print(f"  cut duration: {np.mean(nda):.3f} +/- {np.std(nda):.3f} s")
    print("=" * 60)

    asset_name = Path(args.config).stem
    summary = {
        "asset": asset_name,
        "config": str(args.config),
        "noise_pct": float(plog.get("noise_pct", args.noise_pct)),
        "seed": int(plog.get("seed", args.seed)),
        "nominal_params": plog.get("nominal_params") or plog.get("nominal", {}),
        "noise_trials": {
            "n": int(len(all_forces)),
            "peak_F_mean": float(np.mean(peaks)),
            "peak_F_std": float(np.std(peaks)),
            "peak_F_range": [float(np.min(peaks)), float(np.max(peaks))],
            "cut_dur_mean": float(np.mean(durations)),
            "cut_dur_std": float(np.std(durations)),
        },
        "nominal_trials": (
            {
                "n": int(len(nominal_peaks)),
                "peak_F_mean": float(np.mean(nominal_peaks)),
                "peak_F_std": float(np.std(nominal_peaks)),
                "peak_F_range": [float(np.min(nominal_peaks)),
                                 float(np.max(nominal_peaks))],
                "cut_dur_mean": (float(np.mean(nominal_durs))
                                 if nominal_durs else None),
                "cut_dur_std": (float(np.std(nominal_durs))
                                if nominal_durs else None),
            }
            if nominal_peaks else None
        ),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  >> Summary saved: {OUT_DIR / 'summary.json'}")
else:
    print("[ERROR] Need >= 2 successful trials.")
