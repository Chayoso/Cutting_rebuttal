"""
Velocity Variation Experiment (per asset)
==========================================
Sweeps cutting speed (v_cut) over a grid and collects EEF force curves
for a single asset. Force values reported are raw simulator output
`-force_world_N.z` in Newtons (no display scaling). Plot (a) uses raw
time so different v_cut produce visibly different cut durations.

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
ap.add_argument("--res-kq", type=float, default=None,
                help="Override knife resistance k_quad_per_s (default keeps YAML value, "
                     "set 0 to disable resistance)")
ap.add_argument("--rec-tau", type=float, default=None,
                help="Override knife resistance recovery tau (s)")
args = ap.parse_args()

asset_name = Path(args.config).stem
OUT_DIR = Path(args.out_root) / asset_name
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    if args.res_kq is not None:
        cfg["knife"].setdefault("speed_resistance", {})["k_quad_per_s"] = float(args.res_kq)
    if args.rec_tau is not None:
        cfg["knife"].setdefault("speed_resistance", {})["rec_tau_s"] = float(args.rec_tau)
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

    # Persist the per-trial config (don't delete) for reproducibility
    cfg_snapshot = os.path.join(trial_log_dir, "trial_config.yaml")
    with open(cfg_snapshot, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [sys.executable, "run.py",
           "--config", cfg_snapshot, "--headless", "--run-sim"]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                            encoding="utf-8", errors="replace", env=env)
    elapsed = time.time() - t0

    json_path = os.path.join(trial_log_dir, "ee_force_fps.json")
    if result.returncode != 0 or not os.path.exists(json_path):
        print(f"  [FAIL] {trial_id}: rc={result.returncode} ({elapsed:.0f}s)")
        if result.stderr:
            print(f"         {result.stderr[:300]}")
        return None

    try:
        c = load_force_curve(json_path)
        peak = float(np.max(np.abs(c["f"])))
        dur = float(c["t"][-1]) if len(c["t"]) > 0 else 0.0
        v_act = c["v_actual_contact_mean"]
        k2 = c["k2_eff_at_peak"]
    except Exception:
        peak, dur, v_act, k2 = -1.0, -1.0, 0.0, 0.0
    print(f"  [OK] {trial_id:20s} v_cmd={v_cut:.3f}  v_act={v_act:.3f}  "
          f"{elapsed:5.0f}s  peak|F|={peak:.3f} N  cut={dur:.2f}s  k2_eff={k2:.0f}/s")
    return json_path


def load_force_curve(json_path):
    """Return dict with raw_times, forces, knife_y, actual_speed (numerical
    dy/dt of post-resistance position), and telemetry summaries — all
    trimmed to the first cut window. Time axis is SIMULATION time, not
    rescaled.
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
    knife_y = knife_y[:cut_end]
    forces = np.array([-r["force_world_N"]["z"] for r in data])

    # Actual blade speed from post-resistance y_anim (downward = positive)
    if len(raw_times) >= 2:
        dy = -np.gradient(knife_y, raw_times)  # positive when knife descending
        actual_speed = dy
    else:
        actual_speed = np.zeros_like(raw_times)

    # Mean actual speed in the contact window (where |F| is non-trivial)
    f_thresh = 0.05 * float(np.max(np.abs(forces))) if len(forces) > 0 else 0.0
    contact_mask = np.abs(forces) > f_thresh
    if contact_mask.sum() >= 2:
        v_actual_contact_mean = float(np.mean(actual_speed[contact_mask]))
        v_actual_contact_min = float(np.min(actual_speed[contact_mask]))
    else:
        v_actual_contact_mean = float(np.mean(actual_speed)) if len(actual_speed) else 0.0
        v_actual_contact_min = float(np.min(actual_speed)) if len(actual_speed) else 0.0

    # Telemetry from JSON (knife resistance state at peak-force frame)
    if len(forces) > 0:
        peak_idx = int(np.argmax(np.abs(forces)))
        tel = data[peak_idx].get("telemetry", {}).get("knife_resistance", {})
        k2_eff = float(tel.get("k2_eff_per_s", 0.0))
        c_hat_peak = float(tel.get("c_hat_current", tel.get("c_hat", 0.0)))
    else:
        k2_eff = 0.0
        c_hat_peak = 0.0

    return {
        "t": raw_times,
        "f": forces,
        "y": knife_y,
        "v_actual": actual_speed,
        "v_actual_contact_mean": v_actual_contact_mean,
        "v_actual_contact_min": v_actual_contact_min,
        "k2_eff_at_peak": k2_eff,
        "c_hat_at_peak": c_hat_peak,
    }


def collect(paths):
    """Return dict: velocity -> list of curve-dicts (see load_force_curve)."""
    data = {}
    for v, trial_list in paths.items():
        curves = []
        for jp in trial_list:
            if jp is None or not os.path.exists(jp):
                continue
            try:
                c = load_force_curve(jp)
                if len(c["t"]) >= 2:
                    curves.append(c)
            except Exception:
                pass
        if curves:
            data[v] = curves
    return data


def plot_summary(data, tag=""):
    if not data:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw={"wspace": 0.32})

    cmap = plt.get_cmap("viridis")
    velocities = sorted(data.keys())
    colors = [cmap(i / max(1, len(velocities) - 1)) for i in range(len(velocities))]

    # (a) Force curves overlaid per velocity (raw time)
    ax = axes[0]
    max_t = 0.0
    for v, c in zip(velocities, colors):
        curves = data[v]
        max_dur = max(cc["t"][-1] for cc in curves)
        common_t_v = np.linspace(0, max_dur, N_INTERP)
        resampled = [np.interp(common_t_v, cc["t"], cc["f"], left=0.0, right=cc["f"][-1])
                     for cc in curves]
        arr = np.array(resampled)
        mean_f = np.mean(arr, axis=0)
        ax.plot(common_t_v, mean_f, lw=1.8, color=c, label=f"v={v:.2f} m/s")
        if arr.shape[0] > 1:
            std_f = np.std(arr, axis=0)
            ax.fill_between(common_t_v, mean_f - std_f, mean_f + std_f,
                            alpha=0.15, color=c)
        max_t = max(max_t, max_dur)
    ax.set_title(f"(a) Force vs. raw time — {asset_name}")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Knife cutting force, -F_z (N)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_t * 1.02 if max_t > 0 else 1.0)

    # (b) Peak |F| vs. commanded v_cut, with actual mean speed annotated
    ax = axes[1]
    peaks_mean, peaks_std = [], []
    v_act_mean_list, v_act_min_list, k2_list = [], [], []
    for v in velocities:
        curves = data[v]
        pk = np.array([np.max(np.abs(cc["f"])) for cc in curves])
        v_act = np.array([cc["v_actual_contact_mean"] for cc in curves])
        v_min = np.array([cc["v_actual_contact_min"] for cc in curves])
        k2 = np.array([cc["k2_eff_at_peak"] for cc in curves])
        peaks_mean.append(np.mean(pk))
        peaks_std.append(np.std(pk) if pk.size > 1 else 0.0)
        v_act_mean_list.append(np.mean(v_act))
        v_act_min_list.append(np.mean(v_min))
        k2_list.append(np.mean(k2))
    ax.errorbar(velocities, peaks_mean, yerr=peaks_std,
                fmt="o-", lw=2, ms=7, capsize=4, color="#c0392b")
    ax.set_title(f"(b) Peak |F| vs. commanded v_cut — {asset_name}")
    ax.set_xlabel("Commanded v_cut (m/s)")
    ax.set_ylabel("Peak |F| (N)")
    ax.grid(True, alpha=0.3)

    # (c) Commanded vs actual blade speed (resistance diagnostic)
    ax = axes[2]
    ax.plot(velocities, velocities, "k--", lw=1, label="Identity (no resistance)")
    ax.plot(velocities, v_act_mean_list, "o-", lw=2, color="#2980b9",
            label="actual mean (in-contact)")
    ax.plot(velocities, v_act_min_list, "s--", lw=1.2, color="#16a085",
            label="actual min (in-contact)")
    ax.set_title(f"(c) Resistance diagnostic — k2_eff≈{np.mean(k2_list):.0f}/s")
    ax.set_xlabel("Commanded v_cut (m/s)")
    ax.set_ylabel("Knife speed during contact (m/s)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)

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

    log = {
        "config": str(args.config),
        "asset": asset_name,
        "nominal_E": NOM_E,
        "nominal_v": NOM_V,
        "velocities": args.velocities,
        "trials_per_velocity": args.trials_per_velocity,
        "paths": {f"{v:.3f}": [p for p in paths[v] if p] for v in args.velocities},
    }
    with open(OUT_DIR / "velocity_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
else:
    log_path = OUT_DIR / "velocity_log.json"
    if not log_path.exists():
        print(f"[ERROR] No velocity_log.json in {OUT_DIR}")
        sys.exit(1)
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    src = log.get("paths", log) if isinstance(log, dict) else log
    for vstr, trial_list in src.items():
        v = float(vstr)
        if v not in paths:
            paths[v] = []
        paths[v] = trial_list

# ── Plot ─────────────────────────────────────────────────────────────────────
print("\n--- Summary plot ---")
data = collect(paths)
n_curves = sum(len(curves) for curves in data.values())
print(f"  Loaded {n_curves} curves across {len(data)} velocities")
plot_summary(data, tag="final")

# Print summary table + save summary.json
print("\n" + "=" * 96)
print(f"{'v_cmd':>8s}  {'peak|F| mean':>13s}  {'peak|F| std':>13s}  "
      f"{'v_act_mean':>11s}  {'v_act_min':>10s}  {'cut dur':>9s}  "
      f"{'k2_eff':>8s}  {'n':>3s}")
print("-" * 96)
summary_rows = []
for v in sorted(data.keys()):
    curves = data[v]
    pk = np.array([np.max(np.abs(cc["f"])) for cc in curves])
    durs = np.array([cc["t"][-1] for cc in curves])
    v_act = np.array([cc["v_actual_contact_mean"] for cc in curves])
    v_min = np.array([cc["v_actual_contact_min"] for cc in curves])
    k2 = np.array([cc["k2_eff_at_peak"] for cc in curves])
    print(f"{v:>8.3f}  {np.mean(pk):>13.3f}  {np.std(pk):>13.3f}  "
          f"{np.mean(v_act):>11.4f}  {np.mean(v_min):>10.4f}  "
          f"{np.mean(durs):>9.3f}  {np.mean(k2):>8.0f}  {len(curves):>3d}")
    summary_rows.append({
        "v_cmd": float(v),
        "n": int(len(curves)),
        "peak_F_mean": float(np.mean(pk)),
        "peak_F_std": float(np.std(pk)),
        "peak_F_range": [float(np.min(pk)), float(np.max(pk))],
        "v_actual_contact_mean": float(np.mean(v_act)),
        "v_actual_contact_min": float(np.mean(v_min)),
        "cut_dur_mean": float(np.mean(durs)),
        "k2_eff_at_peak_mean": float(np.mean(k2)),
    })
print("=" * 96)

summary = {
    "asset": asset_name,
    "config": str(args.config),
    "nominal_E_Pa": NOM_E,
    "nominal_v_mps": NOM_V,
    "trials_per_velocity": int(args.trials_per_velocity),
    "res_kq_override": (None if args.res_kq is None else float(args.res_kq)),
    "rec_tau_override": (None if args.rec_tau is None else float(args.rec_tau)),
    "rows": summary_rows,
}
with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"  >> Summary saved: {OUT_DIR / 'summary.json'}")
