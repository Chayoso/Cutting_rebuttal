"""
Friction Variation Experiment (per asset)
==========================================
Sweeps the board-fruit friction coefficient `board.friction` over a grid and
collects EEF force curves + fruit centroid (COM) trajectories. The goal is to
quantify how much the fruit drifts on the cutting board during a single cut
across the realistic friction range — addressing the reviewer concern that
"with no securing the object would jump or move".

Default friction value used in mpmcore is 0.5 (sticky); we sweep down to 0.0
(free slide) to demonstrate that the simulator captures macroscopic object
motion when friction is too low.

Usage:
  python exp_friction_variation.py --config configs/fruits/strawberry.yaml
  python exp_friction_variation.py --config configs/fruits/apple.yaml \
         --frictions 0.0 0.05 0.1 0.2 0.3 0.5 0.8 \
         --trials-per-friction 1
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
ap.add_argument("--frictions", type=float, nargs="+",
                default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8],
                help="Board-fruit friction coefficients to test")
ap.add_argument("--trials-per-friction", type=int, default=1,
                help="Repeat each friction value N times")
ap.add_argument("--out-root", default="logs/friction_variation")
ap.add_argument("--skip-run", action="store_true",
                help="Re-plot only from existing logs")
args = ap.parse_args()

asset_name = Path(args.config).stem
OUT_DIR = Path(args.out_root) / asset_name
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_INTERP = 500

# ── Load base config ─────────────────────────────────────────────────────────
with open(args.config, "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

NOM_E = float(base_cfg["cutting_mesh"]["elasticity"]["youngs_modulus"])
NOM_V = float(base_cfg["knife"]["motion"]["cutting_speed_mps"])
DEFAULT_BOARD_MU = float(base_cfg.get("board", {}).get("friction", 0.5))

print(f"Friction Variation for: {asset_name}")
print("=" * 60)
print(f"  Config:              {args.config}")
print(f"  Nominal E:           {NOM_E:.2e} Pa")
print(f"  Nominal v_cut:       {NOM_V:.3f} m/s")
print(f"  Default board mu:    {DEFAULT_BOARD_MU}")
print(f"  Sweep mu values:     {args.frictions}")
print(f"  Trials per friction: {args.trials_per_friction}")
print(f"  Out dir:             {OUT_DIR}")
print()


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_trial_config(base, mu, trial_id):
    cfg = copy.deepcopy(base)
    cfg.setdefault("board", {})["friction"] = float(mu)
    trial_log_dir = str(OUT_DIR / trial_id)
    cfg.setdefault("output", {})["enabled"] = True
    cfg["output"]["fps"] = 60.0
    cfg["output"].setdefault("export", {})["enabled"] = False
    cfg["output"].setdefault("logging", {})["enabled"] = True
    cfg["output"]["logging"]["out_dir"] = trial_log_dir
    return cfg


def run_trial(mu, trial_id):
    cfg = make_trial_config(base_cfg, mu, trial_id)
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
        c = load_curve(json_path)
        peak = float(np.max(np.abs(c["f"])))
        drift = c["com_drift_max"]
        final_drift = c["com_drift_final"]
    except Exception as e:
        peak, drift, final_drift = -1.0, -1.0, -1.0
    print(f"  [OK] {trial_id:18s} mu={mu:.3f}  {elapsed:5.0f}s  "
          f"peak|F|={peak:.3f} N  drift_max={drift*1000:.1f} mm  "
          f"drift_final={final_drift*1000:.1f} mm")
    return json_path


def load_curve(json_path):
    """Return curve dict trimmed to the first cut window. Includes COM drift."""
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

    # COM (may be missing for older runs without the new logging)
    if "fruit_com" in data[0]:
        com = np.array([[r["fruit_com"]["x"],
                         r["fruit_com"]["y"],
                         r["fruit_com"]["z"]] for r in data])
        com0 = com[0]
        com_xz = com[:, [0, 2]]
        com_xz0 = com0[[0, 2]]
        # XZ-plane drift (lateral motion on cutting board)
        drift_xz = np.linalg.norm(com_xz - com_xz0, axis=1)
        com_drift_max = float(np.max(drift_xz))
        com_drift_final = float(drift_xz[-1])
    else:
        com = np.zeros((len(data), 3))
        com_drift_max = 0.0
        com_drift_final = 0.0

    return {
        "t": raw_times,
        "f": forces,
        "com": com,
        "com_drift_max": com_drift_max,
        "com_drift_final": com_drift_final,
    }


def collect(paths):
    data = {}
    for mu, trial_list in paths.items():
        curves = []
        for jp in trial_list:
            if jp is None or not os.path.exists(jp):
                continue
            try:
                c = load_curve(jp)
                if len(c["t"]) >= 2:
                    curves.append(c)
            except Exception:
                pass
        if curves:
            data[mu] = curves
    return data


def plot_summary(data, tag=""):
    if not data:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw={"wspace": 0.32})

    cmap = plt.get_cmap("viridis")
    mus = sorted(data.keys())
    colors = [cmap(i / max(1, len(mus) - 1)) for i in range(len(mus))]

    # (a) COM XZ-drift over time per mu
    ax = axes[0]
    max_t = 0.0
    for mu, c in zip(mus, colors):
        curves = data[mu]
        max_dur = max(cc["t"][-1] for cc in curves)
        common_t = np.linspace(0, max_dur, N_INTERP)
        # Mean drift trajectory
        drifts = []
        for cc in curves:
            com_xz = cc["com"][:, [0, 2]]
            com0 = com_xz[0]
            d = np.linalg.norm(com_xz - com0, axis=1)
            d_interp = np.interp(common_t, cc["t"], d, left=0.0, right=d[-1])
            drifts.append(d_interp)
        arr = np.array(drifts) * 1000.0  # m → mm
        mean_d = np.mean(arr, axis=0)
        ax.plot(common_t, mean_d, lw=1.8, color=c, label=f"mu={mu:.2f}")
        if arr.shape[0] > 1:
            std_d = np.std(arr, axis=0)
            ax.fill_between(common_t, mean_d - std_d, mean_d + std_d,
                            alpha=0.15, color=c)
        max_t = max(max_t, max_dur)
    ax.set_title(f"(a) COM XZ-drift vs time — {asset_name}")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("COM drift |delta_xz| (mm)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_t * 1.02 if max_t > 0 else 1.0)

    # (b) max & final drift vs mu
    ax = axes[1]
    drift_max_mean, drift_max_std = [], []
    drift_fin_mean, drift_fin_std = [], []
    for mu in mus:
        dmax = np.array([cc["com_drift_max"] for cc in data[mu]]) * 1000.0
        dfin = np.array([cc["com_drift_final"] for cc in data[mu]]) * 1000.0
        drift_max_mean.append(np.mean(dmax))
        drift_max_std.append(np.std(dmax) if dmax.size > 1 else 0.0)
        drift_fin_mean.append(np.mean(dfin))
        drift_fin_std.append(np.std(dfin) if dfin.size > 1 else 0.0)
    ax.errorbar(mus, drift_max_mean, yerr=drift_max_std,
                fmt="o-", lw=2, ms=7, capsize=4, color="#c0392b", label="max during cut")
    ax.errorbar(mus, drift_fin_mean, yerr=drift_fin_std,
                fmt="s--", lw=1.5, ms=6, capsize=3, color="#16a085", label="final after cut")
    ax.axvline(DEFAULT_BOARD_MU, color="k", lw=0.8, linestyle=":", label=f"default mu={DEFAULT_BOARD_MU}")
    ax.set_title(f"(b) COM drift vs board friction — {asset_name}")
    ax.set_xlabel("Board-fruit friction mu")
    ax.set_ylabel("COM drift |delta_xz| (mm)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (c) peak force vs mu
    ax = axes[2]
    pk_mean, pk_std = [], []
    for mu in mus:
        pk = np.array([np.max(np.abs(cc["f"])) for cc in data[mu]])
        pk_mean.append(np.mean(pk))
        pk_std.append(np.std(pk) if pk.size > 1 else 0.0)
    ax.errorbar(mus, pk_mean, yerr=pk_std,
                fmt="o-", lw=2, ms=7, capsize=4, color="#2980b9")
    ax.axvline(DEFAULT_BOARD_MU, color="k", lw=0.8, linestyle=":")
    ax.set_title(f"(c) Peak |F| vs board friction — {asset_name}")
    ax.set_xlabel("Board-fruit friction mu")
    ax.set_ylabel("Peak |F| (N)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Friction Variation — {asset_name}  (E={NOM_E:.1e} Pa)", y=1.02)
    fname = f"friction_variation_{tag}.png" if tag else "friction_variation.png"
    plt.savefig(str(OUT_DIR / fname), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  >> Plot saved: {OUT_DIR / fname}")


# ── Run sweeps ───────────────────────────────────────────────────────────────
paths = {mu: [] for mu in args.frictions}

if not args.skip_run:
    t_start = time.time()
    total = len(args.frictions) * args.trials_per_friction
    done = 0
    for mu in args.frictions:
        for r in range(args.trials_per_friction):
            done += 1
            trial_id = f"mu{mu:.3f}_r{r:02d}".replace(".", "p")
            print(f"[{done}/{total}] mu={mu:.3f}  trial={r}")
            jp = run_trial(mu, trial_id)
            paths[mu].append(jp)
    print(f"\nAll done in {(time.time() - t_start)/60:.1f} min")

    log = {
        "config": str(args.config),
        "asset": asset_name,
        "nominal_E": NOM_E,
        "nominal_v": NOM_V,
        "default_board_mu": DEFAULT_BOARD_MU,
        "frictions": args.frictions,
        "trials_per_friction": args.trials_per_friction,
        "paths": {f"{mu:.3f}": [p for p in paths[mu] if p] for mu in args.frictions},
    }
    with open(OUT_DIR / "friction_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
else:
    log_path = OUT_DIR / "friction_log.json"
    if not log_path.exists():
        print(f"[ERROR] No friction_log.json in {OUT_DIR}")
        sys.exit(1)
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    src = log.get("paths", {})
    for mustr, trial_list in src.items():
        mu = float(mustr)
        if mu not in paths:
            paths[mu] = []
        paths[mu] = trial_list

# ── Plot + summary ───────────────────────────────────────────────────────────
print("\n--- Summary plot ---")
data = collect(paths)
n_curves = sum(len(curves) for curves in data.values())
print(f"  Loaded {n_curves} curves across {len(data)} friction values")
plot_summary(data, tag="final")

print("\n" + "=" * 100)
print(f"{'mu':>8s}  {'peak|F| mean':>13s}  {'drift_max mean':>15s}  "
      f"{'drift_final mean':>17s}  {'n':>3s}")
print("-" * 100)
summary_rows = []
for mu in sorted(data.keys()):
    curves = data[mu]
    pk = np.array([np.max(np.abs(cc["f"])) for cc in curves])
    dmax = np.array([cc["com_drift_max"] for cc in curves]) * 1000.0
    dfin = np.array([cc["com_drift_final"] for cc in curves]) * 1000.0
    print(f"{mu:>8.3f}  {np.mean(pk):>10.3f} N  "
          f"{np.mean(dmax):>11.2f} mm  {np.mean(dfin):>13.2f} mm  {len(curves):>3d}")
    summary_rows.append({
        "mu": float(mu),
        "n": int(len(curves)),
        "peak_F_mean": float(np.mean(pk)),
        "peak_F_std": float(np.std(pk)),
        "drift_max_mm_mean": float(np.mean(dmax)),
        "drift_max_mm_std": float(np.std(dmax)),
        "drift_final_mm_mean": float(np.mean(dfin)),
        "drift_final_mm_std": float(np.std(dfin)),
    })
print("=" * 100)

summary = {
    "asset": asset_name,
    "config": str(args.config),
    "nominal_E_Pa": NOM_E,
    "nominal_v_mps": NOM_V,
    "default_board_mu": DEFAULT_BOARD_MU,
    "trials_per_friction": int(args.trials_per_friction),
    "rows": summary_rows,
}
with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"  >> Summary saved: {OUT_DIR / 'summary.json'}")
