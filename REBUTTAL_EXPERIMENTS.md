# Rebuttal Experiments — Multi-Fruit Force/Velocity Variation

This document describes the rebuttal-phase experiments for evaluating CulinaryCut
across 15 fruits spanning 3 orders of magnitude in Young's modulus.

---

## 1. Experiment Goals

Three experiments were requested by reviewers (or in anticipation of likely reviewer
questions):

1. **Multi-asset support** — Demonstrate that the framework generalizes beyond
   banana to a diverse set of fruits cuttable by vertical knife motion.
2. **Velocity variation per asset** — For each fruit, sweep cutting speed
   (`v_cut`) and report the resulting peak EEF force. Establishes that the
   force model responds appropriately to kinematic changes.
3. **Material variation per asset** — For each fruit, perturb material
   parameters (E, ν, ρ, σ_y, v_cut) by ±10% Gaussian noise and report
   force-curve mean ± std. Establishes robustness to parameter uncertainty.

---

## 2. Files Added

```
assets/
  fruits/                        # 14 procedural fruit meshes (banana stays in assets/)
    apple.obj, avocado.obj, grape.obj, kiwi.obj, lemon.obj,
    mango.obj, orange.obj, peach.obj, pear.obj, persimmon.obj,
    pineapple_slice.obj, strawberry.obj, tomato.obj, watermelon_slice.obj

configs/
  fruits/                        # 15 per-fruit YAML configs (incl. banana)
    apple.yaml, avocado.yaml, banana.yaml, ...

scripts/
  generate_fruit_meshes.py       # Builds the .obj files in assets/fruits/
  generate_fruit_configs.py      # Builds the YAML configs from base + literature props

exp_force_variation.py           # (existing) ±10% material noise, 30 trials
exp_velocity_variation.py        # (new) v_cut grid scan, 7 velocities × N trials

REBUTTAL_EXPERIMENTS.md          # This file
```

---

## 3. Fruit Selection & Material Properties

| Tier   | Fruit            | E (Pa)    | ν     | ρ (kg/m³) | σ_y (kPa) | Reference                    |
|--------|------------------|-----------|-------|-----------|-----------|------------------------------|
| Easy   | banana           | 7.0e3     | 0.35  | 960       | 5         | Jahanbakhshi et al.          |
| Easy   | strawberry       | 2.0e5     | 0.35  | 900       | 10        | Shamaila et al.              |
| Easy   | peach            | 1.2e6     | 0.33  | 980       | 50        | Bourne, food texture refs    |
| Easy   | pear             | 5.5e6     | 0.30  | 1000      | 200       | Bourne, Abbott et al.        |
| Easy   | apple            | 6.5e6     | 0.30  | 850       | 300       | Bourne, classic              |
| Med    | kiwi             | 3.5e5     | 0.33  | 1000      | 20        | Harker et al.                |
| Med    | tomato           | 6.5e5     | 0.33  | 950       | 30        | Li et al.                    |
| Med    | mango            | 1.8e6     | 0.33  | 1000      | 80        | food science literature      |
| Med    | persimmon        | 1.0e6     | 0.33  | 970       | 40        | food science literature      |
| Med    | avocado          | 1.2e6     | 0.33  | 980       | 50        | food science literature      |
| Hard   | grape            | 3.5e5     | 0.35  | 1050      | 15        | Letaief et al.               |
| Hard   | orange           | 1.2e6     | 0.33  | 900       | 80        | citrus mechanical refs       |
| Hard   | lemon            | 1.5e6     | 0.33  | 900       | 100       | citrus mechanical refs       |
| Hard   | pineapple_slice  | 3.0e6     | 0.30  | 950       | 200       | Bourne                       |
| Hard   | watermelon_slice | 1.2e6     | 0.33  | 950       | 60        | watermelon mech refs         |

**Coverage**: E spans 7 kPa → 6.5 MPa (≈3 orders of magnitude).

> **Note**: All values are taken from the median / representative point of
> reported literature ranges. Update `scripts/generate_fruit_configs.py` to
> change them and re-run that script to regenerate YAML configs.

---

## 4. Mesh Generation

All fruit meshes are generated procedurally with `trimesh` (icosphere
subdivisions=2 → ~80 faces per primitive). This keeps `signed_distance`
queries memory-bounded on small assets and avoids the OOM we hit with
denser meshes during SDF voxelization.

```bash
python scripts/generate_fruit_meshes.py
```

Each mesh is centered in XZ and translated so that y_min = 0 (sits on the
cutting board). Sizes match real fruits (cm-level), e.g. strawberry 4 cm,
apple 7 cm, watermelon slice 12 cm.

---

## 5. Configs

```bash
python scripts/generate_fruit_configs.py
```

Per-fruit `sdf_voxel` is auto-computed so that the SDF cell size is
~1.5 mm uniformly across fruits (clamped to [30, 96]). Banana keeps its
original `sdf_voxel=160`.

All other simulation parameters (knife geometry, motion, damping,
plasticity model, board, etc.) are inherited unchanged from
`configs/example_cutting.yaml`.

---

## 6. Running the Experiments

### 6a. Per-fruit material variation (30 trials, ±10% Gaussian noise)

```bash
python exp_force_variation.py \
    --config configs/fruits/strawberry.yaml \
    --out-dir logs/force_variation/strawberry \
    --n-trials 30
```

Outputs: `logs/force_variation/<fruit>/trial_NNNN/ee_force_fps.json` plus
`force_variation_final.png` and `param_log.json`.

### 6b. Per-fruit velocity variation (7 velocities)

```bash
python exp_velocity_variation.py \
    --config configs/fruits/strawberry.yaml \
    --velocities 0.15 0.20 0.25 0.30 0.35 0.40 0.45 \
    --trials-per-velocity 1
```

Outputs: `logs/velocity_variation/<fruit>/v0p150_r00/ee_force_fps.json` per
velocity, plus `velocity_variation_final.png` and `velocity_log.json`.

### 6c. Sequential batch — all 15 fruits

A simple shell loop:

```bash
for f in configs/fruits/*.yaml; do
    name=$(basename "$f" .yaml)
    echo "=== $name (force_variation) ==="
    python exp_force_variation.py --config "$f" \
        --out-dir "logs/force_variation/$name" --n-trials 30 \
    || echo "FAILED $name (force)"

    echo "=== $name (velocity_variation) ==="
    python exp_velocity_variation.py --config "$f" \
    || echo "FAILED $name (velocity)"
done
```

### 6d. Resource estimate

| Quantity              | Value                                    |
|-----------------------|------------------------------------------|
| Trials per fruit      | 31 (force) + 7 (velocity) = 38           |
| Total trials          | 15 × 38 = 570                            |
| Time per trial (CPU)  | ~1–2 min (small fruits) … ~5 min (large) |
| Total wall time       | ~10–20 h on a single machine             |
| GPU                   | not required for MPM solver itself       |
| Disk                  | ~50–200 MB per fruit                     |

---

## 7. Server Deployment Notes

### Setup

```bash
# On the server
git clone <this repo's branch>
cd CPIC
conda create -n cpic python=3.10 -y && conda activate cpic
pip install -r requirements.txt   # if present, otherwise install: trimesh numpy scipy taichi pyyaml matplotlib
```

### Background-safe execution

Use `tmux` (or `nohup`) so the batch survives SSH disconnect:

```bash
tmux new -s cpic-rebuttal
# inside tmux:
bash run_all_fruits.sh 2>&1 | tee logs/batch_run.log
# detach with Ctrl-B then D
```

To check progress later:

```bash
tmux attach -t cpic-rebuttal
# or, without attaching:
tail -f logs/batch_run.log
```

### Pulling results back

```bash
# On the local machine
rsync -avz chayo@hyde06.dabh.io:~/CPIC/logs/force_variation/ ./logs/force_variation/
rsync -avz chayo@hyde06.dabh.io:~/CPIC/logs/velocity_variation/ ./logs/velocity_variation/
```

### Common pitfalls observed during local validation

- **OOM in `trimesh.proximity.signed_distance`** on small fruits when the
  mesh has too many faces. Mitigations already applied: subdivisions=2 for
  spheres; per-fruit `sdf_voxel` capped at 96.
- **Strawberry log lands in `logs/ee_force_fps.json` (root)** when running
  `run.py` directly without overriding `output.logging.out_dir`. The
  experiment scripts override this per-trial; only one-off runs need
  manual handling.
- **Knife motion** is identical across fruits (fixed start/stop Y). Some
  fruits (watermelon slice = 12 cm wide × 4 cm tall) may benefit from a
  fruit-specific `cutting.z_margin_ratio` if reviewers ask for full-cut
  generalization, but that's not required for force-validation purposes.

---

## 8. Status

- [x] 14 procedural fruit meshes generated (`assets/fruits/`)
- [x] 15 per-fruit configs with literature material properties (`configs/fruits/`)
- [x] `exp_velocity_variation.py` script written
- [x] Local pipeline validated end-to-end on **strawberry**
      (peak F = 3.87 N, 72 frames, exit 0, no memory errors)
- [ ] Material variation run on all 15 fruits (server)
- [ ] Velocity variation run on all 15 fruits (server)
- [ ] Aggregate plots: peak-force-vs-E across fruits, peak-force-vs-v_cut per fruit
- [ ] Rebuttal figure(s) prepared

---

## 9. Validation Result (Strawberry, Local)

```
Frames in ee_force_fps.json: 72
Time range: 0.000 ~ 1.183 s
Knife Y range: -0.0360 ~ 0.0986 m  (full down-and-up motion)
Force: peak = 3.87 N at frame 20
Exit code: 0, no OOM
```

Saved to `logs/validation_strawberry/ee_force_fps.json`.
