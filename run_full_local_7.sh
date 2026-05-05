#!/usr/bin/env bash
# Full cutting sim on local (hyde04, single GPU) for 7 fruits.
# Each fruit: exp_force_variation (3 nominal + 30 noise) + exp_velocity_variation
# Default board friction=0.5 (mpmcore default). Generates per-trial JSON
# trajectories under logs/{force,velocity}_variation/<fruit>/.
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
mkdir -p sweep_logs

echo "Full sweep started at $(date)" > sweep_logs/main.log
T0=$(date +%s)

# Order: small/easy first so stiff/long ones don't block early data.
for FRUIT in cherry golden_strawberry tomato kiwi plum lemon pear; do
  CFG="configs/fruits/${FRUIT}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "[$(date +%H:%M:%S)] [SKIP] $FRUIT — config missing" | tee -a sweep_logs/main.log
    continue
  fi

  # ── Force variation (material noise) ─────────────────────────────────────
  TF=$(date +%s)
  FLOG="sweep_logs/${FRUIT}_force.log"
  echo "[$(date +%H:%M:%S)] [START] force_variation $FRUIT" | tee -a sweep_logs/main.log
  $PY -u exp_force_variation.py \
      --config "$CFG" \
      --out-dir "logs/force_variation/${FRUIT}" \
      --n-trials 30 \
      --n-nominal 3 \
      --plot-every 10 \
    > "$FLOG" 2>&1
  RC=$?
  TF1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [DONE]  force_variation $FRUIT  rc=$RC  $((TF1 - TF))s" | tee -a sweep_logs/main.log

  # ── Velocity variation (default 7 v_cut x 1 trial) ───────────────────────
  TV=$(date +%s)
  VLOG="sweep_logs/${FRUIT}_velocity.log"
  echo "[$(date +%H:%M:%S)] [START] velocity_variation $FRUIT" | tee -a sweep_logs/main.log
  $PY -u exp_velocity_variation.py \
      --config "$CFG" \
      --out-root "logs/velocity_variation" \
    > "$VLOG" 2>&1
  RC=$?
  TV1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [DONE]  velocity_variation $FRUIT  rc=$RC  $((TV1 - TV))s" | tee -a sweep_logs/main.log
done

T1=$(date +%s)
echo "Full sweep finished at $(date) — total $(((T1 - T0) / 60))min" | tee -a sweep_logs/main.log
