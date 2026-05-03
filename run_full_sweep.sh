#!/usr/bin/env bash
# Full rebuttal sweep on GPU 3 — 15 fruits × {material variation, velocity variation}.
# Soft → stiff order so early outputs are usable while stiff ones run.
set -u
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=3
export PYTHONIOENCODING=utf-8
PY=.venv/bin/python

mkdir -p sweep_logs

FRUITS=(
  banana
  strawberry
  grape
  kiwi
  tomato
  persimmon
  watermelon_slice
  mango
  avocado
  peach
  orange
  lemon
  pineapple_slice
  pear
  apple
)

MAIN=sweep_logs/main.log
echo "Sweep started at $(date)" | tee -a "$MAIN"
echo "Fruits: ${FRUITS[*]}" | tee -a "$MAIN"
T0=$(date +%s)

for FRUIT in "${FRUITS[@]}"; do
  CFG="configs/fruits/${FRUIT}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "[SKIP] $FRUIT — config missing" | tee -a "$MAIN"
    continue
  fi

  # ── Material variation ────────────────────────────────────────────────
  TF0=$(date +%s)
  FLOG="sweep_logs/${FRUIT}_force.log"
  echo "[$(date +%H:%M:%S)] [START] force_variation $FRUIT" | tee -a "$MAIN"
  $PY -u exp_force_variation.py \
      --config "$CFG" \
      --out-dir "logs/force_variation/${FRUIT}" \
      --n-trials 30 \
      --plot-every 5 > "$FLOG" 2>&1
  RC=$?
  TF1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [DONE]  force_variation $FRUIT  rc=$RC  $((TF1 - TF0))s" | tee -a "$MAIN"

  # ── Velocity variation ────────────────────────────────────────────────
  TV0=$(date +%s)
  VLOG="sweep_logs/${FRUIT}_velocity.log"
  echo "[$(date +%H:%M:%S)] [START] velocity_variation $FRUIT" | tee -a "$MAIN"
  $PY -u exp_velocity_variation.py \
      --config "$CFG" \
      --out-root "logs/velocity_variation" > "$VLOG" 2>&1
  RC=$?
  TV1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [DONE]  velocity_variation $FRUIT  rc=$RC  $((TV1 - TV0))s" | tee -a "$MAIN"

  TNOW=$(date +%s)
  ELAPSED=$((TNOW - T0))
  echo "    cumulative $((ELAPSED / 60))min" | tee -a "$MAIN"
done

T1=$(date +%s)
echo "Sweep finished at $(date) — total $(((T1 - T0) / 60))min" | tee -a "$MAIN"
