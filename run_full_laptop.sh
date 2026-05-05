#!/usr/bin/env bash
# Single-GPU full cutting sim for 5 fruits (lemon, kiwi, plum, grape, tomato).
# Designed for an idle local GPU (e.g. RTX 4090 on the user's laptop).
# Each fruit runs both force_variation (3 nominal + 30 noise) and
# velocity_variation (7 v_cut x 1 trial) sequentially.
#
# Setup once on the laptop:
#   git clone git@github.com:Chayoso/Cutting_rebuttal.git
#   cd Cutting_rebuttal
#   python -m venv .venv && source .venv/bin/activate
#   pip install --upgrade pip
#   pip install trimesh numpy scipy taichi pyyaml matplotlib rtree
#   ./run_full_laptop.sh
#
# Order: small fruits first so early data is available even if interrupted.
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=${PY:-.venv/bin/python}
mkdir -p sweep_logs

echo "Laptop sweep started at $(date)" > sweep_logs/main.log
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> sweep_logs/main.log
T0=$(date +%s)

for FRUIT in grape cherry kiwi tomato plum lemon; do
  CFG="configs/fruits/${FRUIT}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "[$(date +%H:%M:%S)] [SKIP] $FRUIT — config missing" | tee -a sweep_logs/main.log
    continue
  fi
  TF=$(date +%s)
  echo "[$(date +%H:%M:%S)] [START] force_variation $FRUIT" | tee -a sweep_logs/main.log
  $PY -u exp_force_variation.py \
      --config "$CFG" \
      --out-dir "logs/force_variation/${FRUIT}" \
      --n-trials 30 --n-nominal 3 --plot-every 10 \
    > "sweep_logs/${FRUIT}_force.log" 2>&1
  RC=$?
  TF1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [DONE]  force_variation $FRUIT rc=$RC $((TF1 - TF))s" | tee -a sweep_logs/main.log

  TV=$(date +%s)
  echo "[$(date +%H:%M:%S)] [START] velocity_variation $FRUIT" | tee -a sweep_logs/main.log
  $PY -u exp_velocity_variation.py \
      --config "$CFG" \
      --out-root "logs/velocity_variation" \
    > "sweep_logs/${FRUIT}_velocity.log" 2>&1
  RC=$?
  TV1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [DONE]  velocity_variation $FRUIT rc=$RC $((TV1 - TV))s" | tee -a sweep_logs/main.log
done

T1=$(date +%s)
echo "Laptop sweep finished at $(date) — total $(((T1 - T0) / 60))min" | tee -a sweep_logs/main.log
