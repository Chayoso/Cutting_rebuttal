#!/usr/bin/env bash
# Parallel rebuttal sweep across 4 GPUs.
# Plan A: 15 fruits × {force(n-nominal=3, n-trials=30), velocity(default 7v × 1trial)}
# Plan B: 4 soft fruits × velocity(trials-per-velocity=3) extras
# Per-fruit failures don't abort the loop.
#
# Load-balanced assignment (descending stiffness, snake distribution):
#   GPU 0: apple, mango, watermelon_slice            (3 fruits) + grape extras
#   GPU 1: pear, banana, avocado, strawberry         (4 fruits) + kiwi extras
#   GPU 2: pineapple_slice, peach, persimmon, grape  (4 fruits) + tomato extras
#   GPU 3: lemon, orange, tomato, kiwi               (4 fruits) + lemon extras

set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
PY=.venv/bin/python
mkdir -p sweep_logs

run_fruit() {
  # $1 = gpu_id, $2 = fruit
  local GPU=$1 FRUIT=$2
  local CFG="configs/fruits/${FRUIT}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "[GPU${GPU}] [SKIP] $FRUIT — config missing" | tee -a sweep_logs/main.log
    return
  fi

  # Material variation (force)
  local TF0=$(date +%s)
  echo "[$(date +%H:%M:%S)] [GPU${GPU}] [START] force_variation $FRUIT" >> sweep_logs/main.log
  CUDA_VISIBLE_DEVICES=$GPU $PY -u exp_force_variation.py \
      --config "$CFG" \
      --out-dir "logs/force_variation/${FRUIT}" \
      --n-trials 30 \
      --n-nominal 3 \
      --plot-every 10 \
    > "sweep_logs/${FRUIT}_force.log" 2>&1
  local RC=$?
  local TF1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [GPU${GPU}] [DONE]  force_variation $FRUIT  rc=$RC  $((TF1 - TF0))s" >> sweep_logs/main.log

  # Velocity variation (default 7 vels × 1 trial)
  local TV0=$(date +%s)
  echo "[$(date +%H:%M:%S)] [GPU${GPU}] [START] velocity_variation $FRUIT" >> sweep_logs/main.log
  CUDA_VISIBLE_DEVICES=$GPU $PY -u exp_velocity_variation.py \
      --config "$CFG" \
      --out-root "logs/velocity_variation" \
    > "sweep_logs/${FRUIT}_velocity.log" 2>&1
  RC=$?
  local TV1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [GPU${GPU}] [DONE]  velocity_variation $FRUIT  rc=$RC  $((TV1 - TV0))s" >> sweep_logs/main.log
}

run_softfruit_extras() {
  # $1 = gpu_id, $2 = fruit (one of soft fruits to re-run velocity with 3 trials/v)
  local GPU=$1 FRUIT=$2
  local CFG="configs/fruits/${FRUIT}.yaml"
  local OUT="logs/velocity_variation_x3/${FRUIT}"
  local TX0=$(date +%s)
  echo "[$(date +%H:%M:%S)] [GPU${GPU}] [START] velocity_x3 $FRUIT" >> sweep_logs/main.log
  CUDA_VISIBLE_DEVICES=$GPU $PY -u exp_velocity_variation.py \
      --config "$CFG" \
      --trials-per-velocity 3 \
      --out-root "logs/velocity_variation_x3" \
    > "sweep_logs/${FRUIT}_velocity_x3.log" 2>&1
  local RC=$?
  local TX1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [GPU${GPU}] [DONE]  velocity_x3 $FRUIT  rc=$RC  $((TX1 - TX0))s" >> sweep_logs/main.log
}

# ── Worker functions (one per GPU) ──────────────────────────────────────────
worker_gpu0() {
  for f in apple mango watermelon_slice; do run_fruit 0 "$f"; done
  run_softfruit_extras 0 grape
}
worker_gpu1() {
  for f in pear banana avocado strawberry; do run_fruit 1 "$f"; done
  run_softfruit_extras 1 kiwi
}
worker_gpu2() {
  for f in pineapple_slice peach persimmon grape; do run_fruit 2 "$f"; done
  run_softfruit_extras 2 tomato
}
worker_gpu3() {
  for f in lemon orange tomato kiwi; do run_fruit 3 "$f"; done
  run_softfruit_extras 3 lemon
}

# ── Main ────────────────────────────────────────────────────────────────────
echo "Sweep started at $(date)" > sweep_logs/main.log
T0=$(date +%s)

worker_gpu0 &
PID0=$!
worker_gpu1 &
PID1=$!
worker_gpu2 &
PID2=$!
worker_gpu3 &
PID3=$!
echo "workers: gpu0=$PID0 gpu1=$PID1 gpu2=$PID2 gpu3=$PID3" >> sweep_logs/main.log

wait $PID0 $PID1 $PID2 $PID3

T1=$(date +%s)
echo "Sweep finished at $(date) — total $(((T1 - T0) / 60))min" >> sweep_logs/main.log
