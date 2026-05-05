#!/usr/bin/env bash
# 5-fruit full cutting sim on hyde06 across 4 GPUs.
# OOM-safe: at most 1 MPM process per GPU at any time (~600 MB each).
# Distribution (5 fruits / 4 GPUs):
#   GPU 0: lemon            (~80 min,  E=1.5e6, stiffest of the five)
#   GPU 1: plum             (~70 min,  E=1.0e6)
#   GPU 2: tomato           (~60 min,  E=6.5e5)
#   GPU 3: kiwi -> grape    (~60+40 = 100 min,  E=3.5e5 each)
# Bottleneck: GPU 3 ≈ 1.7 hours.
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
PY=.venv/bin/python
mkdir -p sweep_logs

run_fruit() {
  local GPU=$1 FRUIT=$2
  local CFG="configs/fruits/${FRUIT}.yaml"
  if [ ! -f "$CFG" ]; then
    echo "[$(date +%H:%M:%S)] [GPU${GPU}] [SKIP] $FRUIT — config missing" >> sweep_logs/main.log
    return
  fi
  local TF=$(date +%s)
  echo "[$(date +%H:%M:%S)] [GPU${GPU}] [START] force_variation $FRUIT" >> sweep_logs/main.log
  CUDA_VISIBLE_DEVICES=$GPU $PY -u exp_force_variation.py \
      --config "$CFG" \
      --out-dir "logs/force_variation/${FRUIT}" \
      --n-trials 30 --n-nominal 3 --plot-every 10 \
    > "sweep_logs/${FRUIT}_force.log" 2>&1
  local RC=$?
  local TF1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [GPU${GPU}] [DONE]  force_variation $FRUIT rc=$RC $((TF1-TF))s" >> sweep_logs/main.log

  local TV=$(date +%s)
  echo "[$(date +%H:%M:%S)] [GPU${GPU}] [START] velocity_variation $FRUIT" >> sweep_logs/main.log
  CUDA_VISIBLE_DEVICES=$GPU $PY -u exp_velocity_variation.py \
      --config "$CFG" \
      --out-root "logs/velocity_variation" \
    > "sweep_logs/${FRUIT}_velocity.log" 2>&1
  RC=$?
  local TV1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [GPU${GPU}] [DONE]  velocity_variation $FRUIT rc=$RC $((TV1-TV))s" >> sweep_logs/main.log
}

worker_gpu0() { run_fruit 0 lemon; }
worker_gpu1() { run_fruit 1 plum; }
worker_gpu2() { run_fruit 2 tomato; }
worker_gpu3() { run_fruit 3 kiwi; run_fruit 3 grape; }

echo "5-fruit sweep started at $(date)" > sweep_logs/main.log
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
echo "5-fruit sweep finished at $(date) — total $(((T1 - T0) / 60))min" >> sweep_logs/main.log
