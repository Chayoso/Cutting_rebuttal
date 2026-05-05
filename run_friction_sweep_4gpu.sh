#!/usr/bin/env bash
# 4-GPU parallel friction sweep on a small set of "critical" fruits.
# 4 fruits x 7 friction values = 28 cuts, distributed one fruit per GPU.
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
PY=.venv/bin/python
mkdir -p sweep_logs

run_friction() {
  # $1 = gpu_id, $2 = fruit
  local GPU=$1 FRUIT=$2
  local CFG="configs/fruits/${FRUIT}.yaml"
  local LOG="sweep_logs/${FRUIT}_friction.log"
  local T0=$(date +%s)
  echo "[$(date +%H:%M:%S)] [GPU${GPU}] [START] friction $FRUIT" >> sweep_logs/main.log
  CUDA_VISIBLE_DEVICES=$GPU $PY -u exp_friction_variation.py \
      --config "$CFG" \
      --frictions 0.0 0.05 0.1 0.2 0.3 0.5 0.8 \
      --trials-per-friction 1 \
      --out-root logs/friction_variation \
    > "$LOG" 2>&1
  local RC=$?
  local T1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [GPU${GPU}] [DONE]  friction $FRUIT  rc=$RC  $((T1 - T0))s" >> sweep_logs/main.log
}

worker_gpu0() { run_friction 0 watermelon_slice; }   # large flat (most slip-prone)
worker_gpu1() { run_friction 1 apple; }              # round, typical
worker_gpu2() { run_friction 2 banana; }             # asymmetric, elongated
worker_gpu3() { run_friction 3 grape; }              # small, light

echo "Friction sweep started at $(date)" > sweep_logs/main.log
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
echo "Friction sweep finished at $(date) — total $(((T1 - T0) / 60))min" >> sweep_logs/main.log
