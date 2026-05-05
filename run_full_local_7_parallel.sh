#!/usr/bin/env bash
# 2-way parallel full cutting sim on local (hyde04, single RTX A4000).
# Both workers share GPU 0; memory usage: ~600 MB per MPM instance.
# Worker A (3 fruits): cherry, tomato, plum
# Worker B (4 fruits): golden_strawberry, kiwi, lemon, pear
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
mkdir -p sweep_logs

run_fruit_pair() {
  # $1 = worker label, $2... = fruit list
  local LABEL=$1; shift
  for FRUIT in "$@"; do
    local CFG="configs/fruits/${FRUIT}.yaml"
    if [ ! -f "$CFG" ]; then
      echo "[$(date +%H:%M:%S)] [$LABEL] [SKIP] $FRUIT — config missing" >> sweep_logs/main.log
      continue
    fi

    # Force variation
    local TF=$(date +%s)
    echo "[$(date +%H:%M:%S)] [$LABEL] [START] force_variation $FRUIT" >> sweep_logs/main.log
    $PY -u exp_force_variation.py \
        --config "$CFG" \
        --out-dir "logs/force_variation/${FRUIT}" \
        --n-trials 30 --n-nominal 3 --plot-every 10 \
      > "sweep_logs/${FRUIT}_force.log" 2>&1
    local RC=$?
    local TF1=$(date +%s)
    echo "[$(date +%H:%M:%S)] [$LABEL] [DONE]  force_variation $FRUIT rc=$RC $((TF1-TF))s" >> sweep_logs/main.log

    # Velocity variation
    local TV=$(date +%s)
    echo "[$(date +%H:%M:%S)] [$LABEL] [START] velocity_variation $FRUIT" >> sweep_logs/main.log
    $PY -u exp_velocity_variation.py \
        --config "$CFG" \
        --out-root "logs/velocity_variation" \
      > "sweep_logs/${FRUIT}_velocity.log" 2>&1
    RC=$?
    local TV1=$(date +%s)
    echo "[$(date +%H:%M:%S)] [$LABEL] [DONE]  velocity_variation $FRUIT rc=$RC $((TV1-TV))s" >> sweep_logs/main.log
  done
}

echo "Parallel sweep started at $(date)" > sweep_logs/main.log
T0=$(date +%s)

# Worker A: 3 fruits (lighter computational total, since tomato/plum medium-stiff)
run_fruit_pair "A" cherry tomato plum &
PID_A=$!
echo "worker A pid=$PID_A" >> sweep_logs/main.log

# Stagger start by 10s so the two SDF generation phases don't fight over CPU/disk
sleep 10

# Worker B: 4 fruits
run_fruit_pair "B" golden_strawberry kiwi lemon pear &
PID_B=$!
echo "worker B pid=$PID_B" >> sweep_logs/main.log

wait $PID_A $PID_B

T1=$(date +%s)
echo "Parallel sweep finished at $(date) — total $(((T1 - T0) / 60))min" >> sweep_logs/main.log
