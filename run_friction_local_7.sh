#!/usr/bin/env bash
# Sequential single-GPU friction sweep on local (hyde04, RTX A4000)
# 7 fruits x 7 friction values = 49 cuts.
# Order: small/light first (likely biggest drift), then heavier.
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
mkdir -p sweep_logs

echo "Friction sweep started at $(date)" > sweep_logs/main.log
T0=$(date +%s)

for FRUIT in cherry golden_strawberry tomato kiwi plum lemon pear; do
  CFG="configs/fruits/${FRUIT}.yaml"
  LOG="sweep_logs/${FRUIT}_friction.log"
  if [ ! -f "$CFG" ]; then
    echo "[$(date +%H:%M:%S)] [SKIP] $FRUIT — config missing" | tee -a sweep_logs/main.log
    continue
  fi
  TF=$(date +%s)
  echo "[$(date +%H:%M:%S)] [START] friction $FRUIT" | tee -a sweep_logs/main.log
  $PY -u exp_friction_variation.py \
      --config "$CFG" \
      --frictions 0.0 0.05 0.1 0.2 0.3 0.5 0.8 \
      --trials-per-friction 1 \
      --out-root logs/friction_variation \
    > "$LOG" 2>&1
  RC=$?
  TF1=$(date +%s)
  echo "[$(date +%H:%M:%S)] [DONE]  friction $FRUIT  rc=$RC  $((TF1 - TF))s" | tee -a sweep_logs/main.log
done

T1=$(date +%s)
echo "Friction sweep finished at $(date) — total $(((T1 - T0) / 60))min" | tee -a sweep_logs/main.log
