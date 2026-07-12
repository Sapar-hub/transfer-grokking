#!/bin/bash
# Wrapper that captures all errors
SEED=$1
OP=$2
LOG="/home/saparch/playground/grokking/baseline_crash_${SEED}_${OP}.log"
cd /home/saparch/playground/grokking
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
python random_baseline.py "$SEED" "$OP" --logit-lens-only > "$LOG" 2>&1
EXIT=$?
echo "[exit=$EXIT] seed=$SEED op=$OP" >> /home/saparch/playground/grokking/baseline_progress.log
exit $EXIT
