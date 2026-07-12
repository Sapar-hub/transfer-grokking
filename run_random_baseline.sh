#!/bin/bash
# Multi-seed random baseline: random network + one-hot controls (P0 only)
# Seeds 42-46, ops add + mult, logit-lens only (no alpha sweep)
# Run with: nohup bash run_random_baseline.sh > random_baseline.log 2>&1 &

set -e
source .venv/bin/activate
export HF_HUB_OFFLINE=1

BASE_DIR="/home/saparch/playground/grokking"
cd "$BASE_DIR"

echo "=== Random Baseline Pipeline started at $(date) ==="

for op in add mult; do
    for seed in 42 43 44 45 46; do
        suffix=""; [ "$op" = "mult" ] && suffix="_mult"
        csv_path="artifacts/random_baseline/results_seed${seed}${suffix}.csv"
        if [ -f "$csv_path" ] && [ "$csv_path" -nt "random_baseline.py" ]; then
            echo "[skip] random_baseline seed=$seed op=$op (CSV exists and up to date)"
        else
            echo "[run]  random_baseline seed=$seed op=$op"
            python random_baseline.py "$seed" "$op" --logit-lens-only
            echo "[done] random_baseline seed=$seed op=$op"
        fi
    done
done

# Print summary table with named-column awk lookup
echo ""
echo "=== Results ==="
for op in add mult; do
    suf=""; [ "$op" = "mult" ] && suf="_mult"
    echo ""
    echo "--- $op ---"
    echo "seed  onehot_ll   onehot_probe  random-0_ll  random-0_probe  grokked_ll  grokked_probe"
    for seed in 42 43 44 45 46; do
        csv="artifacts/random_baseline/results_seed${seed}${suf}.csv"
        if [ -f "$csv" ]; then
            onehot_ll=$(awk -F',' -v col="onehot" '$1==col {print $2}' "$csv")
            onehot_probe=$(awk -F',' -v col="onehot" '$1==col {print $3}' "$csv")
            random_ll=$(awk -F',' -v col="random-0" '$1==col {print $2}' "$csv")
            random_probe=$(awk -F',' -v col="random-0" '$1==col {print $3}' "$csv")
            grokked_ll=$(awk -F',' -v col="grokked" '$1==col {print $2}' "$csv")
            grokked_probe=$(awk -F',' -v col="grokked" '$1==col {print $3}' "$csv")
            echo "seed${seed}  ${onehot_ll}          ${onehot_probe}            ${random_ll}            ${random_probe}             ${grokked_ll}          ${grokked_probe}"
        else
            echo "seed${seed}  MISSING"
        fi
    done
done

echo ""
echo "=== Pipeline finished at $(date) ==="
