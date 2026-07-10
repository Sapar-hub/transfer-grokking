#!/bin/bash
# Full pipeline: CE projection (L10) + L31 patch alpha sweep
# Seeds 42-46, ops add + mult
# Run with: nohup bash run_pipeline.sh > pipeline.log 2>&1 &

set -e
source .venv/bin/activate

BASE_DIR="/home/saparch/playground/grokking"
cd "$BASE_DIR"

echo "=== Pipeline started at $(date) ==="

# ---- Phase 1: CE Projection ----
for op in add mult; do
    for seed in 42 43 44 45 46; do
        out_dir="artifacts/ce_projection/seeds"
        [ "$op" = "mult" ] && out_dir="${out_dir}_mult"
        w_path="${out_dir}/W_ce_seed${seed}.pth"
        if [ -f "$w_path" ] && [ "$w_path" -nt "ce_projection.py" ]; then
            echo "[skip] ce_projection seed=$seed op=$op (W exists and up to date)"
        else
            echo "[run]  ce_projection seed=$seed op=$op"
            python ce_projection.py "$seed" "$op"
            echo "[done] ce_projection seed=$seed op=$op"
        fi
    done
done

# ---- Phase 2: L31 Patch ----
for op in add mult; do
    for seed in 42 43 44 45 46; do
        out_dir="artifacts/l31_patch"
        [ "$op" = "mult" ] && out_dir="${out_dir}_mult"
        csv_path="${out_dir}/alpha_sweep_l31_seed${seed}.csv"
        if [ -f "$csv_path" ] && [ "$csv_path" -nt "l31_patch.py" ]; then
            echo "[skip] l31_patch seed=$seed op=$op (CSV exists and up to date)"
        else
            echo "[run]  l31_patch seed=$seed op=$op"
            python l31_patch.py "$seed" "$op"
            echo "[done] l31_patch seed=$seed op=$op"
        fi
    done
done

echo "=== Pipeline finished at $(date) ==="
echo "=== Results ==="

# Print summary table
for op in add mult; do
    suf=""; [ "$op" = "mult" ] && suf="_mult"
    echo ""
    echo "--- $op L31 W_CE alpha=0.5 ---"
    for seed in 42 43 44 45 46; do
        csv="artifacts/l31_patch${suf}/alpha_sweep_l31_seed${seed}.csv"
        if [ -f "$csv" ]; then
            val=$(awk -F',' 'NR==3 {print $4}' "$csv")
            echo "seed${seed}: ${val}"
        fi
    done
done
