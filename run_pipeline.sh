#!/bin/bash
# Full pipeline: cache activations → CE projection → L31 patch →
# perplexity eval → cross-model probe → controls
# Seeds 42-46, ops add + mult
# Requires grokked model weights in artifacts/small{,_mult}/best_model.pth
# (committed via LFS; run `git lfs pull` first if missing).
# Run with: nohup bash run_pipeline.sh > pipeline.log 2>&1 &

set -e
source .venv/bin/activate

BASE_DIR="/home/saparch/playground/grokking"
cd "$BASE_DIR"

echo "=== Pipeline started at $(date) ==="

# ---- Step 1: Cache activations ----
for op in add mult; do
  suf=""; [ "$op" = "mult" ] && suf="_mult"
  act_path="artifacts/small_model_activations${suf}.npy"
  lbl_path="artifacts/mod_arithmetic_labels${suf}.npy"
  if [ -f "$act_path" ] && [ -f "$lbl_path" ]; then
    echo "[skip] cache_activations op=$op (cached)"
  else
    echo "[run]  cache_activations op=$op"
    python src/pipeline/cache_activations.py "$op"
    echo "[done] cache_activations op=$op"
  fi
done

# ---- Step 2: CE Projection ----
for op in add mult; do
  for seed in 42 43 44 45 46; do
    if [ "$op" = "mult" ]; then out_dir="artifacts/ce_projection/seeds_mult"; else out_dir="artifacts/ce_projection/seeds"; fi
    w_path="${out_dir}/W_ce_seed${seed}.pth"
    if [ -f "$w_path" ] && [ "$w_path" -nt "src/pipeline/ce_projection.py" ]; then
      echo "[skip] ce_projection seed=$seed op=$op (W exists and up to date)"
    else
      echo "[run]  ce_projection seed=$seed op=$op"
      python src/pipeline/ce_projection.py "$seed" "$op"
      echo "[done] ce_projection seed=$seed op=$op"
    fi
  done
done

# ---- Step 3: L31 Patch ----
for op in add mult; do
  for seed in 42 43 44 45 46; do
    if [ "$op" = "mult" ]; then out_dir="artifacts/l31_patch_mult"; else out_dir="artifacts/l31_patch"; fi
    csv_path="${out_dir}/alpha_sweep_l31_seed${seed}.csv"
    if [ -f "$csv_path" ] && [ "$csv_path" -nt "src/pipeline/l31_patch.py" ]; then
      echo "[skip] l31_patch seed=$seed op=$op (CSV exists and up to date)"
    else
      echo "[run]  l31_patch seed=$seed op=$op"
      python src/pipeline/l31_patch.py "$seed" "$op"
      echo "[done] l31_patch seed=$seed op=$op"
    fi
  done
done

# ---- Step 4: Perplexity eval ----
for op in add mult; do
  suf=""; [ "$op" = "mult" ] && suf="_mult"
  csv_path="artifacts/l31_patch${suf}/perplexity_sweep_seed42.csv"
  if [ -f "$csv_path" ]; then
    echo "[skip] eval_l31_perplexity op=$op (CSV exists)"
  else
    echo "[run]  eval_l31_perplexity op=$op (seeds 42-46)"
    python src/pipeline/eval_l31_perplexity.py "42,43,44,45,46" "$op"
    echo "[done] eval_l31_perplexity op=$op"
  fi
done

# ---- Step 5: Cross-model probe ----
cmp_path="artifacts/cross_model/probe_comparison.md"
if [ -f "$cmp_path" ]; then
  echo "[skip] cross_model_probe (cached)"
else
  echo "[run]  cross_model_probe"
  python src/pipeline/cross_model_probe.py
  echo "[done] cross_model_probe"
fi

# ---- Step 6: Controls (logit-lens only, skips full Phi-2 load) ----
for op in add mult; do
  suf=""; [ "$op" = "mult" ] && suf="_mult"
  csv_path="artifacts/random_baseline/results_seed42${suf}.csv"
  if [ -f "$csv_path" ] && [ "$csv_path" -nt "src/pipeline/random_baseline.py" ]; then
    echo "[skip] random_baseline op=$op (CSV exists)"
  else
    echo "[run]  random_baseline op=$op (seeds 42-46, --partial)"
    for seed in 42 43 44 45 46; do
      python src/pipeline/random_baseline.py "$seed" "$op" --partial
    done
    echo "[done] random_baseline op=$op"
  fi
done

echo "=== Pipeline finished at $(date) ==="

# Print summary
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
