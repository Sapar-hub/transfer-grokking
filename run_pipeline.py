#!/usr/bin/env python3
"""6-step pipeline: cache → CE projection → L31 patch → perplexity → cross-model → controls.
Seeds 42–46, ops add + mult.
Requires grokked model weights committed via LFS (`git lfs pull` first).
"""

import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
SCRIPTS = BASE / "src" / "pipeline"
ARTIFACTS = BASE / "artifacts"
SEEDS = [42, 43, 44, 45, 46]
OPS = ["add", "mult"]
PY = sys.executable


def check_models():
    for op_dir in ("small", "small_mult"):
        path = ARTIFACTS / op_dir / "best_model.pth"
        if not path.exists():
            sys.exit(
                f"Missing {op_dir}/best_model.pth — "
                f"run `git lfs pull` or `python src/pipeline/train_small.py "
                f"{'add' if op_dir == 'small' else 'mult'}`"
            )
        if path.stat().st_size < 1024:
            sys.exit(
                f"Unfetched LFS pointer in {op_dir}/best_model.pth "
                f"({path.stat().st_size} bytes) — run `git lfs pull`"
            )


def run_step(script, *args):
    subprocess.run([PY, str(SCRIPTS / script), *args], check=True)


def main():
    check_models()
    print(f"=== Pipeline started at {datetime.now()} ===")

    # ---- Step 1: Cache activations ----
    for op in OPS:
        suf = "_mult" if op == "mult" else ""
        act = ARTIFACTS / f"small_model_activations{suf}.npy"
        lbl = ARTIFACTS / f"mod_arithmetic_labels{suf}.npy"
        if act.exists() and lbl.exists():
            print(f"[skip] cache_activations op={op}")
        else:
            print(f"[run]  cache_activations op={op}")
            run_step("cache_activations.py", op)
            print(f"[done] cache_activations op={op}")

    # ---- Step 2: CE Projection ----
    for op in OPS:
        out_dir = "seeds_mult" if op == "mult" else "seeds"
        for seed in SEEDS:
            w_path = ARTIFACTS / "ce_projection" / out_dir / f"W_ce_seed{seed}.pth"
            if w_path.exists():
                print(f"[skip] ce_projection seed={seed} op={op}")
            else:
                print(f"[run]  ce_projection seed={seed} op={op}")
                run_step("ce_projection.py", str(seed), op)
                print(f"[done] ce_projection seed={seed} op={op}")

    # ---- Step 3: L31 Patch ----
    for op in OPS:
        out_dir = "l31_patch_mult" if op == "mult" else "l31_patch"
        for seed in SEEDS:
            csv_path = ARTIFACTS / out_dir / f"alpha_sweep_l31_seed{seed}.csv"
            if csv_path.exists():
                print(f"[skip] l31_patch seed={seed} op={op}")
            else:
                print(f"[run]  l31_patch seed={seed} op={op}")
                run_step("l31_patch.py", str(seed), op)
                print(f"[done] l31_patch seed={seed} op={op}")

    # ---- Step 4: Perplexity eval (batches 5 seeds; check last-seed sentinel) ----
    for op in OPS:
        suf = "_mult" if op == "mult" else ""
        csv_path = ARTIFACTS / f"l31_patch{suf}" / f"perplexity_sweep_seed{SEEDS[-1]}.csv"
        if csv_path.exists():
            print(f"[skip] eval_l31_perplexity op={op}")
        else:
            print(f"[run]  eval_l31_perplexity op={op} (seeds 42-46)")
            run_step("eval_l31_perplexity.py", "42,43,44,45,46", op)
            print(f"[done] eval_l31_perplexity op={op}")

    # ---- Step 5: Cross-model probe ----
    cmp_path = ARTIFACTS / "cross_model" / "probe_comparison.md"
    if cmp_path.exists():
        print("[skip] cross_model_probe")
    else:
        print("[run]  cross_model_probe")
        run_step("cross_model_probe.py")
        print("[done] cross_model_probe")

    # ---- Step 6: Controls (check last-seed sentinel) ----
    for op in OPS:
        suf = "_mult" if op == "mult" else ""
        csv_path = ARTIFACTS / "random_baseline" / f"results_seed{SEEDS[-1]}{suf}.csv"
        if csv_path.exists():
            print(f"[skip] random_baseline op={op}")
        else:
            print(f"[run]  random_baseline op={op} (seeds 42-46, --partial)")
            for seed in SEEDS:
                run_step("random_baseline.py", str(seed), op, "--partial")
            print(f"[done] random_baseline op={op}")

    print(f"=== Pipeline finished at {datetime.now()} ===")

    # Print summary
    for op in OPS:
        suf = "_mult" if op == "mult" else ""
        print(f"\n--- {op} L31 W_CE alpha=0.5 ---")
        for seed in SEEDS:
            csv_path = ARTIFACTS / f"l31_patch{suf}" / f"alpha_sweep_l31_seed{seed}.csv"
            if csv_path.exists():
                with open(csv_path) as f:
                    rows = list(csv.reader(f))
                if len(rows) >= 3:
                    val = rows[2][3]  # 3rd row, 4th column = W_CE alpha=0.5
                    print(f"  seed{seed}: {val}")


if __name__ == "__main__":
    main()
