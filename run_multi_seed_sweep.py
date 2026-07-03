"""
Multi-seed sweep orchestrator.

Runs the full pipeline for both addition and multiplication mod 97
across seeds {42, 43, 44, 45, 46}.

Usage:
    python run_multi_seed_sweep.py           # run everything
    python run_multi_seed_sweep.py --mult-only  # multiplication only
    python run_multi_seed_sweep.py --add-only   # addition only
    python run_multi_seed_sweep.py --check      # check what needs to run
"""
import os
import subprocess
import sys
import time

SEEDS = [42, 43, 44, 45, 46]
OPS = ["add", "multiply"]
ARTIFACTS = "artifacts"


def check_flag(path):
    return os.path.exists(path)


def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def run(cmd, timeout=None):
    print(f"  $ {cmd}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  ERROR (returned {result.returncode}) after {elapsed:.0f}s")
        print(f"  stderr: {result.stderr[:500]}")
        sys.exit(1)
    for line in result.stdout.split("\n"):
        if line.strip():
            print(f"  {line}")
    print(f"  Done in {elapsed:.0f}s")
    return result


def main():
    args = set(sys.argv[1:])
    run_add = "add" in OPS if "--mult-only" in args else (OPS[0] in OPS if "--add-only" in args else True)
    run_mult = "multiply" in OPS if "--add-only" in args else (OPS[1] in OPS if "--mult-only" in args else True)

    if "--check" in args:
        run_add = True
        run_mult = True

    ops_to_run = []
    if run_add:
        ops_to_run.append(("add", ""))
    if run_mult:
        ops_to_run.append(("multiply", "_mult"))

    # Phase 1: Train small models
    for op_name, suffix in ops_to_run:
        model_path = f"{ARTIFACTS}/small{suffix}/best_model.pth"
        if check_flag(model_path):
            print(f"[check] {op_name} small model exists: {model_path}")
        else:
            step(f"Training small model for {op_name} mod 97")
            run(f"python train_small.py {op_name}")

    # Phase 2: Cache activations
    for op_name, suffix in ops_to_run:
        acts_path = f"{ARTIFACTS}/small_model_activations{suffix}.npy"
        lbls_path = f"{ARTIFACTS}/mod_arithmetic_labels{suffix}.npy"
        if check_flag(acts_path) and check_flag(lbls_path):
            print(f"[check] {op_name} activations cached")
        else:
            step(f"Caching activations for {op_name}")
            run(f"python cache_activations.py {op_name}")

    # Phase 3: Train W_CE × 5 seeds for each op
    for op_name, suffix in ops_to_run:
        for seed in SEEDS:
            w_path = f"{ARTIFACTS}/ce_projection/seeds{suffix}/W_ce_seed{seed}.pth"
            if check_flag(w_path):
                print(f"[check] {op_name} W_CE seed={seed} exists")
                continue
            step(f"Training W_CE for {op_name}, seed={seed}")
            run(f"python ce_projection.py {seed} {op_name}")

    step("All W_CE training complete")

    # Phase 4: L31 alpha sweep × 5 seeds for each op
    for op_name, suffix in ops_to_run:
        for seed in SEEDS:
            sweep_path = f"{ARTIFACTS}/l31_patch{suffix}/alpha_sweep_l31_seed{seed}.csv"
            if check_flag(sweep_path):
                print(f"[check] {op_name} L31 sweep seed={seed} exists")
                continue
            step(f"L31 alpha sweep for {op_name}, seed={seed}")
            run(f"python l31_patch.py {seed} {op_name}")

    step("All L31 alpha sweeps complete")

    # Phase 5: Perplexity evals (batched per op)
    for op_name, suffix in ops_to_run:
        ppl_path = f"{ARTIFACTS}/l31_patch{suffix}/perplexity_sweep_seed{SEEDS[0]}.csv"
        seeds_str = ",".join(str(s) for s in SEEDS)
        if all(check_flag(f"{ARTIFACTS}/l31_patch{suffix}/perplexity_sweep_seed{s}.csv") for s in SEEDS):
            print(f"[check] {op_name} perplexity all seeds done")
            continue
        step(f"Perplexity eval for {op_name}, seeds={seeds_str}")
        run(f"python eval_l31_perplexity.py {seeds_str} {op_name}")

    step("All perplexity evals complete")
    print("\n✓ Multi-seed sweep finished for all ops and seeds.")


if __name__ == "__main__":
    main()
