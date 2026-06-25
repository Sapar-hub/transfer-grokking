import sys
import os

ARTIFACTS = "artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)


def step1():
    """STEP 1: Train small transformer to grokking.

    Purpose:
        Initial model training. Produces the grokked small model used
        as the steering source in all downstream experiments.
    What:
        Calls train_small.train(), returns whether grokking succeeded.
    Why:
        If grokking fails, no downstream experiments are possible.
    """
    print("=" * 60)
    print("STEP 1: Train small transformer to grokking")
    print("=" * 60)
    from train_small import train
    grokking, best_val = train()
    return grokking


def step2():
    """STEP 2: Verify Fourier structure in small model.

    Purpose:
        Confirm the small model learned circular representations (Fourier
        Hypothesis) before proceeding to probing/steering.
    What:
        Calls verify_fourier.verify(), returns whether probe_acc > 0.95.
    Why:
        Fourier structure is evidence the model learned a genuine
        algorithmic computation, not just memorisation.
    """
    print("=" * 60)
    print("STEP 2: Verify Fourier structure")
    print("=" * 60)
    from verify_fourier import verify
    probe_acc = verify()
    return probe_acc > 0.95


def step3_and_4():
    """STEPS 3-4: Phi-2 baseline + probing.

    Purpose:
        Determine whether Phi-2 encodes modular arithmetic structure and
        which layer is the best steering target.
    What:
        Calls probe_phi2.run_probes(), returns baseline accuracy, best
        layer, max probe accuracy, and per-layer accuracy list.
    Why:
        Establishes the baseline and identifies the steering target layer.
    """
    print("=" * 60)
    print("STEPS 3-4: Phi-2 baseline + probing")
    print("=" * 60)
    from probe_phi2 import run_probes
    baseline_acc, best_layer, max_acc, layer_accs = run_probes()
    return baseline_acc, best_layer, max_acc, layer_accs


def step5_and_6(steering_vec, best_layer):
    """STEPS 5-6: Compute steering vector + apply steering to Phi-2.

    Purpose:
        Extract a steering vector from the small model, project it via
        random orthogonal matrix into Phi-2's space, and test if it
        improves Phi-2's accuracy on mod arithmetic.
    What:
        Calls steering.compute_steering_vector() and
        steering.apply_steering() with alpha sweep.
    Why:
        Phase 4 test of the naive random-projection steering hypothesis.
    """
    print("=" * 60)
    print("STEPS 5-6: Steering vector + apply steering")
    print("=" * 60)
    from steering import compute_steering_vector, apply_steering
    if steering_vec is None:
        steering_vec = compute_steering_vector()
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = apply_steering(steering_vec, best_layer, alphas)
    best_alpha = max(results, key=lambda x: x[1])
    return steering_vec, best_alpha


def step7(steering_vec, best_layer, best_alpha):
    """STEP 7: Measure degradation on downstream benchmarks.

    Purpose:
        Check whether steering degrades Phi-2's performance on general
        language understanding benchmarks.
    What:
        Calls eval_degradation.run_eval() with 4 benchmarks.
    Why:
        Degradation test determines the outcome category. If steering
        destroys general capabilities, it's not a viable method.
    """
    print("=" * 60)
    print("STEP 7: Measure degradation")
    print("=" * 60)
    from eval_degradation import run_eval
    results = run_eval(steering_vec, best_layer, best_alpha[0])
    return results


def step8():
    """STEP 8: Interpretation — determine outcome category.

    Purpose:
        Classify the experiment outcome (A/B/C/D) based on steering
        delta and degradation metrics.
    What:
        Calls interpret.interpret() which reads cached JSON results
        and writes the summary markdown.
    Why:
        Final step that produces the experiment_summary.md with a clear
        verdict on the hypothesis.
    """
    print("=" * 60)
    print("STEP 8: Interpretation")
    print("=" * 60)
    from interpret import interpret
    interpret()


def main():
    """Orchestrate the full experiment pipeline.

    Purpose:
        Top-level entry point for the original project pipeline
        (Phases 1-4 + Experiment A). Runs sequentially through
        all steps, with early stopping if any step fails.
    What:
        step1 -> step2 -> step3_and_4 -> step5_and_6 -> step7 -> step8.
        Each step gates the next: if grokking fails (step1),
        Fourier fails (step2), or Phi-2 probling shows no structure
        (step3_and_4), the pipeline stops early.
    Why:
        Provides a single entry point for the entire experiment.
        The step-by-step structure makes it easy to restart from
        any phase if a step fails.
    """
    if not step1():
        print("Grokking failed — stopping.")
        with open(f"{ARTIFACTS}/experiment_summary.md", "w") as f:
            f.write("# Experiment Summary\n\nGrokking not achieved. Experiment stopped at Step 1.")
        return

    if not step2():
        print("Fourier structure verification failed — probe_acc < 0.95. Retry training.")
        print("Try re-running with different seed or hyperparameters.")
        return

    baseline_acc, best_layer, max_acc, layer_accs = step3_and_4()

    if max_acc < 0.05:
        print(f"Phi-2 probe max acc = {max_acc:.4f} ≈ random. Outcome D.")
        with open(f"{ARTIFACTS}/experiment_summary.md", "w") as f:
            f.write(f"# Experiment Summary\n\nOutcome D: Phi-2 does not encode mod arithmetic. "
                    f"Max probe acc = {max_acc:.4f} ≈ {1/97:.4f}")
        return

    steering_vec = None
    steering_vec, best_alpha = step5_and_6(steering_vec, best_layer)
    print(f"Best alpha: {best_alpha[0]:.1f} with acc = {best_alpha[1]:.4f}")

    step7(steering_vec, best_layer, best_alpha)

    step8()
    print("Experiment complete. See artifacts/ for results.")


if __name__ == "__main__":
    main()
