import json
import os

ARTIFACTS = "artifacts"


def interpret():
    """Interpret the steering experiment results from lm_eval JSON.

    Purpose:
        Determine the outcome category (A/B/C/D) based on steering delta
        and degradation on downstream benchmarks.
    What:
        Loads artifacts/lm_eval_steered.json, computes delta_task = steered
        - baseline for mod arithmetic, checks degradation on Hellaswag,
        writes experiment_summary.md with verdict.
    Why:
        The outcome classification (A = works, no degradation; B = works,
        with degradation; C = no effect; D = no structure) determines
        whether the hypothesis is confirmed. This is the final step of
        the original experiment pipeline.
    """
    if not os.path.exists(f"{ARTIFACTS}/lm_eval_steered.json"):
        print("No steered eval results — experiment likely stopped early.")
        return

    with open(f"{ARTIFACTS}/lm_eval_steered.json") as f:
        r = json.load(f)

    delta_task = r["mod_arithmetic_steered"] - r["mod_arithmetic_baseline"]

    deltas = {}
    for bm in ["hellaswag", "lambada", "winogrande", "boolq"]:
        deltas[bm] = r.get(f"{bm}_delta", 0.0)

    with open(f"{ARTIFACTS}/experiment_summary.md", "w") as f:
        f.write("# Experiment Summary\n\n")
        f.write(f"| Metric | Baseline | Steered | Delta |\n")
        f.write(f"|--------|----------|---------|-------|\n")
        f.write(f"| Mod Arithmetic | {r['mod_arithmetic_baseline']:.4f} | {r['mod_arithmetic_steered']:.4f} | {delta_task:+.4f} |\n")
        for bm in ["hellaswag", "lambada", "winogrande", "boolq"]:
            base = r.get(f"{bm}_baseline", 0)
            steer = r.get(f"{bm}_steered", 0)
            p = r.get(f"{bm}_mcnemar_p", 1.0)
            f.write(f"| {bm} | {base:.4f} | {steer:.4f} | {deltas[bm]:+.4f} (p={p:.4f}) |\n")

        f.write("\n## Outcome\n\n")

        if delta_task > 0.03:
            if deltas["hellaswag"] > -0.02:
                outcome = "A"
                desc = ("Geometry partially compatible. Steering works without degradation. "
                        "Hypothesis partially confirmed.")
            elif deltas["hellaswag"] < -0.05:
                outcome = "B"
                desc = ("Steering works but at cost of degradation. Tradeoff — method not usable as-is.")
            else:
                outcome = "A/B"
                desc = ("Steering improves mod arithmetic with mild degradation on hellaswag. "
                        "Borderline between A and B.")
            f.write(f"### Outcome {outcome}\n{desc}\n")
        else:
            outcome = "C"
            desc = ("Steering had no significant effect. Geometries incompatible or projection "
                    "loses structure. Hypothesis not confirmed.")
            f.write(f"### Outcome {outcome}\n{desc}\n")

        f.write("\n## Raw numbers\n\n")
        f.write(f"```json\n{json.dumps(r, indent=2)}\n```\n")

    print(f"Interpretation: Outcome {'A/B/C/D'[int(delta_task > 0.03) + (deltas['hellaswag'] < -0.05) * 2]}")
    return delta_task, deltas


if __name__ == "__main__":
    interpret()
