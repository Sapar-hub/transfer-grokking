import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

from utils import DEVICE, P

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/baseline_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 256
N_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2000


def main():
    print("=" * 60)
    print("Analyze Baseline Asymmetry: Phi-2 prediction distributions")
    print("=" * 60)

    print("\n[0] Generating random pairs...")
    rng = np.random.RandomState(42)
    a = rng.randint(0, P, size=N_SAMPLES)
    b = rng.randint(0, P, size=N_SAMPLES)
    add_labels = (a + b) % P
    mult_labels = (a * b) % P
    pairs = list(zip(a.tolist(), b.tolist()))
    print(f"  {N_SAMPLES} pairs generated.")

    print("\n[1] Loading Phi-2...")
    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    phi2.eval()
    print("  Phi-2 loaded.")

    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}
    number_ids_list = [number_tokens[n] for n in range(P)]

    def evaluate(prompts):
        all_preds = []
        all_logit_max = []
        for start in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[start:start + BATCH_SIZE]
            tokenized = tokenizer(batch_prompts, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = phi2(**tokenized)
            logits = outputs.logits[:, -1, :]  # (B, V)
            num_logits = logits[:, number_ids_list]  # (B, 97)

            preds = num_logits.argmax(dim=1).numpy()
            logit_max_vals = num_logits.max(dim=1).values.numpy()

            all_preds.extend(preds.tolist())
            all_logit_max.extend(logit_max_vals.tolist())

        return np.array(all_preds), np.array(all_logit_max)

    print("\n[2a] Evaluating Phi-2 on addition format: (a + b) % 97 = ...")
    add_prompts = [f"# ({x} + {y}) % 97 =" for x, y in pairs]
    add_preds, add_logits = evaluate(add_prompts)
    add_acc = (add_preds == add_labels).mean()
    print(f"  Addition prompt accuracy (vs add labels): {add_acc:.4f}")

    print("\n[2b] Evaluating Phi-2 on multiplication format: (a * b) % 97 = ...")
    mult_prompts = [f"# ({x} * {y}) % 97 =" for x, y in pairs]
    mult_preds, mult_logits = evaluate(mult_prompts)
    mult_acc = (mult_preds == mult_labels).mean()
    print(f"  Multiplication prompt accuracy (vs mult labels): {mult_acc:.4f}")

    # Also check: addition prompt vs mult labels
    add_prompt_mult_label_acc = (add_preds == mult_labels).mean()
    mult_prompt_add_label_acc = (mult_preds == add_labels).mean()
    print(f"  Addition prompt vs MULT labels: {add_prompt_mult_label_acc:.4f}")
    print(f"  Multiplication prompt vs ADD labels: {mult_prompt_add_label_acc:.4f}")

    print("\n[3] Analyzing prediction distributions...")

    # Count single-digit predictions (0-9)
    add_single = (add_preds < 10).mean()
    mult_single = (mult_preds < 10).mean()
    print(f"  Addition prompt: single-digit predictions = {add_single:.4f}")
    print(f"  Multiplication prompt: single-digit predictions = {mult_single:.4f}")

    # Count specific token biases (top-5 most predicted classes)
    for name, preds in [("addition prompt", add_preds), ("multiplication prompt", mult_preds)]:
        counts = np.bincount(preds, minlength=P)
        top5 = np.argsort(counts)[-5:][::-1]
        print(f"  {name}: top-5 predicted classes = {top5.tolist()} "
              f"with counts {counts[top5].tolist()} (of {len(preds)})")

    print("\n[4] Plotting...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].hist(add_preds, bins=np.arange(0, P + 1) - 0.5, alpha=0.7, color='steelblue',
                     edgecolor='white', linewidth=0.3, label=f'addition prompt (acc={add_acc:.3f})')
    axes[0, 0].set_title('Predictions: addition prompt')
    axes[0, 0].set_xlabel('Predicted class'); axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(x=9.5, color='red', linestyle='--', alpha=0.5, label='9.5 (single-digit)')
    axes[0, 0].legend()

    axes[0, 1].hist(mult_preds, bins=np.arange(0, P + 1) - 0.5, alpha=0.7, color='coral',
                     edgecolor='white', linewidth=0.3, label=f'mult prompt (acc={mult_acc:.3f})')
    axes[0, 1].set_title('Predictions: multiplication prompt')
    axes[0, 1].set_xlabel('Predicted class'); axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(x=9.5, color='red', linestyle='--', alpha=0.5, label='9.5 (single-digit)')
    axes[0, 1].legend()

    # Correct vs incorrect per class
    add_correct_by_class = np.zeros(P)
    add_total_by_class = np.zeros(P)
    for i in range(len(add_labels)):
        lbl = add_labels[i]
        add_total_by_class[lbl] += 1
        if add_preds[i] == lbl:
            add_correct_by_class[lbl] += 1

    mult_correct_by_class = np.zeros(P)
    mult_total_by_class = np.zeros(P)
    for i in range(len(mult_labels)):
        lbl = mult_labels[i]
        mult_total_by_class[lbl] += 1
        if mult_preds[i] == lbl:
            mult_correct_by_class[lbl] += 1

    add_per_class = np.divide(add_correct_by_class, add_total_by_class,
                              out=np.zeros_like(add_correct_by_class),
                              where=add_total_by_class > 0)
    mult_per_class = np.divide(mult_correct_by_class, mult_total_by_class,
                               out=np.zeros_like(mult_correct_by_class),
                               where=mult_total_by_class > 0)

    axes[1, 0].bar(np.arange(P), add_per_class, alpha=0.7, color='steelblue', label='addition')
    axes[1, 0].bar(np.arange(P), mult_per_class, alpha=0.5, color='coral', label='multiplication')
    axes[1, 0].set_title('Per-class accuracy (correct / total per class)')
    axes[1, 0].set_xlabel('Class'); axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].axvline(x=9.5, color='red', linestyle='--', alpha=0.5, label='single-digit')
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)

    # Logit distribution (max logit values per prompt)
    axes[1, 1].hist(add_logits, bins=50, alpha=0.6, color='steelblue', label=f'addition (mean={add_logits.mean():.1f})')
    axes[1, 1].hist(mult_logits, bins=50, alpha=0.6, color='coral', label=f'mult (mean={mult_logits.mean():.1f})')
    axes[1, 1].set_title('Max logit value distribution')
    axes[1, 1].set_xlabel('Max logit'); axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()

    plt.tight_layout()
    fig_path = f"{OUT_DIR}/prediction_distribution.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")

    # Also check with ORIGINAL eval setup: add prompt, mult labels
    orig_mult_baseline = (add_preds == mult_labels).mean()
    print(f"\n[5] Original eval setup (add prompt, mult labels): {orig_mult_baseline:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Addition prompt → add labels:  {add_acc:.4f}")
    print(f"  Mult prompt    → mult labels:  {mult_acc:.4f}")
    print(f"  Addition prompt → mult labels: {orig_mult_baseline:.4f} (original 'baseline' in paper)")
    print(f"")
    print(f"  Single-digit pred rate (add prompt):  {add_single:.3f}")
    print(f"  Single-digit pred rate (mult prompt): {mult_single:.3f}")
    print(f"")
    print(f"  If Phi-2 had pure single-digit bias → expected acc on add: ~10/97 ≈ 0.103")
    print(f"  Actual add acc: {add_acc:.3f} — >0.103 suggests partial structure beyond bias")

    with open(f"{OUT_DIR}/summary.md", "w") as f:
        f.write(f"# Baseline Asymmetry Analysis\n\n")
        f.write(f"## Setup\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Samples | {N_SAMPLES} |\n")
        f.write(f"| Model | Phi-2 |\n\n")
        f.write(f"## Accuracies\n\n")
        f.write(f"| Condition | Accuracy |\n")
        f.write(f"|-----------|----------|\n")
        f.write(f"| Addition prompt → add labels | {add_acc:.4f} |\n")
        f.write(f"| Mult prompt → mult labels | {mult_acc:.4f} |\n")
        f.write(f"| Addition prompt → mult labels | {orig_mult_baseline:.4f} |\n\n")
        f.write(f"## Single-digit rate\n\n")
        f.write(f"| Condition | Rate |\n")
        f.write(f"|-----------|------|\n")
        f.write(f"| Addition prompt | {add_single:.3f} |\n")
        f.write(f"| Mult prompt | {mult_single:.3f} |\n\n")
        f.write(f"## Interpretation\n\n")
        if add_acc > 0.103:
            f.write("Addition accuracy exceeds pure single-digit bias (0.103), ")
            f.write("suggesting Phi-2 has partial arithmetic structure beyond token bias.\n\n")
        else:
            f.write("Addition accuracy is consistent with single-digit token bias.\n\n")
        f.write(f"_Generated by analyze_baseline_asymmetry.py_\n")

    print(f"  Saved: {OUT_DIR}/summary.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
