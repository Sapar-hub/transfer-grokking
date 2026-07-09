import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, csv, sys

from utils import DEVICE, P

ARTIFACTS = "artifacts"
CE_DIR = f"{ARTIFACTS}/ce_projection"

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
OP = sys.argv[2] if len(sys.argv) > 2 else "add"
SUFFIX = "" if OP == "add" else "_mult"
OP_SYMBOL = {"add": "+", "mult": "*"}

OUT_DIR = f"{ARTIFACTS}/l31_patch{SUFFIX}"
os.makedirs(OUT_DIR, exist_ok=True)

D_SMALL = 128
D_PHI2 = 2560
PATCH_LAYER = 31
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]


def get_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]


def make_patch_hook(W, h_A_batch, attention_mask=None, alpha=1.0):
    W.eval()
    with torch.no_grad():
        patch = W(torch.from_numpy(h_A_batch).float())
    def hook(module, input, output):
        hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            if alpha == 1.0:
                hidden[batch_idx, seq_lens] = patch
            elif alpha > 0:
                orig = hidden[batch_idx, seq_lens].clone()
                hidden[batch_idx, seq_lens] = (1 - alpha) * orig + alpha * patch
        else:
            if alpha == 1.0:
                hidden[:, -1, :] = patch
            elif alpha > 0:
                hidden[:, -1, :] = (1 - alpha) * hidden[:, -1, :] + alpha * patch
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


def evaluate_alpha(model, tokenizer, test_pairs, labels, W, h_A_test, alpha, batch_size=32):
    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}
    correct = 0
    total = 0
    for start in range(0, len(test_pairs), batch_size):
        batch_pairs = test_pairs[start:start + batch_size]
        batch_h_A = h_A_test[start:start + batch_size]
        prompts = [f"# ({a} {OP_SYMBOL[OP]} {b}) % 97 =" for a, b in batch_pairs]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        if alpha == 0.0:
            with torch.no_grad():
                outputs = model(**tokenized)
        else:
            hook = make_patch_hook(W, batch_h_A, tokenized.attention_mask, alpha)
            handle = model.model.layers[PATCH_LAYER].register_forward_hook(hook)
            with torch.no_grad():
                outputs = model(**tokenized)
            handle.remove()
        logits = outputs.logits[:, -1, :]
        for i in range(len(batch_pairs)):
            pred = max(number_tokens, key=lambda n: logits[i, number_tokens[n]].item())
            if pred == labels[start + i]:
                correct += 1
            total += 1
    return correct / total


def main():
    print("=" * 60)
    print(f"L31 Patch: alpha sweep with W_CE and W_MSE at layer {PATCH_LAYER}")
    print("=" * 60)

    print(f"\n[0] Loading data (op={OP}, seed={SEED})...")
    small_acts = np.load(f"{ARTIFACTS}/small_model_activations{SUFFIX}.npy")
    labels = np.load(f"{ARTIFACTS}/mod_arithmetic_labels{SUFFIX}.npy", allow_pickle=True)
    _, test_idx = get_split()

    rng_eval = np.random.RandomState(42)
    eval_idx = rng_eval.choice(test_idx, size=200, replace=False)
    eval_pairs = [(int(i // P), int(i % P)) for i in eval_idx]
    eval_labels = labels[eval_idx]
    eval_h_A = small_acts[eval_idx]
    print(f"  200 eval pairs loaded.")

    print(f"\n[1] Loading seed {SEED} W_ce and W_mse...")
    W_ce = nn.Linear(D_SMALL, D_PHI2, bias=False)
    W_ce.load_state_dict(torch.load(
        f"{CE_DIR}/seeds{SUFFIX}/W_ce_seed{SEED}.pth", map_location=DEVICE, weights_only=True))
    W_mse = nn.Linear(D_SMALL, D_PHI2, bias=False)
    W_mse.load_state_dict(torch.load(
        f"{CE_DIR}/seeds{SUFFIX}/W_mse_seed{SEED}.pth", map_location=DEVICE, weights_only=True))
    print("  Both loaded.")

    print(f"\n[2] Loading Phi-2...")
    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    phi2.eval()
    print("  Phi-2 loaded.")

    print(f"\n[3] Alpha sweep at L{PATCH_LAYER}...")
    results = []
    for W, label in [(W_mse, "MSE"), (W_ce, "CE")]:
        W.requires_grad_(False)
        W.eval()
        row = [label]
        for alpha in ALPHAS:
            acc = evaluate_alpha(phi2, tokenizer, eval_pairs, eval_labels,
                                 W, eval_h_A, alpha)
            row.append(acc)
            print(f"  [{label}] alpha={alpha:.1f}: text_acc = {acc:.4f}")
        results.append(row)

    with open(f"{OUT_DIR}/alpha_sweep_l31_seed{SEED}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["W", *ALPHAS])
        w.writerows(results)

    print(f"\n[4] L10 baseline (from ce_projection seed {SEED})...")
    # Validate metadata before loading L10 CSV
    meta_path = f"{CE_DIR}/seeds{SUFFIX}/metadata_seed{SEED}.pt"
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, weights_only=True)
        if meta.get("LAYER", 0) != 10:
            print(f"  [WARN] metadata LAYER={meta['LAYER']}, expected 10")
        if meta.get("ALPHAS") != ALPHAS:
            print(f"  [WARN] metadata ALPHAS={meta['ALPHAS']}, expected {ALPHAS}")
    else:
        print(f"  [WARN] No metadata file — cannot validate CE training params")

    l10_path = f"{CE_DIR}/seeds{SUFFIX}/alpha_sweep_seed{SEED}.csv"
    with open(l10_path) as f:
        reader = csv.reader(f)
        l10_rows = list(reader)
    if len(l10_rows) < 3:
        raise ValueError(f"CSV {l10_path} has {len(l10_rows)} rows, expected >= 3")
    if l10_rows[1][0] not in ("MSE", "W_MSE"):
        print(f"  [WARN] Expected row[1] label 'MSE'/'W_MSE', got '{l10_rows[1][0]}'")
    if l10_rows[2][0] not in ("CE", "W_CE"):
        print(f"  [WARN] Expected row[2] label 'CE'/'W_CE', got '{l10_rows[2][0]}'")
    l10_mse = [float(v) for v in l10_rows[1][1:]]
    l10_ce = [float(v) for v in l10_rows[2][1:]]
    if len(l10_mse) != len(ALPHAS):
        print(f"  [WARN] CSV has {len(l10_mse)} alphas, expected {len(ALPHAS)}")
    baseline = l10_mse[0]

    l31_mse = [float(v) for v in results[0][1:]]
    l31_ce = [float(v) for v in results[1][1:]]

    print(f"\n[5] Summary...")
    lines = []
    lines.append("# L31 Patch: W_CE / W_MSE alpha sweep at layer 31\n")
    lines.append("## Setup\n")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Patch layer | {PATCH_LAYER} |")
    lines.append(f"| Test pairs | 200 (seed=42) |")
    lines.append(f"| W_CE source | {CE_DIR}/W_ce.pth |")
    lines.append(f"| W_MSE source | {CE_DIR}/W_mse.pth |\n")
    lines.append("## Alpha sweep results\n")
    lines.append("| Alpha | W_MSE L10 | W_MSE L31 | W_CE L10 | W_CE L31 |")
    lines.append("|-------|-----------|-----------|----------|----------|")
    for i, alpha in enumerate(ALPHAS):
        lines.append(f"| {alpha:.1f} | {l10_mse[i]:.4f} | {l31_mse[i]:.4f} | {l10_ce[i]:.4f} | {l31_ce[i]:.4f} |")
    lines.append(f"\nBaseline (alpha=0.0): {baseline:.4f}\n")

    best_l10_mse = max(l10_mse[1:])
    best_l31_mse = max(l31_mse[1:])
    best_l10_ce = max(l10_ce[1:])
    best_l31_ce = max(l31_ce[1:])
    best_all = max(best_l10_mse, best_l31_mse, best_l10_ce, best_l31_ce)

    lines.append("## Best per condition\n")
    lines.append("| Condition | Best α | Best Acc |")
    lines.append("|-----------|--------|----------|")
    for label, vals in [("W_MSE L10", l10_mse), ("W_MSE L31", l31_mse),
                        ("W_CE L10", l10_ce), ("W_CE L31", l31_ce)]:
        best_i = max(range(1, len(vals)), key=lambda i: vals[i])
        lines.append(f"| {label} | {ALPHAS[best_i]:.1f} | {vals[best_i]:.4f} |")
    lines.append(f"\n**Global best**: {best_all:.4f}\n")

    if l31_ce[ALPHAS.index(0.5)] > l10_mse[ALPHAS.index(0.5)]:
        lines.append("**Verdict**: W_CE L31 > W_MSE L10 at α=0.5 — neural function call feasible through last layer.\n")
    elif l31_ce[ALPHAS.index(0.5)] > 0.25:
        lines.append("**Verdict**: L31 patch is not worse than L10 but does not resolve the context/geometry conflict.\n")
    else:
        lines.append("**Verdict**: L31 patch degrades text accuracy — late layers are critical for decoding.\n")

    text = "\n".join(lines)
    with open(f"{OUT_DIR}/comparison_l10_vs_l31_seed{SEED}.md", "w") as f:
        f.write(text)
    print(text)

    print(f"\nDone. Artifacts in {OUT_DIR}/")


if __name__ == "__main__":
    main()
