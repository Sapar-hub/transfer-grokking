import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv

from utils import DEVICE, P, train_probe

BASE = "artifacts"
RP_DIR = f"{BASE}/residual_patch"
OUT_DIR = f"{BASE}/multi_layer_patch"
os.makedirs(OUT_DIR, exist_ok=True)

D_SMALL = 128
D_PHI2 = 2560
LAYERS = [10, 15, 20, 25, 30]
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
BATCH_SIZE = 128
PROBE_SUBSET = 500

# ─── Data split (deterministic, matching residual_patch.py) ────────────

def get_train_test_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]

# ─── Patch hook (copied from residual_patch.py) ────────────────────────

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

# ─── Alpha sweep ──────────────────────────────────────────────────────

def evaluate_alpha(model, tokenizer, test_pairs, labels, Ws, h_A_test, alpha, layers, use_same_W=False, base_W=None):
    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}
    correct = 0
    total = 0

    for start in range(0, len(test_pairs), BATCH_SIZE):
        batch_pairs = test_pairs[start:start + BATCH_SIZE]
        batch_h_A = h_A_test[start:start + BATCH_SIZE]
        prompts = [f"# ({a} + {b}) % 97 =" for a, b in batch_pairs]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")

        if alpha == 0.0:
            with torch.no_grad():
                outputs = model(**tokenized)
        else:
            handles = []
            for l in layers:
                W = base_W if use_same_W else Ws[l]
                hook = make_patch_hook(W, batch_h_A, tokenized.attention_mask, alpha)
                handle = model.model.layers[l].register_forward_hook(hook)
                handles.append(handle)

            with torch.no_grad():
                outputs = model(**tokenized)

            for h in handles:
                h.remove()

        logits = outputs.logits[:, -1, :]
        for i in range(len(batch_pairs)):
            pred = max(number_tokens, key=lambda n: logits[i, number_tokens[n]].item())
            if pred == labels[start + i]:
                correct += 1
            total += 1

    return correct / total

# ─── Logit lens ───────────────────────────────────────────────────────

def collect_patched_activations(model, tokenizer, test_pairs, Ws, h_A_test, patch_layers, collect_layer, alpha=1.0, use_same_W=False, base_W=None):
    all_acts = []
    current_mask = None

    def make_collect_hook():
        def hook(module, input, output):
            nonlocal current_mask
            hidden = output[0] if isinstance(output, tuple) else output
            mask = current_mask.to(hidden.device)
            seq_lens = mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            all_acts.append(hidden[batch_idx, seq_lens].detach().cpu().numpy())
        return hook

    for start in range(0, len(test_pairs), BATCH_SIZE):
        batch_pairs = test_pairs[start:start + BATCH_SIZE]
        batch_h_A = h_A_test[start:start + BATCH_SIZE]
        prompts = [f"# ({a} + {b}) % 97 =" for a, b in batch_pairs]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        current_mask = tokenized.attention_mask

        handles = []
        if alpha > 0:
            for l in patch_layers:
                W = base_W if use_same_W else Ws[l]
                hook = make_patch_hook(W, batch_h_A, tokenized.attention_mask, alpha)
                handles.append(model.model.layers[l].register_forward_hook(hook))

        collect_handle = model.model.layers[collect_layer].register_forward_hook(make_collect_hook())
        handles.append(collect_handle)

        with torch.no_grad():
            model(**tokenized)

        for h in handles:
            h.remove()

    return np.concatenate(all_acts, axis=0)


def logit_lens_accuracy(model, tokenizer, test_pairs, labels, Ws, h_A_test, alpha, patch_layers, use_same_W=False, base_W=None):
    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}
    collect_layer = patch_layers[-1]
    acts = collect_patched_activations(
        model, tokenizer, test_pairs, Ws, h_A_test,
        patch_layers, collect_layer, alpha,
        use_same_W=use_same_W, base_W=base_W
    )
    h = torch.from_numpy(acts).float()
    with torch.no_grad():
        logits = model.lm_head(h)

    correct = 0
    for i in range(len(test_pairs)):
        pred = max(number_tokens, key=lambda n: logits[i, number_tokens[n]].item())
        if pred == labels[i]:
            correct += 1
    return correct / len(test_pairs)

# ─── Probe on collect_layer after multi-layer patch ───────────────────

def probe_after_patch(Ws, h_A_test, labels_test, model, tokenizer, test_pairs, patch_layers, collect_layer, use_same_W=False, base_W=None):
    acts_orig = collect_patched_activations(
        model, tokenizer, test_pairs, Ws, h_A_test,
        patch_layers, collect_layer, alpha=0.0,
        use_same_W=use_same_W, base_W=base_W
    )
    acc_orig, *_ = train_probe(acts_orig, labels_test)

    acts_patched = collect_patched_activations(
        model, tokenizer, test_pairs, Ws, h_A_test,
        patch_layers, collect_layer, alpha=1.0,
        use_same_W=use_same_W, base_W=base_W
    )
    acc_patched, *_ = train_probe(acts_patched, labels_test)

    return acc_orig, acc_patched

# ─── Load existing single-layer results for comparison ────────────────

def load_single_layer_results():
    results = {}
    path = f"{RP_DIR}/results_grid.csv"
    if os.path.exists(path):
        rows = []
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                rows.append((float(row[0]), float(row[1])))
        results["text"] = rows

    path = f"{RP_DIR}/logit_lens.csv"
    if os.path.exists(path):
        rows = []
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                rows.append((float(row[0]), float(row[1])))
        results["logit_lens"] = rows

    path = f"{RP_DIR}/probe_before_after.csv"
    if os.path.exists(path):
        rows = {}
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                rows[row[0]] = float(row[1])
        results["probe"] = rows
    return results


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Multi-Layer Patch: inject at all 5 layers simultaneously")
    print("=" * 60)

    print("\n[0] Loading data...")
    small_acts = np.load(f"{BASE}/small_model_activations.npy")
    labels = np.load(f"{BASE}/mod_arithmetic_labels.npy")
    print(f"  Small activations: {small_acts.shape}")

    train_idx, test_idx = get_train_test_split()
    small_test = small_acts[test_idx]
    labels_test = labels[test_idx]

    rng_eval = np.random.RandomState(42)
    eval_idx = rng_eval.choice(test_idx, size=200, replace=False)
    eval_pairs = [(int(i // P), int(i % P)) for i in eval_idx]
    eval_labels = labels[eval_idx]
    eval_h_A = small_acts[eval_idx]

    print("\n[1] Loading Phi-2...")
    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    phi2.eval()

    print("\n[2] Loading cached Ws from residual patch...")
    Ws = {}
    for l in LAYERS:
        W = nn.Linear(D_SMALL, D_PHI2, bias=False)
        W.load_state_dict(torch.load(f"{RP_DIR}/W_layer{l}.pth", map_location=DEVICE, weights_only=True))
        W.requires_grad_(False)
        Ws[l] = W
        print(f"  Loaded W_layer{l}.pth")
    best_layer = 10
    print(f"  Using W_L{best_layer} as base for 'same W' ablation")

    print("\n[3] Loading single-layer results (baseline)...")
    single = load_single_layer_results()
    print(f"  Single-layer best: {max([r[1] for r in single['text'] if r[0] > 0]):.4f}")

    print("\n[4] Alpha sweep — Per-layer W...")
    per_layer = []
    for alpha in ALPHAS:
        acc = evaluate_alpha(phi2, tokenizer, eval_pairs, eval_labels,
                             Ws, eval_h_A, alpha, LAYERS, use_same_W=False)
        per_layer.append((alpha, acc))
        print(f"  alpha={alpha:.1f}: mod_acc = {acc:.4f}")

    print("\n[5] Alpha sweep — Same W...")
    same_W = []
    for alpha in ALPHAS:
        acc = evaluate_alpha(phi2, tokenizer, eval_pairs, eval_labels,
                             Ws, eval_h_A, alpha, LAYERS,
                             use_same_W=True, base_W=Ws[best_layer])
        same_W.append((alpha, acc))
        print(f"  alpha={alpha:.1f}: mod_acc = {acc:.4f}")

    print("\n[6] Logit lens — Per-layer W...")
    per_layer_logit = []
    for alpha in ALPHAS:
        acc = logit_lens_accuracy(phi2, tokenizer, eval_pairs, eval_labels,
                                  Ws, eval_h_A, alpha, LAYERS, use_same_W=False)
        per_layer_logit.append((alpha, acc))
        print(f"  alpha={alpha:.1f}: logit_lens = {acc:.4f}")

    print("\n[7] Logit lens — Same W...")
    same_W_logit = []
    for alpha in ALPHAS:
        acc = logit_lens_accuracy(phi2, tokenizer, eval_pairs, eval_labels,
                                  Ws, eval_h_A, alpha, LAYERS,
                                  use_same_W=True, base_W=Ws[best_layer])
        same_W_logit.append((alpha, acc))
        print(f"  alpha={alpha:.1f}: logit_lens = {acc:.4f}")

    probe_subset_idx = test_idx[:PROBE_SUBSET]
    test_pairs_full = [(int(i // P), int(i % P)) for i in probe_subset_idx]
    small_test_subset = small_test[:PROBE_SUBSET]
    labels_test_subset = labels_test[:PROBE_SUBSET]
    collect_layer = LAYERS[-1] + 1
    print(f"\n[8] Probe on L{collect_layer} — Per-layer W ({len(test_pairs_full)} pairs)...")
    per_layer_probe = probe_after_patch(
        Ws, small_test_subset, labels_test_subset, phi2, tokenizer,
        test_pairs_full, LAYERS, collect_layer, use_same_W=False
    )
    print(f"    original: {per_layer_probe[0]:.4f}   patched: {per_layer_probe[1]:.4f}")

    print(f"\n[9] Probe on L{collect_layer} — Same W...")
    same_W_probe = probe_after_patch(
        Ws, small_test_subset, labels_test_subset, phi2, tokenizer,
        test_pairs_full, LAYERS, collect_layer,
        use_same_W=True, base_W=Ws[best_layer]
    )
    print(f"    original: {same_W_probe[0]:.4f}   patched: {same_W_probe[1]:.4f}")

    print("\n[10] Writing summary...")
    write_summary(single, per_layer, same_W, best_layer,
                  per_layer_logit, same_W_logit,
                  per_layer_probe, same_W_probe)

    print(f"\nDone. Artifacts in {OUT_DIR}/")


def write_summary(single, per_layer, same_W, best_layer,
                  per_layer_logit, same_W_logit,
                  per_layer_probe, same_W_probe):
    bl = single["text"][0][1]
    lines = []
    lines.append("# Multi-Layer Patch Experiment Summary\n")
    lines.append("## Design\n")
    lines.append(f"- **Single-layer**: inject W_L{best_layer} at layer {best_layer} only (baseline)")
    lines.append(f"- **Per-layer W**: inject Ws[l] at layer l for l in {LAYERS}")
    lines.append(f"- **Same W**: inject W_L{best_layer} at all layers {LAYERS}")
    lines.append(f"- Injection layers: {LAYERS}\n")
    lines.append("## Alpha Sweep (text accuracy)\n")
    lines.append("| Alpha | Single-layer | Per-layer W | Same W |")
    lines.append("|-------|-------------|-------------|--------|")
    for i, alpha in enumerate(ALPHAS):
        s = single["text"][i][1]
        p = per_layer[i][1]
        sw = same_W[i][1]
        lines.append(f"| {alpha:.1f} | {s:.4f} | {p:.4f} | {sw:.4f} |")

    best_s = max([r for r in single["text"] if r[0] > 0], key=lambda x: x[1])[1]
    best_p = max([r for r in per_layer if r[0] > 0], key=lambda x: x[1])[1]
    best_sw = max([r for r in same_W if r[0] > 0], key=lambda x: x[1])[1]
    lines.append(f"\n**Best**: Single={best_s:.4f}  Per-layer={best_p:.4f}  Same-W={best_sw:.4f}")
    lines.append(f"**Delta vs baseline**: Single={best_s-bl:+.4f}  Per-layer={best_p-bl:+.4f}  Same-W={best_sw-bl:+.4f}\n")

    lines.append("## Logit Lens (decode from last patched layer)\n")
    lines.append("| Alpha | Single-layer | Per-layer W | Same W |")
    lines.append("|-------|-------------|-------------|--------|")
    for i, alpha in enumerate(ALPHAS):
        s = single["logit_lens"][i][1]
        p = per_layer_logit[i][1] if len(per_layer_logit) > i else 0
        sw = same_W_logit[i][1] if len(same_W_logit) > i else 0
        lines.append(f"| {alpha:.1f} | {s:.4f} | {p:.4f} | {sw:.4f} |")
    lines.append("")

    lines.append("## Probe on L*+1 (layer after last injection)\n")
    lines.append("| Condition | Single-layer | Per-layer W | Same W |")
    lines.append("|-----------|-------------|-------------|--------|")
    lines.append(f"| original | {single['probe']['original']:.4f} | {per_layer_probe[0]:.4f} | {same_W_probe[0]:.4f} |")
    lines.append(f"| patched  | {single['probe']['patched']:.4f} | {per_layer_probe[1]:.4f} | {same_W_probe[1]:.4f} |")
    lines.append("")

    verdict = ""
    if best_p > best_s + 0.02:
        verdict = "Multi-layer injection BEATS single-layer: cumulative reinforcement works."
    elif best_p < best_s - 0.02:
        verdict = "Multi-layer injection HURTS: too many injections overwhelm Phi-2."
    else:
        verdict = "Multi-layer injection ≈ single-layer: diminishing returns."
    if best_sw > best_p + 0.01:
        verdict += "\nSame-W ablation beats per-layer W: layer-specific alignment is not needed."
    elif best_p > best_sw + 0.01:
        verdict += "\nPer-layer W beats same-W ablation: layer-specific alignment matters."
    else:
        verdict += "\nPer-layer W ≈ same-W: the specific projection per layer does not matter."

    lines.append(f"## Verdict\n{verdict}\n")

    text = "\n".join(lines)
    with open(f"{OUT_DIR}/experiment_summary.md", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
