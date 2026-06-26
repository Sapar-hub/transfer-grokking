import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv

from model import SmallTransformer
from utils import DEVICE, P, generate_all_pairs, train_probe

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/residual_patch"
os.makedirs(OUT_DIR, exist_ok=True)

D_SMALL = 128
D_PHI2 = 2560
BATCH_SIZE = 256
LAYERS = [10, 15, 20, 25, 30]
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]


# ─── Split ─────────────────────────────────────────────────────────────

def get_train_test_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]


# ─── Step 2: Collect Phi-2 activations at all L (single pass) ─────────

def collect_phi2_activations_multi_layer(tokenizer, model, inputs_list, layers):
    path = f"{OUT_DIR}/phi2_activations.npz"
    if os.path.exists(path):
        print("[collect] Loading cached Phi-2 activations...")
        data = np.load(path)
        return {int(k): data[k] for k in data.files}

    activations = {l: [] for l in layers}
    current_mask = None

    handles = []
    for l in layers:
        def make_hook(layer_idx):
            def hook(module, input, output):
                nonlocal current_mask
                hidden = output[0] if isinstance(output, tuple) else output
                mask = current_mask.to(hidden.device)
                seq_lens = mask.sum(dim=1) - 1
                batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
                last_hidden = hidden[batch_idx, seq_lens]
                activations[layer_idx].append(last_hidden.detach().cpu())
            return hook
        handle = model.model.layers[l].register_forward_hook(make_hook(l))
        handles.append(handle)

    model.eval()
    print(f"[collect] Processing {len(inputs_list)} prompts...")
    for start in range(0, len(inputs_list), BATCH_SIZE):
        batch = inputs_list[start:start + BATCH_SIZE]
        prompts = [f"# ({a} + {b}) % 97 =" for a, b in batch]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        current_mask = tokenized.attention_mask
        with torch.no_grad():
            model(**tokenized)

    for h in handles:
        h.remove()

    result = {}
    for l in layers:
        arr = torch.cat(activations[l], dim=0).numpy()
        result[l] = arr
        print(f"  Layer {l}: {arr.shape}")

    np.savez(path, **{str(k): v for k, v in result.items()})
    return result


# ─── Step 3: Train W for a layer ──────────────────────────────────────

def train_W(X_train, X_test, Y_train, Y_test, layer, num_epochs=5000):
    path_w = f"{OUT_DIR}/W_layer{layer}.pth"
    path_cos = f"{OUT_DIR}/cos_sim_layer{layer}.npy"
    path_curve = f"{OUT_DIR}/W_training_layer{layer}.png"

    if os.path.exists(path_w) and os.path.exists(path_cos):
        W = nn.Linear(D_SMALL, D_PHI2, bias=False)
        W.load_state_dict(torch.load(path_w, map_location=DEVICE, weights_only=True))
        cos = np.load(path_cos).item()
        print(f"  [W L{layer}] Loaded. cos_sim={cos:.4f}")
        return W, cos

    X_tr = torch.from_numpy(X_train).float()
    X_te = torch.from_numpy(X_test).float()
    Y_tr = torch.from_numpy(Y_train).float()
    Y_te = torch.from_numpy(Y_test).float()

    W = nn.Linear(D_SMALL, D_PHI2, bias=False)
    opt = optim.AdamW(W.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    lambda_ortho = 0.01
    train_losses, test_losses, cos_sims = [], [], []

    for epoch in range(1, num_epochs + 1):
        pred = W(X_tr)
        loss_mse = mse(pred, Y_tr)
        WtW = W.weight.T @ W.weight
        I = torch.eye(D_SMALL, device=W.weight.device)
        ortho_loss = torch.norm(WtW - I, p='fro') / D_SMALL
        loss = loss_mse + lambda_ortho * ortho_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 500 == 0 or epoch == 1:
            with torch.no_grad():
                p_te = W(X_te)
                tl = mse(p_te, Y_te).item()
                cs = nn.functional.cosine_similarity(p_te, Y_te, dim=1).mean().item()
            train_losses.append(loss.item())
            test_losses.append(tl)
            cos_sims.append(cs)
            print(f"    epoch {epoch:4d}: train_mse={loss.item():.6f} test_mse={tl:.6f} cos_sim={cs:.4f}")

    torch.save(W.state_dict(), path_w)
    np.save(path_cos, cos_sims[-1])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='train_loss')
    plt.plot(range(len(test_losses)), test_losses, label='test_mse')
    plt.xlabel('Epoch (x500)')
    plt.ylabel('MSE')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(len(cos_sims)), cos_sims, 'o-')
    plt.xlabel('Epoch (x500)')
    plt.ylabel('Cosine Sim (test)')
    plt.tight_layout()
    plt.savefig(path_curve)
    plt.close()

    return W, cos_sims[-1]


def evaluate_probe(W, small_acts, labels):
    with torch.no_grad():
        proj = W(torch.from_numpy(small_acts).float()).numpy()
    acc, *_ = train_probe(proj, labels)
    return acc


# ─── Step 4-5: Patch hook and alpha sweep ─────────────────────────────

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


def evaluate_alpha(model, tokenizer, test_pairs, labels, W, h_A_test, alpha, layer, batch_size=32):
    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}
    correct = 0
    total = 0

    for start in range(0, len(test_pairs), batch_size):
        batch_pairs = test_pairs[start:start + batch_size]
        batch_h_A = h_A_test[start:start + batch_size]
        prompts = [f"# ({a} + {b}) % 97 =" for a, b in batch_pairs]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")

        if alpha == 0.0:
            with torch.no_grad():
                outputs = model(**tokenized)
        else:
            hook = make_patch_hook(W, batch_h_A, tokenized.attention_mask, alpha)
            handle = model.model.layers[layer].register_forward_hook(hook)
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


# ─── Logit lens: decode patched residual directly ────────────────────

def logit_lens_accuracy(model, tokenizer, test_pairs, labels, W, h_A_test, alpha, layer):
    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}

    acts = collect_patched_activations(
        model, tokenizer, test_pairs, W, h_A_test,
        patch_layer=layer, collect_layer=layer, alpha=alpha
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


# ─── Step 6: Probe on L+1 ─────────────────────────────────────────────

def collect_patched_activations(model, tokenizer, test_pairs, W, h_A_test, patch_layer, collect_layer, alpha=1.0):
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
            h1 = model.model.layers[patch_layer].register_forward_hook(
                make_patch_hook(W, batch_h_A, tokenized.attention_mask, alpha)
            )
            handles.append(h1)
        h2 = model.model.layers[collect_layer].register_forward_hook(make_collect_hook())
        handles.append(h2)

        with torch.no_grad():
            model(**tokenized)

        for h in handles:
            h.remove()

    return np.concatenate(all_acts, axis=0)


def probe_after_patch(W, h_A_test, labels_test, model, tokenizer, test_pairs, best_layer):
    print(f"  [probe] Layer {best_layer + 1} ...")

    acts_orig = collect_patched_activations(
        model, tokenizer, test_pairs, W, h_A_test,
        patch_layer=best_layer, collect_layer=best_layer + 1, alpha=0.0
    )
    acc_orig, *_ = train_probe(acts_orig, labels_test)
    print(f"    Without patch: probe_acc = {acc_orig:.4f}")

    acts_patched = collect_patched_activations(
        model, tokenizer, test_pairs, W, h_A_test,
        patch_layer=best_layer, collect_layer=best_layer + 1, alpha=1.0
    )
    acc_patched, *_ = train_probe(acts_patched, labels_test)
    print(f"    With patch:    probe_acc = {acc_patched:.4f}")

    return acc_orig, acc_patched


# ─── Summary ──────────────────────────────────────────────────────────

def write_summary(results, logit_lens_results, probe_results, best_layer, cos_sims, probe_accs):
    baseline_acc = results[0][1]
    non_zero = [r for r in results if r[0] > 0]
    best_alpha, best_acc = max(non_zero, key=lambda x: x[1]) if non_zero else (0.0, baseline_acc)
    delta = best_acc - baseline_acc

    lines = []
    lines.append("# Residual Patch Experiment Summary\n")
    lines.append("## Per-layer W training\n")
    lines.append("| Layer | Cos Sim | Probe on W(h_A) |")
    lines.append("|-------|---------|-----------------|")
    for l in LAYERS:
        lines.append(f"| {l} | {cos_sims[l]:.4f} | {probe_accs[l]:.4f} |")
    best_cos_layer = max(cos_sims, key=lambda l: cos_sims[l])
    lines.append(f"\n**By probe**: L*={best_layer} (probe={probe_accs[best_layer]:.4f})")
    lines.append(f"**By cos_sim**: L*={best_cos_layer} (cos_sim={cos_sims[best_cos_layer]:.4f})")
    lines.append(f"**Selected for alpha sweep**: L*={best_layer}\n")
    lines.append("## Alpha sweep (text accuracy)\n")
    lines.append("| Alpha | Accuracy |")
    lines.append("|-------|----------|")
    for alpha, acc in results:
        lines.append(f"| {alpha:.1f} | {acc:.4f} |")
    lines.append(f"\nBaseline (alpha=0.0): {baseline_acc:.4f}")
    lines.append(f"Best (alpha={best_alpha:.1f}): {best_acc:.4f}")
    lines.append(f"Delta: {delta:+.4f}\n")
    lines.append("## Logit lens (direct decode from patched L)\n")
    lines.append("| Alpha | Logit Lens Acc |")
    lines.append("|-------|----------------|")
    for alpha, acc in logit_lens_results:
        lines.append(f"| {alpha:.1f} | {acc:.4f} |")
    lines.append("")
    lines.append("## Probe on L*+1\n")
    lines.append(f"| Condition | Probe Acc |")
    lines.append(f"|-----------|-----------|")
    for cond, acc in probe_results.items():
        lines.append(f"| {cond} | {acc:.4f} |")
    lines.append("")

    if delta > 0.05:
        verdict = "PATCH WORKS: context + correct state activates algorithm"
    elif probe_results.get("patched", 0) > probe_results.get("original", 0) + 0.02:
        verdict = "Geometry changed but output unchanged — Phi-2 ignores patch at decoding"
    else:
        verdict = "No effect — patch does not transfer algorithmic knowledge"
    delta_probe = probe_results.get("patched", 0) - probe_results.get("original", 0)
    verdict += f"\nProbe delta: {delta_probe:+.4f}"

    lines.append(f"## Verdict\n{verdict}\n")

    text = "\n".join(lines)
    with open(f"{OUT_DIR}/experiment_summary.md", "w") as f:
        f.write(text)
    print(text)


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Residual Patch: inject computed state into Phi-2")
    print("=" * 60)

    # ── Load small model activations ──
    print("\n[0] Loading data...")
    small_acts = np.load(f"{ARTIFACTS}/small_model_activations.npy")
    labels = np.load(f"{ARTIFACTS}/mod_arithmetic_labels.npy")
    print(f"  Small activations: {small_acts.shape}")
    print(f"  Labels: {labels.shape}")

    train_idx, test_idx = get_train_test_split()
    small_train = small_acts[train_idx]
    small_test = small_acts[test_idx]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]

    rng_eval = np.random.RandomState(42)
    eval_idx = rng_eval.choice(test_idx, size=200, replace=False)
    eval_pairs = [(int(i // P), int(i % P)) for i in eval_idx]
    eval_labels = labels[eval_idx]
    eval_h_A = small_acts[eval_idx]

    # ── Load Phi-2 (once) ──
    print("\n[1] Loading Phi-2...")
    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    phi2.eval()

    # ── Collect Phi-2 activations (all layers, one pass) ──
    print("\n[2] Collecting Phi-2 activations (all L, one pass)...")
    inputs_all = [(int(i // P), int(i % P)) for i in range(P * P)]
    phi2_acts = collect_phi2_activations_multi_layer(tokenizer, phi2, inputs_all, LAYERS)

    # ── Train W per layer ──
    print("\n[3] Training W per layer...")
    cos_sims = {}
    probe_accs = {}
    best_probe = -1.0
    best_layer = None

    for l in LAYERS:
        print(f"\n  --- Layer {l} ---")
        Y_all = phi2_acts[l]
        Y_train = Y_all[train_idx]
        Y_test = Y_all[test_idx]

        W_l, cos = train_W(small_train, small_test, Y_train, Y_test, l)
        cos_sims[l] = cos

        probe = evaluate_probe(W_l, small_test, labels_test)
        probe_accs[l] = probe
        print(f"  Probe on W(h_A): {probe:.4f}")

        if probe > best_probe:
            best_probe = probe
            best_layer = l

    best_cos_layer = max(cos_sims, key=lambda l: cos_sims[l])
    print(f"\n  By probe: L*={best_layer} ({best_probe:.4f})   By cos_sim: L*={best_cos_layer} ({cos_sims[best_cos_layer]:.4f})")
    print(f"  Using probe-best layer: L*={best_layer}")

    # ── Load best W ──
    W_best = nn.Linear(D_SMALL, D_PHI2, bias=False)
    W_best.load_state_dict(
        torch.load(f"{OUT_DIR}/W_layer{best_layer}.pth", map_location=DEVICE, weights_only=True)
    )
    W_best.requires_grad_(False)

    # ── Alpha sweep ──
    print("\n[4] Alpha sweep...")
    results = []
    for alpha in ALPHAS:
        acc = evaluate_alpha(phi2, tokenizer, eval_pairs, eval_labels,
                             W_best, eval_h_A, alpha, best_layer)
        results.append((alpha, acc))
        print(f"  alpha={alpha:.1f}: mod_acc = {acc:.4f}")

    with open(f"{OUT_DIR}/results_grid.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "mod_accuracy"])
        w.writerows(results)

    # ── Logit lens ──
    print("\n[4b] Logit lens: decode patched residual at L (bypass L+1..31)...")
    logit_lens_results = []
    for alpha in ALPHAS:
        acc = logit_lens_accuracy(phi2, tokenizer, eval_pairs, eval_labels,
                                   W_best, eval_h_A, alpha, best_layer)
        logit_lens_results.append((alpha, acc))
        print(f"  alpha={alpha:.1f}: logit_lens_acc = {acc:.4f}")

    with open(f"{OUT_DIR}/logit_lens.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "logit_lens_accuracy"])
        w.writerows(logit_lens_results)

    # ── Probe on L*+1 ──
    # Use full test set for statistical power
    test_pairs_full = [(int(i // P), int(i % P)) for i in test_idx]
    print(f"\n[5] Probe on L*+1 (layer {best_layer + 1}, {len(test_pairs_full)} pairs)...")
    orig_acc, patched_acc = probe_after_patch(
        W_best, small_test, labels_test, phi2, tokenizer,
        test_pairs_full, best_layer
    )
    probe_data = {"original": orig_acc, "patched": patched_acc}

    with open(f"{OUT_DIR}/probe_before_after.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "probe_acc"])
        w.writerows(probe_data.items())

    # ── Summary ──
    print("\n[6] Writing summary...")
    write_summary(results, logit_lens_results, probe_data, best_layer, cos_sims, probe_accs)

    print(f"\nDone. Artifacts in {OUT_DIR}/")


if __name__ == "__main__":
    main()
