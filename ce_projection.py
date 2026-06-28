import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv

from utils import DEVICE, P, train_probe

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/ce_projection"
os.makedirs(OUT_DIR, exist_ok=True)

D_SMALL = 128
D_PHI2 = 2560
BATCH_SIZE = 256
N_EPOCHS = 5000
LR = 1e-3
LAYER = 10
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]


def get_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]


def collect_phi2_activations(tokenizer, model, inputs_list):
    path = f"{OUT_DIR}/phi2_L10_acts.npy"
    if os.path.exists(path):
        print(f"[collect] Loading cached Phi-2 L{LAYER} activations...")
        return np.load(path)

    all_acts = []
    current_mask = None

    def make_hook():
        def hook(module, input, output):
            nonlocal current_mask
            hidden = output[0] if isinstance(output, tuple) else output
            mask = current_mask.to(hidden.device)
            seq_lens = mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            all_acts.append(hidden[batch_idx, seq_lens].detach().cpu())
        return hook

    handle = model.model.layers[LAYER].register_forward_hook(make_hook())
    model.eval()
    print(f"[collect] Processing {len(inputs_list)} prompts at L{LAYER}...")
    for start in range(0, len(inputs_list), BATCH_SIZE):
        batch = inputs_list[start:start + BATCH_SIZE]
        prompts = [f"# ({a} + {b}) % 97 =" for a, b in batch]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        current_mask = tokenized.attention_mask
        with torch.no_grad():
            model(**tokenized)

    handle.remove()
    acts = torch.cat(all_acts, dim=0).numpy()
    np.save(path, acts)
    print(f"[collect] Saved {acts.shape}")
    return acts


def train_W_mse(X_train, X_test, Y_train, Y_test):
    path_w = f"{OUT_DIR}/W_mse.pth"
    path_cos = f"{OUT_DIR}/cos_sim_mse.npy"
    if os.path.exists(path_w) and os.path.exists(path_cos):
        W = nn.Linear(D_SMALL, D_PHI2, bias=False)
        W.load_state_dict(torch.load(path_w, map_location=DEVICE, weights_only=True))
        cos = np.load(path_cos).item()
        print(f"  [MSE] Loaded W. cos_sim={cos:.4f}")
        return W, cos

    X_tr = torch.from_numpy(X_train).float()
    X_te = torch.from_numpy(X_test).float()
    Y_tr = torch.from_numpy(Y_train).float()
    Y_te = torch.from_numpy(Y_test).float()

    W = nn.Linear(D_SMALL, D_PHI2, bias=False)
    opt = optim.AdamW(W.parameters(), lr=LR)
    mse = nn.MSELoss()
    lambda_ortho = 0.01

    train_losses, test_losses, cos_sims = [], [], []
    for epoch in range(1, N_EPOCHS + 1):
        pred = W(X_tr)
        loss = mse(pred, Y_tr)
        WtW = W.weight.T @ W.weight
        I = torch.eye(D_SMALL, device=W.weight.device)
        ortho_loss = torch.norm(WtW - I, p='fro') / D_SMALL
        loss = loss + lambda_ortho * ortho_loss
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
            print(f"    [MSE] epoch {epoch:4d}: train_mse={loss.item():.6f} test_mse={tl:.6f} cos_sim={cs:.4f}")

    torch.save(W.state_dict(), path_w)
    np.save(path_cos, cos_sims[-1])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='train_mse')
    plt.plot(range(len(test_losses)), test_losses, label='test_mse')
    plt.xlabel('Epoch (x500)'); plt.ylabel('MSE'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(len(cos_sims)), cos_sims, 'o-')
    plt.xlabel('Epoch (x500)'); plt.ylabel('Cosine Sim (test)')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/mse_training.png")
    plt.close()

    return W, cos_sims[-1]


def train_W_ce(X_train, y_train, X_test, y_test, lm_head_sliced):
    path_w = f"{OUT_DIR}/W_ce.pth"
    path_csv = f"{OUT_DIR}/ce_training_log.csv"
    if os.path.exists(path_w):
        W = nn.Linear(D_SMALL, D_PHI2, bias=False)
        W.load_state_dict(torch.load(path_w, map_location=DEVICE, weights_only=True))
        print(f"  [CE] Loaded W.")
        return W

    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    W = nn.Linear(D_SMALL, D_PHI2, bias=False)
    opt = optim.AdamW(W.parameters(), lr=LR, weight_decay=1e-2)

    log_data = []
    for epoch in range(1, N_EPOCHS + 1):
        projected = W(X_tr)
        logits = projected @ lm_head_sliced.T
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 500 == 0 or epoch == 1:
            with torch.no_grad():
                p_te = W(X_te)
                logits_te = p_te @ lm_head_sliced.T
                val_acc = (logits_te.argmax(dim=1) == y_te).float().mean().item()
                train_acc = (logits.argmax(dim=1) == y_tr).float().mean().item()
            log_data.append((epoch, loss.item(), train_acc, val_acc))
            print(f"    [CE] epoch {epoch:4d}: train_loss={loss.item():.6f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    torch.save(W.state_dict(), path_w)
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_acc"])
        w.writerows(log_data)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(log_data)), [r[1] for r in log_data])
    plt.xlabel('Epoch (x500)'); plt.ylabel('CE Loss')
    plt.subplot(1, 2, 2)
    plt.plot(range(len(log_data)), [r[2] for r in log_data], label='train_acc')
    plt.plot(range(len(log_data)), [r[3] for r in log_data], label='val_acc')
    plt.xlabel('Epoch (x500)'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ce_training.png")
    plt.close()

    return W


def logit_lens_accuracy(W, X_test, y_test, lm_head_sliced):
    with torch.no_grad():
        projected = W(torch.from_numpy(X_test).float())
        logits = projected @ lm_head_sliced.T
        acc = (logits.argmax(dim=1) == torch.from_numpy(y_test).long()).float().mean().item()
    return acc


def probe_accuracy(W, X_test, y_test):
    with torch.no_grad():
        proj = W(torch.from_numpy(X_test).float()).numpy()
    acc, _, _ = train_probe(proj, y_test)
    return acc


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


def main():
    print("=" * 60)
    print("CE Projection: train W via CrossEntropy through frozen lm_head")
    print("=" * 60)

    print("\n[0] Loading data...")
    small_acts = np.load(f"{ARTIFACTS}/small_model_activations.npy")
    labels = np.load(f"{ARTIFACTS}/mod_arithmetic_labels.npy", allow_pickle=True)
    train_idx, test_idx = get_split()
    small_train = small_acts[train_idx]
    small_test = small_acts[test_idx]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]
    print(f"  Train: {len(small_train)}  Test: {len(small_test)}")

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
    phi2.lm_head.requires_grad_(False)
    print("  Phi-2 loaded, lm_head frozen.")

    number_ids = [tokenizer.encode(str(n))[0] for n in range(P)]
    lm_head_sliced = phi2.lm_head.weight[number_ids].detach()

    print(f"\n[2] Collecting Phi-2 L{LAYER} activations...")
    inputs_all = [(int(i // P), int(i % P)) for i in range(P * P)]
    phi2_acts = collect_phi2_activations(tokenizer, phi2, inputs_all)
    phi2_train = phi2_acts[train_idx]
    phi2_test = phi2_acts[test_idx]

    print(f"\n[3] Training W_MSE (baseline, L{LAYER} targets)...")
    W_mse, cos_mse = train_W_mse(small_train, small_test, phi2_train, phi2_test)

    print(f"\n[4] Training W_CE (via frozen lm_head, no layernorm)...")
    W_ce = train_W_ce(small_train, labels_train, small_test, labels_test, lm_head_sliced)

    print(f"\n[5] Logit lens comparison...")
    ll_mse = logit_lens_accuracy(W_mse, small_test, labels_test, lm_head_sliced)
    ll_ce = logit_lens_accuracy(W_ce, small_test, labels_test, lm_head_sliced)
    print(f"  W_MSE logit lens: {ll_mse:.4f}")
    print(f"  W_CE  logit lens: {ll_ce:.4f}")

    print(f"\n[5b] Cosine sim & probe on W(h) comparison...")
    probe_mse = probe_accuracy(W_mse, small_test, labels_test)
    probe_ce = probe_accuracy(W_ce, small_test, labels_test)
    print(f"  W_MSE probe: {probe_mse:.4f}")
    print(f"  W_CE  probe: {probe_ce:.4f}")
    with torch.no_grad():
        h_ce = W_ce(torch.from_numpy(small_test).float())
        h_phi2 = torch.from_numpy(phi2_test).float()
        cos_ce = nn.functional.cosine_similarity(h_ce, h_phi2, dim=1).mean().item()
    print(f"  W_CE  cos_sim vs L10 targets: {cos_ce:.4f}")

    print(f"\n[6] Alpha sweep (patch at L{LAYER})...")
    alpha_results = []
    for W, label in [(W_mse, "MSE"), (W_ce, "CE")]:
        W.requires_grad_(False)
        W.eval()
        row = [label]
        for alpha in ALPHAS:
            acc = evaluate_alpha(phi2, tokenizer, eval_pairs, eval_labels,
                                 W, eval_h_A, alpha, LAYER)
            row.append(acc)
            print(f"  [{label}] alpha={alpha:.1f}: text_acc = {acc:.4f}")
        alpha_results.append(row)

    with open(f"{OUT_DIR}/alpha_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["W", *ALPHAS])
        w.writerows(alpha_results)

    print(f"\n[7] Summary...")
    baseline_text = alpha_results[0][1]
    lines = []
    lines.append("# CE Projection Experiment Summary\n")
    lines.append("## Setup\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Layer | {LAYER} |")
    lines.append(f"| D_small → D_phi2 | {D_SMALL} → {D_PHI2} |")
    lines.append(f"| Train / Test | 6586 / 2823 |")
    lines.append(f"| W_CE loss | CE through frozen lm_head (no layernorm) |")
    lines.append(f"| W_MSE loss | MSE + 0.01 × ortho |")
    lines.append(f"| Epochs | {N_EPOCHS} |")
    lines.append(f"| Optimizer | AdamW lr={LR} |\n")
    lines.append("## Logit lens & Probe\n")
    lines.append("| Metric | W_MSE | W_CE | Delta |")
    lines.append("|--------|-------|------|-------|")
    lines.append(f"| Cos sim (test) | {cos_mse:.4f} | {cos_ce:.4f} | {cos_ce - cos_mse:+.4f} |")
    lines.append(f"| Logit lens | {ll_mse:.4f} | {ll_ce:.4f} | {ll_ce - ll_mse:+.4f} |")
    lines.append(f"| Probe on W(h) | {probe_mse:.4f} | {probe_ce:.4f} | {probe_ce - probe_mse:+.4f} |\n")
    lines.append(f"## Alpha sweep (text accuracy at L{LAYER})\n")
    lines.append("| Alpha | W_MSE | W_CE | Delta |")
    lines.append("|-------|-------|------|-------|")
    for i, alpha in enumerate(ALPHAS):
        mse_a = alpha_results[0][i + 1]
        ce_a = alpha_results[1][i + 1]
        lines.append(f"| {alpha:.1f} | {mse_a:.4f} | {ce_a:.4f} | {ce_a - mse_a:+.4f} |")
    lines.append(f"\nBaseline (no patch): {baseline_text:.4f}\n")

    if ll_ce > 0.5:
        lines.append("**Logit lens verdict**: W_CE > 0.5 — MSE was the primary barrier.\n")
    elif ll_ce > 0.1:
        lines.append("**Logit lens verdict**: Partial alignment (0.1–0.5) — MSE contributed but is not the only issue.\n")
    else:
        lines.append("**Logit lens verdict**: < 0.1 — incompatibility is deeper than loss choice.\n")

    ce_alpha_05 = alpha_results[1][ALPHAS.index(0.5) + 1]
    mse_alpha_05 = alpha_results[0][ALPHAS.index(0.5) + 1]
    if ll_ce > 0.5 and ce_alpha_05 > mse_alpha_05 + 0.05:
        lines.append("**Alpha sweep verdict**: CE significantly beats MSE at α=0.5 — MSE was the barrier.\n")
    elif ll_ce > 0.5:
        lines.append("**Alpha sweep verdict**: CE aligns with lm_head but text accuracy remains limited — context/geometry conflict persists.\n")
    else:
        lines.append("**Alpha sweep verdict**: CE does not resolve the transfer problem.\n")

    text = "\n".join(lines)
    with open(f"{OUT_DIR}/comparison_summary.md", "w") as f:
        f.write(text)
    print(text)

    print(f"\nDone. Artifacts in {OUT_DIR}/")


if __name__ == "__main__":
    main()
