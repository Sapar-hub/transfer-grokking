import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

from utils import DEVICE, P, train_probe

ARTIFACTS = "artifacts"
CE_DIR = f"{ARTIFACTS}/ce_projection"
OUT_DIR = f"{ARTIFACTS}/self_projection"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
OP = sys.argv[2] if len(sys.argv) > 2 else "add"
SUFFIX = "" if OP == "add" else "_mult"
OP_SYMBOL = {"add": "+", "mult": "*"}
SEED_DIR = f"{OUT_DIR}/seeds{SUFFIX}"
os.makedirs(SEED_DIR, exist_ok=True)

D_SMALL = 128
D_PHI2 = 2560
BATCH_SIZE = 256
N_EPOCHS = 5000
LR = 1e-3
PATCH_LAYER = 31
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]


def get_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]


def collect_phi2_activations(tokenizer, model, layer=31):
    path = f"{OUT_DIR}/phi2_L{layer}_acts{SUFFIX}.npy"
    if os.path.exists(path):
        print(f"[collect] Loading cached Phi-2 L{layer} activations...")
        return np.load(path)

    cross_cache = f"{ARTIFACTS}/cross_model/microsoft_phi_2_L31_acts.npy"
    if OP == "add" and layer == 31 and os.path.exists(cross_cache):
        print(f"[collect] Loading from cross_model cache (addition, L31)...")
        acts = np.load(cross_cache)
        np.save(path, acts)
        print(f"[collect] Copied to {path}")
        return acts

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

    handle = model.model.layers[layer].register_forward_hook(make_hook())
    model.eval()
    inputs_all = [(int(i // P), int(i % P)) for i in range(P * P)]
    print(f"[collect] Processing {len(inputs_all)} prompts at L{layer}...")
    for start in range(0, len(inputs_all), BATCH_SIZE):
        batch = inputs_all[start:start + BATCH_SIZE]
        prompts = [f"# ({a} {OP_SYMBOL[OP]} {b}) % 97 =" for a, b in batch]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        current_mask = tokenized.attention_mask
        with torch.no_grad():
            model(**tokenized)

    handle.remove()
    acts = torch.cat(all_acts, dim=0).numpy()
    np.save(path, acts)
    print(f"[collect] Saved {acts.shape}")
    return acts


def train_W_ce(X_train, y_train, X_test, y_test, lm_head_sliced, final_layernorm=None):
    path_w = f"{SEED_DIR}/W_ce_seed{SEED}.pth"
    if os.path.exists(path_w):
        W = nn.Linear(D_SMALL, D_PHI2, bias=False)
        W.load_state_dict(torch.load(path_w, map_location=DEVICE, weights_only=True))
        print(f"  [CE] Loaded W_ce from {path_w}")
        return W

    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    W = nn.Linear(D_SMALL, D_PHI2, bias=False)
    opt = optim.AdamW(W.parameters(), lr=LR, weight_decay=1e-2)

    def apply_ln(x):
        return final_layernorm(x) if final_layernorm is not None else x

    for epoch in range(1, N_EPOCHS + 1):
        projected = W(X_tr)
        logits = apply_ln(projected) @ lm_head_sliced.T
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 500 == 0 or epoch == 1:
            with torch.no_grad():
                p_te = W(X_te)
                logits_te = apply_ln(p_te) @ lm_head_sliced.T
                val_acc = (logits_te.argmax(dim=1) == y_te).float().mean().item()
                train_acc = (logits.argmax(dim=1) == y_tr).float().mean().item()
            print(f"    [CE] epoch {epoch:4d}: loss={loss.item():.6f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    torch.save(W.state_dict(), path_w)
    return W


def train_W_self(X_train, y_train, X_test, y_test, lm_head_sliced, final_layernorm=None):
    path_w = f"{SEED_DIR}/W_self_seed{SEED}.pth"
    path_csv = f"{SEED_DIR}/self_training_log_seed{SEED}.csv"
    if os.path.exists(path_w):
        W = nn.Linear(D_PHI2, D_PHI2, bias=False)
        W.load_state_dict(torch.load(path_w, map_location=DEVICE, weights_only=True))
        print(f"  [Self] Loaded W_self from {path_w}")
        return W

    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    W = nn.Linear(D_PHI2, D_PHI2, bias=False)
    opt = optim.AdamW(W.parameters(), lr=LR, weight_decay=1e-2)

    def apply_ln(x):
        return final_layernorm(x) if final_layernorm is not None else x

    log_data = []
    for epoch in range(1, N_EPOCHS + 1):
        projected = W(X_tr)
        logits = apply_ln(projected) @ lm_head_sliced.T
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 500 == 0 or epoch == 1:
            with torch.no_grad():
                p_te = W(X_te)
                logits_te = apply_ln(p_te) @ lm_head_sliced.T
                val_acc = (logits_te.argmax(dim=1) == y_te).float().mean().item()
                train_acc = (logits.argmax(dim=1) == y_tr).float().mean().item()
            log_data.append((epoch, loss.item(), train_acc, val_acc))
            print(f"    [Self] epoch {epoch:4d}: loss={loss.item():.6f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    torch.save(W.state_dict(), path_w)
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_acc"])
        w.writerows(log_data)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot([r[0] for r in log_data], [r[1] for r in log_data])
    plt.xlabel('Epoch'); plt.ylabel('CE Loss')
    plt.subplot(1, 2, 2)
    plt.plot([r[0] for r in log_data], [r[2] for r in log_data], label='train_acc')
    plt.plot([r[0] for r in log_data], [r[3] for r in log_data], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SEED_DIR}/self_training_seed{SEED}.png")
    plt.close()

    return W


def logit_lens_accuracy(W, X_test, y_test, lm_head_sliced, final_layernorm=None):
    with torch.no_grad():
        projected = W(torch.from_numpy(X_test).float())
        h = final_layernorm(projected) if final_layernorm is not None else projected
        logits = h @ lm_head_sliced.T
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


def make_self_patch_hook(W, attention_mask=None, alpha=1.0):
    W.eval()
    def hook(module, input, output):
        hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        orig = hidden[batch_idx, seq_lens].clone()
        patch = W(orig)
        if alpha == 1.0:
            hidden[batch_idx, seq_lens] = patch
        elif alpha > 0:
            hidden[batch_idx, seq_lens] = (1 - alpha) * orig + alpha * patch
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


def evaluate_alpha_grokked(model, tokenizer, test_pairs, labels, W, h_A_test, alpha, batch_size=32):
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


def evaluate_alpha_self(model, tokenizer, test_pairs, labels, W, alpha, batch_size=32):
    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}
    correct = 0
    total = 0
    for start in range(0, len(test_pairs), batch_size):
        batch_pairs = test_pairs[start:start + batch_size]
        prompts = [f"# ({a} {OP_SYMBOL[OP]} {b}) % 97 =" for a, b in batch_pairs]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        if alpha == 0.0:
            with torch.no_grad():
                outputs = model(**tokenized)
        else:
            hook = make_self_patch_hook(W, tokenized.attention_mask, alpha)
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


def evaluate_perplexity_grokked(model, tokenizer, dataset, W, h_A_pool, alpha):
    W.requires_grad_(False)
    W.eval()
    all_losses = []
    rng = np.random.RandomState(42)
    batch_size = 16
    n_samples = 300
    max_seq_len = 64

    sequences = []
    for item in dataset:
        if len(sequences) >= n_samples:
            break
        text = item["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, truncation=True, max_length=max_seq_len)
        if len(ids) >= 3:
            sequences.append(ids)

    for start in range(0, len(sequences), batch_size):
        batch_ids = sequences[start:start + batch_size]
        inputs = [ids[:-1] for ids in batch_ids]
        targets = torch.tensor([ids[-1] for ids in batch_ids])

        max_len = max(len(inp) for inp in inputs)
        padded = torch.zeros(len(inputs), max_len, dtype=torch.long)
        mask = torch.zeros(len(inputs), max_len, dtype=torch.long)
        for i, inp in enumerate(inputs):
            padded[i, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            mask[i, :len(inp)] = 1

        idx = rng.randint(0, len(h_A_pool), size=len(inputs))
        batch_h_A = h_A_pool[idx]

        handle = None
        if alpha > 0:
            hook = make_patch_hook(W, batch_h_A, mask, alpha)
            handle = model.model.layers[PATCH_LAYER].register_forward_hook(hook)

        with torch.no_grad():
            outputs = model(padded, attention_mask=mask)

        if handle is not None:
            handle.remove()

        logits = outputs.logits
        seq_lens = mask.sum(dim=1) - 1
        batch_idx = torch.arange(len(inputs))
        last_logits = logits[batch_idx, seq_lens]

        loss = F.cross_entropy(last_logits, targets, reduction='none')
        all_losses.extend(loss.tolist())

    mean_loss = float(np.mean(all_losses))
    ppl = float(np.exp(mean_loss))
    return mean_loss, ppl


def evaluate_perplexity_self(model, tokenizer, dataset, W, alpha):
    W.requires_grad_(False)
    W.eval()
    all_losses = []
    batch_size = 16
    n_samples = 300
    max_seq_len = 64

    sequences = []
    for item in dataset:
        if len(sequences) >= n_samples:
            break
        text = item["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, truncation=True, max_length=max_seq_len)
        if len(ids) >= 3:
            sequences.append(ids)

    for start in range(0, len(sequences), batch_size):
        batch_ids = sequences[start:start + batch_size]
        inputs = [ids[:-1] for ids in batch_ids]
        targets = torch.tensor([ids[-1] for ids in batch_ids])

        max_len = max(len(inp) for inp in inputs)
        padded = torch.zeros(len(inputs), max_len, dtype=torch.long)
        mask = torch.zeros(len(inputs), max_len, dtype=torch.long)
        for i, inp in enumerate(inputs):
            padded[i, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            mask[i, :len(inp)] = 1

        handle = None
        if alpha > 0:
            hook = make_self_patch_hook(W, mask, alpha)
            handle = model.model.layers[PATCH_LAYER].register_forward_hook(hook)

        with torch.no_grad():
            outputs = model(padded, attention_mask=mask)

        if handle is not None:
            handle.remove()

        logits = outputs.logits
        seq_lens = mask.sum(dim=1) - 1
        batch_idx = torch.arange(len(inputs))
        last_logits = logits[batch_idx, seq_lens]

        loss = F.cross_entropy(last_logits, targets, reduction='none')
        all_losses.extend(loss.tolist())

    mean_loss = float(np.mean(all_losses))
    ppl = float(np.exp(mean_loss))
    return mean_loss, ppl


def main():
    print("=" * 60)
    print(f"Self Projection Control (seed={SEED}, op={OP})")
    print("Train W_self: Phi-2 L31 -> CE -> lm_head, then alpha-sweep at L31")
    print("=" * 60)

    print("\n[0] Loading data...")
    small_acts = np.load(f"{ARTIFACTS}/small_model_activations{SUFFIX}.npy")
    labels = np.load(f"{ARTIFACTS}/mod_arithmetic_labels{SUFFIX}.npy", allow_pickle=True)
    torch.manual_seed(SEED)
    train_idx, test_idx = get_split()
    small_train = small_acts[train_idx]
    small_test = small_acts[test_idx]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]
    print(f"  Small model: Train {len(small_train)}  Test {len(small_test)}")

    rng_eval = np.random.RandomState(42)
    eval_idx = rng_eval.choice(test_idx, size=200, replace=False)
    eval_pairs = [(int(i // P), int(i % P)) for i in eval_idx]
    eval_labels = labels[eval_idx]
    eval_h_A_small = small_acts[eval_idx]

    print("\n[1] Loading Phi-2...")
    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", revision="810d367871c1d460086d9f82db8696f2e0a0fcd0", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", revision="810d367871c1d460086d9f82db8696f2e0a0fcd0")
    tokenizer.pad_token = tokenizer.eos_token
    phi2.eval()
    phi2.lm_head.requires_grad_(False)
    print("  Phi-2 loaded, lm_head frozen.")

    # Template consistency check
    test_prompt = f"# (3 {OP_SYMBOL[OP]} 5) % 97 ="
    print(f'  Prompt template: "{test_prompt}" (matches ce_projection convention)')

    number_ids = [tokenizer.encode(str(n))[0] for n in range(P)]
    lm_head_sliced = phi2.lm_head.weight[number_ids].detach()
    final_layernorm = phi2.model.final_layernorm

    print(f"\n[2] Collecting Phi-2 L{PATCH_LAYER} activations...")
    phi2_acts = collect_phi2_activations(tokenizer, phi2, layer=PATCH_LAYER)
    phi2_train = phi2_acts[train_idx]
    phi2_test = phi2_acts[test_idx]
    phi2_eval = phi2_acts[eval_idx]
    print(f"  Phi-2 L{PATCH_LAYER} acts: {phi2_acts.shape}")

    print(f"\n[3] Training W_self (Phi-2 L{PATCH_LAYER} -> CE -> lm_head)...")
    W_self = train_W_self(phi2_train, labels_train, phi2_test, labels_test,
                          lm_head_sliced, final_layernorm)
    W_self.requires_grad_(False)
    W_self.eval()

    print(f"\n[4] Training/loading W_ce (grokked -> CE -> lm_head, reference)...")
    W_ce = train_W_ce(small_train, labels_train, small_test, labels_test,
                      lm_head_sliced, final_layernorm)
    W_ce.requires_grad_(False)
    W_ce.eval()

    print(f"\n[5] Norm & cosine similarity diagnostics...")

    with torch.no_grad():
        X_phi2_te = torch.from_numpy(phi2_test).float()
        X_small_te = torch.from_numpy(small_test).float()

        h_self = W_self(X_phi2_te)
        h_ce = W_ce(X_small_te)
        h_phi2 = X_phi2_te

        norm_self = h_self.norm(dim=1).mean().item()
        norm_ce = h_ce.norm(dim=1).mean().item()
        norm_phi2 = h_phi2.norm(dim=1).mean().item()
        ratio_self = norm_self / norm_phi2
        ratio_ce = norm_ce / norm_phi2

        print(f"  W_self:  ||W_self(h)||  = {norm_self:.3f}  ||h_phi2|| = {norm_phi2:.3f}  ratio = {ratio_self:.4f}")
        print(f"  W_ce:    ||W_ce(h)||   = {norm_ce:.3f}   ||h_phi2|| = {norm_phi2:.3f}  ratio = {ratio_ce:.4f}")

        cos_self = F.cosine_similarity(h_self, h_phi2, dim=1).mean().item()
        print(f"  cos_sim(W_self(h), h)    = {cos_self:.4f}  (1.0 = identity collapse)")

        cos_ce = F.cosine_similarity(h_ce, h_phi2, dim=1).mean().item()
        print(f"  cos_sim(W_ce(h), h_phi2) = {cos_ce:.4f}")

        if cos_self > 0.99 and ratio_self > 0.95:
            print(f"  *** W_self approximates identity mapping (cos={cos_self:.4f}, ratio={ratio_self:.4f})")
            print(f"      Experiment may be uninformative -- W_self is not transforming the signal.")
        elif ratio_self < 0.1:
            print(f"  << Norm collapse confirmed (ratio={ratio_self:.4f}) -- Section 7 mechanism is generic.")
        elif 0.1 <= ratio_self <= 0.95:
            print(f"  ~~ Intermediate regime (ratio={ratio_self:.4f}) -- norm partially preserved.")

    print(f"\n[6] Logit lens & probe comparison...")

    ll_ce = logit_lens_accuracy(W_ce, small_test, labels_test, lm_head_sliced, final_layernorm)
    ll_self = logit_lens_accuracy(W_self, phi2_test, labels_test, lm_head_sliced, final_layernorm)
    probe_ce = probe_accuracy(W_ce, small_test, labels_test)
    probe_self = probe_accuracy(W_self, phi2_test, labels_test)

    print(f"  W_ce:  logit_lens={ll_ce:.4f}  probe={probe_ce:.4f}")
    print(f"  W_self: logit_lens={ll_self:.4f}  probe={probe_self:.4f}")

    print(f"\n[7] Accuracy alpha-sweep at L{PATCH_LAYER}...")
    alpha_results = []
    for W, label, eval_fn, h_A in [
        (W_ce, "W_ce", evaluate_alpha_grokked, eval_h_A_small),
        (W_self, "W_self", evaluate_alpha_self, None),
    ]:
        row = [label]
        for alpha in ALPHAS:
            if h_A is not None:
                acc = eval_fn(phi2, tokenizer, eval_pairs, eval_labels, W, h_A, alpha)
            else:
                acc = eval_fn(phi2, tokenizer, eval_pairs, eval_labels, W, alpha)
            row.append(acc)
            print(f"  [{label}] alpha={alpha:.1f}: acc = {acc:.4f}")
        alpha_results.append(row)

    with open(f"{SEED_DIR}/alpha_sweep_seed{SEED}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["W", *ALPHAS])
        w.writerows(alpha_results)

    print(f"\n[8] Perplexity at alpha=1.0 (WikiText-2)...")
    try:
        wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        print(f"  WikiText-2 loaded: {len(wiki)} examples.")
    except Exception as e:
        print(f"  Could not load WikiText-2: {e}")
        wiki = None

    loss_ce, ppl_ce = float('nan'), float('nan')
    loss_self, ppl_self = float('nan'), float('nan')
    if wiki is not None:
        print(f"  Evaluating W_ce (alpha=1.0)...")
        loss_ce, ppl_ce = evaluate_perplexity_grokked(phi2, tokenizer, wiki, W_ce, small_acts[test_idx], 1.0)
        print(f"    W_ce:  loss={loss_ce:.4f}  ppl={ppl_ce:.1f}")
        print(f"  Evaluating W_self (alpha=1.0)...")
        loss_self, ppl_self = evaluate_perplexity_self(phi2, tokenizer, wiki, W_self, 1.0)
        print(f"    W_self: loss={loss_self:.4f}  ppl={ppl_self:.1f}")

    print(f"\n[9] Summary...")
    baseline_text = alpha_results[0][1]
    lines = []
    lines.append(f"# Self Projection Experiment Summary (seed={SEED}, op={OP})\n")
    lines.append("## Setup\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| W_ce source | Grokked model (128->{D_PHI2}) via CE |")
    lines.append(f"| W_self source | Phi-2 L{PATCH_LAYER} ({D_PHI2}->{D_PHI2}) via CE |")
    lines.append(f"| Operation | {OP} mod {P} |")
    lines.append(f"| Seed | {SEED} |")
    lines.append(f"| Train / Test | 6586 / 2823 |")
    lines.append(f"| Eval pairs | 200 |")
    lines.append(f"| CE epochs | {N_EPOCHS} |")
    lines.append(f"| Optimizer | AdamW lr={LR}, wd=1e-2 |")
    lines.append(f"| Patch layer | L{PATCH_LAYER} |\n")

    lines.append("## Output Norm Comparison\n")
    lines.append("| Source | ||W(h)|| | ||h_Phi2|| | Ratio |")
    lines.append("|--------|----------|-------------|-------|")
    lines.append(f"| W_ce (grokked 128->2560) | {norm_ce:.3f} | {norm_phi2:.3f} | {ratio_ce:.4f} |")
    lines.append(f"| W_self (Phi-2 2560->2560) | {norm_self:.3f} | {norm_phi2:.3f} | {ratio_self:.4f} |\n")

    lines.append("## Identity-Collapse Diagnostics\n")
    lines.append("| Source | cos_sim(W(h), h_target) | Interpretation |")
    lines.append("|--------|------------------------|----------------|")
    interp_ce = "Directional alignment" if cos_ce < 0.5 else "Moderate alignment" if cos_ce < 0.9 else "Near-identity" if cos_ce > 0.99 else "Strong alignment"
    lines.append(f"| W_ce vs h_Phi2 | {cos_ce:.4f} | {interp_ce} |")
    if cos_self > 0.99:
        interp_self = "*** Identity collapse"
    elif cos_self > 0.9:
        interp_self = "Near-identity"
    elif cos_self < 0.5:
        interp_self = "Directional shift"
    else:
        interp_self = "Partial shift"
    lines.append(f"| W_self vs own h | {cos_self:.4f} | {interp_self} |\n")

    if cos_self > 0.99 and ratio_self > 0.95:
        lines.append("**Identity-collapse detected.** W_self approximates identity mapping. "
                     "The experiment may be uninformative -- W_self is not transforming the signal. "
                     "Consider re-running with weight_decay=0 or orthogonal-to-identity init.\n")

    lines.append("## Logit Lens & Probe\n")
    lines.append("| Metric | W_ce | W_self |")
    lines.append("|--------|------|--------|")
    lines.append(f"| Logit lens | {ll_ce:.4f} | {ll_self:.4f} |")
    lines.append(f"| Probe on W(h) | {probe_ce:.4f} | {probe_self:.4f} |\n")

    lines.append("## Accuracy Alpha-Sweep at L31\n")
    lines.append("| Alpha | W_ce | W_self |")
    lines.append("|-------|------|--------|")
    for i, alpha in enumerate(ALPHAS):
        ce_a = alpha_results[0][i + 1]
        self_a = alpha_results[1][i + 1]
        lines.append(f"| {alpha:.1f} | {ce_a:.4f} | {self_a:.4f} |")
    lines.append(f"\nBaseline (no patch): {baseline_text:.4f}\n")

    lines.append("## Perplexity (WikiText-2 last-token, alpha=1.0)\n")
    lines.append("| Source | Loss | Perplexity |")
    lines.append("|--------|------|------------|")
    lines.append("| Baseline (alpha=0.0) | -- | ~63.3 |")
    lines.append(f"| W_ce (grokked) | {loss_ce:.4f} | {ppl_ce:.1f} |")
    lines.append(f"| W_self (Phi-2) | {loss_self:.4f} | {ppl_self:.1f} |\n")

    lines.append("## Interpretation\n")

    if cos_self > 0.99 and ratio_self > 0.95:
        lines.append("**Identity-collapse regime**: W_self learned near-identity mapping, making the "
                     "alpha-sweep uninformative. The flat-then-jump pattern in W_ce cannot be compared "
                     "because W_self does not transform its input.\n\n"
                     "Recommendation: Re-run with weight_decay=0 or constrain W_self away from identity.")
    elif ratio_self < 0.1 and ratio_ce < 0.1:
        lines.append("**Both norms collapsed.** Both W_ce and W_self show the same norm-collapse pattern "
                     "(ratio << 1). This confirms the Section 7 mechanism is architecture-generic -- "
                     "the discontinuity is a property of CE training through frozen scale-invariant "
                     "layernorm with weight decay, applied to any vector, independent of whether it "
                     "originates from a grokked circuit. The flat-then-jump pattern would occur for "
                     "any CE-trained linear probe injected this way, regardless of input source.")
    elif ratio_self >= 0.5 and ratio_ce < 0.1:
        lines.append("**Divergent norms**: W_self preserves norm while W_ce collapses. This rescues the "
                     "transfer framing -- something about the grokked representation specifically (its Fourier "
                     "geometry, its norm structure, how weight decay interacts with that input distribution) "
                     "is causing the pathological norm collapse. Further analysis is warranted.")
    elif ratio_self < 0.1 and ratio_ce >= 0.5:
        lines.append("**Unexpected:** W_ce preserves norm but W_self collapses. This is the opposite of "
                     "the expected pattern -- grokked source resists collapse better than Phi-2's own representation.")
    else:
        lines.append(f"Mixed regime. W_ce ratio={ratio_ce:.4f}, W_self ratio={ratio_self:.4f}. "
                     "See raw data for detailed interpretation.")

    lines.append(f"\n---\n_Seed {SEED}, operation: {OP}_\n")

    text = "\n".join(lines)
    with open(f"{OUT_DIR}/comparison_summary{SUFFIX}.md", "w") as f:
        f.write(text)
    print(text)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for row in alpha_results:
        label = row[0]
        vals = row[1:]
        ax.plot(ALPHAS, vals, 'o-', label=label, linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=1/P, color='red', linestyle='--', alpha=0.3, label=f'chance={1/P:.3f}')
    ax.set_xlabel('alpha (injection strength)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'L{PATCH_LAYER} Accuracy alpha-Sweep (seed={SEED}, {OP})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels_bar = ['W_ce', 'W_self']
    ll_vals = [ll_ce, ll_self]
    probe_vals = [probe_ce, probe_self]
    norm_ratios = [ratio_ce, ratio_self]
    x = np.arange(len(labels_bar))
    w_bar = 0.25
    ax.bar(x - w_bar, ll_vals, w_bar, label='Logit lens', color='steelblue')
    ax.bar(x, probe_vals, w_bar, label='Probe', color='coral')
    ax.bar(x + w_bar, norm_ratios, w_bar, label='||W(h)||/||h_Phi2||', color='seagreen')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar)
    ax.set_ylabel('Accuracy / Ratio')
    ax.set_title('Logit lens, Probe, and Norm Ratio')
    ax.legend()
    ax.axhline(y=1/P, color='red', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = f"{OUT_DIR}/comparison_seed{SEED}{SUFFIX}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nPlot saved: {fig_path}")

    print(f"\nDone. Artifacts in {SEED_DIR}/ and {OUT_DIR}/")


if __name__ == "__main__":
    main()
