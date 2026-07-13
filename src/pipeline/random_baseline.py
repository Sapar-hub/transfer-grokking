import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

from utils import DEVICE, P, train_probe
from model import SmallTransformer

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/random_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
OP = sys.argv[2] if len(sys.argv) > 2 else "add"
SUFFIX = "" if OP == "add" else "_mult"
OP_SYMBOL = {"add": "+", "mult": "*"}
PARTIAL = "--partial" in sys.argv  # skip Phi-2 loading entirely, use caches
LOGIT_LENS_ONLY = PARTIAL or "--logit-lens-only" in sys.argv  # skip L31 alpha sweep

D_SMALL = 128
D_WIDE = 256
D_PHI2 = 2560
D_ONEHOT = 194
BATCH_SIZE = 256
N_EPOCHS_CE = 2000
LR = 1e-3
PATCH_LAYER = 31
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]


def get_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]


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


def make_patch_hook(W, h_A_batch, attention_mask=None, alpha=1.0, adapter=None):
    W.eval()
    if adapter is not None:
        adapter.eval()
    with torch.no_grad():
        h = torch.from_numpy(h_A_batch).float()
        if adapter is not None:
            h = adapter(h)
        patch = W(h)
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


def evaluate_alpha(model, tokenizer, test_pairs, labels, W, h_A_test, alpha,
                   adapter=None, batch_size=32):
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
            hook = make_patch_hook(W, batch_h_A, tokenized.attention_mask, alpha, adapter)
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


def cache_random_activations(seed, batch_size=256):
    path = f"{OUT_DIR}/random_acts_seed{seed}{SUFFIX}.npy"
    if os.path.exists(path):
        print(f"  [rand] Loading cached activations...")
        return np.load(path)

    torch.manual_seed(seed)
    rand_model = SmallTransformer()
    rand_model.eval()

    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    inputs = torch.stack([a, b], dim=1)

    all_acts = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            x = inputs[i:i + batch_size]
            _, acts = rand_model(x, return_activations=True)
            batch_acts = acts["blocks.1.hook_resid_post"][:, 1, :].numpy()
            all_acts.append(batch_acts)

    acts = np.concatenate(all_acts, axis=0)
    np.save(path, acts)
    print(f"  [rand] Saved {acts.shape} (seed={seed})")
    return acts


def train_onehot_end_to_end(X_train, y_train, X_test, y_test, lm_head_sliced,
                            n_epochs=N_EPOCHS_CE, lr=1e-3, final_layernorm=None):
    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    def apply_ln(x):
        return final_layernorm(x) if final_layernorm is not None else x

    W_oh = nn.Linear(D_ONEHOT, D_SMALL, bias=False)
    W_ce = nn.Linear(D_SMALL, D_PHI2, bias=False)

    params = list(W_oh.parameters()) + list(W_ce.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-2)

    for epoch in range(1, n_epochs + 1):
        h = W_oh(X_tr)
        projected = W_ce(h)
        logits = apply_ln(projected) @ lm_head_sliced.T
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 500 == 0 or epoch == 1:
            with torch.no_grad():
                h_te = W_oh(X_te)
                proj_te = W_ce(h_te)
                logits_te = apply_ln(proj_te) @ lm_head_sliced.T
                val_acc = (logits_te.argmax(dim=1) == y_te).float().mean().item()
                train_acc = (logits.argmax(dim=1) == y_tr).float().mean().item()
            chance = np.log(P)
            print(f"    [onehot] epoch {epoch:4d}: loss={loss.item():.6f} (chance={chance:.3f}) train={train_acc:.4f} val={val_acc:.4f}")
            if val_acc > 0.99 and train_acc > 0.99:
                print(f"    [onehot] Early stop at epoch {epoch} (val={val_acc:.4f})")
                break

    # Mode collapse check
    with torch.no_grad():
        all_logits = apply_ln(W_ce(W_oh(X_te))) @ lm_head_sliced.T
        preds = all_logits.argmax(dim=1)
        unique, counts = preds.unique(return_counts=True)
        top_k = min(10, len(unique))
        top_idx = counts.argsort(descending=True)[:top_k]
        top_str = ", ".join([f"{unique[i].item()}:{counts[i].item()}" for i in top_idx])
        print(f"    [onehot] pred distribution (top-{top_k}): {top_str}")
        print(f"    [onehot] unique classes predicted: {len(unique)} / {P}")

    return W_oh, W_ce


def train_onehot_mlp_end_to_end(X_train, y_train, X_test, y_test, lm_head_sliced,
                                n_epochs=N_EPOCHS_CE, lr=1e-3, final_layernorm=None):
    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    def apply_ln(x):
        return final_layernorm(x) if final_layernorm is not None else x

    W_oh = nn.Linear(D_ONEHOT, D_SMALL, bias=True)
    nonlinear = nn.ReLU()
    W_ce = nn.Linear(D_SMALL, D_PHI2, bias=False)

    params = list(W_oh.parameters()) + list(W_ce.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-2)

    for epoch in range(1, n_epochs + 1):
        h = nonlinear(W_oh(X_tr))
        projected = W_ce(h)
        logits = apply_ln(projected) @ lm_head_sliced.T
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 500 == 0 or epoch == 1:
            with torch.no_grad():
                h_te = nonlinear(W_oh(X_te))
                proj_te = W_ce(h_te)
                logits_te = apply_ln(proj_te) @ lm_head_sliced.T
                val_acc = (logits_te.argmax(dim=1) == y_te).float().mean().item()
                train_acc = (logits.argmax(dim=1) == y_tr).float().mean().item()
            chance = np.log(P)
            print(f"    [onehot-mlp] epoch {epoch:4d}: loss={loss.item():.6f} (chance={chance:.3f}) train={train_acc:.4f} val={val_acc:.4f}")
            if val_acc > 0.99 and train_acc > 0.99:
                print(f"    [onehot-mlp] Early stop at epoch {epoch} (val={val_acc:.4f})")
                break

    with torch.no_grad():
        all_logits = apply_ln(W_ce(nonlinear(W_oh(X_te)))) @ lm_head_sliced.T
        preds = all_logits.argmax(dim=1)
        unique, counts = preds.unique(return_counts=True)
        top_k = min(10, len(unique))
        top_idx = counts.argsort(descending=True)[:top_k]
        top_str = ", ".join([f"{unique[i].item()}:{counts[i].item()}" for i in top_idx])
        print(f"    [onehot-mlp] pred distribution (top-{top_k}): {top_str}")
        print(f"    [onehot-mlp] unique classes predicted: {len(unique)} / {P}")

    return W_oh, nonlinear, W_ce


def train_onehot_wide_end_to_end(X_train, y_train, X_test, y_test, lm_head_sliced,
                                 n_epochs=N_EPOCHS_CE, lr=1e-3, final_layernorm=None):
    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    def apply_ln(x):
        return final_layernorm(x) if final_layernorm is not None else x

    W_oh = nn.Linear(D_ONEHOT, D_WIDE, bias=False)
    W_ce = nn.Linear(D_WIDE, D_PHI2, bias=False)

    params = list(W_oh.parameters()) + list(W_ce.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-2)

    for epoch in range(1, n_epochs + 1):
        h = W_oh(X_tr)
        projected = W_ce(h)
        logits = apply_ln(projected) @ lm_head_sliced.T
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 500 == 0 or epoch == 1:
            with torch.no_grad():
                h_te = W_oh(X_te)
                proj_te = W_ce(h_te)
                logits_te = apply_ln(proj_te) @ lm_head_sliced.T
                val_acc = (logits_te.argmax(dim=1) == y_te).float().mean().item()
                train_acc = (logits.argmax(dim=1) == y_tr).float().mean().item()
            chance = np.log(P)
            print(f"    [onehot-wide] epoch {epoch:4d}: loss={loss.item():.6f} (chance={chance:.3f}) train={train_acc:.4f} val={val_acc:.4f}")
            if val_acc > 0.99 and train_acc > 0.99:
                print(f"    [onehot-wide] Early stop at epoch {epoch} (val={val_acc:.4f})")
                break

    with torch.no_grad():
        all_logits = apply_ln(W_ce(W_oh(X_te))) @ lm_head_sliced.T
        preds = all_logits.argmax(dim=1)
        unique, counts = preds.unique(return_counts=True)
        top_k = min(10, len(unique))
        top_idx = counts.argsort(descending=True)[:top_k]
        top_str = ", ".join([f"{unique[i].item()}:{counts[i].item()}" for i in top_idx])
        print(f"    [onehot-wide] pred distribution (top-{top_k}): {top_str}")
        print(f"    [onehot-wide] unique classes predicted: {len(unique)} / {P}")

    return W_oh, W_ce


def train_random_W_ce(X_train, y_train, X_test, y_test, lm_head_sliced,
                      n_epochs=N_EPOCHS_CE, lr=1e-3, final_layernorm=None):
    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    def apply_ln(x):
        return final_layernorm(x) if final_layernorm is not None else x

    W = nn.Linear(D_SMALL, D_PHI2, bias=False)
    opt = optim.AdamW(W.parameters(), lr=lr, weight_decay=1e-2)

    for epoch in range(1, n_epochs + 1):
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
            print(f"    [W] epoch {epoch:4d}: loss={loss.item():.6f} train={train_acc:.4f} val={val_acc:.4f}")

    return W


def main():
    print("=" * 60)
    print(f"Random Baseline Control (seed={SEED}, op={OP})")
    print("=" * 60)

    print("\n[0] Loading data...")
    small_acts = np.load(f"{ARTIFACTS}/small_model_activations{SUFFIX}.npy")
    labels = np.load(f"{ARTIFACTS}/mod_arithmetic_labels{SUFFIX}.npy", allow_pickle=True)
    train_idx, test_idx = get_split()
    small_train = small_acts[train_idx]
    small_test = small_acts[test_idx]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]
    print(f"  Train: {len(small_train)}  Test: {len(small_test)}")

    torch.manual_seed(SEED)
    rng_eval = np.random.RandomState(42)
    eval_idx = rng_eval.choice(test_idx, size=200, replace=False)
    eval_pairs = [(int(i // P), int(i % P)) for i in eval_idx]
    eval_labels = labels[eval_idx]
    eval_h_A_grokked = small_acts[eval_idx]

    all_pairs = [(int(i // P), int(i % P)) for i in range(P * P)]

    lm_head_cache = f"{ARTIFACTS}/lm_head_sliced.pt"
    ln_cache = f"{ARTIFACTS}/final_layernorm.pt"
    phi2 = None
    tokenizer = None
    partial_mode = PARTIAL

    # In logit-lens-only mode, use cached files if available (avoids Phi-2 load)
    if LOGIT_LENS_ONLY and os.path.exists(lm_head_cache) and os.path.exists(ln_cache):
        partial_mode = True
        print("  [--logit-lens-only] Cached lm_head and layernorm found — skipping Phi-2 load.")

    if not partial_mode:
        try:
            print("\n[1] Loading full Phi-2 model...")
            phi2 = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2", dtype=torch.float32, device_map=None
            )
            tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
            tokenizer.pad_token = tokenizer.eos_token
            phi2.eval()
            phi2.lm_head.requires_grad_(False)
            print("  Phi-2 loaded, lm_head frozen.")

            final_layernorm = phi2.model.final_layernorm
            final_layernorm.requires_grad_(False)
            final_layernorm.eval()
            torch.save(final_layernorm.state_dict(), ln_cache)
            print(f"  Saved final_layernorm.pt cache.")

            number_ids = [tokenizer.encode(str(n))[0] for n in range(P)]
            lm_head_sliced = phi2.lm_head.weight[number_ids].detach()
            torch.save(lm_head_sliced, lm_head_cache)
            print(f"  Saved lm_head_sliced.pt cache.")
        except Exception as e:
            print(f"  Could not load full Phi-2: {e}")
            print("  Falling back to cached lm_head_sliced.pt (logit-lens only)...")
            partial_mode = True

    if partial_mode:
        print("\n[1] Loading cached lm_head and final_layernorm...")
        if not os.path.exists(lm_head_cache):
            print("  Cache not found — loading Phi-2 to generate it...")
            phi2 = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2", dtype=torch.float32, device_map=None
            )
            tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
            tokenizer.pad_token = tokenizer.eos_token
            phi2.eval()
            phi2.lm_head.requires_grad_(False)
            number_ids = [tokenizer.encode(str(n))[0] for n in range(P)]
            lm_head_sliced = phi2.lm_head.weight[number_ids].detach()
            torch.save(lm_head_sliced, lm_head_cache)
            print(f"  Generated and saved lm_head_sliced.pt ({lm_head_sliced.shape}).")

            final_layernorm = phi2.model.final_layernorm
            final_layernorm.requires_grad_(False)
            final_layernorm.eval()
            torch.save(final_layernorm.state_dict(), ln_cache)
            print(f"  Generated and saved final_layernorm.pt.")
            del phi2
        lm_head_sliced = torch.load(lm_head_cache, map_location=DEVICE, weights_only=True).float()
        print(f"  Loaded lm_head_sliced: {lm_head_sliced.shape}")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        tokenizer.pad_token = tokenizer.eos_token

        if os.path.exists(ln_cache):
            final_layernorm = nn.LayerNorm(D_PHI2)
            final_layernorm.load_state_dict(torch.load(ln_cache, map_location=DEVICE, weights_only=True))
            final_layernorm.requires_grad_(False)
            final_layernorm.eval()
            print(f"  Loaded final_layernorm.pt from cache.")
        else:
            print("  WARNING: final_layernorm.pt not found — training without layernorm.")
            final_layernorm = None

    results = {}

    # ==========================================
    # Part A: One-hot control
    # ==========================================
    print("\n" + "=" * 60)
    print("PART A: One-hot encoding control")
    print("=" * 60)

    one_hot_a = np.eye(P, dtype=np.float32)[np.array([p[0] for p in all_pairs])]
    one_hot_b = np.eye(P, dtype=np.float32)[np.array([p[1] for p in all_pairs])]
    one_hot_all = np.concatenate([one_hot_a, one_hot_b], axis=1)
    one_hot_train = one_hot_all[train_idx]
    one_hot_test = one_hot_all[test_idx]
    one_hot_eval = one_hot_all[eval_idx]

    print(f"\n  One-hot shape: {one_hot_all.shape}")

    print("\n[2] Training one-hot → W_CE end-to-end...")
    W_oh, W_ce_oh = train_onehot_end_to_end(
        one_hot_train, labels_train, one_hot_test, labels_test, lm_head_sliced,
        final_layernorm=final_layernorm
    )

    print("\n[3] Logit lens (one-hot)...")
    ll_oh = logit_lens_accuracy(
        lambda x: W_ce_oh(W_oh(x)), one_hot_test, labels_test, lm_head_sliced,
        final_layernorm=final_layernorm
    )
    probe_oh = probe_accuracy(
        lambda x: W_ce_oh(W_oh(x)), one_hot_test, labels_test
    )
    print(f"  Logit lens: {ll_oh:.4f}")
    print(f"  Probe:      {probe_oh:.4f}")

    W_ce_oh.requires_grad_(False)
    W_ce_oh.eval()
    W_oh.requires_grad_(False)
    W_oh.eval()

    oh_results = []
    oh_row = ["onehot", ll_oh, probe_oh]
    if not LOGIT_LENS_ONLY and phi2 is not None:
        print("\n[4] L31 alpha-sweep (one-hot)...")
        for alpha in ALPHAS:
            acc = evaluate_alpha(phi2, tokenizer, eval_pairs, eval_labels,
                                 W_ce_oh, one_hot_eval, alpha, adapter=W_oh)
            oh_row.append(acc)
            print(f"  alpha={alpha:.1f}: acc = {acc:.4f}")
    else:
        oh_row.extend([0.0] * len(ALPHAS))
    oh_results.append(oh_row)

    results["onehot"] = {"ll": ll_oh, "probe": probe_oh, "alpha": oh_row[3:]}

    # ==========================================
    # Part B: Random untrained network
    # ==========================================
    print("\n" + "=" * 60)
    print("PART B: Random untrained network control")
    print("=" * 60)

    rand_seeds = list(range(5))
    for rseed in rand_seeds:
        print(f"\n--- Random seed {rseed} ---")

        print(f"\n[5] Caching random network activations (seed={rseed})...")
        rand_acts = cache_random_activations(rseed)
        rand_train = rand_acts[train_idx]
        rand_test = rand_acts[test_idx]
        rand_eval = rand_acts[eval_idx]

        print(f"\n[6] Training W_CE from random activations (seed={rseed})...")
        W_ce_rand = train_random_W_ce(
            rand_train, labels_train, rand_test, labels_test, lm_head_sliced,
            final_layernorm=final_layernorm
        )

        print(f"\n[7] Logit lens (random, seed={rseed})...")
        ll_rand = logit_lens_accuracy(W_ce_rand, rand_test, labels_test, lm_head_sliced,
                                       final_layernorm=final_layernorm)
        probe_rand = probe_accuracy(W_ce_rand, rand_test, labels_test)
        print(f"  Logit lens: {ll_rand:.4f}")
        print(f"  Probe:      {probe_rand:.4f}")

        W_ce_rand.requires_grad_(False)
        W_ce_rand.eval()

        rand_row = [f"random-{rseed}", ll_rand, probe_rand]
        if not LOGIT_LENS_ONLY and phi2 is not None:
            print(f"\n[8] L31 alpha-sweep (random, seed={rseed})...")
            for alpha in ALPHAS:
                acc = evaluate_alpha(phi2, tokenizer, eval_pairs, eval_labels,
                                     W_ce_rand, rand_eval, alpha)
                rand_row.append(acc)
                print(f"  alpha={alpha:.1f}: acc = {acc:.4f}")
        else:
            rand_row.extend([0.0] * len(ALPHAS))
        oh_results.append(rand_row)

        results[f"random-{rseed}"] = {"ll": ll_rand, "probe": probe_rand, "alpha": rand_row[3:]}

    # ==========================================
    # Part C: One-hot MLP control
    # ==========================================
    print("\n" + "=" * 60)
    print("PART C: One-hot MLP control (194→128→ReLU→2560)")
    print("=" * 60)

    print("\n[9] Training one-hot MLP → W_CE end-to-end...")
    W_oh_mlp, nonlinear_mlp, W_ce_mlp = train_onehot_mlp_end_to_end(
        one_hot_train, labels_train, one_hot_test, labels_test, lm_head_sliced,
        final_layernorm=final_layernorm
    )

    def logit_lens_mlp(X, y):
        with torch.no_grad():
            h = nonlinear_mlp(W_oh_mlp(torch.from_numpy(X).float()))
            projected = W_ce_mlp(h)
            h_ln = final_layernorm(projected) if final_layernorm is not None else projected
            logits = h_ln @ lm_head_sliced.T
            acc = (logits.argmax(dim=1) == torch.from_numpy(y).long()).float().mean().item()
        return acc

    def probe_mlp(X, y):
        with torch.no_grad():
            h = nonlinear_mlp(W_oh_mlp(torch.from_numpy(X).float())).numpy()
        acc, _, _ = train_probe(h, y)
        return acc

    ll_oh_mlp = logit_lens_mlp(one_hot_test, labels_test)
    probe_oh_mlp = probe_mlp(one_hot_test, labels_test)
    print(f"  Logit lens: {ll_oh_mlp:.4f}")
    print(f"  Probe:      {probe_oh_mlp:.4f}")

    W_ce_mlp.requires_grad_(False)
    W_ce_mlp.eval()
    W_oh_mlp.requires_grad_(False)
    W_oh_mlp.eval()

    mlp_adapter = nn.Sequential(W_oh_mlp, nonlinear_mlp)
    mlp_adapter.eval()

    oh_mlp_row = ["onehot-mlp", ll_oh_mlp, probe_oh_mlp]
    if not LOGIT_LENS_ONLY and phi2 is not None:
        print("\n[10] L31 alpha-sweep (one-hot MLP)...")
        for alpha in ALPHAS:
            acc = evaluate_alpha(phi2, tokenizer, eval_pairs, eval_labels,
                                 W_ce_mlp, one_hot_eval, alpha, adapter=mlp_adapter)
            oh_mlp_row.append(acc)
            print(f"  alpha={alpha:.1f}: acc = {acc:.4f}")
    else:
        oh_mlp_row.extend([0.0] * len(ALPHAS))
    oh_results.append(oh_mlp_row)

    results["onehot-mlp"] = {"ll": ll_oh_mlp, "probe": probe_oh_mlp, "alpha": oh_mlp_row[3:]}

    # ==========================================
    # Part D: One-hot wide linear control (194→256→2560)
    # ==========================================
    print("\n" + "=" * 60)
    print("PART D: One-hot wide linear control (194→256→2560)")
    print("=" * 60)

    print("\n[11] Training one-hot wide → W_CE end-to-end...")
    W_oh_wide, W_ce_wide = train_onehot_wide_end_to_end(
        one_hot_train, labels_train, one_hot_test, labels_test, lm_head_sliced,
        final_layernorm=final_layernorm
    )

    ll_oh_wide = logit_lens_accuracy(
        lambda x: W_ce_wide(W_oh_wide(x)), one_hot_test, labels_test, lm_head_sliced,
        final_layernorm=final_layernorm
    )
    probe_oh_wide = probe_accuracy(
        lambda x: W_ce_wide(W_oh_wide(x)), one_hot_test, labels_test
    )
    print(f"  Logit lens: {ll_oh_wide:.4f}")
    print(f"  Probe:      {probe_oh_wide:.4f}")

    W_ce_wide.requires_grad_(False)
    W_ce_wide.eval()
    W_oh_wide.requires_grad_(False)
    W_oh_wide.eval()

    oh_wide_row = ["onehot-wide", ll_oh_wide, probe_oh_wide]
    if not LOGIT_LENS_ONLY and phi2 is not None:
        print("\n[12] L31 alpha-sweep (one-hot wide)...")
        for alpha in ALPHAS:
            acc = evaluate_alpha(phi2, tokenizer, eval_pairs, eval_labels,
                                 W_ce_wide, one_hot_eval, alpha, adapter=W_oh_wide)
            oh_wide_row.append(acc)
            print(f"  alpha={alpha:.1f}: acc = {acc:.4f}")
    else:
        oh_wide_row.extend([0.0] * len(ALPHAS))
    oh_results.append(oh_wide_row)

    results["onehot-wide"] = {"ll": ll_oh_wide, "probe": probe_oh_wide, "alpha": oh_wide_row[3:]}

    # ==========================================
    # Reference: grokked model (recompute for comparison)
    # ==========================================
    print("\n" + "=" * 60)
    print("REFERENCE: Grokked model W_CE")
    print("=" * 60)

    print("\n[13] Loading/training W_CE from grokked activations...")
    cached_w_path = f"{ARTIFACTS}/ce_projection/seeds{SUFFIX}/W_ce_seed{SEED}.pth"
    if os.path.exists(cached_w_path):
        W_ce_grokked = nn.Linear(D_SMALL, D_PHI2, bias=False)
        W_ce_grokked.load_state_dict(torch.load(cached_w_path, map_location=DEVICE, weights_only=True))
        print(f"  Loaded cached W_ce from {cached_w_path}")
    else:
        W_ce_grokked = train_random_W_ce(
            small_train, labels_train, small_test, labels_test, lm_head_sliced,
            n_epochs=2000, final_layernorm=final_layernorm
        )

    ll_grokked = logit_lens_accuracy(W_ce_grokked, small_test, labels_test, lm_head_sliced,
                                      final_layernorm=final_layernorm)
    probe_grokked = probe_accuracy(W_ce_grokked, small_test, labels_test)
    print(f"  Logit lens: {ll_grokked:.4f}")
    print(f"  Probe:      {probe_grokked:.4f}")

    # Mode collapse check (grokked reference)
    with torch.no_grad():
        X_te = torch.from_numpy(small_test).float()
        y_te = torch.from_numpy(labels_test).long()
        h_ln = final_layernorm(W_ce_grokked(X_te)) if final_layernorm is not None else W_ce_grokked(X_te)
        all_logits = h_ln @ lm_head_sliced.T
        preds = all_logits.argmax(dim=1)
        unique, counts = preds.unique(return_counts=True)
        top_k = min(10, len(unique))
        top_idx = counts.argsort(descending=True)[:top_k]
        top_str = ", ".join([f"{unique[i].item()}:{counts[i].item()}" for i in top_idx])
        print(f"    [grokked] pred distribution (top-{top_k}): {top_str}")
        print(f"    [grokked] unique classes predicted: {len(unique)} / {P}")

    W_ce_grokked.requires_grad_(False)
    W_ce_grokked.eval()

    grokked_row = ["grokked", ll_grokked, probe_grokked]
    if not LOGIT_LENS_ONLY and phi2 is not None:
        print("\n[14] L31 alpha-sweep (grokked)...")
        for alpha in ALPHAS:
            acc = evaluate_alpha(phi2, tokenizer, eval_pairs, eval_labels,
                                 W_ce_grokked, eval_h_A_grokked, alpha)
            grokked_row.append(acc)
            print(f"  alpha={alpha:.1f}: acc = {acc:.4f}")
    else:
        grokked_row.extend([0.0] * len(ALPHAS))
    oh_results.append(grokked_row)

    results["grokked"] = {"ll": ll_grokked, "probe": probe_grokked, "alpha": grokked_row[3:]}

    # ==========================================
    # Save results
    # ==========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    header = ["condition", "logit_lens", "probe"] + [f"alpha_{a}" for a in ALPHAS]
    summary_rows = []
    for name, data in results.items():
        row = [name, f"{data['ll']:.4f}", f"{data['probe']:.4f}"] + \
              [f"{v:.4f}" for v in data["alpha"]]
        summary_rows.append(row)
        print(f"  {name:20s} ll={data['ll']:.4f}  probe={data['probe']:.4f}  "
              f"a0.5={data['alpha'][ALPHAS.index(0.5)]:.4f}  a1.0={data['alpha'][-1]:.4f}")

    csv_path = f"{OUT_DIR}/results_seed{SEED}{SUFFIX}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(summary_rows)
    print(f"\n  Saved: {csv_path}")

    md_path = f"{OUT_DIR}/summary_seed{SEED}{SUFFIX}.md"
    lines = [
        f"# Random Baseline Control Results (seed={SEED}, op={OP})\n",
        "## Setup\n",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Training split | 70/30 ({len(small_train)}/{len(small_test)}) |",
        f"| CE epochs | {N_EPOCHS_CE} |",
        f"| Learning rate | {LR} |",
        f"| Patch layer | L{PATCH_LAYER} |",
        f"| Eval pairs | 200 |\n",
        "## Results\n",
        f"| Condition | Logit lens | Probe | " +
        " | ".join([f"α={a}" for a in ALPHAS]) + " |",
        f"|-----------|------------|-------|" + "-|" * len(ALPHAS),
    ]
    for row in oh_results:
        cond = row[0]
        vals = row[1:]
        lines.append(f"| {cond} | " + " | ".join([f"{v:.4f}" for v in vals]) + " |")

    interpretation = []
    ll_oh_val = results["onehot"]["ll"]
    ll_rand_val = results["random-0"]["ll"]
    ll_grokked_val = results["grokked"]["ll"]

    if abs(ll_oh_val - ll_grokked_val) < 0.05 and abs(ll_rand_val - ll_grokked_val) < 0.05:
        interpretation.append(
            "**CRITICAL**: Both one-hot and random controls achieve logit-lens accuracy "
            f"comparable to grokked ({ll_oh_val:.4f} vs {ll_grokked_val:.4f}). "
            "Fourier structure in the grokked model is NOT necessary for CE projection. "
            "The paper's central claim requires reformulation."
        )
    elif ll_oh_val > 0.9 and ll_rand_val < 0.1:
        interpretation.append(
            "One-hot achieves high logit-lens (trivially provides class info) but "
            "random network does not — Fourier structure provides unique signal. "
            "Mixed result: one-hot is a strong baseline but random-init is the cleaner control."
        )
    elif ll_oh_val < 0.1 and ll_rand_val < 0.1:
        interpretation.append(
            "Both controls fail — only the grokked model's structured activations "
            "(which include circular Fourier features) enable CE projection. "
            "This is consistent with the Fourier hypothesis but does not isolate "
            "Fourier structure as the sole causal factor: one-hot adds a trainable "
            "bottleneck, and random activations lack all structure."
        )
    else:
        interpretation.append(
            f"One-hot={ll_oh_val:.4f}, random={ll_rand_val:.4f} — mixed result, "
            "interpretation requires careful analysis of the α-sweep curves."
        )

    lines.append("\n## Interpretation\n")
    lines.append("\n".join(interpretation))
    lines.append(f"\n\n_Generated by random_baseline.py, seed={SEED}, op={OP}_\n")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {md_path}")

    # ==========================================
    # Plot
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for row in oh_results:
        cond = row[0]
        alpha_vals = [float(v) for v in row[3:]]
        ax.plot(ALPHAS, alpha_vals, 'o-', label=cond, linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=1/P, color='red', linestyle='--', alpha=0.3, label='random')
    ax.set_xlabel('alpha')
    ax.set_ylabel('Accuracy')
    ax.set_title('L31 Patch Accuracy: one-hot vs random vs grokked')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels_bar = [r[0] for r in oh_results]
    ll_vals = [results[r[0]]["ll"] for r in oh_results]
    probe_vals = [results[r[0]]["probe"] for r in oh_results]
    x = np.arange(len(labels_bar))
    w_bar = 0.35
    ax.bar(x - w_bar/2, ll_vals, w_bar, label='Logit lens', color='steelblue')
    ax.bar(x + w_bar/2, probe_vals, w_bar, label='Probe', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, rotation=15)
    ax.set_ylabel('Accuracy')
    ax.set_title('Logit lens vs Probe accuracy')
    ax.legend()
    ax.axhline(y=1/P, color='red', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = f"{OUT_DIR}/comparison_seed{SEED}{SUFFIX}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")

    print(f"\nDone. All results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
