import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, sys, csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import DEVICE, P, train_probe

ARTIFACTS = "artifacts"
D_WIDE = 256
D_PHI2 = 2560
D_ONEHOT = 194
BATCH_SIZE = 256
N_EPOCHS = 5000
LR = 1e-3
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]

SEEDS = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [42]
OP = sys.argv[2] if len(sys.argv) > 2 else "mult"
SUFFIX = "" if OP == "add" else "_mult"
OP_SYMBOL = {"add": "+", "mult": "*"}
OUT_DIR = f"{ARTIFACTS}/random_baseline"
os.makedirs(OUT_DIR, exist_ok=True)


def get_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]


def train_wide(X_train, y_train, X_test, y_test, lm_head_sliced):
    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    W_oh = nn.Linear(D_ONEHOT, D_WIDE, bias=False)
    W_ce = nn.Linear(D_WIDE, D_PHI2, bias=False)
    opt = optim.AdamW(list(W_oh.parameters()) + list(W_ce.parameters()), lr=LR, weight_decay=1e-2)

    for epoch in range(1, N_EPOCHS + 1):
        h = W_oh(X_tr)
        logits = W_ce(h) @ lm_head_sliced.T
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 500 == 0 or epoch == 1:
            with torch.no_grad():
                h_te = W_oh(X_te)
                l_te = W_ce(h_te) @ lm_head_sliced.T
                va = (l_te.argmax(dim=1) == y_te).float().mean().item()
                ta = (logits.argmax(dim=1) == y_tr).float().mean().item()
            print(f"    [wide] epoch {epoch:4d}: loss={loss.item():.6f} train={ta:.4f} val={va:.4f}")
            if va > 0.99 and ta > 0.99:
                break
    return W_oh, W_ce


def logit_lens(W_oh, W_ce, X, y, lm_head_sliced):
    with torch.no_grad():
        logits = W_ce(W_oh(torch.from_numpy(X).float())) @ lm_head_sliced.T
        acc = (logits.argmax(dim=1) == torch.from_numpy(y).long()).float().mean().item()
    return acc


def probe_acc(W_oh, W_ce, X, y):
    with torch.no_grad():
        h = W_ce(W_oh(torch.from_numpy(X).float())).numpy()
    acc, _, _ = train_probe(h, y)
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


def evaluate_alpha(model, tokenizer, test_pairs, labels, W, h_A_test, alpha, adapter=None, batch_size=32):
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
            handle = model.model.layers[31].register_forward_hook(hook)
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
    print(f"Wide Linear Multi-Seed Control (seeds={SEEDS}, op={OP})")
    print("=" * 60)

    print("\n[0] Loading data...")
    labels = np.load(f"{ARTIFACTS}/mod_arithmetic_labels{SUFFIX}.npy", allow_pickle=True)
    _, test_idx = get_split()
    rng_eval = np.random.RandomState(42)
    eval_idx = rng_eval.choice(test_idx, size=200, replace=False)
    eval_pairs = [(int(i // P), int(i % P)) for i in eval_idx]
    eval_labels = labels[eval_idx]

    all_pairs = [(int(i // P), int(i % P)) for i in range(P * P)]
    one_hot_a = np.eye(P, dtype=np.float32)[np.array([p[0] for p in all_pairs])]
    one_hot_b = np.eye(P, dtype=np.float32)[np.array([p[1] for p in all_pairs])]
    one_hot_all = np.concatenate([one_hot_a, one_hot_b], axis=1)
    one_hot_test = one_hot_all[test_idx]
    one_hot_eval = one_hot_all[eval_idx]

    print("\n[1] Loading Phi-2...")
    phi2 = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", dtype=torch.float32, device_map=None)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    phi2.eval()
    number_ids = [tokenizer.encode(str(n))[0] for n in range(P)]
    lm_head_sliced = phi2.lm_head.weight[number_ids].detach()
    print("  Phi-2 loaded.")

    # Training data uses full 70/30 split
    train_idx, test_idx = get_split()
    one_hot_train = one_hot_all[train_idx]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]

    results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)

        W_oh, W_ce = train_wide(one_hot_train, labels_train, one_hot_test, labels_test, lm_head_sliced)

        ll = logit_lens(W_oh, W_ce, one_hot_test, labels_test, lm_head_sliced)
        pr = probe_acc(W_oh, W_ce, one_hot_test, labels_test)
        print(f"  Logit lens: {ll:.4f}  Probe: {pr:.4f}")

        W_ce.requires_grad_(False); W_ce.eval()
        W_oh.requires_grad_(False); W_oh.eval()

        row = [seed, ll, pr]
        for alpha in ALPHAS:
            acc = evaluate_alpha(phi2, tokenizer, eval_pairs, eval_labels,
                                 W_ce, one_hot_eval, alpha, adapter=W_oh)
            row.append(acc)
            print(f"  alpha={alpha:.1f}: acc = {acc:.4f}")
        results.append(row)

    path = f"{OUT_DIR}/wide_linear_multi_seed{SUFFIX}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "logit_lens", "probe"] + [f"alpha_{a}" for a in ALPHAS])
        w.writerows(results)
    print(f"\nSaved: {path}")

    print("\nSUMMARY")
    print(f"{'seed':>6}  {'ll':>8}  {'probe':>8}  {'a0.5':>8}  {'a1.0':>8}")
    for r in results:
        print(f"{r[0]:>6}  {r[1]:>8.4f}  {r[2]:>8.4f}  {r[3+ALPHAS.index(0.5)]:>8.4f}  {r[3+ALPHAS.index(1.0)]:>8.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
