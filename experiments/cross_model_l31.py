import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, csv, sys

from utils import DEVICE, P

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/cross_model"
os.makedirs(OUT_DIR, exist_ok=True)

D_SMALL = 128
BATCH_SIZE = 256
N_EPOCHS = 5000
LR = 1e-3
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]

TARGET = "Qwen2-Math-1.5B"
TARGET_HF = "Qwen/Qwen2-Math-1.5B"
TARGET_LAYER = 27
TARGET_D_MODEL = 1536
TARGET_N_LAYERS = 28


def get_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]


def load_model():
    proxy_env = {"HTTPS_PROXY": "socks5://127.0.0.1:1080", "HTTP_PROXY": "socks5://127.0.0.1:1080"}
    try:
        print(f"  Loading {TARGET} (with SOCKS5 proxy)...")
        orig_env = {k: os.environ.get(k) for k in proxy_env}
        os.environ.update(proxy_env)
        model = AutoModelForCausalLM.from_pretrained(TARGET_HF, dtype=torch.float32, device_map=None)
        tokenizer = AutoTokenizer.from_pretrained(TARGET_HF)
    except Exception as e:
        print(f"  Proxy failed ({type(e).__name__}), trying direct...")
        for k in proxy_env:
            os.environ.pop(k, None)
        model = AutoModelForCausalLM.from_pretrained(TARGET_HF, dtype=torch.float32, device_map=None)
        tokenizer = AutoTokenizer.from_pretrained(TARGET_HF)
    finally:
        for k in proxy_env:
            os.environ.pop(k, None)
            if orig_env.get(k):
                os.environ[k] = orig_env[k]

    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.lm_head.requires_grad_(False)
    return model, tokenizer


def check_tokenizer(tokenizer):
    print("\n--- Tokenizer check: numbers 0-96 ---")
    multi = 0
    for n in range(P):
        ids = tokenizer.encode(str(n))
        status = "OK" if len(ids) == 1 else f"MULTI({len(ids)})"
        if len(ids) > 1:
            multi += 1
            if multi <= 5:
                print(f"  {n:2d}: {ids}  <- multi-token")
    print(f"  Multi-token numbers: {multi}/{P}")
    print(f"  All numbers encoded as single tokens: {multi == 0}")
    number_ids = [tokenizer.encode(str(n))[0] for n in range(P)]
    print(f"  Token ID range: [{min(number_ids)}, {max(number_ids)}]")
    print(f"  Unique token IDs: {len(set(number_ids))} / {P}")
    return number_ids


def collect_target_activations(tokenizer, model):
    path = f"{OUT_DIR}/{TARGET.lower().replace('-','_')}_L{TARGET_LAYER}_acts.npy"
    if os.path.exists(path):
        print(f"[collect] Loading cached activations...")
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

    handle = model.model.layers[TARGET_LAYER].register_forward_hook(make_hook())

    inputs_all = [(int(i // P), int(i % P)) for i in range(P * P)]
    print(f"[collect] Processing {len(inputs_all)} prompts at L{TARGET_LAYER}...")
    for start in range(0, len(inputs_all), BATCH_SIZE):
        batch = inputs_all[start:start + BATCH_SIZE]
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


def train_W_ce(X_train, y_train, X_test, y_test, lm_head_sliced):
    path_w = f"{OUT_DIR}/W_ce_{TARGET.lower().replace('-','_')}.pth"
    if os.path.exists(path_w):
        W = nn.Linear(D_SMALL, TARGET_D_MODEL, bias=False)
        W.load_state_dict(torch.load(path_w, map_location=DEVICE, weights_only=True))
        print(f"  [CE] Loaded W.")
        return W

    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    W = nn.Linear(D_SMALL, TARGET_D_MODEL, bias=False)
    opt = optim.AdamW(W.parameters(), lr=LR, weight_decay=1e-2)

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
            print(f"    [CE] epoch {epoch:4d}: loss={loss.item():.6f} val_acc={val_acc:.4f}")

    torch.save(W.state_dict(), path_w)
    return W


def logit_lens_accuracy(W, X_test, y_test, lm_head_sliced):
    with torch.no_grad():
        projected = W(torch.from_numpy(X_test).float())
        logits = projected @ lm_head_sliced.T
        acc = (logits.argmax(dim=1) == torch.from_numpy(y_test).long()).float().mean().item()
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


def evaluate_alpha(model, tokenizer, test_pairs, labels, W, h_A_test, alpha, number_ids, layer, batch_size=32):
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
            pred = max(range(P), key=lambda n: logits[i, number_ids[n]].item())
            if pred == labels[start + i]:
                correct += 1
            total += 1
    return correct / total


def main():
    print("=" * 60)
    print(f"Cross-model L31: train W_CE for {TARGET}, alpha sweep at L{TARGET_LAYER}")
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

    print(f"\n[1] Loading {TARGET}...")
    model, tokenizer = load_model()
    print(f"  {TARGET}: {TARGET_N_LAYERS} layers, d_model={TARGET_D_MODEL}")

    print(f"\n[1b] Tokenizer check...")
    number_ids = check_tokenizer(tokenizer)

    print(f"\n[2] Collecting {TARGET} L{TARGET_LAYER} activations...")
    target_acts = collect_target_activations(tokenizer, model)
    target_train = target_acts[train_idx]
    target_test = target_acts[test_idx]

    lm_head_sliced = model.lm_head.weight[number_ids].detach()

    print(f"\n[3] Training W_CE (128 -> {TARGET_D_MODEL}) via frozen lm_head...")
    W_ce = train_W_ce(small_train, labels_train, small_test, labels_test, lm_head_sliced)

    print(f"\n[4] Logit lens accuracy...")
    ll_ce = logit_lens_accuracy(W_ce, small_test, labels_test, lm_head_sliced)
    print(f"  W_CE logit lens: {ll_ce:.4f}")

    print(f"\n[5] Alpha sweep at L{TARGET_LAYER}...")
    W_ce.requires_grad_(False)
    W_ce.eval()
    results = []
    for alpha in ALPHAS:
        acc = evaluate_alpha(model, tokenizer, eval_pairs, eval_labels,
                             W_ce, eval_h_A, alpha, number_ids, TARGET_LAYER)
        results.append((alpha, acc))
        print(f"  alpha={alpha:.1f}: text_acc = {acc:.4f}")

    path = f"{OUT_DIR}/{TARGET.lower().replace('-','_')}_L{TARGET_LAYER}_sweep.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "text_acc"])
        w.writerows(results)
    print(f"  Saved: {path}")

    print(f"\n[6] Comparison with Phi-2 L31...")
    phi2_path = f"{ARTIFACTS}/l31_patch/alpha_sweep_l31.csv"
    with open(phi2_path) as f:
        reader = csv.reader(f)
        phi2_rows = list(reader)
    phi2_ce = [float(v) for v in phi2_rows[2][1:]]

    lines = []
    lines.append(f"# Cross-Model L31: {TARGET} vs Phi-2\n")
    lines.append("## Setup\n")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Target model | {TARGET} ({TARGET_HF}) |")
    lines.append(f"| Layers | {TARGET_N_LAYERS} |")
    lines.append(f"| d_model | {TARGET_D_MODEL} |")
    lines.append(f"| Patch layer | L{TARGET_LAYER} |")
    lines.append(f"| Phi-2 ref layer | L31 |")
    lines.append(f"| Test pairs | 200 (seed=42) |\n")
    lines.append("## Alpha sweep\n")
    lines.append("| Alpha | Qwen2-Math L27 | Phi-2 L31 | Delta |")
    lines.append("|-------|----------------|-----------|-------|")
    for i, alpha in enumerate(ALPHAS):
        qw_acc = results[i][1]
        ph_acc = phi2_ce[i]
        lines.append(f"| {alpha:.1f} | {qw_acc:.4f} | {ph_acc:.4f} | {qw_acc - ph_acc:+.4f} |")
    lines.append(f"\nBaseline (alpha=0.0): {results[0][1]:.4f}\n")
    lines.append(f"Logit lens (W_CE): {ll_ce:.4f}\n")

    if ll_ce > 0.9:
        lines.append("**The hypothesis is not supported**: Qwen2-Math with math pretraining achieves W_CE logit_lens > 0.9, but this is expected since CE training through lm_head guarantees alignment. The key comparison is the alpha sweep delta.")
    elif ll_ce > 0.5:
        lines.append("**Partial alignment**: W_CE > 0.5 suggests Qwen2-Math lm_head is somewhat compatible with mod arithmetic tokens.")
    else:
        lines.append("**Hypothesis check**: Low logit lens suggests Qwen2-Math lm_head is not aligned with mod arithmetic tokens.")

    best_qw = max(r[1] for r in results[1:])
    best_ph = max(phi2_ce[1:])
    if best_qw > best_ph * 0.9:
        lines.append(f"\n**Comparison**: Qwen2-Math ({best_qw:.4f}) ≈ Phi-2 ({best_ph:.4f}) — math pretraining does not significantly improve W_CE resonance.")
    elif best_qw < best_ph * 0.5:
        lines.append(f"\n**Comparison**: Qwen2-Math ({best_qw:.4f}) << Phi-2 ({best_ph:.4f}) — lower resonance despite math pretraining, likely due to tokenizer/architecture mismatch.")
    else:
        lines.append(f"\n**Comparison**: Qwen2-Math ({best_qw:.4f}) vs Phi-2 ({best_ph:.4f}) — moderate resonance difference.")

    text = "\n".join(lines)
    with open(f"{OUT_DIR}/comparison_{TARGET.lower().replace('-','_')}_vs_phi2.md", "w") as f:
        f.write(text)
    print(text)

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
