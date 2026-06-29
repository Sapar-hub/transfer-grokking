import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

from utils import DEVICE, P

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/cross_model"
os.makedirs(OUT_DIR, exist_ok=True)

D_SMALL = 128
BATCH_SIZE = 256
N_EPOCHS = 5000
LR = 1e-3
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
PROBE_ALPHAS = [0.0, 0.5, 1.0]

CANDIDATES = [
    ("Phi-2", "microsoft/phi-2"),
    ("Qwen2-Math-1.5B", "Qwen/Qwen2-Math-1.5B"),
    ("Phi-3-mini-4k", "microsoft/Phi-3-mini-4k-instruct"),
]

RESULTS_CACHE = f"{OUT_DIR}/probe_results_cache.npy"


def get_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]


def load_model(name, hf_name):
    proxy_env = {"HTTPS_PROXY": "socks5://127.0.0.1:1080", "HTTP_PROXY": "socks5://127.0.0.1:1080"}
    orig_env = {k: os.environ.get(k) for k in proxy_env}
    try:
        print(f"  Loading {name} (with SOCKS5 proxy)...")
        os.environ.update(proxy_env)
        model = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.float32, device_map=None)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
    except Exception as e:
        print(f"  Proxy failed ({type(e).__name__}), trying direct...")
        for k in proxy_env:
            os.environ.pop(k, None)
        model = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.float32, device_map=None)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
    finally:
        for k in proxy_env:
            os.environ.pop(k, None)
            if orig_env.get(k):
                os.environ[k] = orig_env[k]
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def get_model_info(model):
    config = model.config
    if hasattr(config, "num_hidden_layers"):
        n_layers = config.num_hidden_layers
    elif hasattr(config, "num_layers"):
        n_layers = config.num_layers
    else:
        n_layers = len(model.model.layers)
    d_model = config.hidden_size if hasattr(config, "hidden_size") else config.d_model
    return n_layers, d_model


def check_tokenizer(tokenizer, name):
    multi = sum(1 for n in range(P) if len(tokenizer.encode(str(n))) > 1)
    number_ids = [tokenizer.encode(str(n))[0] for n in range(P)]
    uniq = len(set(number_ids))
    print(f"  Tokenizer: {multi}/{P} numbers multi-token, {uniq}/{P} unique first tokens")
    return number_ids


def collect_clean_activations(tokenizer, model, layer):
    safe = lambda s: s.lower().replace('-', '_').replace('/', '_')
    path = f"{OUT_DIR}/{safe(model.config._name_or_path)}_L{layer}_acts.npy"
    known_old = [
        f"{OUT_DIR}/qwen2_math_1.5b_L{layer}_acts.npy",
    ]
    for p in [path] + known_old:
        if os.path.exists(p):
            print(f"  Loading cached clean activations from {p}")
            return np.load(p)

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
    inputs_all = [(int(i // P), int(i % P)) for i in range(P * P)]
    print(f"  Collecting {len(inputs_all)} clean activations at L{layer}...")
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
    print(f"  Saved clean activations: {acts.shape}")
    return acts


def train_W_ce(X_train, y_train, X_test, y_test, lm_head_sliced, name, d_model):
    safe = lambda s: s.lower().replace('-', '_').replace('/', '_')
    path_w = f"{OUT_DIR}/W_ce_{safe(name)}.pth"
    # Also check alternative locations
    alt_paths = []
    if "phi-2" in name.lower():
        alt_paths.append(f"{ARTIFACTS}/ce_projection/W_ce.pth")
    for p in [path_w] + alt_paths:
        if os.path.exists(p):
            W = nn.Linear(D_SMALL, d_model, bias=False)
            W.load_state_dict(torch.load(p, map_location=DEVICE, weights_only=True))
            print(f"  Loaded W_CE from {p}")
            return W

    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    W = nn.Linear(D_SMALL, d_model, bias=False)
    opt = optim.AdamW(W.parameters(), lr=LR, weight_decay=1e-2)

    best_val = 0.0
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
                if val_acc > best_val:
                    best_val = val_acc
            print(f"    [CE] epoch {epoch:4d}: loss={loss.item():.6f} val_acc={val_acc:.4f}")

    torch.save(W.state_dict(), path_w)
    print(f"  Saved W_CE to {path_w}, best val_acc={best_val:.4f}")
    return W


def collect_patched_for_probe(tokenizer, model, layer, W, h_A_all, alpha, name):
    safe = lambda s: s.lower().replace('-', '_').replace('/', '_')
    W.requires_grad_(False)
    all_acts = []

    inputs_all = [(int(i // P), int(i % P)) for i in range(P * P)]
    for start in range(0, len(inputs_all), BATCH_SIZE):
        batch_pairs = inputs_all[start:start + BATCH_SIZE]
        batch_h_A = h_A_all[start:start + BATCH_SIZE]
        prompts = [f"# ({a} + {b}) % 97 =" for a, b in batch_pairs]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        mask = tokenized.attention_mask

        with torch.no_grad():
            patch = W(torch.from_numpy(batch_h_A).float())

        storage = []

        def make_hook(patch, mask, alpha, storage):
            def hook(module, input, output):
                hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                seq_lens = mask.sum(dim=1) - 1
                batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
                if alpha == 1.0:
                    hidden[batch_idx, seq_lens] = patch
                elif alpha > 0:
                    orig = hidden[batch_idx, seq_lens].clone()
                    hidden[batch_idx, seq_lens] = (1 - alpha) * orig + alpha * patch
                storage.append(hidden[batch_idx, seq_lens].detach().cpu().numpy())
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
            return hook

        hook = make_hook(patch, mask, alpha, storage)
        handle = model.model.layers[layer].register_forward_hook(hook)

        with torch.no_grad():
            model(**tokenized)

        handle.remove()
        all_acts.append(np.concatenate(storage, axis=0))

    return np.concatenate(all_acts, axis=0)


def train_probe(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    probe = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=42)
    probe.fit(X_train, y_train)
    acc = probe.score(X_test, y_test)
    return acc


def process_model(name, hf_name, small_acts, labels, train_idx, test_idx):
    print(f"\n{'='*60}")
    print(f"Processing {name} ({hf_name})")
    print(f"{'='*60}")

    model, tokenizer = load_model(name, hf_name)
    n_layers, d_model = get_model_info(model)
    last_layer = n_layers - 1
    print(f"  Architecture: {n_layers} layers, d_model={d_model}, last_layer=L{last_layer}")

    number_ids = check_tokenizer(tokenizer, name)

    print(f"\n  [1] Collecting clean L{last_layer} activations...")
    clean_acts = collect_clean_activations(tokenizer, model, last_layer)

    print(f"\n  [2] Training/loading W_CE...")
    lm_head_sliced = model.lm_head.weight[number_ids].detach()
    small_train = small_acts[train_idx]
    small_test = small_acts[test_idx]
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]
    W_ce = train_W_ce(small_train, labels_train, small_test, labels_test,
                       lm_head_sliced, name, d_model)
    W_ce.requires_grad_(False)

    with torch.no_grad():
        p_te = W_ce(torch.from_numpy(small_test).float())
        logits_te = p_te @ lm_head_sliced.T
        ll_acc = (logits_te.argmax(dim=1) == torch.from_numpy(labels_test).long()).float().mean().item()
    print(f"  Logit lens (reference): {ll_acc:.4f}")

    print(f"\n  [3] Probe on patched activations...")
    results = {}
    for alpha in PROBE_ALPHAS:
        if alpha == 0.0:
            patched = clean_acts.copy()
        else:
            print(f"  Collecting patched activations alpha={alpha:.1f}...")
            patched = collect_patched_for_probe(tokenizer, model, last_layer,
                                                W_ce, small_acts, alpha, name)
        print(f"  Training probe alpha={alpha:.1f} (X={patched.shape})...")
        acc = train_probe(patched, labels)
        results[alpha] = acc
        print(f"    Probe acc = {acc:.4f}")

    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "name": name,
        "n_layers": n_layers,
        "d_model": d_model,
        "last_layer": last_layer,
        "logit_lens": ll_acc,
        "probe_results": results,
        "unique_tokens": len(set(number_ids)),
    }


def main():
    print("=" * 60)
    print("Cross-Model Probe: probe on W_CE-injected residual states")
    print("=" * 60)

    print("\n[0] Loading data...")
    small_acts = np.load(f"{ARTIFACTS}/small_model_activations.npy")
    labels = np.load(f"{ARTIFACTS}/mod_arithmetic_labels.npy", allow_pickle=True)
    train_idx, test_idx = get_split()
    print(f"  {len(small_acts)} activations, 70/30 split")

    all_results = []
    for name, hf_name in CANDIDATES:
        try:
            res = process_model(name, hf_name, small_acts, labels, train_idx, test_idx)
            all_results.append(res)
        except Exception as e:
            print(f"\nERROR processing {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")

    plt.figure(figsize=(10, 6))
    lines = []
    lines.append("# Cross-Model Probe Comparison\n")
    lines.append("| Model | Layers | d_model | L_last | Unique tokens | Logit lens | Probe(α=0.0) | Probe(α=0.5) | Probe(α=1.0) |")
    lines.append("|-------|--------|---------|--------|---------------|------------|--------------|--------------|--------------|")

    colors = {'Phi-2': 'blue', 'Qwen2-Math-1.5B': 'orange', 'Phi-3-mini-4k': 'green'}
    for res in all_results:
        p0 = res["probe_results"].get(0.0, 0.0)
        p5 = res["probe_results"].get(0.5, 0.0)
        p1 = res["probe_results"].get(1.0, 0.0)
        lines.append(f"| {res['name']} | {res['n_layers']} | {res['d_model']} | L{res['last_layer']} | {res['unique_tokens']}/97 | {res['logit_lens']:.4f} | {p0:.4f} | {p5:.4f} | {p1:.4f} |")

        alphas = sorted(res["probe_results"].keys())
        accs = [res["probe_results"][a] for a in alphas]
        c = colors.get(res['name'], 'gray')
        plt.plot(alphas, accs, 'o-', label=f"{res['name']} (L{res['last_layer']})", color=c, linewidth=2, markersize=8)

    plt.axhline(y=1/P, color='gray', linestyle='--', alpha=0.4, label=f'random={1/P:.4f}')
    plt.xlabel('Alpha (injection strength)')
    plt.ylabel('Probe Accuracy (97-class LogisticRegression)')
    plt.title('Cross-Model Comparison: Probe on W_CE-injected L_last')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path_plot = f"{OUT_DIR}/probe_comparison.png"
    plt.savefig(path_plot)
    plt.close()
    print(f"\nPlot saved: {path_plot}")

    text = "\n".join(lines)
    print("\n" + text)
    path_md = f"{OUT_DIR}/probe_comparison.md"
    with open(path_md, "w") as f:
        f.write(text)
    print(f"\nSaved: {path_md}")
    print("\nDone.")


if __name__ == "__main__":
    main()
