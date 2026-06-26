import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv

from utils import DEVICE, P

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/natural_adapter"
os.makedirs(OUT_DIR, exist_ok=True)

D_PHI2 = 2560
LAYERS = [20, 25, 28, 30]
BATCH_SIZE = 256
N_EPOCHS = 1000
LR = 1e-3
WD = 1e-2

TEMPLATES = [
    "what is ({a} + {b}) mod 97?",
    "calculate ({a} + {b}) modulo 97",
    "{a} + {b} mod 97 =",
    "if I add {a} and {b} and take remainder when divided by 97 what do I get",
]


# ─── Step 0: Train / test split ────────────────────────────────────────

def get_train_test_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]


# ─── Step 1: Generate pairs + assign random templates ──────────────────

def generate_pairs_with_templates():
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    labels = (a + b) % P
    rng = np.random.RandomState(42)
    templates = rng.randint(0, len(TEMPLATES), size=P * P)
    pairs = [(int(ai), int(bi), int(t)) for ai, bi, t in zip(a, b, templates)]
    return pairs, labels.numpy()


# ─── Step 2: Collect Phi-2 activations (4 layers, one pass) ────────────

def collect_activations(tokenizer, model, pairs_with_templates, layers):
    outs = {}
    n_total = len(pairs_with_templates)
    for l in layers:
        path = f"{OUT_DIR}/phi2_natural_L{l}.npy"
        if os.path.exists(path):
            print(f"  [cache] Loading cached L{l} activations...")
            outs[l] = np.load(path)
    if len(outs) == len(layers):
        print(f"  All {len(layers)} layers loaded from cache.")
        return outs

    activations = {l: [] for l in layers}
    current_mask = [None]

    handles = []
    for l in layers:
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                mask = current_mask[0].to(hidden.device)
                seq_lens = mask.sum(dim=1) - 1
                batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
                activations[layer_idx].append(hidden[batch_idx, seq_lens].detach().cpu())
            return hook
        handle = model.model.layers[l].register_forward_hook(make_hook(l))
        handles.append(handle)

    model.eval()
    print(f"[collect] Processing {n_total} prompts through {len(layers)} layers...")
    for start in range(0, n_total, BATCH_SIZE):
        batch = pairs_with_templates[start:start + BATCH_SIZE]
        prompts = [TEMPLATES[t].format(a=a, b=b) for a, b, t in batch]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        current_mask[0] = tokenized.attention_mask
        with torch.no_grad():
            model(**tokenized)

    for h in handles:
        h.remove()

    for l in layers:
        arr = torch.cat(activations[l], dim=0).numpy()
        path = f"{OUT_DIR}/phi2_natural_L{l}.npy"
        np.save(path, arr)
        outs[l] = arr
        print(f"  Saved L{l}: {arr.shape}")

    return outs


# ─── Step 3: Train adapter (nn.Linear + AdamW) ─────────────────────────

def train_adapter(X_train, y_train, X_test, y_test, layer):
    acc_path = f"{OUT_DIR}/adapter_acc_L{layer}.txt"
    if os.path.exists(acc_path):
        with open(acc_path) as f:
            test_acc = float(f.read().strip())
        print(f"  [adapter L{layer}] Loaded cached: acc={test_acc:.4f}")
        return test_acc

    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    adapter = nn.Linear(D_PHI2, P)
    opt = optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(1, N_EPOCHS + 1):
        adapter.train()
        logits = adapter(X_tr)
        loss = loss_fn(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 100 == 0 or epoch == 1:
            adapter.eval()
            with torch.no_grad():
                tr_pred = logits.argmax(dim=1)
                tr_acc = (tr_pred == y_tr).float().mean().item()

                te_logits = adapter(X_te)
                te_pred = te_logits.argmax(dim=1)
                te_acc = (te_pred == y_te).float().mean().item()

            train_losses.append(loss.item())
            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            print(f"    epoch {epoch:4d}: loss={loss.item():.4f}  train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")

    test_acc = test_accs[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(range(1, len(train_losses) + 1), train_losses, 'o-')
    ax1.set_xlabel('Epoch (x100)')
    ax1.set_ylabel('CrossEntropy Loss')
    ax2.plot(range(1, len(train_accs) + 1), train_accs, 'o-', label='train')
    ax2.plot(range(1, len(test_accs) + 1), test_accs, 's-', label='test')
    ax2.set_xlabel('Epoch (x100)')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.suptitle(f'Adapter L{layer} (2560 → 97)')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/training_curve_L{layer}.png")
    plt.close()

    torch.save(adapter.state_dict(), f"{OUT_DIR}/adapter_L{layer}.pth")
    with open(acc_path, "w") as f:
        f.write(f"{test_acc:.6f}")
    print(f"  [adapter L{layer}] Done. Test acc = {test_acc:.4f}")
    return test_acc


# ─── Step 3b: sklearn LogisticRegression baseline ──────────────────────

def train_probe_sklearn(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)
    probe = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, random_state=42)
    probe.fit(X_scaled, y_train)
    acc = probe.score(X_te_scaled, y_test)
    return acc


# ─── Step 5: Template generalization ───────────────────────────────────

def collect_template_acts_single_template(tokenizer, model, pairs, t_idx, layers):
    """Collect activations for one template on all pairs. Returns dict[l] -> [N, D]."""
    result = {}
    for l in layers:
        path = f"{OUT_DIR}/template_gen_T{t_idx}_L{l}.npy"
        if os.path.exists(path):
            result[l] = np.load(path)
    if len(result) == len(layers):
        return result

    activations = {l: [] for l in layers}
    current_mask = [None]
    template_str = TEMPLATES[t_idx]

    handles = []
    for l in layers:
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                mask = current_mask[0].to(hidden.device)
                seq_lens = mask.sum(dim=1) - 1
                batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
                activations[layer_idx].append(hidden[batch_idx, seq_lens].detach().cpu())
            return hook
        handle = model.model.layers[l].register_forward_hook(make_hook(l))
        handles.append(handle)

    model.eval()
    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start:start + BATCH_SIZE]
        prompts = [template_str.format(a=a, b=b) for a, b in batch]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        current_mask[0] = tokenized.attention_mask
        with torch.no_grad():
            model(**tokenized)

    for h in handles:
        h.remove()

    for l in layers:
        arr = torch.cat(activations[l], dim=0).numpy()
        np.save(f"{OUT_DIR}/template_gen_T{t_idx}_L{l}.npy", arr)
        result[l] = arr

    return result


def template_generalization(tokenizer, model, layers, n_pairs=500):
    print("\n[5] Template generalization test...")
    rng = np.random.RandomState(42)
    pairs = [(int(rng.randint(0, P)), int(rng.randint(0, P))) for _ in range(n_pairs)]
    labels = np.array([(a + b) % P for a, b in pairs])

    # Collect activations per template (separate runs)
    template_acts = {}
    for t_idx in range(len(TEMPLATES)):
        print(f"  Collecting T{t_idx} ({n_pairs} pairs)...")
        template_acts[t_idx] = collect_template_acts_single_template(
            tokenizer, model, pairs, t_idx, layers
        )

    results = []
    for t_train in range(len(TEMPLATES)):
        for t_test in range(len(TEMPLATES)):
            layer_accs = []
            for l in layers:
                X_tr = template_acts[t_train][l]
                X_te = template_acts[t_test][l]
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_te_sc = scaler.transform(X_te)
                probe = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, random_state=42)
                probe.fit(X_tr_sc, labels)
                acc = probe.score(X_te_sc, labels)
                layer_accs.append((l, acc))

            if layer_accs:
                best = max(layer_accs, key=lambda x: x[1])
                results.append({
                    "train_template": t_train,
                    "test_template": t_test,
                    "best_layer": best[0],
                    "best_acc": best[1],
                })

    path = f"{OUT_DIR}/template_generalization.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["train_template", "test_template", "best_layer", "best_acc"])
        w.writeheader()
        w.writerows(results)
    print(f"  Saved to {path}")
    for r in results:
        print(f"    T{r['train_template']} → T{r['test_template']}: "
              f"L{r['best_layer']} acc={r['best_acc']:.4f}")
    return results


# ─── Summary ────────────────────────────────────────────────────────────

def write_summary(acc_text_baseline, layer_results, gen_results):
    best_adapter = max(layer_results, key=lambda r: r["adapter_acc"])
    best_sklearn = max(layer_results, key=lambda r: r["sklearn_acc"])

    lines = []
    lines.append("# Natural Adapter Summary\n")
    lines.append("## Baseline\n")
    lines.append(f"| Condition | Accuracy |")
    lines.append(f"|-----------|----------|")
    lines.append(f"| Phi-2 LM head (text) | {acc_text_baseline:.4f} |")
    lines.append(f"| Random (1/{P}) | {1/P:.4f} |\n")
    lines.append("## Per-layer adapter accuracy\n")
    lines.append("| Layer | nn.Linear (AdamW) | LogisticRegression |")
    lines.append("|-------|-------------------|--------------------|")
    for r in layer_results:
        lines.append(f"| {r['layer']} | {r['adapter_acc']:.4f} | {r['sklearn_acc']:.4f} |")
    lines.append(f"\n**Best nn.Linear**: L{best_adapter['layer']} acc={best_adapter['adapter_acc']:.4f}")
    lines.append(f"**Best sklearn**: L{best_sklearn['layer']} acc={best_sklearn['sklearn_acc']:.4f}\n")

    lines.append("## Template generalization (LogisticRegression, best per L)\n")
    if gen_results:
        lines.append("| Train → Test | Best L | Acc |")
        lines.append("|--------------|--------|-----|")
        for r in gen_results:
            lines.append(f"| T{r['train_template']} → T{r['test_template']} | {r['best_layer']} | {r['best_acc']:.4f} |")

    # Interpretation
    max_adapter = best_adapter["adapter_acc"]
    lines.append("\n## Interpretation\n")
    if max_adapter > 0.30:
        lines.append("**adapter acc >> 0.235** → Phi-2 содержит ответ в residual stream.")
        lines.append("LM head был bottleneck, не знание. Вся архитектура с маленькой моделью не нужна.")
    elif max_adapter > 0.25:
        lines.append("**adapter acc ≈ 0.235** → информации на уровне линейного разделения не хватает.")
        lines.append("Нужен нелинейный adapter (MLP).")
    else:
        lines.append("**adapter acc ≈ baseline** → Phi-2 не кодирует ответ линейно в residual stream.")

    if gen_results:
        t0 = [r for r in gen_results if r["train_template"] == 0]
        if t0:
            same_template = next((r for r in t0 if r["test_template"] == 0), None)
            cross_templates = [r for r in t0 if r["test_template"] != 0]
            if same_template and cross_templates:
                in_domain = same_template["best_acc"]
                cross_mean = np.mean([r["best_acc"] for r in cross_templates])
                if in_domain > 0.30 and cross_mean > in_domain * 0.8:
                    lines.append("\n**Template generalization высокая** → Phi-2 реально понимает задачу через язык.")
                    lines.append("Это работающий 'language-grounded residual tool'.")
                elif in_domain > 0.30 and cross_mean < in_domain * 0.5:
                    lines.append("\n**Template generalization низкая** → adapter выучил поверхностный паттерн.")
                    lines.append("Не обобщается на новые формулировки.")
                else:
                    lines.append("\n**Template generalization умеренная** → частичное обобщение.")
    lines.append("")

    text = "\n".join(lines)
    path = f"{OUT_DIR}/experiment_summary.md"
    with open(path, "w") as f:
        f.write(text)
    print(text)


# ─── Helper: Phi-2 text baseline ───────────────────────────────────────

def baseline_accuracy(phi2, tokenizer, n_pairs=200):
    rng = np.random.RandomState(42)
    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}

    correct = 0
    B = 64
    for start in range(0, n_pairs, B):
        bs = min(B, n_pairs - start)
        batch = [(int(rng.randint(0, P)), int(rng.randint(0, P))) for _ in range(bs)]
        prompts = [f"# ({a} + {b}) % 97 =" for a, b in batch]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = phi2(**inputs).logits
        seq_lens = inputs.attention_mask.sum(dim=1) - 1
        for i, (a, b) in enumerate(batch):
            y = (a + b) % P
            pred_logits = logits[i, seq_lens[i], :]
            pred = max(number_tokens, key=lambda n: pred_logits[number_tokens[n]].item())
            if pred == y:
                correct += 1
    return correct / n_pairs


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Natural Adapter: Phi-2 residual stream from natural language")
    print("=" * 60)

    # ── Step 0-1: data ──
    print("\n[0-1] Generating pairs + assigning templates...")
    pairs_with_templates, labels = generate_pairs_with_templates()
    train_idx, test_idx = get_train_test_split()
    print(f"  {len(pairs_with_templates)} pairs, {len(train_idx)} train, {len(test_idx)} test")

    # ── Load Phi-2 ──
    print("\n[2] Loading Phi-2...")
    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    phi2.eval()

    # ── Baseline ──
    print("\n[2b] Baseline: Phi-2 LM head accuracy...")
    acc_text = baseline_accuracy(phi2, tokenizer)
    print(f"  Phi-2 text baseline: {acc_text:.4f}")

    # ── Collect activations ──
    print("\n[3] Collecting Phi-2 activations (one pass, 4 layers)...")
    phi2_acts = collect_activations(tokenizer, phi2, pairs_with_templates, LAYERS)

    # ── Train adapters ──
    print("\n[4] Training adapters per layer...")
    layer_results = []
    for l in LAYERS:
        print(f"\n  --- Layer {l} ---")
        X_all = phi2_acts[l]
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]

        adapter_acc = train_adapter(X_tr, y_tr, X_te, y_te, l)
        sklearn_acc = train_probe_sklearn(X_tr, y_tr, X_te, y_te)
        print(f"  nn.Linear:       {adapter_acc:.4f}")
        print(f"  LogisticReg:     {sklearn_acc:.4f}")

        layer_results.append({
            "layer": l,
            "adapter_acc": adapter_acc,
            "sklearn_acc": sklearn_acc,
        })

    # ── Save per-layer results ──
    path_csv = f"{OUT_DIR}/results_per_layer.csv"
    with open(path_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "adapter_acc", "sklearn_acc"])
        w.writeheader()
        w.writerows(layer_results)
    print(f"\n  Saved {path_csv}")

    # ── Template generalization ──
    gen_results = template_generalization(tokenizer, phi2, LAYERS)

    # ── Summary ──
    print("\n[6] Writing summary...")
    write_summary(acc_text, layer_results, gen_results)

    print(f"\nDone. Artifacts in {OUT_DIR}/")


if __name__ == "__main__":
    main()
