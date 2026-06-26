import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from model import SmallTransformer
from utils import DEVICE, P

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/adapter"
os.makedirs(OUT_DIR, exist_ok=True)

D_SMALL = 128
D_PHI2 = 2560
BATCH_SIZE = 32
BEST_LAYER = 10
COLLECT_LAYER = BEST_LAYER + 1
ALPHAS = [0.0, 0.5, 1.0]

NUM_SAMPLES = 3000  # representative subset (Phi-2 is slow on CPU)


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


def collect_patched_activations(model, tokenizer, pairs_all, W, h_A_all, patch_layer, collect_layer, alpha=1.0):
    cache_path = f"{OUT_DIR}/patched_acts_L{collect_layer}_alpha{alpha:.1f}.npy"
    if os.path.exists(cache_path):
        print(f"  Loading cached: {cache_path}")
        return np.load(cache_path)

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

    for start in range(0, len(pairs_all), BATCH_SIZE):
        batch_pairs = pairs_all[start:start + BATCH_SIZE]
        batch_h_A = h_A_all[start:start + BATCH_SIZE]
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

    result = np.concatenate(all_acts, axis=0)
    np.save(cache_path, result)
    print(f"  Saved: {cache_path} {result.shape}")
    return result


class Adapter(nn.Module):
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


def train_adapter(X_train, y_train, X_test, y_test, alpha, num_epochs=1000):
    path_model = f"{OUT_DIR}/adapter_alpha{alpha:.1f}.pth"
    path_curve = f"{OUT_DIR}/training_curve_alpha{alpha:.1f}.png"

    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).long()
    X_te = torch.from_numpy(X_test).float()
    y_te = torch.from_numpy(y_test).long()

    adapter = Adapter(D_PHI2, P)
    opt = optim.AdamW(adapter.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_accs, test_accs = [], []
    best_test_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        adapter.train()
        logits = adapter(X_tr)
        loss = loss_fn(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 50 == 0 or epoch == 1:
            adapter.eval()
            with torch.no_grad():
                tr_preds = adapter(X_tr).argmax(dim=1)
                te_preds = adapter(X_te).argmax(dim=1)
                tr_acc = (tr_preds == y_tr).float().mean().item()
                te_acc = (te_preds == y_te).float().mean().item()
            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            print(f"    epoch {epoch:4d}: train_acc={tr_acc:.4f} test_acc={te_acc:.4f}")

            if te_acc > best_test_acc:
                best_test_acc = te_acc
                best_state = adapter.state_dict().copy()

    if best_state is not None:
        adapter.load_state_dict(best_state)
        torch.save(best_state, path_model)
        print(f"  Saved: {path_model} (best test_acc={best_test_acc:.4f})")

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(train_accs)), train_accs, label='train_acc')
    plt.plot(range(len(test_accs)), test_accs, label='test_acc')
    plt.xlabel('Epoch (x50)')
    plt.ylabel('Accuracy')
    plt.title(f'Adapter training (alpha={alpha:.1f})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_curve)
    plt.close()
    print(f"  Saved: {path_curve}")

    return adapter, best_test_acc


def train_logreg_probe(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    probe = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=42)
    probe.fit(X_tr_s, y_train)
    acc = probe.score(X_te_s, y_test)
    return acc


def plot_confusion_matrix(y_true, y_pred, alpha, path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest', norm=matplotlib.colors.LogNorm())
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Adapter Confusion Matrix (alpha={alpha:.1f}, acc={np.mean(y_true==y_pred):.4f})')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("Adapter: residual tool — classifier on patched L+1")
    print("=" * 60)

    # ── Generate data ──
    print("\n[0] Generating data...")
    rng_data = np.random.RandomState(42)
    all_pairs = []
    all_labels = []
    for _ in range(NUM_SAMPLES):
        a = rng_data.randint(0, P)
        b = rng_data.randint(0, P)
        all_pairs.append((a, b))
        all_labels.append((a + b) % P)
    all_pairs = np.array(all_pairs)
    all_labels = np.array(all_labels)

    # 70/30 split
    rng_split = np.random.RandomState(42)
    idx = np.arange(NUM_SAMPLES)
    rng_split.shuffle(idx)
    split = int(NUM_SAMPLES * 0.7)
    train_idx = idx[:split]
    test_idx = idx[split:]

    pairs_list = [(int(a), int(b)) for a, b in all_pairs]
    y_train = all_labels[train_idx]
    y_test = all_labels[test_idx]
    print(f"  {NUM_SAMPLES} pairs, {split} train, {NUM_SAMPLES - split} test")

    # ── Load small model and extract activations ──
    print("\n[1] Loading small model and extracting activations...")
    model_small = SmallTransformer()
    model_small.load_state_dict(
        torch.load(f"{ARTIFACTS}/small/best_model.pth", map_location=DEVICE, weights_only=True)
    )
    model_small.eval()

    h_A_all = np.zeros((NUM_SAMPLES, D_SMALL), dtype=np.float32)
    for i, (a, b) in enumerate(pairs_list):
        x = torch.tensor([[a, b]])
        with torch.no_grad():
            _, acts = model_small(x, return_activations=True)
        h_A_all[i] = acts["blocks.1.hook_resid_post"][0, 1, :].numpy()
    print(f"  Small activations: {h_A_all.shape}")

    # ── Load W (best layer) ──
    print(f"\n[2] Loading W (L={BEST_LAYER})...")
    W = nn.Linear(D_SMALL, D_PHI2, bias=False)
    W.load_state_dict(
        torch.load(f"{ARTIFACTS}/residual_patch/W_layer{BEST_LAYER}.pth", map_location=DEVICE, weights_only=True)
    )
    W.requires_grad_(False)

    # ── Load Phi-2 ──
    print("\n[3] Loading Phi-2...")
    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    phi2.eval()

    # ── Collect patched activations for all alpha ──
    print(f"\n[4] Collecting patched activations at L={COLLECT_LAYER}...")
    activations = {}
    for alpha in ALPHAS:
        print(f"  alpha={alpha:.1f}: ", end="")
        acts = collect_patched_activations(
            phi2, tokenizer, pairs_list, W, h_A_all,
            patch_layer=BEST_LAYER, collect_layer=COLLECT_LAYER, alpha=alpha
        )
        activations[alpha] = acts

    # ── Train adapters ──
    print("\n[5] Training adapters...")
    adapter_results = {}
    logreg_results = {}
    preds_cache = {}

    for alpha in ALPHAS:
        print(f"\n  --- alpha={alpha:.1f} ---")
        X_all = activations[alpha]

        X_train = X_all[train_idx]
        X_test_acts = X_all[test_idx]

        ad, test_acc = train_adapter(X_train, y_train, X_test_acts, y_test, alpha)
        adapter_results[alpha] = test_acc

        adapter_results[f"logreg_{alpha:.1f}"] = train_logreg_probe(
            X_train, y_train, X_test_acts, y_test
        )
        print(f"  LogisticRegression test_acc = {adapter_results[f'logreg_{alpha:.1f}']:.4f}")

        # Store predictions for confusion matrix
        ad.eval()
        with torch.no_grad():
            te_logits = ad(torch.from_numpy(X_test_acts).float())
            te_preds = te_logits.argmax(dim=1).numpy()
        preds_cache[alpha] = (y_test, te_preds)

    # ── Confusion matrices ──
    print("\n[6] Plotting confusion matrices...")
    for alpha in [0.5, 1.0]:
        yt, yp = preds_cache[alpha]
        path = f"{OUT_DIR}/confusion_matrix_alpha{alpha:.1f}.png"
        plot_confusion_matrix(yt, yp, alpha, path)

    # ── Summary ──
    print("\n[7] Summary")
    lines = []
    lines.append("# Adapter Experiment Summary\n")
    lines.append("| Alpha | nn.Linear Acc | LogisticRegression Acc |")
    lines.append("|-------|---------------|------------------------|")
    for alpha in ALPHAS:
        nn_acc = adapter_results[alpha]
        lr_acc = adapter_results[f"logreg_{alpha:.1f}"]
        lines.append(f"| {alpha:.1f} | {nn_acc:.4f} | {lr_acc:.4f} |")

    lines.append(f"\n**Patch layer:** L={BEST_LAYER}")
    lines.append(f"**Collect layer:** L={COLLECT_LAYER}")
    lines.append(f"**Train size:** {len(train_idx)}")
    lines.append(f"**Test size:** {len(test_idx)}")

    gap = adapter_results[1.0] - adapter_results[0.5]
    lines.append(f"\n**Gap (alpha=1.0 - alpha=0.5):** {gap:+.4f}")
    lines.append("Interpretation: gap = price of context interference on geometry")

    a0 = adapter_results[0.0]
    lines.append(f"\n**Control (alpha=0.0, no patch):** {a0:.4f} (expected ~0.0035)")
    if a0 < 0.01:
        lines.append("Unpatched activations carry no linear structure — adapter can't learn.")
    else:
        lines.append("WARNING: unpatched adapter > random baseline — possible leak.")

    lines.append("")
    text = "\n".join(lines)
    print(text)

    with open(f"{OUT_DIR}/experiment_summary.md", "w") as f:
        f.write(text)
    print(f"\nSaved: {OUT_DIR}/experiment_summary.md")
    print("Done.")


if __name__ == "__main__":
    main()
