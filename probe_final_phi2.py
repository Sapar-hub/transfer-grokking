import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import os

from utils import DEVICE, P

BASE = "artifacts"
RP_DIR = f"{BASE}/residual_patch"
OUT_DIR = f"{BASE}/probe_final_phi2"
os.makedirs(OUT_DIR, exist_ok=True)

D_SMALL = 128
D_PHI2 = 2560
N_EPOCHS = 1000
BATCH_SIZE = 256
LR = 1e-3
N_COLLECT = 2000


def collect_data(phi2, tokenizer):
    """Generate N_COLLECT pairs, collect L31 activations + labels."""
    acts_path = f"{OUT_DIR}/phi2_L31_acts.npy"
    lbl_path = f"{OUT_DIR}/phi2_L31_labels.npy"
    if os.path.exists(acts_path) and os.path.exists(lbl_path):
        return np.load(acts_path), np.load(lbl_path)

    rng = np.random.RandomState(42)
    a = rng.randint(0, P, size=N_COLLECT)
    b = rng.randint(0, P, size=N_COLLECT)
    pairs = list(zip(a, b))
    labels = np.array([(ai + bi) % P for ai, bi in pairs])

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

    handle = phi2.model.layers[31].register_forward_hook(make_hook())
    phi2.eval()

    for start in range(0, N_COLLECT, BATCH_SIZE):
        batch = pairs[start:start + BATCH_SIZE]
        prompts = [f"# ({a} + {b}) % 97 =" for a, b in batch]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        current_mask = tokenized.attention_mask
        with torch.no_grad():
            phi2(**tokenized)

    handle.remove()
    acts = torch.cat(all_acts, dim=0).numpy()
    np.save(acts_path, acts)
    np.save(lbl_path, labels)
    print(f"  Collected {len(acts)} L31 activations")
    return acts, labels


def train_linear(X, y, label, d_in):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    X_tr = torch.from_numpy(X_tr).float()
    y_tr = torch.from_numpy(y_tr).long()
    X_te = torch.from_numpy(X_te).float()
    y_te = torch.from_numpy(y_te).long()

    adapter = nn.Linear(d_in, P)
    opt = torch.optim.AdamW(adapter.parameters(), lr=LR)

    best_acc = 0
    for epoch in range(1, N_EPOCHS + 1):
        adapter.train()
        logits = adapter(X_tr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 200 == 0 or epoch == 1:
            adapter.eval()
            with torch.no_grad():
                acc = (adapter(X_te).argmax(dim=1) == y_te).float().mean().item()
            best_acc = max(best_acc, acc)
            print(f"    [{label}] epoch {epoch:4d}: test_acc={acc:.4f}")

    return best_acc


def main():
    print("=" * 60)
    print("Probe Final Phi-2: Linear on L31 activations (single template)")
    print("=" * 60)

    print("\n[0] Loading Phi-2...")
    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\n[1] Text baseline (batched)...")
    rng = np.random.RandomState(42)
    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}
    n_baseline = 200
    a_all = rng.randint(0, P, size=n_baseline)
    b_all = rng.randint(0, P, size=n_baseline)
    labels_all = (a_all + b_all) % P
    correct = 0
    for start in range(0, n_baseline, 64):
        end = min(start + 64, n_baseline)
        prompts = [f"# ({a} + {b}) % 97 =" for a, b in zip(a_all[start:end], b_all[start:end])]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = phi2(**inputs).logits
        seq_lens = inputs.attention_mask.sum(dim=1) - 1
        for i, idx in enumerate(range(start, end)):
            pred = max(number_tokens, key=lambda n: logits[i, seq_lens[i], number_tokens[n]].item())
            if pred == labels_all[idx]:
                correct += 1
    text_baseline = correct / n_baseline
    print(f"  Phi-2 text baseline: {text_baseline:.4f}")

    print("\n[2] Collecting L31 activations...")
    h_phi2, labels = collect_data(phi2, tokenizer)

    print("\n[3] Loading W and small_acts (for conditions A, B)...")
    W = nn.Linear(D_SMALL, D_PHI2, bias=False)
    W.load_state_dict(torch.load(f"{RP_DIR}/W_layer10.pth", map_location=DEVICE, weights_only=True))
    W.requires_grad_(False)
    W.eval()
    small_acts = np.load(f"{BASE}/small_model_activations.npy")
    with torch.no_grad():
        h_proj = W(torch.from_numpy(small_acts).float()).numpy()

    # Re-generate the same pairs to index into small_acts
    rng2 = np.random.RandomState(42)
    a_vals = rng2.randint(0, P, size=N_COLLECT)
    b_vals = rng2.randint(0, P, size=N_COLLECT)
    indices = a_vals * P + b_vals
    h_small = small_acts[indices]
    h_W = h_proj[indices]

    print("\n[4] Training conditions...")
    results = {}

    print("\n  A: small_acts (128) → Linear(128→97)")
    results["A: small_acts→97"] = train_linear(h_small, labels, "A: small_acts", D_SMALL)

    print("\n  B: W(small_acts) (2560) → Linear(2560→97)")
    results["B: W(h_A)→97"] = train_linear(h_W, labels, "B: W→97", D_PHI2)

    print("\n  C: h_phi2_L31 (2560) → Linear(2560→97)  [THE QUESTION]")
    results["C: phi2_L31→97"] = train_linear(h_phi2, labels, "C: phi2_L31→97", D_PHI2)

    print("\n[5] Summary")
    natural_best = 0.0446
    lines = []
    lines.append("# Probe Final Phi-2: Linear on L31 (single template)\n")
    lines.append("## Parameters\n")
    lines.append(f"- Pairs collected: {N_COLLECT}")
    lines.append(f"- Phi-2 text baseline: {text_baseline:.4f}")
    lines.append(f"- Natural adapter best (L30, mixed templates): {natural_best:.4f}\n")
    lines.append("## Results\n")
    lines.append("| Cond | Input | Adapter | Test Acc |")
    lines.append("|------|-------|---------|----------|")
    for k, v in results.items():
        lines.append(f"| {k} | — | Linear→97 | {v:.4f} |")
    lines.append("")

    verdict = ""
    c = results["C: phi2_L31→97"]
    if c < 0.05:
        verdict = "CONFIRMED: Phi-2 does NOT encode answer linearly at L31."
        verdict += " Single template, no mixed confound. Fundamental result."
    elif c < text_baseline + 0.03:
        verdict = "Phi-2 encodes ≈ what lm_head reads linearly."
    elif c > 0.5:
        verdict = "SURPRISE: Linear L31 readout >> Phi-2's own lm_head."
    lines.append(f"## Verdict\n{verdict}\n")

    text = "\n".join(lines)
    with open(f"{OUT_DIR}/experiment_summary.md", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
