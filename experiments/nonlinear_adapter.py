import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from utils import DEVICE, P

BASE = "artifacts"
RP_DIR = f"{BASE}/residual_patch"
OUT_DIR = f"{BASE}/nonlinear_adapter"
os.makedirs(OUT_DIR, exist_ok=True)

D_SMALL = 128
D_PHI2 = 2560
D_HIDDEN = 256
N_EPOCHS = 500
LR = 1e-3


def get_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]


class MLP(nn.Module):
    def __init__(self, d_in=D_PHI2, d_hidden=D_HIDDEN, d_out=D_PHI2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def train_and_eval(adapter, h_train, y_train, h_test, y_test, lm_head_sliced, label):
    opt = torch.optim.AdamW(adapter.parameters(), lr=LR)
    best_acc = 0
    for epoch in range(1, N_EPOCHS + 1):
        adapter.train()
        h_adapted = adapter(h_train)
        logits = h_adapted @ lm_head_sliced.T
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 100 == 0 or epoch == 1:
            adapter.eval()
            with torch.no_grad():
                h_a = adapter(h_test)
                logits = h_a @ lm_head_sliced.T
                acc = (logits.argmax(dim=1) == y_test).float().mean().item()
            best_acc = max(best_acc, acc)
            print(f"    [{label}] epoch {epoch:4d}: test_acc={acc:.4f}")
    return best_acc


def main():
    print("=" * 60)
    print("Nonlinear Adapter: MLP between W(h_A) and frozen lm_head")
    print("=" * 60)

    print("\n[0] Loading data...")
    small_acts = np.load(f"{BASE}/small_model_activations.npy")
    labels = np.load(f"{BASE}/mod_arithmetic_labels.npy")
    train_idx, test_idx = get_split()
    X_train = torch.from_numpy(small_acts[train_idx]).float()
    y_train = torch.from_numpy(labels[train_idx]).long()
    X_test = torch.from_numpy(small_acts[test_idx]).float()
    y_test = torch.from_numpy(labels[test_idx]).long()
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    print("\n[1] Precomputing h_proj = W(small_acts)...")
    W = nn.Linear(D_SMALL, D_PHI2, bias=False)
    W.load_state_dict(torch.load(f"{RP_DIR}/W_layer10.pth", map_location=DEVICE, weights_only=True))
    W.requires_grad_(False)
    W.eval()
    with torch.no_grad():
        h_train = W(X_train)
        h_test = W(X_test)
    print(f"  h_train: {h_train.shape}  h_test: {h_test.shape}")

    print("\n[2] Loading Phi-2 lm_head...")
    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    number_ids = [tokenizer.encode(str(n))[0] for n in range(P)]
    lm_head_sliced = phi2.lm_head.weight[number_ids].detach()
    print(f"  lm_head sliced: {lm_head_sliced.shape}")

    print("\n[3] A: Logit lens (W → lm_head, frozen)")
    with torch.no_grad():
        logits = h_test @ lm_head_sliced.T
        acc_a = (logits.argmax(dim=1) == y_test).float().mean().item()
    print(f"  test_acc = {acc_a:.4f}")

    print("\n[4] B: Trained Linear(2560→2560) → lm_head (frozen)")
    linear_adapter = nn.Linear(D_PHI2, D_PHI2, bias=True)
    acc_b = train_and_eval(linear_adapter, h_train, y_train, h_test, y_test,
                           lm_head_sliced, "Linear")
    torch.save(linear_adapter.state_dict(), f"{OUT_DIR}/linear_adapter.pth")

    print("\n[5] C: Trained MLP(2560→256→2560) → lm_head (frozen)")
    mlp_adapter = MLP(D_PHI2, D_HIDDEN, D_PHI2)
    acc_c = train_and_eval(mlp_adapter, h_train, y_train, h_test, y_test,
                           lm_head_sliced, "MLP")
    torch.save(mlp_adapter.state_dict(), f"{OUT_DIR}/mlp_adapter.pth")

    print("\n[6] D: Trained MLP + trainable lm_head (ceiling)")
    mlp2 = MLP(D_PHI2, D_HIDDEN, D_PHI2)
    lm_head_tune = nn.Linear(D_PHI2, P, bias=False)
    opt = torch.optim.AdamW(list(mlp2.parameters()) + list(lm_head_tune.parameters()), lr=LR)
    for epoch in range(1, N_EPOCHS + 1):
        mlp2.train()
        lm_head_tune.train()
        h_adapted = mlp2(h_train)
        logits = lm_head_tune(h_adapted)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 100 == 0 or epoch == 1:
            mlp2.eval()
            lm_head_tune.eval()
            with torch.no_grad():
                h_a = mlp2(h_test)
                logits = lm_head_tune(h_a)
                acc = (logits.argmax(dim=1) == y_test).float().mean().item()
            print(f"    [MLP+lm_head] epoch {epoch:4d}: test_acc={acc:.4f}")
    acc_d = acc

    print("\n[7] Summary")
    lines = []
    lines.append("# Nonlinear Adapter Experiment Summary\n")
    lines.append("## Conditions\n")
    lines.append("| # | Adapter | lm_head |")
    lines.append("|---|---------|---------|")
    lines.append("| A | none (logit lens) | frozen |")
    lines.append("| B | trained Linear(2560→2560) | frozen |")
    lines.append("| C | trained MLP(2560→256→2560) | frozen |")
    lines.append("| D | trained MLP(2560→256→2560) | trainable |\n")
    lines.append("## Results\n")
    lines.append("| Cond | Test Acc | vs logit lens | vs linear adapter |")
    lines.append("|------|----------|---------------|-------------------|")
    lines.append(f"| A | {acc_a:.4f} | — | — |")
    lines.append(f"| B | {acc_b:.4f} | {acc_b-acc_a:+.4f} | — |")
    lines.append(f"| C | {acc_c:.4f} | {acc_c-acc_a:+.4f} | {acc_c-acc_b:+.4f} |")
    lines.append(f"| D | {acc_d:.4f} | {acc_d-acc_a:+.4f} | {acc_d-acc_b:+.4f} |")
    lines.append("")

    verdict = ""
    if acc_c > 0.8:
        verdict = "CONFIRMED: MLP reshapes Fourier features for frozen lm_head."
    elif acc_c > 0.5:
        verdict = "PARTIAL: MLP helps but lm_head still struggles to read."
    elif acc_c > acc_b + 0.02:
        verdict = "Nonlinear > linear, but both far below usable."
    else:
        verdict = "REJECTED: Nonlinear adapter does not help lm_head."

    if acc_d > acc_c + 0.1:
        verdict += "\nTrainable lm_head >> frozen: receiver must adapt to injected state."
    elif acc_d > acc_c:
        verdict += "\nTrainable lm_head helps modestly."

    lines.append(f"## Verdict\n{verdict}\n")
    text = "\n".join(lines)
    with open(f"{OUT_DIR}/experiment_summary.md", "w") as f:
        f.write(text)
    print(text)

    # Save curves
    print(f"\nDone. Artifacts in {OUT_DIR}/")


if __name__ == "__main__":
    main()
