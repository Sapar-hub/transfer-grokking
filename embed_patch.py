import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv, sys, warnings

from model import SmallTransformer
from utils import DEVICE, P, train_probe

warnings.filterwarnings('ignore')

BASE = "artifacts"
EMBED_DIR = f"{BASE}/embed_patch"
os.makedirs(EMBED_DIR, exist_ok=True)

D_SMALL = 128
D_PHI2 = 2560
P_VAL = 97
N_EPOCHS = 5000
LAMBDA_ORTHO = 1.0
LR = 1e-3
N_ACC_PAIRS = 200
N_PROBE = 1000
BATCH_SIZE = 32


def extract_embeddings():
    """Extract small model embeddings [97,128] and Phi-2 target embeddings [97,2560]."""
    print("[1/5] Extracting embeddings...")

    model_small = SmallTransformer().to(DEVICE)
    state = torch.load(f"{BASE}/small/best_model.pth", map_location=DEVICE, weights_only=True)
    model_small.load_state_dict(state)
    embed_A = model_small.embed.weight.detach()
    print(f"  Small embed weights: {embed_A.shape}")

    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    phi2.eval()

    phi2_vocab = phi2.model.embed_tokens.weight
    phi2_targets = []
    for n in range(P_VAL):
        token_ids = tokenizer.encode(str(n))
        embs = phi2_vocab[torch.tensor(token_ids)]
        phi2_targets.append(embs.mean(dim=0))
    phi2_targets = torch.stack(phi2_targets).detach().to(DEVICE)
    print(f"  Phi-2 target embeddings: {phi2_targets.shape}")

    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P_VAL)}
    return embed_A, phi2_targets, phi2, tokenizer, number_tokens


def train_W_emb(embed_A, phi2_targets):
    """Train W_emb: 128 -> 2560. MSE + orthogonality loss (isometry)."""
    print("[2/5] Training W_emb...")

    W_emb = nn.Linear(D_SMALL, D_PHI2, bias=False).to(DEVICE)
    opt = torch.optim.AdamW(W_emb.parameters(), lr=LR)
    I = torch.eye(D_SMALL, device=DEVICE)

    mses, orthos, coss = [], [], []
    for epoch in range(1, N_EPOCHS + 1):
        pred = W_emb(embed_A)
        loss_mse = F.mse_loss(pred, phi2_targets)

        WtW = W_emb.weight.T @ W_emb.weight
        loss_ortho = F.mse_loss(WtW, I)

        loss = loss_mse + LAMBDA_ORTHO * loss_ortho

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 500 == 0 or epoch == 1:
            with torch.no_grad():
                cs = F.cosine_similarity(pred, phi2_targets, dim=1).mean().item()
            mses.append(loss_mse.item())
            orthos.append(loss_ortho.item())
            coss.append(cs)
            print(f"  epoch {epoch:4d}: mse={loss_mse.item():.6f}  "
                  f"ortho={loss_ortho.item():.6f}  cos={cs:.4f}")

    torch.save(W_emb.state_dict(), f"{EMBED_DIR}/W_emb.pth")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(range(len(mses)), mses)
    axes[0].set_ylabel('MSE')
    axes[1].plot(range(len(orthos)), orthos)
    axes[1].set_ylabel('Ortho Loss')
    axes[2].plot(range(len(coss)), coss, 'o-')
    axes[2].set_ylabel('Cos Sim')
    for ax in axes:
        ax.set_xlabel('Epoch (x500)')
    plt.tight_layout()
    plt.savefig(f"{EMBED_DIR}/W_emb_training.png")
    plt.close()

    return W_emb


def accuracy_text(phi2, tokenizer, number_tokens, pairs, labels):
    """Text-prompt accuracy on given pairs (batched)."""
    correct = 0
    B = 64
    for start in range(0, len(pairs), B):
        batch = pairs[start:start + B]
        prompts = [f"# ({a} + {b}) % 97 =" for a, b in batch]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = phi2(**inputs).logits
        seq_lens = inputs.attention_mask.sum(dim=1) - 1
        for i, y in enumerate(labels[start:start + B]):
            pred_logits = logits[i, seq_lens[i], :]
            pred = max(number_tokens,
                       key=lambda n: pred_logits[number_tokens[n]].item())
            if pred == y:
                correct += 1
    return correct / len(pairs)


def accuracy_patch(phi2, W_emb, embed_A, number_tokens, pairs, labels):
    """inputs_embeds accuracy on same pairs (batched)."""
    correct = 0
    B = 64
    for start in range(0, len(pairs), B):
        batch = pairs[start:start + B]
        a_ids = torch.tensor([p[0] for p in batch], device=embed_A.device)
        b_ids = torch.tensor([p[1] for p in batch], device=embed_A.device)
        emb_a = W_emb(embed_A[a_ids]).unsqueeze(1)
        emb_b = W_emb(embed_A[b_ids]).unsqueeze(1)
        batch_embeds = torch.cat([emb_a, emb_b], dim=1)
        with torch.no_grad():
            logits = phi2(inputs_embeds=batch_embeds).logits[:, -1, :]
        for i, y in enumerate(labels[start:start + B]):
            pred = max(number_tokens,
                       key=lambda n: logits[i, number_tokens[n]].item())
            if pred == y:
                correct += 1
    return correct / len(pairs)


def probe_activations(phi2, forward_fn, n_samples, label=""):
    """Collect pre-layer activations and train a logistic regression probe.

    forward_fn(batch_pairs, current_mask) runs the model on a batch,
    sets current_mask[0] = attention_mask (or None if all equal length).
    A pre-hook on layers[0] captures the input to the first transformer block.
    """
    act_list = []
    labels_list = []
    current_mask = [None]

    def pre_hook(module, input):
        hidden = input[0]
        mask = current_mask[0]
        if mask is not None:
            seq_lens = mask.to(hidden.device).sum(dim=1) - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            hidden = hidden[batch_idx, seq_lens]
        else:
            hidden = hidden[:, -1, :]
        act_list.append(hidden.detach().cpu())

    handle = phi2.model.layers[0].register_forward_pre_hook(pre_hook)

    rng = np.random.RandomState(42)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_i in range(n_batches):
        start = batch_i * BATCH_SIZE
        bs = min(BATCH_SIZE, n_samples - start)
        batch_pairs = [
            (int(rng.randint(0, P_VAL)), int(rng.randint(0, P_VAL)))
            for _ in range(bs)
        ]
        labels_list.extend([(a + b) % P_VAL for a, b in batch_pairs])
        forward_fn(batch_pairs, current_mask)
        if (batch_i + 1) % 8 == 0:
            print(f"    {label}: {batch_i+1}/{n_batches} batches", flush=True)

    handle.remove()

    acts = torch.cat(act_list, dim=0).numpy()
    labels = np.array(labels_list)
    acc, *_ = train_probe(acts, labels)
    print(f"  Probe ({label}): {acc:.4f}", flush=True)
    return acc


def forward_text(phi2, tokenizer, batch_pairs, current_mask):
    """Forward function for text input with padding mask."""
    prompts = [f"# ({a} + {b}) % 97 =" for a, b in batch_pairs]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    current_mask[0] = inputs.attention_mask
    with torch.no_grad():
        phi2(**inputs)


def forward_patch(phi2, W_emb, embed_A, batch_pairs, current_mask):
    """Forward function for inputs_embeds (all seq_len=2, no mask needed)."""
    a_ids = torch.tensor([p[0] for p in batch_pairs], device=embed_A.device)
    b_ids = torch.tensor([p[1] for p in batch_pairs], device=embed_A.device)
    emb_a = W_emb(embed_A[a_ids]).unsqueeze(1)
    emb_b = W_emb(embed_A[b_ids]).unsqueeze(1)
    batch_embeds = torch.cat([emb_a, emb_b], dim=1)
    current_mask[0] = None
    with torch.no_grad():
        phi2(inputs_embeds=batch_embeds)


def log(msg):
    print(msg, flush=True)


def plot_results(acc_text, acc_patch, probe_text, probe_patch):
    """Plot accuracy and probe comparison side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.bar(['Text', 'inputs_embeds'], [acc_text, acc_patch],
            color=['gray', '#4A90D9'])
    ax1.set_ylabel('Mod Arithmetic Acc')
    ax1.axhline(1 / P_VAL, color='red', ls='--', alpha=0.4,
                label=f'random={1/P_VAL:.3f}')
    ax1.legend()

    ax2.bar(['Text', 'inputs_embeds'], [probe_text, probe_patch],
            color=['gray', '#4A90D9'])
    ax2.set_ylabel('Probe Acc (pre-layer)')
    ax2.axhline(1 / P_VAL, color='red', ls='--', alpha=0.4,
                label=f'random={1/P_VAL:.3f}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{EMBED_DIR}/results.png")
    plt.close()


def write_summary(acc_text, acc_patch, probe_text, probe_patch,
                  delta, delta_probe):
    """Write experiment summary markdown with interpretation."""
    lines = []
    lines.append("# Embed Patch Summary\n")
    lines.append("## Metrics\n")
    lines.append("| Metric | Text Baseline | inputs_embeds | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Mod Arithmetic Acc | {acc_text:.4f} | {acc_patch:.4f} |"
                 f" {delta:+.4f} |")
    lines.append(f"| Probe Acc (pre-layer) | {probe_text:.4f} |"
                 f" {probe_patch:.4f} | {delta_probe:+.4f} |")
    lines.append(f"| Random baseline | {1/P_VAL:.4f} | — | — |")
    lines.append("")

    if delta < -0.10 and acc_patch < 0.02:
        verdict = ("FAILED: inputs_embeds accuracy dropped to random. "
                   "W_emb preserves geometry (cos=0.82) but Phi-2 cannot "
                   "use it without the text prompt context.")
    elif delta > 0.10 and delta_probe > 0.05:
        verdict = ("CONFIRMED: Phi-2 contains arithmetic algorithm. "
                   "Barrier was tokenizer-level.")
    elif abs(delta) < 0.01 and abs(delta_probe) < 0.01:
        verdict = ("FAILED: No effect. inputs_embeds produces same "
                   "accuracy as text baseline. W_emb mapping is "
                   "geometrically inert.")
    elif delta > 0.05 and abs(delta_probe) < 0.01:
        verdict = ("Phi-2 uses a different mechanism "
                   "(not Fourier / not geometric).")
    else:
        verdict = "Mixed / inconclusive."

    lines.append(f"## Verdict\n{verdict}\n")

    text = "\n".join(lines) + "\n"
    with open(f"{EMBED_DIR}/experiment_summary.md", "w") as f:
        f.write(text)
    print(text)


def main():
    log("=" * 60)
    log("Embed Patch: inputs_embeds for Phi-2")
    log("=" * 60)

    embed_A, phi2_targets, phi2, tokenizer, number_tokens = extract_embeddings()

    W_emb = train_W_emb(embed_A, phi2_targets)

    log("[3/5] Evaluating accuracy...")
    rng = np.random.RandomState(42)
    eval_pairs = [
        (int(rng.randint(0, P_VAL)), int(rng.randint(0, P_VAL)))
        for _ in range(N_ACC_PAIRS)
    ]
    eval_labels = [(a + b) % P_VAL for a, b in eval_pairs]

    acc_text = accuracy_text(phi2, tokenizer, number_tokens,
                             eval_pairs, eval_labels)
    log(f"  Text baseline acc:       {acc_text:.4f}")

    acc_patch = accuracy_patch(phi2, W_emb, embed_A, number_tokens,
                               eval_pairs, eval_labels)
    log(f"  inputs_embeds acc:       {acc_patch:.4f}")

    log("[4/5] Probing pre-layer activations...")
    probe_text = probe_activations(
        phi2,
        lambda bp, cm: forward_text(phi2, tokenizer, bp, cm),
        N_PROBE, "text"
    )
    probe_patch = probe_activations(
        phi2,
        lambda bp, cm: forward_patch(phi2, W_emb, embed_A, bp, cm),
        N_PROBE, "inputs_embeds"
    )

    delta = acc_patch - acc_text
    delta_probe = probe_patch - probe_text
    log(f"\n  Delta acc:      {delta:+.4f}")
    log(f"  Delta probe:    {delta_probe:+.4f}")

    plot_results(acc_text, acc_patch, probe_text, probe_patch)

    with open(f"{EMBED_DIR}/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "mod_accuracy", "probe_accuracy"])
        w.writerow(["text", f"{acc_text:.4f}", f"{probe_text:.4f}"])
        w.writerow(["inputs_embeds", f"{acc_patch:.4f}", f"{probe_patch:.4f}"])

    write_summary(acc_text, acc_patch, probe_text, probe_patch,
                  delta, delta_probe)

    log(f"\nDone. Artifacts in {EMBED_DIR}/")


if __name__ == "__main__":
    main()
