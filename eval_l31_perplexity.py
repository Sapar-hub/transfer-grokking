import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os, csv

from utils import DEVICE, P

ARTIFACTS = "artifacts"
CE_DIR = f"{ARTIFACTS}/ce_projection"
OUT_DIR = f"{ARTIFACTS}/l31_patch"
os.makedirs(OUT_DIR, exist_ok=True)

D_SMALL = 128
D_PHI2 = 2560
PATCH_LAYER = 31
ALPHAS = [0.0, 0.5, 1.0]
BATCH_SIZE = 16
MAX_SEQ_LEN = 64
NUM_SAMPLES = 300


def get_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]


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


def eval_last_token_loss(model, tokenizer, dataset, W, h_A_pool, alpha):
    model.eval()
    all_losses = []
    rng = np.random.RandomState(42)

    # Tokenize texts, limit to NUM_SAMPLES
    sequences = []
    for item in dataset:
        if len(sequences) >= NUM_SAMPLES:
            break
        text = item["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, truncation=True, max_length=MAX_SEQ_LEN)
        if len(ids) >= 3:
            sequences.append(ids)

    print(f"  {len(sequences)} sequences loaded.")

    n_batches = (len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_i, start in enumerate(range(0, len(sequences), BATCH_SIZE)):
        if batch_i % 10 == 0:
            print(f"    batch {batch_i}/{n_batches}")

        batch_ids = sequences[start:start + BATCH_SIZE]

        # Input: all tokens except last. Target: last token.
        inputs = [ids[:-1] for ids in batch_ids]
        targets = torch.tensor([ids[-1] for ids in batch_ids])

        # Pad inputs
        max_len = max(len(inp) for inp in inputs)
        padded = torch.zeros(len(inputs), max_len, dtype=torch.long)
        mask = torch.zeros(len(inputs), max_len, dtype=torch.long)
        for i, inp in enumerate(inputs):
            padded[i, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            mask[i, :len(inp)] = 1

        # Sample h_A vectors from pool
        idx = rng.randint(0, len(h_A_pool), size=len(inputs))
        batch_h_A = h_A_pool[idx]

        # Register patch hook if needed
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


def main():
    print("=" * 60)
    print("L31 Degradation Eval: last-token perplexity on WikiText-2")
    print("=" * 60)

    print("\n[0] Loading data...")
    small_acts = np.load(f"{ARTIFACTS}/small_model_activations.npy")
    _, test_idx = get_split()
    h_A_test = small_acts[test_idx]
    print(f"  Small model test activations: {h_A_test.shape}")

    print("\n[1] Loading W_ce.pth...")
    W_ce = nn.Linear(D_SMALL, D_PHI2, bias=False)
    W_ce.load_state_dict(torch.load(f"{CE_DIR}/W_ce.pth", map_location=DEVICE, weights_only=True))
    W_ce.requires_grad_(False)
    print("  W_CE loaded.")

    print("\n[2] Loading Phi-2...")
    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    phi2.eval()
    print("  Phi-2 loaded.")

    print("\n[3] Loading WikiText-2 validation set...")
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    print(f"  {len(wiki)} examples.")

    print(f"\n[4] Alpha sweep (last-token loss at L{PATCH_LAYER})...")
    results = []
    for alpha in ALPHAS:
        loss, ppl = eval_last_token_loss(phi2, tokenizer, wiki, W_ce, h_A_test, alpha)
        results.append((alpha, loss, ppl))
        print(f"  alpha={alpha:.1f}: last_token_loss={loss:.4f}  last_token_ppl={ppl:.4f}")

    path = f"{OUT_DIR}/perplexity_sweep.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "last_token_loss", "last_token_perplexity"])
        w.writerows(results)
    print(f"  Saved: {path}")

    baseline_loss = results[0][1]
    lines = ["# L31 Degradation: WikiText-2 last-token perplexity\n"]
    lines.append("| Alpha | Last-Token Loss | Last-Token PPL |")
    lines.append("|-------|----------------|----------------|")
    for alpha, loss, ppl in results:
        delta = loss - baseline_loss
        lines.append(f"| {alpha:.1f} | {loss:.4f} ({delta:+.4f}) | {ppl:.4f} |")
    lines.append(f"\nBaseline (no patch): loss={baseline_loss:.4f}, ppl={np.exp(baseline_loss):.4f}")
    if results[-1][1] > baseline_loss + 0.1:
        lines.append("\n**Verdict**: α=1.0 degrades last-token prediction — patch corrupts general LM.")
    elif results[-1][1] > baseline_loss + 0.01:
        lines.append("\n**Verdict**: Slight degradation at α=1.0 — patch has mild negative effect.")
    else:
        lines.append("\n**Verdict**: No meaningful degradation — patch at L31 does not harm general LM.")

    text = "\n".join(lines)
    with open(f"{OUT_DIR}/perplexity_eval.md", "w") as f:
        f.write(text)
    print(text)

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
