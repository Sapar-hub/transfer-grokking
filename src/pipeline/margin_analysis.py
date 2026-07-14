import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv, pickle

from utils import DEVICE, P

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/self_projection"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else "42,43,44")]
OP = sys.argv[2] if len(sys.argv) > 2 else "add"
SUFFIX = "" if OP == "add" else "_mult"
OP_SYMBOL = {"add": "+", "mult": "*"}
D_SMALL = 128
D_PHI2 = 2560
PATCH_LAYER = 31
BATCH_SIZE = 64

# α grids per model
GRID_WCE_RAW = sorted(set(np.linspace(0, 0.8, 33)) | set(np.linspace(0.8, 1.0, 41)))
GRID_WCE_SCALED = list(np.linspace(0, 0.3, 121))
GRID_SELF = list(np.linspace(0, 1.0, 101))
GRID_PCA = list(np.linspace(0, 1.0, 101))

# Perplexity grid
PPL_ALPHAS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0]


def get_split():
    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    return idx[:split], idx[split:]


def load_lin(path, indim, outdim):
    W = nn.Linear(indim, outdim, bias=False)
    W.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    W.eval()
    W.requires_grad_(False)
    return W


def crossing_alphas(h_L31_t, h_patch_t, alphas, y_true, lm_head_w_num, final_ln):
    """Per-example crossing-α: first α where prediction matches label.
    
    Returns array of length N; NaN if never correct.
    """
    N = h_L31_t.shape[0]
    crossing = np.full(N, np.nan)
    for a in sorted(alphas):
        h_p = (1 - a) * h_L31_t + a * h_patch_t
        with torch.no_grad():
            h_n = final_ln(h_p)
            logits = h_n @ lm_head_w_num.T
        preds = logits.argmax(dim=1).numpy()
        correct = (preds == y_true)
        first = np.isnan(crossing) & correct
        crossing[first] = a
    return crossing


def per_class_summary(crossing, y_true):
    """Per-class statistics of crossing-α.
    
    Returns dict: class -> {n, n_unreachable, n_baseline, mean, std, median, ...}
    """
    stats = {}
    for c in range(P):
        mask = y_true == c
        x = crossing[mask]
        n_total = len(x)
        n_unreachable = int(np.isnan(x).sum())
        n_baseline = int((x == 0.0).sum())
        x_v = x[~np.isnan(x)]
        stats[c] = {
            "n": n_total,
            "n_unreachable": n_unreachable,
            "n_baseline": n_baseline,
            "frac_unreachable": n_unreachable / n_total,
            "frac_baseline": n_baseline / n_total,
            "mean": float(x_v.mean()) if len(x_v) > 0 else None,
            "std": float(x_v.std()) if len(x_v) > 0 else None,
            "median": float(np.median(x_v)) if len(x_v) > 0 else None,
        }
    return stats


def report_model(name, crossing, y_true):
    n_test = len(y_true)
    n_reach = int((~np.isnan(crossing)).sum())
    n_unreach = int(np.isnan(crossing).sum())
    v = crossing[~np.isnan(crossing)]
    c_censored = crossing.copy()
    c_censored[np.isnan(c_censored)] = 1.0

    print(f"  Reachable: {n_reach}/{n_test}  Unreachable: {n_unreach} ({100*n_unreach/n_test:.1f}%)")
    if len(v) > 0:
        print(f"  Uncensored: μ={v.mean():.4f} σ={v.std():.4f} med={np.median(v):.4f}")
        print(f"    P5={np.percentile(v,5):.4f} P25={np.percentile(v,25):.4f} "
              f"P75={np.percentile(v,75):.4f} P95={np.percentile(v,95):.4f}")
    print(f"  Censored@1.0: μ={c_censored.mean():.4f} σ={c_censored.std():.4f} "
          f"med={np.median(c_censored):.4f}")

    pc = per_class_summary(crossing, y_true)
    pc_stds = [pc[c]["std"] for c in range(P) if pc[c]["std"] is not None]
    pc_frac_baseline = [pc[c]["frac_baseline"] for c in range(P)]
    pc_frac_unreach = [pc[c]["frac_unreachable"] for c in range(P)]

    print(f"  Per-class within-σ: μ={np.mean(pc_stds):.4f} med={np.median(pc_stds):.4f} "
          f"P25={np.percentile(pc_stds,25):.4f} P75={np.percentile(pc_stds,75):.4f}")
    print(f"  Per-class frac_baseline: μ={np.mean(pc_frac_baseline):.4f} σ={np.std(pc_frac_baseline):.4f}")
    print(f"  Per-class frac_unreach: μ={np.mean(pc_frac_unreach):.4f} σ={np.std(pc_frac_unreach):.4f}")

    # Key diagnostic: within-class σ distribution
    bins_small = sum(1 for s in pc_stds if s < 0.1)
    bins_medium = sum(1 for s in pc_stds if 0.1 <= s < 0.3)
    bins_large = sum(1 for s in pc_stds if s >= 0.3)
    print(f"  Within-σ dist: small(<0.1)={bins_small} mid(0.1-0.3)={bins_medium} "
          f"large(≥0.3)={bins_large}")

    classes_any_unreach = sum(1 for c in range(P) if pc[c]["n_unreachable"] > 0)
    classes_high_baseline = sum(1 for c in range(P) if pc[c]["frac_baseline"] > 0.5)
    print(f"  Classes with any unreachable: {classes_any_unreach}/{P}")
    print(f"  Classes with >50% baseline-correct: {classes_high_baseline}/{P}")

    return pc


def main():
    print("=" * 60)
    print(f"Margin Analysis (seeds={SEEDS}, op={OP})")
    print("=" * 60)

    print("\n[0] Loading data...")
    labels = np.load(f"{ARTIFACTS}/mod_arithmetic_labels{SUFFIX}.npy", allow_pickle=True)
    _, test_idx = get_split()
    y_true = labels[test_idx]
    n_test = len(test_idx)
    print(f"  Test set: {n_test}")

    small_acts = np.load(f"{ARTIFACTS}/small_model_activations{SUFFIX}.npy")
    phi2_acts = np.load(f"{ARTIFACTS}/cross_model/microsoft_phi_2_L31_acts.npy")

    norm_small = np.linalg.norm(small_acts[test_idx], axis=1).mean()
    norm_phi2 = np.linalg.norm(phi2_acts[test_idx], axis=1).mean()
    SCALE = norm_phi2 / norm_small
    print(f"  Scale factor: {SCALE:.1f}")

    print("\n[1] Fitting PCA (128 components)...")
    rng = np.random.RandomState(42)
    full_idx = np.arange(P * P)
    rng.shuffle(full_idx)
    pca = PCA(n_components=128)
    pca.fit(phi2_acts[full_idx[:int(len(full_idx) * 0.7)]])
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    print("\n[2] Loading Phi-2...")
    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", revision="810d367871c1d460086d9f82db8696f2e0a0fcd0",
        torch_dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2", revision="810d367871c1d460086d9f82db8696f2e0a0fcd0"
    )
    tokenizer.pad_token = tokenizer.eos_token
    phi2.eval()
    print("  Phi-2 loaded.")

    number_ids = [tokenizer.encode(str(n))[0] for n in range(P)]
    num_tid = torch.tensor(number_ids)
    lm_head_w_num = phi2.lm_head.weight[num_tid]
    final_ln = phi2.model.final_layernorm

    print("\n[3] Extracting L31 hidden states (test set)...")
    h_L31 = np.zeros((n_test, D_PHI2), dtype=np.float32)
    for start in range(0, n_test, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_test)
        batch = test_idx[start:end]
        pairs = [(int(i // P), int(i % P)) for i in batch]
        prompts = [f"# ({a} {OP_SYMBOL[OP]} {b}) % 97 =" for a, b in pairs]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        mask_len = tokenized.attention_mask.sum(dim=1) - 1
        with torch.no_grad():
            outputs = phi2(**tokenized, output_hidden_states=True)
            h = outputs.hidden_states[PATCH_LAYER]
        for i in range(end - start):
            h_L31[start + i] = h[i, mask_len[i]].float().numpy()
        print(f"  {end}/{n_test}")
    h_L31_t = torch.from_numpy(h_L31).float()
    print("  L31 extraction done.")

    h_A_t = torch.from_numpy(small_acts[test_idx]).float()
    phi2_t = torch.from_numpy(phi2_acts[test_idx]).float()
    pca_t = torch.from_numpy(pca.transform(phi2_acts[test_idx]).astype(np.float32))

    # Build model list
    model_defs = []

    # W_ce (raw) — multi-seed
    for seed in SEEDS:
        path = f"{ARTIFACTS}/ce_projection/seeds/W_ce_seed{seed}.pth"
        if os.path.exists(path):
            model_defs.append((f"W_ce (raw) seed={seed}", path, D_SMALL, D_PHI2, "h_A", GRID_WCE_RAW))
            print(f"  Added: W_ce (raw) seed={seed}")
        else:
            print(f"  WARNING: {path} not found, skipping seed={seed}")

    # Control models (seed=42 only)
    controls = [
        ("W_ce (1370x) seed=42",
         f"{OUT_DIR}/seeds/W_ce_scaled_seed42.pth", D_SMALL, D_PHI2, "h_A_scaled", GRID_WCE_SCALED),
        ("W_self seed=42",
         f"{OUT_DIR}/seeds/W_self_seed42.pth", D_PHI2, D_PHI2, "phi2", GRID_SELF),
        ("PCA-128 seed=42",
         f"{OUT_DIR}/seeds/W_ce_pca_seed42.pth", D_SMALL, D_PHI2, "pca", GRID_PCA),
    ]
    for name, path, idim, odim, src, grid in controls:
        if os.path.exists(path):
            model_defs.append((name, path, idim, odim, src, grid))
            print(f"  Added: {name}")

    # Compute patches
    print("\n[4] Computing patches...")
    all_patches = {}
    for name, path, indim, outdim, source, grid in model_defs:
        W = load_lin(path, indim, outdim)
        with torch.no_grad():
            if source == "h_A":
                patch = W(h_A_t).numpy().astype(np.float32)
            elif source == "h_A_scaled":
                patch = W(h_A_t * SCALE).numpy().astype(np.float32)
            elif source == "phi2":
                patch = W(phi2_t).numpy().astype(np.float32)
            elif source == "pca":
                patch = W(pca_t).numpy().astype(np.float32)
        all_patches[name] = (patch, grid)
        print(f"  {name}: {patch.shape} done")

    # Crossing-α analysis
    print("\n[5] Running crossing-α analysis...")
    results = {}
    for name in all_patches:
        patch, grid = all_patches[name]
        print(f"\n--- {name} ({len(grid)} α values) ---")
        h_patch_t = torch.from_numpy(patch).float()
        crossing = crossing_alphas(h_L31_t, h_patch_t, grid, y_true, lm_head_w_num, final_ln)
        pc = report_model(name, crossing, y_true)
        results[name] = {"crossing": crossing, "grid": grid, "per_class": pc}

    # Save
    out_pkl = f"{OUT_DIR}/crossing_alpha_data{SUFFIX}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved: {out_pkl}")

    # Per-class CSV
    csv_path = f"{OUT_DIR}/per_class_crossing{SUFFIX}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "class", "n", "n_unreachable", "n_baseline",
                     "frac_unreachable", "frac_baseline", "mean", "std", "median"])
        for name in results:
            pc = results[name]["per_class"]
            for c in range(P):
                s = pc[c]
                w.writerow([name, c, s["n"], s["n_unreachable"], s["n_baseline"],
                           f"{s['frac_unreachable']:.4f}", f"{s['frac_baseline']:.4f}",
                           f"{s['mean']:.4f}" if s["mean"] is not None else "",
                           f"{s['std']:.4f}" if s["std"] is not None else "",
                           f"{s['median']:.4f}" if s["median"] is not None else ""])
    print(f"  Saved: {csv_path}")

    # --- Plot: crossing-α histograms ---
    n_models = len(results)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    colors = plt.cm.Set1(np.linspace(0, 1, n_models))
    for idx, name in enumerate(results):
        ax = axes[idx]
        crossing = results[name]["crossing"]
        grid = results[name]["grid"]
        valid = crossing[~np.isnan(crossing)]
        n_unreach = np.isnan(crossing).sum()
        res = grid[1] - grid[0] if len(grid) > 1 else 0.01

        ax.hist(valid, bins=np.linspace(-0.02, 1.02, 105),
                color=colors[idx], alpha=0.7, edgecolor='white', linewidth=0.3)
        ax.axvline(valid.mean(), color=colors[idx], linestyle='--',
                   label=f"μ={valid.mean():.3f}" if len(valid) > 0 else "no reachable")
        ax.set_title(f"{name}\n(n={len(valid)}, resid={res:.4f})", fontsize=9)
        ax.set_xlabel("Crossing α"); ax.set_ylabel("Count")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        if n_unreach > 0:
            ax.text(0.95, 0.95, f"unreach={n_unreach}\n({100*n_unreach/len(crossing):.1f}%)",
                    transform=ax.transAxes, ha='right', va='top', fontsize=7, color='red')

    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.suptitle("Per-Example Crossing-α Distributions", fontsize=13)
    plt.tight_layout()
    fig_path = f"{OUT_DIR}/crossing_alpha_hist{SUFFIX}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {fig_path}")
    plt.close(fig)

    # Summary table
    print("\n\n=== Summary (uncensored) ===")
    hdr = f"{'Model':<30} {'N':>5} {'μ':>7} {'σ':>7} {'med':>7} {'P5':>7} {'P25':>7} {'P75':>7} {'P95':>7} {'unreach':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name in results:
        c = results[name]["crossing"]
        v = c[~np.isnan(c)]
        unreach = np.isnan(c).sum()
        if len(v) > 0:
            print(f"{name:<30} {len(v):>5} {v.mean():>7.4f} {v.std():>7.4f} "
                  f"{np.median(v):>7.4f} {np.percentile(v,5):>7.4f} "
                  f"{np.percentile(v,25):>7.4f} {np.percentile(v,75):>7.4f} "
                  f"{np.percentile(v,95):>7.4f} {unreach:>8}")

    print("\n=== Within-class σ summary ===")
    for name in results:
        pc = results[name]["per_class"]
        pc_stds = [pc[c]["std"] for c in range(P) if pc[c]["std"] is not None]
        print(f"  {name:<30} μ(σ)={np.mean(pc_stds):.4f} med(σ)={np.median(pc_stds):.4f} "
              f"P25={np.percentile(pc_stds,25):.4f} P75={np.percentile(pc_stds,75):.4f}")

    # --- Perplexity fine grid ---
    print("\n\n[6] Perplexity fine grid (WikiText-2)...")
    try:
        wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        print(f"  WikiText-2 loaded: {len(wiki)} examples.")
    except Exception as e:
        print(f"  Could not load WikiText-2: {e}")
        wiki = None

    if wiki is not None:
        rng_ppl = np.random.RandomState(42)
        n_samples = 300
        max_seq_len = 64

        # Collect sequences (same approach as self_projection.py)
        sequences = []
        for item in wiki:
            if len(sequences) >= n_samples:
                break
            text = item["text"].strip()
            if not text:
                continue
            ids = tokenizer.encode(text, truncation=True, max_length=max_seq_len)
            if len(ids) >= 3:
                sequences.append(ids)
        print(f"  Collected {len(sequences)} sequences.")

        # Run once to extract L31 for each sequence + collect targets
        all_h_L31_ppl = []
        all_targets = []
        for ids in sequences:
            inp = ids[:-1]
            tgt = ids[-1]
            prompt_ids = torch.tensor([inp])
            with torch.no_grad():
                outputs = phi2(prompt_ids, output_hidden_states=True)
                h = outputs.hidden_states[PATCH_LAYER]
                h_last = h[0, -1].float().numpy()
            all_h_L31_ppl.append(h_last)
            all_targets.append(tgt)

        h_L31_ppl = np.array(all_h_L31_ppl, dtype=np.float32)
        h_L31_ppl_t = torch.from_numpy(h_L31_ppl).float()
        targets_t = torch.tensor(all_targets)
        n_ppl = len(sequences)
        print(f"  Extracted L31 states: {h_L31_ppl.shape}")

        # Pre-extract lm_head weight for full vocab
        lm_head_w_full = phi2.lm_head.weight
        h_A_pool = small_acts  # pool of grokked activations for random pairing

        ppl_results = []
        for name in all_patches:
            patch_arr, _ = all_patches[name]
            print(f"\n  Perplexity: {name}")

            if "W_self" in name:
                # W_self: patch depends on the input itself, need per-sample
                print("    W_self perplexity not implemented in accelerated mode (needs per-sample patching)")
                ppl_results.append({"model": name, "alpha": 0.0, "loss": float('nan'), "ppl": float('nan')})
                continue

            W = None
            for mname, path, indim, outdim, source, _ in model_defs:
                if mname == name:
                    W = load_lin(path, indim, outdim)
                    break
            if W is None:
                continue

            # For W_ce (raw) and variants: use random h_A from pool
            for alpha in PPL_ALPHAS:
                rng_ppl = np.random.RandomState(42)
                losses = []
                batch_size = 32
                for start in range(0, n_ppl, batch_size):
                    end = min(start + batch_size, n_ppl)
                    b_h_L31 = h_L31_ppl_t[start:end].clone()
                    b_targets = targets_t[start:end]
                    b_n = end - start

                    # Random h_A from pool
                    idx = rng_ppl.randint(0, len(h_A_pool), size=b_n)
                    b_h_A = torch.from_numpy(h_A_pool[idx]).float()

                    with torch.no_grad():
                        b_patch = W(b_h_A)
                        if alpha == 0.0:
                            h_p = b_h_L31
                        elif alpha == 1.0:
                            h_p = b_patch
                        else:
                            h_p = (1 - alpha) * b_h_L31 + alpha * b_patch
                        h_n = final_ln(h_p)
                        logits = h_n @ lm_head_w_full.T
                    loss = F.cross_entropy(logits, b_targets, reduction='none')
                    losses.extend(loss.tolist())

                mean_loss = float(np.mean(losses))
                ppl = float(np.exp(mean_loss))
                ppl_results.append({"model": name, "alpha": alpha, "loss": mean_loss, "ppl": ppl})
                print(f"    α={alpha:.4f}: loss={mean_loss:.4f} ppl={ppl:.1f}")

        # Save perplexity CSV
        ppl_csv = f"{OUT_DIR}/perplexity_fine_grid{SUFFIX}.csv"
        with open(ppl_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "alpha", "loss", "ppl"])
            for r in ppl_results:
                w.writerow([r["model"], r["alpha"], f"{r['loss']:.4f}", f"{r['ppl']:.4f}"])
        print(f"\n  Saved: {ppl_csv}")

        # Perplexity plot
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        models_ppl = set(r["model"] for r in ppl_results)
        for m in models_ppl:
            rows = [r for r in ppl_results if r["model"] == m and not np.isnan(r["ppl"])]
            if rows:
                alphas = [r["alpha"] for r in rows]
                ppls = [r["ppl"] for r in rows]
                ax2.semilogy(alphas, ppls, 'o-', label=m)
        ax2.axhline(y=63.3, color='gray', linestyle='--', alpha=0.5, label='baseline ~63.3')
        ax2.set_xlabel("α"); ax2.set_ylabel("Perplexity (log)")
        ax2.set_title("Perplexity Fine Grid (WikiText-2)")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
        fig_path2 = f"{OUT_DIR}/perplexity_fine_grid{SUFFIX}.png"
        fig2.savefig(fig_path2, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fig_path2}")
        plt.close(fig2)

    print("\nDone.")


if __name__ == "__main__":
    main()
