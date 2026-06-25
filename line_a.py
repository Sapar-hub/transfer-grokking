import torch, torch.nn as nn, numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv, time

from model import make_model, CFG_SMALL, CFG_BIG
from utils import DEVICE, P

BASE = "artifacts"
ACT_DIR = f"{BASE}/activations"
SMALL_DIR = f"{BASE}/small"
BIG_DIR = f"{BASE}/big"
PROJ_DIR = f"{BASE}/projection"
LINEA_DIR = f"{BASE}/line_a"
os.makedirs(LINEA_DIR, exist_ok=True)

SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.50]
ALPHAS_DEGRAD = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
NUM_EVAL = 500
RNG_EVAL = np.random.RandomState(42)
SVCCA_K = 20


def svcca_corr(X, Y, k=SVCCA_K):
    """First SVCCA canonical correlation between two activation spaces.

    Purpose:
        Measure functional alignment between layers of models with different
        dimensions (128 vs 512). Standard CCA overfits when n_samples is not
        much larger than n_features; PCA truncation regularizes by keeping
        only the top k variance components.
    What:
        PCA-truncate both X and Y to k dims, run CCA(n_components=1) on the
        truncated spaces, return Pearson r of the first canonical pair.
    Why:
        Raw cosine similarity is meaningless across unequal dimensions.
        Full CCA with d1=128,d2=512 on N=2823 samples gives near-1.0
        correlations (overfitting). SVCCA(k=20) gives a bounded [0,1] measure
        that reflects only the top shared variance structure.
    """
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    X_pca = PCA(n_components=min(k, Xs.shape[1])).fit_transform(Xs)
    Y_pca = PCA(n_components=min(k, Ys.shape[1])).fit_transform(Ys)
    cca = CCA(n_components=1, max_iter=2000)
    cca.fit(X_pca, Y_pca)
    X_c, Y_c = cca.transform(X_pca, Y_pca)
    return float(np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])


def compute_cca_heatmap(small_acts, big_acts):
    """Build a 2x6 SVCCA heatmap over all layer pairs.

    Purpose:
        Determine whether layers align by position (A[i]↔B[i]) or
        cross-functionally (A[early]↔B[late]).
    What:
        Iterates over all 2×6=12 layer pairs, computes SVCCA correlation
        for each, returns a matrix.
    Why:
        The existing W was trained on A[layer=1]→B[layer=5] (last layers).
        If SVCCA shows maximal correlation at a different pair, retraining
        W on that pair might improve geometry transfer.
    """
    n_la, n_lb = small_acts.shape[0], big_acts.shape[0]
    heatmap = np.zeros((n_la, n_lb))
    print(f"\n[SVCCA(k={SVCCA_K})] Computing {n_la}x{n_lb} heatmap...")
    for la in range(n_la):
        for lb in range(n_lb):
            t0 = time.time()
            heatmap[la, lb] = svcca_corr(small_acts[la], big_acts[lb])
            print(f"  SVCCA(A[{la}],B[{lb}]) = {heatmap[la, lb]:.4f}  ({time.time()-t0:.1f}s)")
    return heatmap


def plot_cca_heatmap(heatmap):
    """Visualise the SVCCA heatmap as a colour-coded 2x6 grid.

    Purpose:
        Communicate layer alignment patterns in a single figure.
    What:
        Uses matplotlib imshow with annotations; saves to
        artifacts/line_a/cca_heatmap.png.
    Why:
        The heatmap immediately shows whether alignment is positional (diagonal)
        or cross-layer (off-diagonal peaks).
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(heatmap, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(heatmap.shape[1]))
    ax.set_yticks(range(heatmap.shape[0]))
    ax.set_xticklabels([f'B[{i}]' for i in range(heatmap.shape[1])])
    ax.set_yticklabels([f'A[{i}]' for i in range(heatmap.shape[0])])
    ax.set_title(f'SVCCA Correlation (k={SVCCA_K}): A layers vs B layers')
    fig.colorbar(im, ax=ax)
    for la in range(heatmap.shape[0]):
        for lb in range(heatmap.shape[1]):
            ax.text(lb, la, f'{heatmap[la, lb]:.3f}', ha='center', va='center',
                    color='white' if heatmap[la, lb] > 0.5 else 'black')
    plt.tight_layout()
    path = f"{LINEA_DIR}/cca_heatmap.png"
    plt.savefig(path)
    plt.close()
    print(f"[CCA] Heatmap saved to {path}")


def plot_pca_alignment(small_acts, big_acts):
    """Project all layer activations to 2D via PCA to visualise clustering.

    Purpose:
        Qualitative check: do layers from different models cluster by
        functional role (early vs late) rather than by model identity?
    What:
        Random-projects A's 128-dim activations → 512, concatenates with B's
        native activations, fits PCA(2), colours each layer separately.
    Why:
        If A[1] and B[5] overlap in PCA space while A[1] and B[0] are far
        apart, this visually confirms positional alignment.
    """
    n_la, n_lb = small_acts.shape[0], big_acts.shape[0]
    rng = np.random.RandomState(0)
    R = rng.randn(512, 128)
    R, _ = np.linalg.qr(R)
    R = R.T
    small_proj = small_acts @ R
    all_acts = []
    labels_list = []
    for la in range(n_la):
        all_acts.append(small_proj[la]); labels_list.append(f'A{la}')
    for lb in range(n_lb):
        all_acts.append(big_acts[lb]); labels_list.append(f'B{lb}')
    combined = np.concatenate(all_acts, axis=0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(StandardScaler().fit_transform(combined))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels_list)))
    fig, ax = plt.subplots(figsize=(8, 6))
    offset = 0
    for i, (acts_i, label) in enumerate(zip(all_acts, labels_list)):
        n = acts_i.shape[0]
        ax.scatter(proj[offset:offset+n, 0], proj[offset:offset+n, 1],
                   c=[colors[i]], label=label, s=5, alpha=0.5)
        offset += n
    ax.set_title('PCA of all layer activations (A->512 via random projection + B)')
    ax.legend(markerscale=3)
    plt.tight_layout()
    path = f"{LINEA_DIR}/pca_alignment.png"
    plt.savefig(path); plt.close()
    print(f"[PCA] Alignment plot saved to {path}")


def extract_eval_set():
    """Generate a fixed set of 500 random (a,b) pairs for evaluation.

    Purpose:
        Provide a reproducible test set for steering experiments.
    What:
        Draws NUM_EVAL=500 pairs from Uniform(0,96) with fixed seed,
        returns token tensor (500,2) and label tensor (500,).
    Why:
        Steering accuracy must be measured on the same examples across
        conditions (sigma, alpha, steer_type). A fixed seed ensures
        comparability.
    """
    pairs = [(int(RNG_EVAL.randint(0, P)), int(RNG_EVAL.randint(0, P))) for _ in range(NUM_EVAL)]
    tokens = torch.tensor(pairs)
    labels = torch.tensor([(a + b) % P for a, b in pairs])
    return tokens, labels


def manual_forward(model, tokens, noise_sigma=0.0, steer_vec=None, steer_alpha=1.0, steer_layer=None):
    """Custom forward pass with optional noise injection and steering.

    Purpose:
        Support steering experiments that require modifying the model's
        internal activations and/or inputs, without using hooks (which add
        complexity and can be fragile with repeated register/remove cycles).
    What:
        Replicates Transformer.forward() but injects Gaussian noise at the
        embedding stage (before blocks) and adds a steering vector at a
        chosen layer's last token position.
    Why:
        Noise injection needs fine-grained control at the embedding level.
        Steering at arbitrary layers needs per-layer hooks; manual forward
        is simpler and more explicit than managing multiple hook handles.
    """
    B, T = tokens.shape
    pos = torch.arange(T, device=tokens.device).unsqueeze(0)
    h = model.embed(tokens) + model.pos_embed(pos)
    if noise_sigma > 0:
        h = h + torch.randn_like(h) * noise_sigma
    n_layers = len(model.blocks)
    if steer_layer is None:
        steer_layer = n_layers - 1
    for bi, block in enumerate(model.blocks):
        h = block(h)
        if bi == steer_layer and steer_vec is not None:
            h[:, -1, :] = h[:, -1, :] + steer_alpha * steer_vec.unsqueeze(0)
    h = model.ln_final(h)
    logits = model.unembed(h)
    return logits


def eval_accuracy(model, tokens, labels, noise_sigma=0.0, steer_vec=None, steer_alpha=1.0):
    """Compute accuracy under a given noise + steering configuration.

    Purpose:
        Central metric for steering experiments. Used by both
        noise_injection_test and degradation_test.
    What:
        Runs manual_forward, takes argmax over logits at position 1,
        compares with ground-truth labels, returns mean accuracy.
    Why:
        Factorises out the common accuracy computation so each experiment
        function only varies the experimental parameters (sigma, alpha,
        steer_type).
    """
    with torch.no_grad():
        logits = manual_forward(model, tokens, noise_sigma, steer_vec, steer_alpha)
    preds = logits[:, 1, :].argmax(dim=1)
    return (preds == labels).float().mean().item()


def noise_injection_test(model_b, steer_learned):
    """Test whether projected steering recovers accuracy under embedding noise.

    Purpose:
        Since B's baseline accuracy is 1.0 (ceiling effect), we degrade B
        by adding Gaussian noise to its input embeddings. If W(steering_A)
        carries algorithmic information, it should recover accuracy better
        than a random vector of the same norm.
    What:
        For each sigma in SIGMAS, measures accuracy under three steering
        conditions: none (just noise), random (unit vector), learned
        (W(steering_A)). Fixed alpha=1.0 for steering.
    Why:
        The ceiling effect (baseline=1.0) makes standard steering meaningless.
        Noise injection creates headroom to measure recovery. Random steering
        controls for non-specific effects (any vector might add signal).
    """
    print(f"\n{'='*50}")
    print("Noise Injection Test")
    print(f"{'='*50}")
    tokens, labels = extract_eval_set()
    steer_random = torch.from_numpy(RNG_EVAL.randn(CFG_BIG['d_model']).astype(np.float32))
    steer_random = steer_random / steer_random.norm()

    rows = []
    for sigma in SIGMAS:
        acc_none = eval_accuracy(model_b, tokens, labels, sigma, None)
        acc_rand = eval_accuracy(model_b, tokens, labels, sigma, steer_random)
        acc_learn = eval_accuracy(model_b, tokens, labels, sigma, steer_learned)
        rows.append((sigma, 'none', acc_none))
        rows.append((sigma, 'random', acc_rand))
        rows.append((sigma, 'learned', acc_learn))
        print(f"  sigma={sigma:.2f}: none={acc_none:.4f}  random={acc_rand:.4f}  learned={acc_learn:.4f}")

    path = f"{LINEA_DIR}/noise_injection.csv"
    with open(path, 'w') as f:
        f.write("sigma,steer_type,accuracy\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f}\n")
    print(f"Saved to {path}")

    fig, ax = plt.subplots(figsize=(8, 4))
    for stype, marker, color in [('none', 'o', 'gray'), ('random', 's', 'orange'), ('learned', '^', 'blue')]:
        xs = [r[0] for r in rows if r[1] == stype]
        ys = [r[2] for r in rows if r[1] == stype]
        ax.plot(xs, ys, marker + '-', label=stype, color=color)
    ax.set_xlabel('Noise sigma')
    ax.set_ylabel('Accuracy')
    ax.set_title('Steering under Noise Injection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{LINEA_DIR}/noise_injection.png")
    plt.close()
    return rows


def degradation_test(model_b, steer_learned):
    """Test whether steering degrades accuracy on clean inputs.

    Purpose:
        If W(steering_A) is well-aligned with B's internal solution direction,
        adding it should not hurt accuracy (or hurt minimally). If it is
        orthogonal or adversarial, accuracy should drop significantly.
    What:
        For each alpha in ALPHAS_DEGRAD, measures accuracy on clean inputs
        (sigma=0) with learned steering at varying strengths.
    Why:
        Even though baseline=1.0 prevents measuring improvement, degradation
        is informative: a large drop would indicate misalignment; no drop
        indicates the steering vector lives in a functionally neutral subspace
        of B's residual stream.
    """
    print(f"\n{'='*50}")
    print("Degradation Test (sigma=0, varying alpha)")
    print(f"{'='*50}")
    tokens, labels = extract_eval_set()

    rows = []
    for alpha in ALPHAS_DEGRAD:
        acc = eval_accuracy(model_b, tokens, labels, 0.0, steer_learned, alpha)
        rows.append((alpha, acc))
        delta = acc - 1.0
        print(f"  alpha={alpha:.1f}: acc={acc:.4f}  (delta={delta:+.4f})")

    path = f"{LINEA_DIR}/degradation.csv"
    with open(path, 'w') as f:
        f.write("alpha,accuracy\n")
        for r in rows:
            f.write(f"{r[0]:.1f},{r[1]:.4f}\n")
    print(f"Saved to {path}")

    fig, ax = plt.subplots(figsize=(8, 4))
    alphas, accs = zip(*rows)
    ax.plot(alphas, accs, 'o-', color='blue')
    ax.axhline(y=1.0, color='green', ls='--', alpha=0.5, label='baseline=1.0')
    ax.axvline(x=1.0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Steering alpha')
    ax.set_ylabel('Accuracy')
    ax.set_title('Degradation: Steering Effect on Clean Inputs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{LINEA_DIR}/degradation.png")
    plt.close()
    return rows


def write_summary(heatmap, noise_rows, deg_rows):
    """Write a markdown summary of all Line A results.

    Purpose:
        Consolidate the SVCCA heatmap, noise injection results, and
        degradation results into a single human-readable report.
    What:
        Writes artifacts/line_a/line_a_summary.md with tables for each
        experiment and an interpretation section that computes recovery
        and specificity metrics.
    Why:
        The summary is the deliverable of Line A — it encapsulates the
        answer to whether layer alignment and steering transfer work.
    """
    best_la, best_lb = np.unravel_index(heatmap.argmax(), heatmap.shape)
    best_val = heatmap[best_la, best_lb]

    lines = []
    lines.append("# Line A: Multi-layer Alignment Summary\n")
    lines.append(f"## SVCCA Heatmap (k={SVCCA_K})\n")
    lines.append(f"Existing W was trained on A[1] to B[5] (last layers).\n")
    header = "| A\\B | " + " | ".join(f"B[{j}]" for j in range(heatmap.shape[1])) + " |"
    sep = "|" + "---|" * (heatmap.shape[1] + 1)
    lines.append(header)
    lines.append(sep)
    for la in range(heatmap.shape[0]):
        vals = " | ".join(f"{heatmap[la, lb]:.4f}" for lb in range(heatmap.shape[1]))
        lines.append(f"| A[{la}] | {vals} |")
    lines.append(f"\nBest aligned: **A[{best_la}] <-> B[{best_lb}]** (CCA = {best_val:.4f})\n")

    if best_la != CFG_SMALL['n_layers'] - 1 or best_lb != CFG_BIG['n_layers'] - 1:
        lines.append(f"> Note: Existing W was trained on A[{CFG_SMALL['n_layers']-1}]->B[{CFG_BIG['n_layers']-1}] (last layers), "
                     f"but CCA best is A[{best_la}]<->B[{best_lb}]. Steering uses existing W.\n")

    lines.append("## Noise Injection\n")
    lines.append("| sigma | steer_type | accuracy |")
    lines.append("|---|------------|----------|")
    for sigma, stype, acc in noise_rows:
        lines.append(f"| {sigma} | {stype} | {acc:.4f} |")

    lines.append("\n## Degradation (sigma=0)\n")
    lines.append("| alpha | accuracy | delta |")
    lines.append("|---|----------|---|")
    for alpha, acc in deg_rows:
        delta = acc - 1.0
        lines.append(f"| {alpha:.1f} | {acc:.4f} | {delta:+.4f} |")

    lines.append("\n### Interpretation\n")
    noise_data = {sigma: {st: acc for sig, st, acc in noise_rows if sig == sigma} for sigma in SIGMAS}
    for sigma in SIGMAS:
        if sigma == 0:
            continue
        none_acc = noise_data[sigma]['none']
        rand_acc = noise_data[sigma]['random']
        learn_acc = noise_data[sigma]['learned']
        recovery = learn_acc - none_acc
        spec = learn_acc - rand_acc
        lines.append(f"- sigma={sigma}: baseline={none_acc:.4f}, random={rand_acc:.4f}, learned={learn_acc:.4f}, "
                     f"recovery={recovery:+.4f}, specificity={spec:+.4f}")

    deg_alphas = [r for r in deg_rows if r[0] > 0]
    max_drop = max((1.0 - acc) for _, acc in deg_alphas) if deg_alphas else 0
    lines.append(f"\n- Max degradation from steering: {max_drop:.4f}")
    if max_drop < 0.02:
        lines.append("  -> Steering is nearly lossless (aligned with model's solution).")
    elif max_drop < 0.10:
        lines.append("  -> Mild degradation (partial misalignment).")
    else:
        lines.append("  -> Significant degradation (steering interferes with computation).")

    text = "\n".join(lines) + "\n"
    with open(f"{LINEA_DIR}/line_a_summary.md", "w") as f:
        f.write(text)
    print(f"\nSummary saved to {LINEA_DIR}/line_a_summary.md")


def main():
    """Orchestrate Line A: load activations, compute SVCCA heatmap,
    run noise injection and degradation tests, write summary.

    Purpose:
        Top-level entry point. Executes the full Line A experiment pipeline.
    What:
        1. Load cached activations (small_acts_test, big_acts_test)
        2. Compute and plot SVCCA heatmap (2x6)
        3. Compute and plot PCA alignment visualisation
        4. Load B model and existing W (used for A[1]->B[5] projection)
        5. Load pre-computed steering vector from A, project through W
        6. Run noise injection test
        7. Run degradation test
        8. Write summary markdown
    Why:
        Line A tests the layer-alignment hypothesis: does matching layers
        by functional similarity (SVCCA) improve geometry transfer vs
        naive last-layer pairing? And does steering via W work when
        measured by recovery under noise?
    """
    print("=" * 60)
    print("Line A: Multi-layer Alignment via CCA + Steering")
    print("=" * 60)

    small_acts = np.load(f"{ACT_DIR}/small_acts_test.npy")
    big_acts = np.load(f"{ACT_DIR}/big_acts_test.npy")
    print(f"Activations: A {small_acts.shape}, B {big_acts.shape}")

    heatmap = compute_cca_heatmap(small_acts, big_acts)
    plot_cca_heatmap(heatmap)
    plot_pca_alignment(small_acts, big_acts)
    best_la, best_lb = np.unravel_index(heatmap.argmax(), heatmap.shape)
    print(f"\nBest SVCCA pair: A[{best_la}] <-> B[{best_lb}] = {heatmap[best_la, best_lb]:.4f}")

    model_b = make_model(CFG_BIG).to(DEVICE)
    model_b.load_state_dict(torch.load(f"{BIG_DIR}/best_model.pth", map_location=DEVICE))
    model_b.eval()

    W = nn.Linear(CFG_SMALL['d_model'], CFG_BIG['d_model'], bias=False)
    W.load_state_dict(torch.load(f"{PROJ_DIR}/W.pth", map_location=DEVICE))
    W.eval()
    print(f"\nLoaded W: {CFG_SMALL['d_model']}->{CFG_BIG['d_model']} (from A[last]->B[last])")

    steer_128 = np.load(f"{BASE}/steering/steering_vec.npy")
    with torch.no_grad():
        steer_512 = W(torch.from_numpy(steer_128).float()).numpy()
    steer_512 = steer_512 / (np.linalg.norm(steer_512) + 1e-10)
    steer_tensor = torch.from_numpy(steer_512).float()
    print(f"Steering: 128-dim norm={np.linalg.norm(steer_128):.4f} -> "
          f"512-dim norm={np.linalg.norm(steer_512):.4f}")

    noise_rows = noise_injection_test(model_b, steer_tensor)
    deg_rows = degradation_test(model_b, steer_tensor)

    write_summary(heatmap, noise_rows, deg_rows)

    print(f"\n{'='*60}")
    print(f"Line A complete. Results in {LINEA_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
