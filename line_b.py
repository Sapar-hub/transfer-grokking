import torch, torch.nn as nn, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from model import make_model, CFG_SMALL, CFG_BIG
from utils import DEVICE, P

BASE = "artifacts"
ACT_DIR = f"{BASE}/activations"
SMALL_DIR = f"{BASE}/small"
PROJ_DIR = f"{BASE}/projection"
LINEB_DIR = f"{BASE}/line_b"
os.makedirs(LINEB_DIR, exist_ok=True)


def compute_projected_acts():
    """Project small model activations through learned W into B's space.

    Purpose:
        Obtain W(A_acts) — the images of A's representations in B's
        residual stream dimension. These are used to test whether linear
        separability (probe accuracy) transfers across model scales.
    What:
        Loads W: 128->512 from artifacts/projection/W.pth, applies it
        to small_acts_test[layer=-1] (the last layer, probe=1.0),
        returns the projected array (2823, 512) and corresponding labels.
    Why:
        This is the proxy tokenization step: if W preserves the modular
        arithmetic structure, a probe on W(A_acts) should achieve accuracy
        close to B's native probe (1.0). The existing probe accuracy of
        0.94 suggests partial geometry transfer.
    """
    W = nn.Linear(CFG_SMALL['d_model'], CFG_BIG['d_model'], bias=False)
    W.load_state_dict(torch.load(f"{PROJ_DIR}/W.pth", map_location=DEVICE))
    W.requires_grad_(False)

    small_acts = np.load(f"{ACT_DIR}/small_acts_test.npy")
    labels = np.load(f"{ACT_DIR}/small_labels_test.npy")
    proj_acts = W(torch.from_numpy(small_acts[-1]).float()).numpy()
    print(f"Projected activations: {proj_acts.shape}")
    print(f"Labels: {labels.shape}, classes: {len(np.unique(labels))}")
    return proj_acts, labels


def train_and_eval_probe(X, y):
    """Train a logistic regression probe on projected activations.

    Purpose:
        Measure how much of A's algorithmic structure survives projection
        through W into B's dimension.
    What:
        Splits X,y 70/30, standardises, trains LogisticRegression(97 classes),
        returns the probe object and test predictions for further analysis.
    Why:
        Probe accuracy is the primary metric for geometry preservation.
        If probe acc is high (>> random 1/97), then linear separability
        of mod arithmetic is preserved through the projection, even if
        cosine similarity between W(A_acts) and B_acts is low (0.30).
        This tells us that the information survives in some subspace.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    probe = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=42)
    probe.fit(X_tr_s, y_tr)
    acc = probe.score(X_te_s, y_te)
    y_pred = probe.predict(X_te_s)
    print(f"Probe accuracy: {acc:.4f}")
    return probe, scaler, y_te, y_pred, X_te_s


def plot_confusion_matrix(y_true, y_pred):
    """Plot a confusion matrix of the probe's predictions.

    Purpose:
        Identify which classes are systematically confused after projection.
    What:
        Computes sklearn confusion_matrix (97x97), renders with matplotlib
        using LogNorm to handle the wide dynamic range (diagonal ~30,
        off-diagonal ~0-5).
    Why:
        Accuracy aggregates over all classes, but the confusion matrix
        reveals structure: do errors cluster around specific residues?
        Are nearby classes (k, k+1) confused more often? This gives insight
        into what information W preserves and what it loses.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest', norm=matplotlib.colors.LogNorm())
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (97 classes, acc={np.mean(y_true==y_pred):.4f})')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    path = f"{LINEB_DIR}/confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    print(f"Confusion matrix saved to {path}")


def per_class_analysis(y_true, y_pred):
    """Compute and plot per-class accuracy.

    Purpose:
        Identify which modular arithmetic classes are easiest/hardest for
        the projected probe. Uneven accuracy reveals class-specific
        information loss in the projection.
    What:
        Extracts diagonal of confusion matrix, normalises by class support,
        plots a bar chart, prints best/worst 5 classes.
    Why:
        Some classes may be inherently harder (e.g., boundary effects at
        class 0 or 96). If the hardest classes correspond to specific
        arithmetic patterns (e.g., carry operations), this suggests the
        projection selectively preserves certain computational substructures.
    """
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
    top5 = np.argsort(class_acc)[-5:][::-1]
    bottom5 = np.argsort(class_acc)[:5]

    print(f"\nPer-class accuracy: mean={class_acc.mean():.4f} +/- {class_acc.std():.4f}")
    print(f"  Best 5 classes: {[(i, class_acc[i]) for i in top5]}")
    print(f"  Worst 5 classes: {[(i, class_acc[i]) for i in bottom5]}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(P), class_acc, color='steelblue', alpha=0.7)
    ax.axhline(y=class_acc.mean(), color='red', ls='--', alpha=0.5, label=f'mean={class_acc.mean():.3f}')
    ax.set_xlabel('Class ((a+b) mod 97)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-class Accuracy of Projected Probe')
    ax.legend()
    plt.tight_layout()
    path = f"{LINEB_DIR}/per_class_accuracy.png"
    plt.savefig(path)
    plt.close()
    print(f"Per-class accuracy plot saved to {path}")

    return class_acc


def plot_probe_comparison():
    """Compare probe accuracy across all layers of both models.

    Purpose:
        Provide context for the projected probe accuracy: how well does
        each individual layer linearly separate the 97 classes?
    What:
        For each layer in A (2) and B (6), trains a logistic regression
        probe on the raw activations and records accuracy. Saves to
        artifacts/line_b/layer_comparison.txt.
    Why:
        This shows the progression of algorithmic structure across depths:
        - A[0] probe=0.0, A[1] probe=1.0 (alg suddenly forms)
        - B[0] probe=0.0, B[1]=0.008, ..., B[5]=1.0 (gradual formation)
        The projected probe acc (0.93) can be compared against these to see
        which native layer's performance it matches.
    """
    accs_file = f"{LINEB_DIR}/layer_comparison.txt"
    small_acts = np.load(f"{ACT_DIR}/small_acts_test.npy")
    big_acts = np.load(f"{ACT_DIR}/big_acts_test.npy")
    labels = np.load(f"{ACT_DIR}/small_labels_test.npy")

    lines = []
    lines.append("# Layer-wise Probe Comparison\n")
    lines.append(f"| Model | Layer | Probe Acc |")
    lines.append(f"|-------|-------|-----------|")

    for la in range(small_acts.shape[0]):
        X = small_acts[la]
        X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.3, random_state=42)
        sc = StandardScaler().fit(X_tr)
        probe = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=42)
        probe.fit(sc.transform(X_tr), y_tr)
        acc = probe.score(sc.transform(X_te), y_te)
        lines.append(f"| A | {la} | {acc:.4f} |")

    for lb in range(big_acts.shape[0]):
        X = big_acts[lb]
        X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.3, random_state=42)
        sc = StandardScaler().fit(X_tr)
        probe = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=42)
        probe.fit(sc.transform(X_tr), y_tr)
        acc = probe.score(sc.transform(X_te), y_te)
        lines.append(f"| B | {lb} | {acc:.4f} |")

    text = "\n".join(lines) + "\n"
    with open(accs_file, "w") as f:
        f.write(text)
    print(f"Layer comparison saved to {accs_file}")


def main():
    """Orchestrate Line B: project activations, train probe, analyse errors.

    Purpose:
        Top-level entry point for the proxy tokenisation analysis.
    What:
        1. Compare probe accuracy per layer across both models
        2. Project A[last] through W into B's space
        3. Train/evaluate logistic regression probe
        4. Plot confusion matrix and per-class accuracy
        5. Write summary markdown
    Why:
        Line B drills into why the projected probe achieves 0.94 accuracy
        despite W having only 0.30 cosine similarity with B's activations.
        The per-class analysis and confusion matrix reveal which aspects of
        the algorithm survive projection and which are lost.
    """
    print("=" * 60)
    print("Line B: Proxy Tokenization — Projected Probe Analysis")
    print("=" * 60)

    plot_probe_comparison()

    proj_acts, labels = compute_projected_acts()
    probe, scaler, y_te, y_pred, X_te_s = train_and_eval_probe(proj_acts, labels)

    plot_confusion_matrix(y_te, y_pred)
    class_acc = per_class_analysis(y_te, y_pred)

    summary_lines = []
    summary_lines.append("# Line B: Proxy Tokenization Summary\n")
    summary_lines.append(f"## Projected Probe (W: {CFG_SMALL['d_model']}->{CFG_BIG['d_model']})\n")
    summary_lines.append(f"| Metric | Value |")
    summary_lines.append(f"|--------|-------|")
    summary_lines.append(f"| Test accuracy | {np.mean(y_te == y_pred):.4f} |")
    summary_lines.append(f"| Per-class mean | {class_acc.mean():.4f} |")
    summary_lines.append(f"| Per-class std | {class_acc.std():.4f} |")
    summary_lines.append(f"| Per-class min | {class_acc.min():.4f} |")
    summary_lines.append(f"| Per-class max | {class_acc.max():.4f} |")
    summary_lines.append(f"| Random baseline | {1/P:.4f} |")
    summary_lines.append(f"\n## Interpretation\n")
    summary_lines.append(f"Projected probe achieves {np.mean(y_te == y_pred):.4f} accuracy on 97-class mod arithmetic.")
    if np.mean(y_te == y_pred) > 0.9:
        summary_lines.append("Linear separability is preserved through W -> geometry partially transfers.")
    else:
        summary_lines.append("Moderate linear separability loss through projection.")

    text = "\n".join(summary_lines) + "\n"
    with open(f"{LINEB_DIR}/line_b_summary.md", "w") as f:
        f.write(text)
    print(f"\nSummary saved to {LINEB_DIR}/line_b_summary.md")
    print("Line B complete.")


if __name__ == "__main__":
    main()
