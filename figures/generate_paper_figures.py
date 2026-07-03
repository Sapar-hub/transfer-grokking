import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/paper_figures"


def probe_accuracy(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    probe = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=42)
    probe.fit(X_tr, y_tr)
    return probe.score(X_te, y_te)


def figure_a(ax, small_acts, labels, phi2_acts):
    print("  Small->Small probe...")
    probe_small = probe_accuracy(small_acts, labels)
    print(f"    Probe acc: {probe_small:.4f}")

    print("  Phi-2->Phi-2 probe (L31)...")
    probe_phi2 = probe_accuracy(phi2_acts, labels)
    print(f"    Probe acc: {probe_phi2:.4f}")

    # (cos_sim, probe_acc, label, marker, color, ha_offset, va_offset)
    points = [
        (1.00, probe_small, "Small->Small\n(self)", 'o', '#2c7bb6', 0, 0.04),
        (1.00, probe_phi2,  "Phi-2->Phi-2\n(self, L31)", 's', '#d7191c', 0, -0.08),
        (0.30, 0.9362, "Clean Exp.\n(Small->Big)", '^', '#fdae61', 0, 0.04),
        (0.82, 0.0100, "Embed Patch\n(Small->Phi-2)", 'v', '#abd9e9', 0, -0.06),
    ]

    for x, y, label, marker, color, hoff, voff in points:
        ax.scatter(x, y, marker=marker, s=120, c=color, edgecolors='black', linewidths=0.5, zorder=5)
        ax.annotate(label, (x, y), (x + hoff, y + voff),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='y=x (cos=probe)')
    ax.set_xlabel('Cosine similarity', fontsize=12)
    ax.set_ylabel('Probe accuracy (97-class LogisticRegression)', fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_title('Figure A: Geometry vs. Linear Separability', fontsize=13)


def figure_b(ax):
    groups = ['MSE', 'CE']
    x = np.arange(len(groups))
    width = 0.3

    logit_lens = [0.0117, 1.0000]
    probe_accs = [1.0000, 1.0000]

    bars1 = ax.bar(x - width/2, logit_lens, width, label='Logit lens', color='#fdae61', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, probe_accs, width, label='Probe on W(h)', color='#2c7bb6', edgecolor='black', linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=0.235, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.text(1.5, 0.24, 'Phi-2 baseline (0.235)', fontsize=9, color='gray', ha='right', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    ax.set_title('Figure B: MSE vs. CE projection', fontsize=13)


def main():
    print("Loading data...")
    small_acts = np.load(f"{ARTIFACTS}/small_model_activations.npy")
    labels = np.load(f"{ARTIFACTS}/mod_arithmetic_labels.npy", allow_pickle=True)
    phi2_acts = np.load(f"{ARTIFACTS}/cross_model/microsoft_phi_2_L31_acts.npy")
    print(f"  Small acts: {small_acts.shape}")
    print(f"  Phi-2 L31 acts: {phi2_acts.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    figure_a(axes[0], small_acts, labels, phi2_acts)
    figure_b(axes[1])
    plt.tight_layout()
    path = f"{OUT_DIR}/paper_figures_panel.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    fig_a, ax_a = plt.subplots(figsize=(7, 6))
    figure_a(ax_a, small_acts, labels, phi2_acts)
    fig_a.tight_layout()
    path_a = f"{OUT_DIR}/figure_a_cos_probe.png"
    fig_a.savefig(path_a, dpi=150)
    plt.close()
    print(f"Saved: {path_a}")

    fig_b, ax_b = plt.subplots(figsize=(6, 5))
    figure_b(ax_b)
    fig_b.tight_layout()
    path_b = f"{OUT_DIR}/figure_b_mse_vs_ce.png"
    fig_b.savefig(path_b, dpi=150)
    plt.close()
    print(f"Saved: {path_b}")

    print("Done.")


if __name__ == "__main__":
    main()
