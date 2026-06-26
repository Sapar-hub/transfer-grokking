import torch
import numpy as np
from sklearn.decomposition import PCA
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import SmallTransformer
from utils import DEVICE, P

ARTIFACTS = "artifacts"
SMALL_ACT_PATH = f"{ARTIFACTS}/natural_adapter/phi2_natural_L30.npy"


def get_small_residual_diag():
    model = SmallTransformer().to(DEVICE)
    state = torch.load(f"{ARTIFACTS}/small/best_model.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    inputs = torch.stack([a, b], dim=1)

    with torch.no_grad():
        _, acts = model(inputs, return_activations=True)

    resid = acts["blocks.1.hook_resid_post"]  # [P^2, 2, 128]
    resid_pos1 = resid[:, 1, :].numpy()        # [P^2, 128]

    diag_idx = [n * P + 0 for n in range(P)]
    return resid_pos1[diag_idx]  # [97, 128]


def main():
    print("=" * 60)
    print("PCA vs UMAP: Small model residual (diag) + Phi-2 L30 (all pairs)")
    print("=" * 60)

    # ── Left panel: small model residual PCA ──
    print("\n[1] Small model: extracting residual activations for diagonal pairs...")
    small_diag = get_small_residual_diag()  # [97, 128]
    pca_small = PCA(n_components=2)
    small_pca = pca_small.fit_transform(small_diag)
    var_small = pca_small.explained_variance_ratio_.sum()
    print(f"  Small PCA: {small_diag.shape}, var={var_small:.1%}")

    # ── Right panel: Phi-2 L30 UMAP ──
    print("\n[2] Phi-2 L30: loading cached activations...")
    phi2_all = np.load(SMALL_ACT_PATH)  # [9409, 2560]
    print(f"  Phi-2 L30: {phi2_all.shape}")

    indices = np.arange(P * P)
    labels = ((indices // P) + (indices % P)) % P  # [9409]
    diag_idx = [n * P + 0 for n in range(P)]

    print("\n[3] Running UMAP on all 9409 points (this may take a minute)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        random_state=42,
        metric='euclidean',
        verbose=False,
    )
    phi2_umap = reducer.fit_transform(phi2_all)  # [9409, 2]
    print(f"  UMAP done: {phi2_umap.shape}")

    # ── Plot ──
    print("\n[4] Plotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Small model PCA
    sc1 = ax1.scatter(
        small_pca[:, 0], small_pca[:, 1],
        c=np.arange(P), cmap='hsv', s=80, edgecolors='black', linewidth=0.5
    )
    ax1.set_title(
        f'Small Model (layer 1, d=128)\n'
        f'PCA of diagonal pairs (n, 0) — var={var_small:.1%}',
        fontsize=12
    )
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.set_aspect('equal')
    cbar1 = fig.colorbar(sc1, ax=ax1, ticks=[0, P // 2, P - 1])
    cbar1.set_label('n', fontsize=10)

    # Right: Phi-2 UMAP
    sc2 = ax2.scatter(
        phi2_umap[:, 0], phi2_umap[:, 1],
        c=labels, cmap='hsv', s=5, alpha=0.6
    )
    ax2.scatter(
        phi2_umap[diag_idx, 0], phi2_umap[diag_idx, 1],
        c='black', marker='*', s=50, edgecolors='white', linewidth=0.3,
        label='Diagonal (n, 0)'
    )
    ax2.set_title(
        f'Phi-2 (layer 30, d=2560)\n'
        f'UMAP of all 9409 pairs (4 natural prompts)',
        fontsize=12
    )
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.legend(loc='upper right', framealpha=0.8, fontsize=9)
    cbar2 = fig.colorbar(sc2, ax=ax2, ticks=[0, P // 2, P - 1])
    cbar2.set_label('(a + b) mod 97', fontsize=10)

    plt.tight_layout()
    path = f"{ARTIFACTS}/pca_umap_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved to {path}")


if __name__ == "__main__":
    main()
