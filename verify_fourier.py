import torch
import numpy as np
from sklearn.decomposition import PCA
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import SmallTransformer
from utils import DEVICE, P, generate_all_pairs, train_probe

ARTIFACTS = "artifacts"


def verify():
    """Confirm the small model learns circular Fourier representations.

    Purpose:
        Validate that the grokked model encodes modular arithmetic as
        circular features in its residual stream (Fourier Hypothesis).
    What:
        1. Runs the model on all P^2 diagonal pairs (n, 0).
        2. PCA of last-layer activations -> should show circular structure.
        3. Logistic regression probe on all pairs -> should give ~1.0.
    Why:
        Fourier Hypothesis predicts that neural networks trained on
        modular arithmetic learn discrete Fourier basis functions.
        Circular PCA structure + probe_acc=1.0 confirms the model has
        learned the algorithmic computation and not just memorised.
    """
    model = SmallTransformer().to(DEVICE)
    state = torch.load(f"{ARTIFACTS}/small/best_model.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    inputs, labels = generate_all_pairs()

    with torch.no_grad():
        _, activations = model(inputs, return_activations=True)
    resid_post = activations["blocks.1.hook_resid_post"]  # [P^2, 2, d_model]
    acts_pos1 = resid_post[:, 1, :].numpy()  # [P^2, d_model]

    # --- Side-by-side: Embedding PCA (circle) + Residual PCA (star) ---
    diag_indices = [n * P + 0 for n in range(P)]
    diag_acts = acts_pos1[diag_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 1. Embedding PCA (should show clean circle — Fourier basis)
    embed_w = model.embed.weight.data.numpy()
    pca_emb = PCA(n_components=2)
    pca2_emb = pca_emb.fit_transform(embed_w)
    sc1 = ax1.scatter(pca2_emb[:, 0], pca2_emb[:, 1], c=range(P), cmap='hsv', s=30, alpha=0.8)
    ax1.set_title(f'PCA of token embeddings — circular Fourier structure\n(var={pca_emb.explained_variance_ratio_.sum():.1%})')
    fig.colorbar(sc1, ax=ax1, label='n')

    # 2. Residual stream PCA (star — multi-frequency superposition)
    pca_res = PCA(n_components=2)
    pca2_res = pca_res.fit_transform(diag_acts)
    sc2 = ax2.scatter(pca2_res[:, 0], pca2_res[:, 1], c=range(P), cmap='hsv', s=30, alpha=0.8)
    ax2.set_title(f'PCA of residual stream activations\n(var={pca_res.explained_variance_ratio_.sum():.1%})')
    fig.colorbar(sc2, ax=ax2, label='n')

    plt.tight_layout()
    plt.savefig(f"{ARTIFACTS}/pca_fourier_structure.png")
    plt.close()

    # --- Logistic regression probe on all P^2 pairs ---
    X = acts_pos1
    y = labels.numpy()
    acc, probe, scaler = train_probe(X, y, test_size=0.3)
    print(f"Probe accuracy on small model: {acc:.4f}")

    if acc > 0.95:
        print("PASS: probe_acc > 0.95 — model has learned the algorithm")
    else:
        print(f"FAIL: probe_acc = {acc:.4f} < 0.95")

    return acc


if __name__ == "__main__":
    verify()
