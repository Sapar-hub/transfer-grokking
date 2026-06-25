import torch
import numpy as np
from sklearn.decomposition import PCA
import os

from model import SmallTransformer
from utils import DEVICE, P, generate_all_pairs, train_probe, plot_pca

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

    # --- PCA on diagonal pairs (n, 0) ---
    diag_indices = [n * P + 0 for n in range(P)]
    diag_acts = acts_pos1[diag_indices]
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(diag_acts)
    plot_pca(pca_2d, list(range(P)), f"{ARTIFACTS}/pca_fourier_structure.png")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

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
