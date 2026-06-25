import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEVICE = torch.device("cpu")
P = 97


def generate_data(num_pairs: int, seed: int = 42):
    """Generate random (a, b) pairs with labels.

    Purpose:
        Create train/eval data for modular arithmetic task.
    What:
        Samples num_pairs pairs from Uniform(0, P-1), computes (a+b) mod P.
    Why:
        Used by steering.py, probe_phi2.py, and experiment_a.py for
        computing steering vectors and extracting activations.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    a = torch.randint(0, P, (num_pairs,), generator=rng)
    b = torch.randint(0, P, (num_pairs,), generator=rng)
    inputs = torch.stack([a, b], dim=1)
    labels = (a + b) % P
    return inputs, labels


def generate_all_pairs():
    """Generate all P^2 = 9409 exhaustive (a, b) pairs.

    Purpose:
        Full Cartesian product of inputs for exhaustive evaluation.
    What:
        Creates all pairs (a, b) for a,b in [0, P-1] and computes labels.
    Why:
        Used by verify_fourier.py, train.py, train_small.py, and
        clean_test.py for full-batch probing and training.
    """
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    inputs = torch.stack([a, b], dim=1)
    labels = (a + b) % P
    return inputs, labels


def train_probe(X, y, test_size=0.3, C=1.0, max_iter=1000):
    """Train a logistic regression probe for modular arithmetic.

    Purpose:
        Measure how linearly separable the 97 classes are in a given
        activation space. Used to detect algorithmic structure.
    What:
        Standardises X, splits train/test 70/30, fits LogisticRegression
        with 97 classes, returns accuracy + trained probe + scaler.
    Why:
        Probe accuracy is the primary metric for detecting algorithmic
        encoding across all experiments. A probe acc >> 1/P (~0.01)
        indicates the model encodes class information at that layer.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )
    probe = LogisticRegression(
        max_iter=max_iter,
        solver='lbfgs', C=C, random_state=42
    )
    probe.fit(X_train, y_train)
    acc = probe.score(X_test, y_test)
    return acc, probe, scaler


def plot_curves(train_accs, val_accs, path):
    """Plot training and validation accuracy curves.

    Purpose:
        Visualise grokking dynamics (memorisation phase + generalisation
        phase).
    What:
        Plots train_accs and val_accs vs epoch with a horizontal dashed
        line at 0.99 (grokking threshold).
    Why:
        Used by train_small.py to save training curves showing the
        characteristic grokking double-descent pattern.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='train_acc', alpha=0.8)
    plt.plot(val_accs, label='val_acc', alpha=0.8)
    plt.axhline(y=0.99, color='g', linestyle='--', alpha=0.4, label='val_acc=0.99')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Grokking curves')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_pca(pca_2d, labels, path):
    """Plot 2D PCA of residual stream activations.

    Purpose:
        Visual check for circular Fourier structure in diagonal pairs
        (n, 0). A circular arrangement of points along n supports the
        Fourier Hypothesis.
    What:
        Scatter plot with colour-coded class labels (n from 0 to P-1).
    Why:
        Used by verify_fourier.py to confirm the small model learns
        circular frequency representations.
    """
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=labels, cmap='hsv', s=20, alpha=0.8)
    plt.colorbar(scatter, label='n')
    plt.title('PCA of blocks.1.hook_resid_post on diagonal pairs (n, 0)')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def make_steering_hook(steering_tensor, alpha):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        batch_size = hidden.shape[0]
        noise = alpha * steering_tensor.unsqueeze(0).expand(batch_size, -1)
        hidden[:, -1, :] = hidden[:, -1, :] + noise
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook


def plot_probe_per_layer(layer_accs, path):
    """Plot probe accuracy per layer for an LLM.

    Purpose:
        Show how algorithmic structure builds across the depth of a
        pre-trained model.
    What:
        Plots (layer, probe_acc) pairs with a horizontal line at
        random baseline (1/P).
    Why:
        Used by probe_phi2.py to visualise which layers encode modular
        arithmetic in Phi-2 and similar models.
    """
    layers, accs = zip(*layer_accs)
    plt.figure(figsize=(10, 5))
    plt.plot(layers, accs, 'o-')
    plt.axhline(y=1/P, color='r', linestyle='--', alpha=0.4, label=f'random={1/P:.3f}')
    plt.xlabel('Layer')
    plt.ylabel('Probe Accuracy')
    plt.title('Probe accuracy per Phi-2 layer')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
