import torch
import torch.optim as optim
import numpy as np
import os

from model import SmallTransformer
from utils import DEVICE, P, generate_all_pairs, plot_curves

ARTIFACTS = "artifacts"
SMALL_DIR = f"{ARTIFACTS}/small"
os.makedirs(SMALL_DIR, exist_ok=True)


def train():
    """Train small model to grok (30% data, detect by sliding window).

    Purpose:
        Early-phase training script that produces the grokked small model
        used for Fourier verification and as the source for steering vectors.
    What:
        Uses 30% of P^2 pairs for training (memorisation phase), AdamW with
        weight decay 1.0, detects grokking via rolling window of val_acc.
        Saves best model to artifacts/small/best_model.pth.
    Why:
        30% data split is standard in grokking literature (Power et al.)
        to ensure a clear memorisation-to-generalisation transition.
        The 500-epoch window detects consistent generalisation.
    """
    torch.manual_seed(42)
    model = SmallTransformer().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1.0)

    num_total = P * P
    num_train = int(num_total * 0.3)
    indices = torch.randperm(num_total)
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    all_inputs, all_labels = generate_all_pairs()

    train_inputs = all_inputs[train_idx]
    train_labels = all_labels[train_idx]
    val_inputs = all_inputs[val_idx]
    val_labels = all_labels[val_idx]

    batch_size = 256
    num_epochs = 30000

    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    grokking_detected = False
    train_at_1_epoch = None
    window_size = 500

    for epoch in range(1, num_epochs + 1):
        model.train()
        perm = torch.randperm(num_train)
        total_correct = 0
        for start in range(0, num_train, batch_size):
            idx = perm[start:start + batch_size]
            x = train_inputs[idx]
            y = train_labels[idx]
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits[:, 1, :], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits[:, 1, :].argmax(dim=1)
            total_correct += (preds == y).sum().item()

        train_acc = total_correct / num_train
        train_accs.append(train_acc)

        model.eval()
        with torch.no_grad():
            val_logits = model(val_inputs)
            val_preds = val_logits[:, 1, :].argmax(dim=1)
            val_acc = (val_preds == val_labels).sum().item() / (num_total - num_train)
        val_accs.append(val_acc)

        if train_acc == 1.0 and train_at_1_epoch is None:
            train_at_1_epoch = epoch

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{SMALL_DIR}/best_model.pth")

        if (train_at_1_epoch is not None
                and epoch - train_at_1_epoch >= 3000
                and epoch >= window_size):
            recent_val = np.mean(val_accs[-window_size:])
            if recent_val >= 0.99:
                grokking_detected = True
                print(f"Grokking detected at epoch {epoch} (mean val_acc over last {window_size}: {recent_val:.4f})")
                break

        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}")

    if not grokking_detected:
        print("Grokking NOT detected within 30000 epochs.")
        print(f"Final train_acc: {train_accs[-1]:.4f}, val_acc: {val_accs[-1]:.4f}")

    plot_curves(train_accs, val_accs, f"{SMALL_DIR}/train_val_curves.png")
    np.savez(f"{SMALL_DIR}/curves.npz", train_accs=train_accs, val_accs=val_accs)
    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Grokking: {'YES' if grokking_detected else 'NO'}")

    return grokking_detected, best_val_acc


if __name__ == "__main__":
    train()
