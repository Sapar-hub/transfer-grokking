import torch, torch.optim as optim, numpy as np, os, time
from model import make_model, Transformer, CFG_SMALL, CFG_BIG
from utils import DEVICE, P, generate_all_pairs

ARTIFACTS = "artifacts"
SMALL_DIR = f"{ARTIFACTS}/small"
BIG_DIR = f"{ARTIFACTS}/big"
for d in [SMALL_DIR, BIG_DIR]:
    os.makedirs(d, exist_ok=True)

BATCH_SIZE = 256
MAX_EPOCHS = 50000
WINDOW = 2000


def train_model(cfg, seed=42):
    """Train a transformer to grokking on modular arithmetic.

    Purpose:
        Train either Small (128-dim) or Big (512-dim) model until it
        generalises (grokking). Used to produce the trained models for
        all downstream experiments.
    What:
        Weight decay 1.0 (essential for grokking), AdamW lr=1e-4,
        early stopping when val_acc averaged over WINDOW epochs exceeds
        0.99 (grokking threshold). Saves best model checkpoint.
    Why:
        Weight decay of 1.0 is critical — it forces the model to use
        cleaner circuits by penalising large weights, which accelerates
        the transition from memorisation to generalisation. Without it,
        grokking may not occur or may take significantly longer.
    """
    torch.manual_seed(seed)
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"Training Model {name}")
    print(f"{'='*60}")

    model = make_model(cfg).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1.0)

    all_inputs, all_labels = generate_all_pairs()
    num_total = len(all_inputs)
    num_train = int(num_total * 0.7)
    indices = torch.randperm(num_total)
    train_idx, val_idx = indices[:num_train], indices[num_train:]

    train_inputs = all_inputs[train_idx]
    train_labels = all_labels[train_idx]
    val_inputs = all_inputs[val_idx]
    val_labels = all_labels[val_idx]

    train_accs, val_accs = [], []
    best_val_acc = 0.0
    grokking = False
    train_at_1 = None
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(num_train)
        total_correct = 0
        for start in range(0, num_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            x, y = train_inputs[idx], train_labels[idx]
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
            preds = model(val_inputs)[:, 1, :].argmax(dim=1)
            val_acc = (preds == val_labels).sum().item() / len(val_labels)
        val_accs.append(val_acc)

        if train_acc == 1.0 and train_at_1 is None:
            train_at_1 = epoch

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_dir = SMALL_DIR if name == "small" else BIG_DIR
            torch.save(model.state_dict(), f"{model_dir}/best_model.pth")

        if train_at_1 is not None and epoch >= WINDOW:
            recent_val = np.mean(val_accs[-WINDOW:])
            if recent_val > 0.99:
                grokking = True
                print(f"  Grokking at epoch {epoch} (mean={recent_val:.4f})")
                break

        if epoch % 1000 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            rate = epoch / elapsed
            eta = (MAX_EPOCHS - epoch) / rate if rate > 0 else 0
            print(f"  epoch {epoch:5d} | train={train_acc:.4f} val={val_acc:.4f} | {elapsed:.0f}s | ETA {eta:.0f}s")

    total_time = time.time() - start_time
    model_dir = SMALL_DIR if name == "small" else BIG_DIR
    np.savez(f"{model_dir}/curves.npz", train_accs=train_accs, val_accs=val_accs)

    print(f"  Best val_acc: {best_val_acc:.4f}")
    print(f"  Grokking: {'YES' if grokking else 'NO (ran {MAX_EPOCHS} epochs, val={val_accs[-1]:.4f})'}")
    print(f"  Time: {total_time:.0f}s")

    return grokking, best_val_acc, model, train_accs, val_accs


def main():
    ok_a, acc_a, model_a, *_ = train_model(CFG_SMALL)
    if ok_a:
        ok_b, acc_b, model_b, *_ = train_model(CFG_BIG)


if __name__ == "__main__":
    main()
