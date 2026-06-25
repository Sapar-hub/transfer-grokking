import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv

from model import SmallTransformer
from utils import DEVICE, P, generate_data, make_steering_hook

ARTIFACTS = "artifacts"
EXPERIMENT_DIR = f"{ARTIFACTS}/experiment_a"
BATCH_SIZE = 256
BEST_LAYER = 30

D_SMALL = 128
D_PHI2 = 2560

os.makedirs(EXPERIMENT_DIR, exist_ok=True)


# ─── Step 1: Collect paired activations ─────────────────────────────

def collect_phi2_activations():
    """Extract Phi-2 layer 30 activations on all P^2 pairs.

    Purpose:
        Generate the target activations for training the learned projection
        W: 128 -> 2560. These are the activations of Phi-2's best layer
        (layer 30, determined by probe_phi2.py) on all 9409 inputs.
    What:
        Loads Phi-2, registers hook on layer 30, runs all pairs with
        coded prompts ("a b"), captures last-token hidden states.
        Caches to artifacts/experiment_a/phi2_layer30_activations.npy.
    Why:
        The learned projection W is trained via MSE to map small model
        activations -> Phi-2 activations. Without matched paired data,
        we cannot train W. The hook approach captures activations at the
        exact token position where the answer is predicted.
    """
    path = f"{EXPERIMENT_DIR}/phi2_layer30_activations.npy"
    if os.path.exists(path):
        print("[step 1] Phi-2 activations already cached, loading...")
        return np.load(path)

    print("[step 1] Loading Phi-2...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token

    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    inputs_list = [(int(ai), int(bi)) for ai, bi in zip(a, b)]
    all_acts = []
    current_mask = None

    def make_hook():
        def hook(module, input, output):
            nonlocal current_mask
            hidden = output[0] if isinstance(output, tuple) else output
            mask = current_mask.to(hidden.device)
            seq_lens = mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            last_hidden = hidden[batch_idx, seq_lens]
            all_acts.append(last_hidden.detach().cpu())
        return hook

    handle = model.model.layers[BEST_LAYER].register_forward_hook(make_hook())
    model.eval()

    print(f"[step 1] Extracting Phi-2 layer {BEST_LAYER} activations...")
    for start in range(0, len(inputs_list), BATCH_SIZE):
        batch = inputs_list[start:start+BATCH_SIZE]
        prompts = [f"{a} {b}" for a, b in batch]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        current_mask = tokenized.attention_mask
        with torch.no_grad():
            model(**tokenized)

    handle.remove()
    acts = torch.cat(all_acts, dim=0).numpy()
    np.save(path, acts)
    print(f"[step 1] Saved {acts.shape}")
    return acts


# ─── Step 2: Train projection W ─────────────────────────────────────

def train_projection(small_acts, target_acts):
    """Train a linear projection W: 128 -> 2560 via MSE.

    Purpose:
        Learn a linear map from small model activations to Phi-2
        activations (layer 30). The goal is to transfer the geometric
        structure of modular arithmetic from the small model to Phi-2.
    What:
        Shuffles data, splits 70/30, trains nn.Linear(128, 2560) with
        MSELoss and AdamW for 5000 epochs. Monitors test MSE and cosine
        similarity. Saves W and training curve.
    Why:
        Replaces the random orthogonal projection from Phase 4 with a
        data-driven map. If the geometries are compatible, W should
        achieve high cos_sim (> 0.85) and enable effective steering.
        The result (cos_sim = 0.33) showed tokenizer mismatch.
    """
    path_w = f"{EXPERIMENT_DIR}/projection_W.pth"
    path_curve = f"{EXPERIMENT_DIR}/projection_training_curve.png"

    if os.path.exists(path_w) and os.path.exists(path_curve):
        print("[step 2] W already trained, loading...")
        W = nn.Linear(D_SMALL, D_PHI2, bias=False)
        W.load_state_dict(torch.load(path_w, map_location=DEVICE, weights_only=True))
        with open(f"{EXPERIMENT_DIR}/final_metrics.txt") as f:
            print(f"[step 2] " + f.readline().strip())
        return W

    idx = np.arange(len(small_acts))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train = torch.from_numpy(small_acts[train_idx]).float()
    X_test = torch.from_numpy(small_acts[test_idx]).float()
    Y_train = torch.from_numpy(target_acts[train_idx]).float()
    Y_test = torch.from_numpy(target_acts[test_idx]).float()

    W = nn.Linear(D_SMALL, D_PHI2, bias=False)
    optimizer = optim.AdamW(W.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_losses, test_losses, cos_sims = [], [], []
    num_epochs = 5000

    print(f"[step 2] Training W: {D_SMALL} -> {D_PHI2}, {num_epochs} epochs")
    for epoch in range(1, num_epochs + 1):
        pred_train = W(X_train)
        loss = loss_fn(pred_train, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == 1:
            with torch.no_grad():
                pred_test = W(X_test)
                test_loss = loss_fn(pred_test, Y_test).item()
                cos_sim = nn.functional.cosine_similarity(pred_test, Y_test, dim=1).mean().item()

            train_losses.append(loss.item())
            test_losses.append(test_loss)
            cos_sims.append(cos_sim)
            print(f"  epoch {epoch:4d}: train_mse={loss.item():.6f}, test_mse={test_loss:.6f}, cos_sim={cos_sim:.4f}")

    torch.save(W.state_dict(), path_w)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    epochs_plotted = list(range(0, num_epochs, 500))
    if 0 not in epochs_plotted:
        epochs_plotted = [0] + epochs_plotted
    plt.plot(range(len(train_losses)), train_losses, label='train_mse')
    plt.plot(range(len(test_losses)), test_losses, label='test_mse')
    plt.xlabel('Epoch (x500)'); plt.ylabel('MSE'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(len(cos_sims)), cos_sims, 'o-')
    plt.xlabel('Epoch (x500)'); plt.ylabel('Mean Cosine Sim (test)')
    plt.tight_layout()
    plt.savefig(path_curve)
    plt.close()

    final_cos = cos_sims[-1]
    with open(f"{EXPERIMENT_DIR}/final_metrics.txt", "w") as f:
        f.write(f"cosine_similarity={final_cos:.4f}\n")
    print(f"[step 2] Done. Final cosine similarity (test): {final_cos:.4f}")

    return W


# ─── Step 3: Verify geometry ────────────────────────────────────────

def verify_geometry(W, small_acts, target_acts):
    """Check geometry preservation via probe accuracy on projected acts.

    Purpose:
        Measure whether W(A_acts) preserves the algorithmic structure of
        modular arithmetic, compared to the target (Phi-2) activations.
    What:
        Projects small_acts through W, trains logistic regression probes
        on both the projected and the original target activations.
        Compares accuracy ratio.
    Why:
        The probe accuracy ratio tells us how much of Phi-2's linear
        separability is recovered by W(A). A ratio > 0.8 indicates good
        geometry preservation. (Result: projected acc = 0.997, but this
        was misleading — W preserved small structure, not Phi-2's.)
    """
    print("[step 3] Verifying geometry preservation...")

    with torch.no_grad():
        W_small = W(torch.from_numpy(small_acts).float()).numpy()

    labels = np.load(f"{ARTIFACTS}/mod_arithmetic_labels.npy")

    def run_probe(X, y, label=""):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        probe = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, random_state=42)
        probe.fit(X_tr, y_tr)
        acc = probe.score(X_te, y_te)
        print(f"  {label}: probe_acc = {acc:.4f}")
        return acc

    acc_target = run_probe(target_acts, labels, "original (Phi-2 layer 30)")
    acc_proj = run_probe(W_small, labels, "projected (W@small)")

    ratio = acc_proj / acc_target if acc_target > 0 else 0
    print(f"  Ratio projected/original: {ratio:.3f}")

    verdict = ""
    if ratio > 0.8:
        verdict = "Geometry preserved (ratio > 0.8)"
    elif ratio > 0.5:
        verdict = "Partially preserved (0.5 < ratio <= 0.8)"
    else:
        verdict = "Geometry lost (ratio <= 0.5)"

    print(f"  Verdict: {verdict}")

    with open(f"{EXPERIMENT_DIR}/projected_probe_acc.txt", "w") as f:
        f.write(f"original_probe_acc: {acc_target:.4f}\n")
        f.write(f"projected_probe_acc: {acc_proj:.4f}\n")
        f.write(f"ratio: {ratio:.3f}\n")
        f.write(f"verdict: {verdict}\n")

    return acc_target, acc_proj, ratio


# ─── Step 4: Project steering vector ──────────────────────────────────

def project_steering_vector(W):
    """Project the small model's steering vector through W.

    Purpose:
        Map the 128-dim steering vector (high - low confidence activations
        from the small model) into Phi-2's 2560-dim space via the learned
        projection W.
    What:
        Recomputes the steering vector from the small model (10k pairs),
        projects through W, normalises to unit norm.
    Why:
        The projected steering vector is applied to Phi-2 via hooks.
        If W correctly aligns the geometries, the projected steering
        should improve Phi-2's mod arithmetic accuracy.
    """
    print("[step 4] Computing steering vector from small model...")

    model = SmallTransformer().to(DEVICE)
    state = torch.load(f"{ARTIFACTS}/small/best_model.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    num_total = 10000
    inputs_all, labels_all = generate_data(num_total, seed=42)

    acts = []
    confs = []
    for i in range(num_total):
        x = inputs_all[i:i+1]
        y = labels_all[i].item()
        with torch.no_grad():
            logits, activations = model(x, return_activations=True)
        logits_pos1 = logits[0, 1, :]
        probs = torch.softmax(logits_pos1, dim=0)
        act = activations["blocks.1.hook_resid_post"][0, 1, :].numpy()
        correct_prob = probs[y].item()
        acts.append(act)
        confs.append(correct_prob)

    acts = np.array(acts)
    confs = np.array(confs)
    idx_sorted = np.argsort(confs)
    n = 200
    high_conf = acts[idx_sorted[-n:]]
    low_conf = acts[idx_sorted[:n]]

    steering_small = high_conf.mean(axis=0) - low_conf.mean(axis=0)
    steering_small = steering_small / np.linalg.norm(steering_small)

    with torch.no_grad():
        steering_projected = W(torch.from_numpy(steering_small).float()).numpy()
    steering_projected = steering_projected / np.linalg.norm(steering_projected)

    print(f"  Steering vector: 128 -> {steering_projected.shape[0]}")
    return steering_projected


# ─── Step 5: Apply steering ──────────────────────────────────────────

def apply_steering(steering_projected, alphas):
    """Apply the projected steering vector to Phi-2 via hooks.

    Purpose:
        Test whether the learned projection W enables effective steering
        of Phi-2 on mod arithmetic.
    What:
        Registers forward hook on Phi-2 layer 30 that adds
        alpha * steering_projected to the residual stream, measures
        mod arithmetic accuracy for each alpha.
    Why:
        If W correctly aligns the activation geometries, the projected
        steering vector should improve Phi-2's accuracy beyond the
        baseline (~0.24). A delta of 0.03+ is considered meaningful.
    """
    print("[step 5] Applying steering with learned projection...")

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token

    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}
    rng = np.random.RandomState(42)
    test_pairs = [(rng.randint(0, P), rng.randint(0, P)) for _ in range(200)]

    steering_tensor = torch.from_numpy(steering_projected).float()

    results = []
    for alpha in alphas:
        handle = model.model.layers[BEST_LAYER].register_forward_hook(
            make_steering_hook(steering_tensor, alpha)
        )

        correct = 0
        for a, b in test_pairs:
            prompt = f"# ({a} + {b}) % 97 ="
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[:, -1, :][0]
            predicted = max(number_tokens, key=lambda n: logits[number_tokens[n]].item())
            if predicted == (a + b) % P:
                correct += 1

        handle.remove()
        acc = correct / len(test_pairs)
        results.append((alpha, acc))
        print(f"  alpha={alpha:.1f}: mod_acc = {acc:.4f}")

    with open(f"{EXPERIMENT_DIR}/steering_results_per_alpha.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "mod_accuracy"])
        writer.writerows(results)

    return results


# ─── Step 6: Compare ──────────────────────────────────────────────────

def compare(results, acc_target, acc_proj, ratio):
    """Compare learned projection vs random projection results.

    Purpose:
        Determine whether the learned projection W beats the random
        projection baseline from Phase 4, and assign a verdict.
    What:
        Compares steering delta, probe accuracy ratio, and assigns a
        textual verdict (PARTIALLY CONFIRMED / failed / etc.).
    Why:
        The comparison is the final output of Experiment A. It tells us
        whether a learned linear map can bridge the activation geometry
        gap between the small model and Phi-2.
    """
    best_alpha, best_acc = max(results, key=lambda x: x[1])
    baseline = 0.24
    random_proj = 0.245

    delta = best_acc - baseline

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Baseline (no steering):            {baseline:.4f}")
    print(f"  Random projection (best):          {random_proj:.4f}")
    print(f"  Learned projection (alpha={best_alpha:.1f}):  {best_acc:.4f}")
    print(f"  Delta (learned vs baseline):       {delta:+.4f}")
    print()
    print(f"  Geometry preservation:")
    print(f"    Original probe acc (Phi-2 L{BEST_LAYER}):   {acc_target:.4f}")
    print(f"    Projected probe acc (W@small):    {acc_proj:.4f}")
    print(f"    Ratio:                            {ratio:.3f}")

    verdict = ""
    if ratio >= 0.8 and delta > 0.03:
        verdict = ("PARTIALLY CONFIRMED: Geometry preserved, steering works.\n"
                   "Learned projection beats random projection.")
    elif ratio >= 0.8 and delta <= 0.03:
        verdict = ("Geometry preserved correctly, but steering fundamentally\n"
                   "does not transfer algorithmic knowledge via linear projection.")
    elif ratio < 0.5:
        verdict = "W failed: dimensions are fundamentally incompatible."
    else:
        verdict = "Partial geometry preservation but steering effect insufficient."

    print(f"\n  VERDICT: {verdict}")

    summary = f"""# Experiment A Summary
## Learned Projection: {D_SMALL} -> {D_PHI2}

| Metric | Value |
|--------|-------|
| Baseline mod acc | {baseline:.4f} |
| Random projection (best) | {random_proj:.4f} |
| Learned projection (alpha={best_alpha:.1f}) | {best_acc:.4f} |
| Delta vs baseline | {delta:+.4f} |
| Original probe acc (Phi-2 L{BEST_LAYER}) | {acc_target:.4f} |
| Projected probe acc (W@small) | {acc_proj:.4f} |
| Geometry preservation ratio | {ratio:.3f} |
| Best alpha | {best_alpha:.1f} |

## Verdict
{verdict}
"""
    with open(f"{EXPERIMENT_DIR}/experiment_summary.md", "w") as f:
        f.write(summary)
    print(summary)

    return delta, verdict


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    """Orchestrate Experiment A: train W, evaluate geometry, test steering.

    Purpose:
        Top-level entry point for the learned projection experiment.
        Tests whether a linear map trained to align small model -> Phi-2
        activations enables geometry transfer and effective steering.
    What:
        1. Load small model activations
        2. Collect Phi-2 activations (or load cached)
        3. Train W: 128 -> 2560 with MSE
        4. Verify geometry (probe accuracy on W(A) vs Phi-2 targets)
        5. Project steering vector through W
        6. Apply steering to Phi-2 with alpha sweep
        7. Compare results with random projection baseline
    Why:
        Experiment A is the key test of whether learned projections can
        overcome the geometry gap. The result (cos_sim = 0.33, delta = 0)
        motivated the Clean Experiment to isolate the tokenizer confound.
    """
    print("=" * 60)
    print("Experiment A: Learned Projection")
    print("=" * 60)

    small_acts = np.load(f"{ARTIFACTS}/small_model_activations.npy")
    labels = np.load(f"{ARTIFACTS}/mod_arithmetic_labels.npy")
    print(f"Small activations: {small_acts.shape}")
    print(f"Labels: {labels.shape}")

    target_acts = collect_phi2_activations()

    W = train_projection(small_acts, target_acts)

    acc_target, acc_proj, ratio = verify_geometry(W, small_acts, target_acts)

    steering_projected = project_steering_vector(W)

    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = apply_steering(steering_projected, alphas)

    compare(results, acc_target, acc_proj, ratio)

    print("\nDone. Artifacts in", EXPERIMENT_DIR)


if __name__ == "__main__":
    main()
