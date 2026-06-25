import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import os

from model import SmallTransformer
from utils import DEVICE, P, generate_data, make_steering_hook

ARTIFACTS = "artifacts"
D_SMALL = 128
D_PHI2 = 2560


def compute_steering_vector():
    """Compute steering vector from small model: high_conf - low_conf.

    Purpose:
        Extract a direction in the small model's residual stream that
        represents "correct computation" (high-confidence activations minus
        low-confidence activations at the last layer).
    What:
        Runs 10k random pairs, records activations for layer 1 (blocks.1),
        picks top/bottom 200 by confidence on correct answer, subtracts
        means, normalises to unit vector.
    Why:
        The steering vector captures the direction that separates confident
        correct answers from uncertain ones. When projected into another
        model's space, it should guide activations toward correct arithmetic.
        This is the standard activation engineering approach (Turner et al.).
    """
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
    n_high = 200
    n_low = 200
    high_conf_acts = acts[idx_sorted[-n_high:]]
    low_conf_acts = acts[idx_sorted[:n_low]]

    steering_vec = high_conf_acts.mean(axis=0) - low_conf_acts.mean(axis=0)
    steering_vec = steering_vec / np.linalg.norm(steering_vec)
    print(f"Steering vector computed: high_conf mean={high_conf_acts.mean():.4f}, low_conf mean={low_conf_acts.mean():.4f}")
    print(f"Confidence range: {confs.min():.4f} - {confs.max():.4f}")
    return steering_vec


def random_orthogonal_projection(d_src, d_dst, seed=42):
    """Create a random orthogonal matrix R: d_dst x d_src.

    Purpose:
        Project a low-dimensional steering vector into a high-dimensional
        target space while (approximately) preserving angles.
    What:
        Samples Gaussian(d_dst, d_src), QR-decomposes for orthonormal
        columns, returns R.T (d_dst x d_src) as float32.
    Why:
        Used for initial steering tests (Phase 4) to embed the small
        model's 128-dim steering vector into Phi-2's 2560-dim residual
        stream before we had a learned projection. Random projection
        preserves dot products in expectation (Johnson-Lindenstrauss).
    """
    rng = np.random.RandomState(seed)
    R = rng.randn(d_dst, d_src)
    R, _ = np.linalg.qr(R)
    return R.astype(np.float32)


def apply_steering(steering_vec, best_layer, alphas):
    """Apply steering vector to Phi-2 via random projection and hooks.

    Purpose:
        Test whether a steering vector from the small model can improve
        Phi-2's accuracy on mod arithmetic.
    What:
        Random-projects the 128-dim steering vector into 2560-dim Phi-2
        space, then registers forward hooks at best_layer to add
        alpha * steering_proj at each forward pass. Measures mod acc.
    Why:
        Phase 4 experiment. Tests the naive hypothesis that any steering
        direction compatible with the small model would generalise to
        a large LLM via random projection. Result: delta = +0.005
        (negligible), motivating the learned projection experiment.
    """
    R = random_orthogonal_projection(D_SMALL, D_PHI2)
    steering_proj = R @ steering_vec

    model_name = "microsoft/phi-2"
    print("Loading Phi-2 for steering...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}
    rng = np.random.RandomState(42)
    test_pairs = [(rng.randint(0, P), rng.randint(0, P)) for _ in range(200)]

    results = []
    steering_tensor = torch.from_numpy(steering_proj).float()

    for alpha in alphas:
        handle = model.model.layers[best_layer].register_forward_hook(
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

    with open(f"{ARTIFACTS}/steering_results_per_alpha.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "mod_accuracy"])
        writer.writerows(results)

    return results


if __name__ == "__main__":
    best_layer = 12
    vec = compute_steering_vector()
    results = apply_steering(vec, best_layer, [0.1, 0.5, 1.0, 2.0, 5.0])
    print(results)
