import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from utils import DEVICE, P, generate_data, train_probe, plot_probe_per_layer

ARTIFACTS = "artifacts"
# Phi-2 has 32 layers. Sample every 2nd from 0-30 + last.
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]


def make_prompt(a, b):
    """Format a (a, b) pair as a prompt string for Phi-2.

    Purpose:
        Standard prompt template used for all Phi-2 evaluations.
    What:
        Returns f"# ({a} + {b}) % 97 =".
    Why:
        Consistent prompt formatting ensures fair comparison between
        baseline accuracy and steered accuracy.
    """
    return f"# ({a} + {b}) % 97 ="


def phi2_baseline():
    """Compute Phi-2's baseline accuracy on mod arithmetic.

    Purpose:
        Measure how well Phi-2 performs on (a+b) mod 97 out of the box
        (without any steering or intervention).
    What:
        Generates 200 random (a, b) pairs, prompts Phi-2, checks if the
        predicted token matches the correct answer. Uses the number token
        that appears in Phi-2's vocabulary (BPE token for each number).
    Why:
        Baseline accuracy (~0.24) establishes the floor for steering
        experiments. If steering improves this, it indicates the steering
        vector carries useful information.
    """
    model_name = "microsoft/phi-2"
    print("Loading Phi-2...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    rng = np.random.RandomState(42)
    pairs = [(rng.randint(0, P), rng.randint(0, P)) for _ in range(200)]

    correct = 0
    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}
    for a, b in pairs:
        prompt = make_prompt(a, b)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[:, -1, :][0]
        predicted = max(number_tokens, key=lambda n: logits[number_tokens[n]].item())
        if predicted == (a + b) % P:
            correct += 1

    acc = correct / len(pairs)
    print(f"Phi-2 baseline accuracy (mod arithmetic): {acc:.4f}")
    return acc, model, tokenizer


def extract_activations(model, tokenizer, layer_indices):
    """Extract residual stream activations from Phi-2 layers.

    Purpose:
        Collect activations at specified layers for probe training.
    What:
        Registers forward hooks on each target layer, runs 2000 random
        pairs through Phi-2, captures last-token activations.
    Why:
        These activations are used by run_probes() to train logistic
        regression probes per layer, measuring how much modular arithmetic
        structure each layer encodes.
    """
    model.eval()
    num_samples = 2000
    inputs_all, labels_all = generate_data(num_samples, seed=0)
    activations_per_layer = {l: [] for l in layer_indices}
    all_labels = []

    handles = []
    for l in layer_indices:
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                activations_per_layer[layer_idx].append(
                    hidden[:, -1, :].detach().numpy()
                )
            return hook
        target_layer = model.model.layers[l]
        handle = target_layer.register_forward_hook(make_hook(l))
        handles.append(handle)

    for i in range(len(inputs_all)):
        a, b = inputs_all[i].tolist()
        prompt = make_prompt(a, b)
        tokenized = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            model(**tokenized)
        all_labels.append(labels_all[i].item())

    for h in handles:
        h.remove()

    X_per_layer = {}
    for l in layer_indices:
        arr = np.concatenate(activations_per_layer[l], axis=0)
        X_per_layer[l] = arr

    y = np.array(all_labels)
    return X_per_layer, y


def run_probes():
    """Run full Phi-2 probing pipeline: baseline -> activations -> probes.

    Purpose:
        Top-level entry point for Phase 3 experiment. Determines whether
        Phi-2 encodes modular arithmetic structure and which layer is best.
    What:
        1. Compute baseline accuracy
        2. Extract activations from LAYERS
        3. Train probe per layer, record accuracy
        4. Plot probe accuracy across layers
        5. Return best layer and max accuracy
    Why:
        Results inform the steering layer choice (best_layer) and establish
        whether Phi-2 has enough structure (max_acc > 0.05) for steering.
    """
    baseline_acc, model, tokenizer = phi2_baseline()

    print("Extracting activations from Phi-2 layers...")
    X_per_layer, y = extract_activations(model, tokenizer, LAYERS)

    layer_accs = []
    for l in LAYERS:
        acc, _, _ = train_probe(X_per_layer[l], y, test_size=0.3)
        layer_accs.append((l, acc))
        print(f"  Layer {l:2d}: probe acc = {acc:.4f}")

    plot_probe_per_layer(layer_accs, f"{ARTIFACTS}/probe_accuracy_per_layer_phi2.png")

    max_acc = max(acc for _, acc in layer_accs)
    best_layer = max(layer_accs, key=lambda x: x[1])[0]
    print(f"\nBest layer: {best_layer} with acc = {max_acc:.4f}")

    if max_acc > 0.05:
        print("Structure detected in Phi-2 — continuing to steering")
    else:
        print("No structure in Phi-2 — experiment ended (Outcome D)")

    return baseline_acc, best_layer, max_acc, layer_accs


if __name__ == "__main__":
    run_probes()
