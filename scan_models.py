import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json, sys, time

from model import SmallTransformer
from utils import DEVICE, P

ARTIFACTS = "artifacts"
RESULTS_DIR = f"{ARTIFACTS}/probe_results"
BATCH_SIZE = 256

os.makedirs(RESULTS_DIR, exist_ok=True)

CANDIDATES = [
    ("Qwen2-Math-1.5B", "Qwen/Qwen2-Math-1.5B"),
    ("DeepSeek-Math-7B", "deepseek-ai/deepseek-math-7b"),
    ("Phi-3-mini-4k", "microsoft/Phi-3-mini-4k-instruct"),
]

THRESHOLDS = {"great": 0.70, "better_than_phi2": 0.41}


# ─── Step 1: Cache small model activations ─────────────────────────────

def cache_small_activations():
    """Cache small model activations on all P^2 pairs for probe training.

    Purpose:
        Pre-compute the reference activations used as features for
        logistic regression probes across multiple LLMs.
    What:
        Runs the small model on all 9409 (a, b) pairs, saves the last
        layer's residual stream activations (pos 1, i.e. after seeing
        both tokens) to artifacts/small_model_activations.npy.
    Why:
        Avoids re-running the small model for each LLM probe. These
        activations serve as the positive control: if a probe on them
        gives acc=1.0, the probe setup is correct.
    """
    path_acts = f"{ARTIFACTS}/small_model_activations.npy"
    path_lbls = f"{ARTIFACTS}/mod_arithmetic_labels.npy"
    if os.path.exists(path_acts) and os.path.exists(path_lbls):
        print("[cache] small_model_activations already cached, loading...")
        return np.load(path_acts), np.load(path_lbls)

    print("[cache] Running small model on all P^2 pairs...")
    model = SmallTransformer().to(DEVICE)
    state = torch.load(f"{ARTIFACTS}/small/best_model.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    inputs = torch.stack([a, b], dim=1)
    labels = (a + b) % P

    all_acts = []
    for i in range(0, len(inputs), BATCH_SIZE):
        x = inputs[i:i+BATCH_SIZE]
        with torch.no_grad():
            _, acts = model(x, return_activations=True)
        batch_acts = acts["blocks.1.hook_resid_post"][:, 1, :].numpy()
        all_acts.append(batch_acts)

    acts_arr = np.concatenate(all_acts, axis=0)
    lbls_arr = labels.numpy()

    np.save(path_acts, acts_arr)
    np.save(path_lbls, lbls_arr)
    print(f"[cache] Saved {acts_arr.shape[0]} activations [{acts_arr.shape[1]}] + labels")
    return acts_arr, lbls_arr


# ─── Step 2-3: Probe a model ──────────────────────────────────────────

def probe_model(name, hf_name):
    """Probe a HuggingFace model for modular arithmetic structure.

    Purpose:
        For a given LLM, extract activations at every layer on all P^2
        pairs and train logistic regression probes to measure class
        separability.
    What:
        Loads model + tokenizer, registers hooks on all layers, processes
        all pairs with prompts like "a b", captures last-token activations,
        trains 97-class LogisticRegression per layer.
    Why:
        Core measurement for Phase 3: do LLMs encode modular arithmetic
        in their residual stream? The probe accuracy across layers reveals
        both whether (max > 1/97) and where (which layer) the structure
        exists.
    """
    print(f"\n{'='*60}")
    print(f"Probing {name} ({hf_name})")
    print(f"{'='*60}")

    result_path = f"{RESULTS_DIR}/{name.lower().replace('-','_')}_probe_per_layer.npy"
    if os.path.exists(result_path):
        print(f"[probe] {result_path} exists, skipping.")
        data = np.load(result_path, allow_pickle=True).item()
        return data

    t0 = time.time()

    proxy_env = {"HTTPS_PROXY": "socks5://127.0.0.1:1080", "HTTP_PROXY": "socks5://127.0.0.1:1080"}
    try:
        print("  Loading model (with SOCKS5 proxy)...")
        orig_env = {k: os.environ.get(k) for k in proxy_env}
        os.environ.update(proxy_env)
        model = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.float32, device_map=None)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
    except Exception as e:
        print(f"  Proxy failed ({type(e).__name__}), trying direct...")
        for k in proxy_env:
            os.environ.pop(k, None)
        model = AutoModelForCausalLM.from_pretrained(hf_name, dtype=torch.float32, device_map=None)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
    finally:
        for k in proxy_env:
            os.environ.pop(k, None)
            if orig_env.get(k):
                os.environ[k] = orig_env[k]

    tokenizer.pad_token = tokenizer.eos_token

    # Determine model architecture
    config = model.config
    if hasattr(config, "num_hidden_layers"):
        n_layers = config.num_hidden_layers
    elif hasattr(config, "num_layers"):
        n_layers = config.num_layers
    else:
        n_layers = len(model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else 0
    d_model = config.hidden_size if hasattr(config, "hidden_size") else config.d_model
    print(f"  Architecture: {n_layers} layers, d_model={d_model}")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Generate all P^2 pairs
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    inputs_list = [(int(ai), int(bi)) for ai, bi in zip(a, b)]
    labels_arr = ((a + b) % P).numpy()
    num_pairs = len(inputs_list)

    # Prepare hooks
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_module = model.model.layers
    else:
        raise ValueError(f"Unknown layer structure for {name}")

    activations_per_layer = {l: [] for l in range(n_layers)}
    current_attention_mask = None

    def make_hook(layer_idx):
        def hook(module, input, output):
            nonlocal current_attention_mask
            hidden = output[0] if isinstance(output, tuple) else output
            mask = current_attention_mask.to(hidden.device)
            seq_lens = mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            last_hidden = hidden[batch_idx, seq_lens]
            activations_per_layer[layer_idx].append(last_hidden.detach().cpu())
        return hook

    handles = []
    for i in range(n_layers):
        h = layer_module[i].register_forward_hook(make_hook(i))
        handles.append(h)

    model.eval()
    print(f"  Processing {num_pairs} pairs in batches of {BATCH_SIZE}...")
    t0_batch = time.time()
    for start in range(0, num_pairs, BATCH_SIZE):
        batch = inputs_list[start:start+BATCH_SIZE]
        prompts = [f"{a} {b}" for a, b in batch]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt")
        current_attention_mask = tokenized.attention_mask
        with torch.no_grad():
            model(**tokenized)
        if (start // BATCH_SIZE) % 20 == 0 and start > 0:
            elapsed = time.time() - t0_batch
            rate = start / elapsed
            remaining = (num_pairs - start) / rate
            print(f"    batch {start//BATCH_SIZE}/{num_pairs//BATCH_SIZE} | {rate:.0f} pairs/s | ETA {remaining:.0f}s")

    for h in handles:
        h.remove()

    total_time = time.time() - t0_batch
    print(f"  Batch processing done in {total_time:.0f}s ({num_pairs/total_time:.0f} pairs/s)")

    # Train probe per layer
    print("  Training probes...")
    layer_accs = []
    for l in range(n_layers):
        X = torch.cat(activations_per_layer[l], dim=0).numpy()
        y = labels_arr

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        probe = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0, random_state=42)
        probe.fit(X_train, y_train)
        acc = probe.score(X_test, y_test)
        layer_accs.append((l, acc))

        if l % 4 == 0 or l == n_layers - 1:
            print(f"    Layer {l:3d}: probe_acc = {acc:.4f}")

    max_acc = max(acc for _, acc in layer_accs)
    best_layer = max(layer_accs, key=lambda x: x[1])[0]
    print(f"  Best: layer {best_layer} with acc = {max_acc:.4f}")

    data = {
        "name": name,
        "hf_name": hf_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "max_probe_acc": max_acc,
        "best_layer": best_layer,
        "probe_acc_per_layer": np.array([acc for _, acc in layer_accs]),
        "total_time_s": time.time() - t0,
    }
    np.save(result_path, data)
    print(f"  Saved to {result_path}")
    return data


# ─── Step 3: Comparison plot ──────────────────────────────────────────

def plot_comparison(results_list):
    """Plot probe accuracy per layer for all probed models.

    Purpose:
        Visual comparison of how much modular arithmetic structure each
        LLM encodes, and at which layers.
    What:
        Overlays per-layer probe accuracy for all models, with horizontal
        lines at random baseline, Phi-2 threshold, and "great" threshold.
    Why:
        Enables quick identification of the best model and the best layer
        for downstream steering experiments.
    """
    plt.figure(figsize=(12, 6))
    for res in results_list:
        accs = res["probe_acc_per_layer"]
        layers = np.arange(len(accs))
        plt.plot(layers, accs, 'o-', label=f"{res['name']} (max={res['max_probe_acc']:.3f})", markersize=3)

    plt.axhline(y=1/P, color='gray', linestyle='--', alpha=0.4, label=f'random={1/P:.3f}')
    plt.axhline(y=THRESHOLDS["better_than_phi2"], color='orange', linestyle=':', alpha=0.6, label=f'better_than_phi2={THRESHOLDS["better_than_phi2"]}')
    plt.axhline(y=THRESHOLDS["great"], color='green', linestyle=':', alpha=0.6, label=f'great={THRESHOLDS["great"]}')

    plt.xlabel('Layer')
    plt.ylabel('Probe Accuracy')
    plt.title('Probe Accuracy per Layer — Model Comparison')
    plt.legend()
    plt.tight_layout()
    path = f"{RESULTS_DIR}/comparison_plot.png"
    plt.savefig(path)
    plt.close()
    print(f"Comparison plot saved to {path}")


# ─── Step 4: Select best ──────────────────────────────────────────────

def select_best(results_list):
    """Select the best-probed model and save results.

    Purpose:
        Determine which LLM encodes the most modular arithmetic structure
        and record the decision for downstream experiments.
    What:
        Sorts by max_probe_acc descending, picks the best, saves to
        artifacts/probe_results/best_model.txt with verdict.
    Why:
        The best model is used as the steering target in Experiment A.
        The verdict (GREAT / OK / WEAK) determines whether steering is
        attempted.
    """
    results_list.sort(key=lambda r: r["max_probe_acc"], reverse=True)
    best = results_list[0]

    path = f"{RESULTS_DIR}/best_model.txt"
    with open(path, "w") as f:
        f.write(f"Best model: {best['name']}\n")
        f.write(f"HF name: {best['hf_name']}\n")
        f.write(f"Best layer: {best['best_layer']}\n")
        f.write(f"Max probe acc: {best['max_probe_acc']:.4f}\n")
        f.write(f"d_model: {best['d_model']}\n")
        f.write(f"n_layers: {best['n_layers']}\n")
        f.write(f"\nAll results:\n")
        for r in results_list:
            verdict = ""
            if r["max_probe_acc"] > THRESHOLDS["great"]:
                verdict = "=> GREAT (>=0.70, go to A)"
            elif r["max_probe_acc"] > THRESHOLDS["better_than_phi2"]:
                verdict = f"=> OK (>0.41, better than Phi-2)"
            else:
                verdict = f"=> WEAK (<=0.41, no better than Phi-2)"
            f.write(f"  {r['name']}: max={r['max_probe_acc']:.4f} layer={r['best_layer']} {verdict}\n")

    print(f"Best model: {best['name']}, layer {best['best_layer']}, probe_acc = {best['max_probe_acc']:.4f}")

    if best["max_probe_acc"] >= THRESHOLDS["great"]:
        print(f"  => Outcome: GREAT. Go to experiment A.")
    elif best["max_probe_acc"] > THRESHOLDS["better_than_phi2"]:
        print(f"  => Outcome: OK but weak ceiling.")
    else:
        print(f"  => Outcome: No model better than Phi-2.")

    return best


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    """Orchestrate scanning: cache small activations, probe all candidates.

    Purpose:
        Top-level entry point for Phase 3. Runs through all candidate
        LLMs, probes each for modular arithmetic structure, selects the
        best one for Experiment A.
    What:
        1. Cache small model activations (positive control)
        2. For each candidate (Qwen2-Math, DeepSeek-Math, Phi-3):
           probe all layers, train classifiers, record accuracy
        3. Plot comparison chart
        4. Select best model and save decision
    Why:
        Phase 3 answers: which LLM encodes modular arithmetic best?
        The best model becomes the target for steering experiments.
    """
    print("Step 1: Cache small model activations")
    cache_small_activations()

    print("\nStep 2-4: Probe candidate models")
    results = []
    for name, hf_name in CANDIDATES:
        try:
            data = probe_model(name, hf_name)
            results.append(data)
            if data["max_probe_acc"] >= THRESHOLDS["great"]:
                print(f"\n{name} already exceeds threshold (>={THRESHOLDS['great']}). Skipping remaining models.")
                break
        except Exception as e:
            print(f"\nError probing {name}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

    if not results:
        print("No models probed successfully.")
        return

    print("\nStep 3: Comparison plot")
    plot_comparison(results)

    print("\nStep 4: Select best model")
    best = select_best(results)

    print(f"\n{'='*60}")
    print(f"RESULT: {best['name']}, layer {best['best_layer']}, probe_acc = {best['max_probe_acc']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
