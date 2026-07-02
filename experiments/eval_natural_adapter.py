"""Evaluate cached natural_adapter results + template generalization + summary.

Run this after natural_adapter.py collected activations and trained adapters.
Skips Phi-2 loading entirely — loads precomputed .npy caches.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os, csv

from utils import P

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/natural_adapter"
LAYERS = [20, 25, 28, 30]
N_TEMPLATES = 4

ACC_TEXT_BASELINE = 0.235


def load_phi2_natural():
    acts = {}
    for l in LAYERS:
        path = f"{OUT_DIR}/phi2_natural_L{l}.npy"
        acts[l] = np.load(path)
        print(f"  phi2_natural_L{l}: {acts[l].shape}")
    return acts


def load_template_gen():
    acts = {}
    for t in range(N_TEMPLATES):
        acts[t] = {}
        for l in LAYERS:
            path = f"{OUT_DIR}/template_gen_T{t}_L{l}.npy"
            acts[t][l] = np.load(path)
    print(f"  template_gen: {N_TEMPLATES} templates × {len(LAYERS)} layers")
    return acts


def run_template_generalization(acts, n_pairs=500):
    print("\n[template_gen] Running 16 cross-evaluations...")
    rng = np.random.RandomState(42)
    pairs = [(int(rng.randint(0, P)), int(rng.randint(0, P))) for _ in range(n_pairs)]
    labels = np.array([(a + b) % P for a, b in pairs])

    results = []
    # Train on T0 only (user spec: "what is (a + b) mod 97?")
    t_train = 0
    for t_test in range(N_TEMPLATES):
        layer_accs = []
        for l in LAYERS:
            X_tr = acts[t_train][l]
            X_te = acts[t_test][l]
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
            probe = LogisticRegression(max_iter=200, solver='lbfgs', C=1.0, random_state=42)
            probe.fit(X_tr_sc, labels)
            acc = probe.score(X_te_sc, labels)
            layer_accs.append((l, acc))

        best = max(layer_accs, key=lambda x: x[1])
        results.append({
            "train_template": t_train,
            "test_template": t_test,
            "best_layer": best[0],
            "best_acc": best[1],
        })
        print(f"    T{t_train} -> T{t_test}: L{best[0]} acc={best[1]:.4f}", flush=True)

    # Also add in-domain accuracy for T1/T2/T3 (train/test on same template, split 70/30)
    for t in range(1, N_TEMPLATES):
        layer_accs = []
        for l in LAYERS:
            X = acts[t][l]
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.3, random_state=42)
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
            probe = LogisticRegression(max_iter=200, solver='lbfgs', C=1.0, random_state=42)
            probe.fit(X_tr_sc, y_tr)
            acc = probe.score(X_te_sc, y_te)
            layer_accs.append((l, acc))
        best = max(layer_accs, key=lambda x: x[1])
        results.append({
            "train_template": t,
            "test_template": t,
            "best_layer": best[0],
            "best_acc": best[1],
        })
        print(f"    T{t} -> T{t} (in-domain): L{best[0]} acc={best[1]:.4f}", flush=True)

    path = f"{OUT_DIR}/template_generalization.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["train_template", "test_template", "best_layer", "best_acc"])
        w.writeheader()
        w.writerows(results)
    print(f"  Saved {path}")
    return results


def write_summary(layer_results, gen_results):
    best_adapter = max(layer_results, key=lambda r: r["adapter_acc"])
    best_sklearn = max(layer_results, key=lambda r: r["sklearn_acc"])

    lines = []
    lines.append("# Natural Adapter Summary\n")
    lines.append("## Baseline\n")
    lines.append("| Condition | Accuracy |")
    lines.append("|-----------|----------|")
    lines.append(f"| Phi-2 LM head (text) | {ACC_TEXT_BASELINE:.4f} |")
    lines.append(f"| Random (1/{P}) | {1/P:.4f} |\n")
    lines.append("## Per-layer adapter accuracy\n")
    lines.append("| Layer | nn.Linear (AdamW) | LogisticRegression |")
    lines.append("|-------|-------------------|--------------------|")
    for r in layer_results:
        lines.append(f"| {r['layer']} | {r['adapter_acc']:.4f} | {r['sklearn_acc']:.4f} |")
    lines.append(f"\n**Best nn.Linear**: L{best_adapter['layer']} acc={best_adapter['adapter_acc']:.4f}")
    lines.append(f"**Best sklearn**: L{best_sklearn['layer']} acc={best_sklearn['sklearn_acc']:.4f}\n")

    lines.append("## Template generalization (LogisticRegression, best per L)\n")
    if gen_results:
        lines.append("| Train → Test | Best L | Acc |")
        lines.append("|--------------|--------|-----|")
        for r in gen_results:
            lines.append(f"| T{r['train_template']} → T{r['test_template']} | {r['best_layer']} | {r['best_acc']:.4f} |")

    max_adapter = best_adapter["adapter_acc"]
    lines.append("\n## Interpretation\n")
    if max_adapter > 0.30:
        lines.append("**adapter acc >> 0.235** → Phi-2 содержит ответ в residual stream.")
        lines.append("LM head был bottleneck, не знание. Вся архитектура с маленькой моделью не нужна.")
    elif max_adapter > 0.25:
        lines.append("**adapter acc ≈ 0.235** → информации на уровне линейного разделения не хватает.")
        lines.append("Нужен нелинейный adapter (MLP).")
    else:
        lines.append("**adapter acc ≈ random** → Phi-2 не кодирует ответ линейно в residual stream через естественный язык.")

    if gen_results:
        t0 = [r for r in gen_results if r["train_template"] == 0]
        if t0:
            same_template = next((r for r in t0 if r["test_template"] == 0), None)
            cross_templates = [r for r in t0 if r["test_template"] != 0]
            if same_template and cross_templates:
                in_domain = same_template["best_acc"]
                cross_mean = np.mean([r["best_acc"] for r in cross_templates])
                if in_domain > 0.06 and cross_mean > in_domain * 0.8:
                    lines.append("\n**Template generalization высокая** → Phi-2 реально понимает задачу через язык.")
                    lines.append("Это работающий 'language-grounded residual tool'.")
                elif in_domain > 0.06 and cross_mean < in_domain * 0.5:
                    lines.append("\n**Template generalization низкая** → adapter выучил поверхностный паттерн.")
                    lines.append("Не обобщается на новые формулировки.")
                else:
                    lines.append("\n**Template generalization умеренная** → частичное обобщение.")
    lines.append("")

    text = "\n".join(lines)
    path = f"{OUT_DIR}/experiment_summary.md"
    with open(path, "w") as f:
        f.write(text)
    print(text)


def main():
    print("Natural Adapter — evaluation (no Phi-2 load)")

    layer_results = []
    for l in LAYERS:
        txt_path = f"{OUT_DIR}/adapter_acc_L{l}.txt"
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                adapter_acc = float(f.read().strip())
        else:
            print(f"  Warning: adapter_acc_L{l}.txt not found, using 0")
            adapter_acc = 0.0
        layer_results.append({
            "layer": l,
            "adapter_acc": adapter_acc,
            "sklearn_acc": None,
        })

    # Fill sklearn acc from results CSV
    if os.path.exists(f"{OUT_DIR}/results_per_layer.csv"):
        with open(f"{OUT_DIR}/results_per_layer.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                l = int(row["layer"])
                for r in layer_results:
                    if r["layer"] == l:
                        r["sklearn_acc"] = float(row["sklearn_acc"])
    for r in layer_results:
        print(f"  L{r['layer']}: adapter={r['adapter_acc']:.4f} sklearn={r['sklearn_acc']}")

    gen_results = []
    if os.path.exists(f"{OUT_DIR}/template_generalization.csv"):
        print("\n[template_gen] Loading cached results...")
        with open(f"{OUT_DIR}/template_generalization.csv") as f:
            reader = csv.DictReader(f)
            gen_results = list(reader)
            for r in gen_results:
                r["train_template"] = int(r["train_template"])
                r["test_template"] = int(r["test_template"])
                r["best_layer"] = int(r["best_layer"])
                r["best_acc"] = float(r["best_acc"])
            print(f"  Loaded {len(gen_results)} results")
    else:
        print("\n[template_gen] Running from cached activations...")
        template_acts = load_template_gen()
        gen_results = run_template_generalization(template_acts)

    print("\n[summary] Writing experiment_summary.md ...")
    write_summary(layer_results, gen_results)

    print(f"\nDone. See {OUT_DIR}/experiment_summary.md")


if __name__ == "__main__":
    main()
