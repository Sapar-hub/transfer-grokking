import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import chi2
import json
import os

from steering import compute_steering_vector, random_orthogonal_projection, D_SMALL, D_PHI2
from utils import DEVICE, P, make_steering_hook

ARTIFACTS = "artifacts"


def mcnemar_test(baseline_correct, steered_correct):
    """McNemar's test for paired binary outcomes.

    Purpose:
        Determine whether steering significantly changes accuracy at the
        sample level (not just aggregate). More sensitive than a simple
        accuracy comparison.
    What:
        Computes n01 = baseline_right & steered_wrong, n10 = baseline_wrong
        & steered_right, applies McNemar chi-squared with continuity
        correction, returns p-value.
    Why:
        Standard test for paired binary outcomes. Used to detect if steering
        systematically flips predictions in one direction, even when the
        aggregate accuracy change is small.
    """
    n01 = np.sum(baseline_correct & ~steered_correct)
    n10 = np.sum(~baseline_correct & steered_correct)
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10 + 1e-10)
    p_value = 1 - chi2.cdf(statistic, 1)
    return p_value


def eval_on_mod_arithmetic(model, tokenizer, steering_vec_proj=None, alpha=0.0, best_layer=None):
    """Evaluate Phi-2 on mod arithmetic with optional steering.

    Purpose:
        Measure accuracy on (a+b) mod 97 under a given steering condition.
    What:
        Generates 200 random pairs, hooks steering at best_layer if
        provided, evaluates Phi-2, returns accuracy and per-sample results.
    Why:
        Primary metric for steering effect. Can be used at baseline (no
        steering) or with learned/random projection. Returns per-sample
        for McNemar test.
    """
    rng = np.random.RandomState(42)
    pairs = [(rng.randint(0, P), rng.randint(0, P)) for _ in range(200)]

    handle = None
    if steering_vec_proj is not None and alpha > 0 and best_layer is not None:
        if not isinstance(steering_vec_proj, torch.Tensor):
            steering_tensor = torch.from_numpy(steering_vec_proj).float()
        else:
            steering_tensor = steering_vec_proj.float()

        handle = model.model.layers[best_layer].register_forward_hook(
            make_steering_hook(steering_tensor, alpha)
        )

    correct = 0
    per_sample_correct = []
    for a, b in pairs:
        prompt = f"# ({a} + {b}) % 97 ="
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        predicted = logits.argmax(dim=-1).item()
        decoded = tokenizer.decode(predicted).strip()
        try:
            answer = int(decoded.split()[0] if len(decoded.split()) else decoded)
        except ValueError:
            answer = -1
        is_correct = answer == (a + b) % P
        per_sample_correct.append(is_correct)
        if is_correct:
            correct += 1

    if handle:
        handle.remove()

    return correct / len(pairs), np.array(per_sample_correct)


def eval_on_benchmark(model, tokenizer, name, num_samples=100, steering_vec_proj=None, alpha=0.0, best_layer=None):
    """Evaluate Phi-2 on a downstream benchmark with/without steering.

    Purpose:
        Measure whether steering degrades performance on other tasks
        (Hellaswag, LAMBADA, Winogrande, Boolq).
    What:
        Delegates to the appropriate benchmark evaluator, hooks steering
        at best_layer if provided.
    Why:
        Degradation test: if steering improves mod arithmetic but destroys
        general language understanding, it's not a viable method. We need
        steering to be specific to the target computation.
    """
    handle = None
    if steering_vec_proj is not None and alpha > 0 and best_layer is not None:
        if not isinstance(steering_vec_proj, torch.Tensor):
            steering_tensor = torch.from_numpy(steering_vec_proj).float()
        else:
            steering_tensor = steering_vec_proj.float()

        handle = model.model.layers[best_layer].register_forward_hook(
            make_steering_hook(steering_tensor, alpha)
        )

    rng = np.random.RandomState(42)

    if name == "hellaswag":
        correct, per_sample = eval_hellaswag(model, tokenizer, rng, num_samples)
    elif name == "lambada":
        correct, per_sample = eval_lambada(model, tokenizer, rng, num_samples)
    elif name == "winogrande":
        correct, per_sample = eval_winogrande(model, tokenizer, rng, num_samples)
    elif name == "boolq":
        correct, per_sample = eval_boolq(model, tokenizer, rng, num_samples)
    else:
        raise ValueError(f"Unknown benchmark: {name}")

    if handle:
        handle.remove()

    return correct, per_sample


def eval_hellaswag(model, tokenizer, rng, num_samples):
    """Evaluate on Hellaswag (commonsense NLI).

    Purpose:
        Measure general language understanding under steering.
    What:
        Uses datasets library to load Hellaswag validation, computes
        accuracy by scoring each ending and picking the max.
    Why:
        One of four benchmark tasks for degradation testing.
    """
    from datasets import load_dataset
    ds = load_dataset("hellaswag", split="validation", streaming=True)
    indices = rng.choice(len(ds), min(num_samples, len(ds)), replace=False)
    correct = 0
    per_sample = []
    for i in indices:
        item = ds[int(i)]
        ctx = item["ctx"]
        endings = item["endings"]
        scores = []
        for end in endings:
            text = ctx + " " + end
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            scores.append(logits.softmax(dim=-1).max().item())
        predicted = np.argmax(scores)
        is_correct = predicted == int(item["label"])
        per_sample.append(is_correct)
        if is_correct:
            correct += 1
    return correct / len(indices), np.array(per_sample)


def eval_lambada(model, tokenizer, rng, num_samples):
    """Evaluate on LAMBADA (next-word prediction).

    Purpose:
        Measure language modelling capability under steering.
    What:
        Uses datasets library to load LAMBADA validation, computes
        accuracy by predicting the last token of each passage.
    Why:
        One of four benchmark tasks for degradation testing.
    """
    from datasets import load_dataset
    ds = load_dataset("lambada", split="validation", streaming=True)
    indices = rng.choice(min(len(ds), 1000), min(num_samples, len(ds)), replace=False)
    correct = 0
    per_sample = []
    for i in indices:
        item = ds[int(i)]
        text = item["text"]
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids
        target_id = input_ids[0, -1].item()
        with torch.no_grad():
            outputs = model(input_ids[:, :-1])
        logits = outputs.logits[:, -1, :]
        predicted = logits.argmax(dim=-1).item()
        is_correct = predicted == target_id
        per_sample.append(is_correct)
        if is_correct:
            correct += 1
    return correct / len(indices), np.array(per_sample)


def eval_winogrande(model, tokenizer, rng, num_samples):
    """Evaluate on Winogrande (pronoun resolution).

    Purpose:
        Measure coreference resolution ability under steering.
    What:
        Uses datasets library to load Winogrande XL validation, scores
        each option by max logit softmax, picks the higher-scoring one.
    Why:
        One of four benchmark tasks for degradation testing.
    """
    from datasets import load_dataset
    ds = load_dataset("winogrande", "winogrande_xl", split="validation", streaming=True)
    indices = rng.choice(min(len(ds), 1000), min(num_samples, len(ds)), replace=False)
    correct = 0
    per_sample = []
    for i in indices:
        item = ds[int(i)]
        sentence = item["sentence"]
        option1 = item["option1"]
        option2 = item["option2"]
        scores = []
        for opt in [option1, option2]:
            text = sentence.replace("_", opt)
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            scores.append(logits.softmax(dim=-1).max().item())
        predicted = np.argmax(scores) + 1
        is_correct = predicted == int(item["answer"])
        per_sample.append(is_correct)
        if is_correct:
            correct += 1
    return correct / len(indices), np.array(per_sample)


def eval_boolq(model, tokenizer, rng, num_samples):
    """Evaluate on Boolq (yes/no reading comprehension).

    Purpose:
        Measure reading comprehension ability under steering.
    What:
        Uses datasets library to load Boolq validation, compares logits
        for "Yes" vs "No" tokens.
    Why:
        One of four benchmark tasks for degradation testing.
    """
    from datasets import load_dataset
    ds = load_dataset("boolq", split="validation", streaming=True)
    indices = rng.choice(min(len(ds), 1000), min(num_samples, len(ds)), replace=False)
    correct = 0
    per_sample = []
    for i in indices:
        item = ds[int(i)]
        question = item["question"]
        passage = item["passage"]
        text = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        yes_id = tokenizer.encode("Yes")[0]
        no_id = tokenizer.encode("No")[0]
        predicted = yes_id if logits[0, yes_id] > logits[0, no_id] else no_id
        is_correct = (predicted == yes_id) == item["answer"]
        per_sample.append(is_correct)
        if is_correct:
            correct += 1
    return correct / len(indices), np.array(per_sample)


def run_eval(steering_vec, best_layer, alpha):
    """Run full degradation evaluation: mod arithmetic + 4 benchmarks.

    Purpose:
        Top-level entry point for Phase 4+ steering evaluation.
    What:
        1. Load Phi-2
        2. Random-project steering vector
        3. Evaluate mod arithmetic baseline + steered
        4. Evaluate 4 benchmarks (Hellaswag, LAMBADA, Winogrande, Boolq)
           with McNemar test on each
        5. Save results to JSON
    Why:
        Comprehensive evaluation of steering impact. Tests both improvement
        on target task (mod arith) and degradation on general tasks.
    """
    model_name = "microsoft/phi-2"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    R = random_orthogonal_projection(D_SMALL, D_PHI2)
    steering_proj = R @ steering_vec

    baseline_mod, _ = eval_on_mod_arithmetic(model, tokenizer)
    steered_mod, _ = eval_on_mod_arithmetic(model, tokenizer, steering_proj, alpha, best_layer)

    benchmarks = ["hellaswag", "lambada", "winogrande", "boolq"]
    results = {"mod_arithmetic_baseline": baseline_mod, "mod_arithmetic_steered": steered_mod}

    for bm in benchmarks:
        print(f"  Evaluating {bm}...")
        base_correct, base_per = eval_on_benchmark(
            model, tokenizer, bm, num_samples=100
        )
        steer_correct, steer_per = eval_on_benchmark(
            model, tokenizer, bm, num_samples=100,
            steering_vec_proj=steering_proj, alpha=alpha, best_layer=best_layer
        )
        p_val = mcnemar_test(base_per, steer_per)
        results[f"{bm}_baseline"] = base_correct
        results[f"{bm}_steered"] = steer_correct
        results[f"{bm}_delta"] = steer_correct - base_correct
        results[f"{bm}_mcnemar_p"] = p_val
        print(f"    {bm}: baseline={base_correct:.4f}, steered={steer_correct:.4f}, delta={steer_correct - base_correct:.4f}, p={p_val:.4f}")

    with open(f"{ARTIFACTS}/lm_eval_baseline.json", "w") as f:
        json.dump({k: v for k, v in results.items() if "steered" not in k}, f, indent=2)
    with open(f"{ARTIFACTS}/lm_eval_steered.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    vec = compute_steering_vector()
    best_layer = 12
    alpha = 1.0
    results = run_eval(vec, best_layer, alpha)
    print(json.dumps(results, indent=2))
