import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import os, sys, math

from utils import DEVICE, P

ARTIFACTS = "artifacts"
OUT_DIR = f"{ARTIFACTS}/trivial_baselines"
os.makedirs(OUT_DIR, exist_ok=True)

OP = sys.argv[1] if len(sys.argv) > 1 else "add"
SUFFIX = "" if OP == "add" else "_mult"

N_FEWSHOT = 5
BATCH_SIZE = 32
N_EVAL = 500
LORA_R = 8
LORA_EPOCHS = 10


def get_test_pairs(n=500):
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    labels = (a + b) % P if OP == "add" else (a * b) % P
    rng = np.random.RandomState(42)
    idx = rng.choice(P * P, size=n, replace=False)
    pairs = [(int(a[i]), int(b[i])) for i in idx]
    return pairs, labels[idx].numpy()


def build_fewshot_prompt(test_a, test_b, num_shots, tokenizer):
    rng = np.random.RandomState(42)
    shot_ids = rng.choice(P * P, size=num_shots, replace=False)
    a_all = torch.arange(P).repeat_interleave(P)
    b_all = torch.arange(P).repeat(P)
    label_all = (a_all + b_all) % P if OP == "add" else (a_all * b_all) % P

    op_str = "+" if OP == "add" else "*"
    examples = []
    for sid in shot_ids:
        sa, sb = int(a_all[sid]), int(b_all[sid])
        slabel = int(label_all[sid])
        examples.append(f"# ({sa} {op_str} {sb}) % 97 = {slabel}")
    examples.append(f"# ({test_a} {op_str} {test_b}) % 97 =")
    return "\n".join(examples)


def evaluate_fewshot(model, tokenizer):
    print(f"\n[Few-shot] Evaluating Phi-2 with {N_FEWSHOT}-shot prompting...")
    model.eval()
    test_pairs, test_labels = get_test_pairs(N_EVAL)
    number_tokens = {n: tokenizer.encode(str(n))[0] for n in range(P)}
    correct = 0
    for i, ((a, b), lbl) in enumerate(zip(test_pairs, test_labels)):
        prompt = build_fewshot_prompt(a, b, N_FEWSHOT, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        pred = max(number_tokens, key=lambda n: logits[number_tokens[n]].item())
        if pred == lbl:
            correct += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{N_EVAL}: acc so far = {correct/(i+1):.4f}")
    acc = correct / N_EVAL
    print(f"  Few-shot ({N_FEWSHOT}-shot) accuracy: {acc:.4f}")
    return acc


def train_lora(model, tokenizer):
    print(f"\n[LoRA] Training Phi-2 with LoRA r={LORA_R}...")

    a_all = torch.arange(P).repeat_interleave(P)
    b_all = torch.arange(P).repeat(P)
    labels_all = (a_all + b_all) % P if OP == "add" else (a_all * b_all) % P

    rng = np.random.RandomState(42)
    idx = np.arange(P * P)
    rng.shuffle(idx)
    split = int(len(idx) * 0.7)
    train_idx, test_idx = idx[:split], idx[split:]

    op_str = "+" if OP == "add" else "*"
    train_prompts = [f"# ({int(a_all[i])} {op_str} {int(b_all[i])}) % 97 = "
                     for i in train_idx]
    test_prompts = [f"# ({int(a_all[i])} {op_str} {int(b_all[i])}) % 97 = "
                    for i in test_idx]
    train_labels = labels_all[train_idx].numpy()
    test_labels = labels_all[test_idx].numpy()

    # Tokenize all at once
    train_enc = tokenizer(train_prompts, padding=True, return_tensors="pt")
    test_enc = tokenizer(test_prompts, padding=True, return_tensors="pt")

    train_labels_t = torch.from_numpy(train_labels).long()
    test_labels_t = torch.from_numpy(test_labels).long()

    number_ids = [tokenizer.encode(str(n))[0] for n in range(P)]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model_lora = get_peft_model(model, lora_cfg)
    model_lora.train()

    opt = torch.optim.AdamW(model_lora.parameters(), lr=5e-5)

    train_dataset = TensorDataset(
        train_enc["input_ids"], train_enc["attention_mask"], train_labels_t
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    best_val_acc = 0.0
    best_state = None
    patience = 2
    no_improve = 0

    for epoch in range(1, LORA_EPOCHS + 1):
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
            input_ids, attn_mask, lbls = batch
            outputs = model_lora(input_ids=input_ids, attention_mask=attn_mask,
                                 labels=input_ids)
            logits = outputs.logits[:, -1, :]  # last token
            num_logits = logits[:, number_ids]
            loss = F.cross_entropy(num_logits, lbls)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        model_lora.eval()
        with torch.no_grad():
            out = model_lora(input_ids=test_enc["input_ids"],
                             attention_mask=test_enc["attention_mask"])
            logits = out.logits[:, -1, :]
            num_logits = logits[:, number_ids]
            val_acc = (num_logits.argmax(dim=1) == test_labels_t).float().mean().item()
        model_lora.train()

        print(f"  [LoRA] epoch {epoch}: loss={total_loss/n_batches:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model_lora.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [LoRA] Early stopping at epoch {epoch}")
                break

    # Restore best
    if best_state is not None:
        model_lora.load_state_dict(best_state)

    model_lora.eval()

    # Full test set evaluation
    with torch.no_grad():
        out = model_lora(input_ids=test_enc["input_ids"],
                         attention_mask=test_enc["attention_mask"])
        logits = out.logits[:, -1, :]
        num_logits = logits[:, number_ids]
        test_acc = (num_logits.argmax(dim=1) == test_labels_t).float().mean().item()

    # PPL on WikiText-2
    print("  [LoRA] Evaluating WikiText-2 PPL...")
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    ppl = compute_ppl(model_lora, tokenizer, wiki)

    print(f"  [LoRA] Test acc: {test_acc:.4f}, WikiText-2 PPL: {ppl:.2f}")

    # Save model
    path = f"{OUT_DIR}/lora_r{LORA_R}_{OP}.pt"
    torch.save(model_lora.state_dict(), path)
    print(f"  [LoRA] Saved: {path}")

    return test_acc, ppl


def compute_ppl(model, tokenizer, dataset, max_samples=300, max_len=64):
    model.eval()
    losses = []
    rng = np.random.RandomState(42)
    seen = 0
    for item in dataset:
        if seen >= max_samples:
            break
        text = item["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, truncation=True, max_length=max_len)
        if len(ids) < 3:
            continue
        inputs = ids[:-1]
        targets = torch.tensor([ids[-1]])
        padded = torch.tensor([inputs])
        mask = torch.ones((1, len(inputs)), dtype=torch.long)

        with torch.no_grad():
            out = model(input_ids=padded, attention_mask=mask)
            logits = out.logits[0, -1, :]
            loss = F.cross_entropy(logits.unsqueeze(0), targets)
        losses.append(loss.item())
        seen += 1

    mean_loss = float(np.mean(losses))
    return float(np.exp(mean_loss))


def main():
    print("=" * 60)
    print(f"Trivial Baselines: Few-shot + LoRA (op={OP})")
    print("=" * 60)

    print("\n[0] Loading Phi-2...")
    phi2 = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", dtype=torch.float32, device_map=None, use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    tokenizer.pad_token = tokenizer.eos_token
    phi2.eval()
    print("  Phi-2 loaded.")

    results = {}

    # Few-shot
    fs_acc = evaluate_fewshot(phi2, tokenizer)
    results["fewshot"] = fs_acc

    # LoRA
    lora_acc, lora_ppl = train_lora(phi2, tokenizer)
    results["lora_acc"] = lora_acc
    results["lora_ppl"] = lora_ppl

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Few-shot ({N_FEWSHOT}-shot) accuracy: {results['fewshot']:.4f}")
    print(f"  LoRA (r={LORA_R}) test accuracy: {results['lora_acc']:.4f}")
    print(f"  LoRA WikiText-2 PPL: {results['lora_ppl']:.2f}")

    with open(f"{OUT_DIR}/results_{OP}.md", "w") as f:
        f.write(f"# Trivial Baselines: {OP}\n\n")
        f.write(f"| Baseline | Accuracy | PPL (WikiText-2) |\n")
        f.write(f"|----------|----------|-------------------|\n")
        f.write(f"| {N_FEWSHOT}-shot prompting | {results['fewshot']:.4f} | - |\n")
        f.write(f"| LoRA r={LORA_R} | {results['lora_acc']:.4f} | {results['lora_ppl']:.2f} |\n")
        f.write(f"\n_Generated by trivial_baselines.py_\n")

    print(f"\nResults saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
