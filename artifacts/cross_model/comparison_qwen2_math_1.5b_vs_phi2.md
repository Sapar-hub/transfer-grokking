# Cross-Model L31: Qwen2-Math-1.5B vs Phi-2

## Setup

| Parameter | Value |
|-----------|-------|
| Target model | Qwen2-Math-1.5B (Qwen/Qwen2-Math-1.5B) |
| Layers | 28 |
| d_model | 1536 |
| Patch layer | L27 |
| Phi-2 ref layer | L31 |
| Test pairs | 200 (seed=42) |

## Alpha sweep

| Alpha | Qwen2-Math L27 | Phi-2 L31 | Delta |
|-------|----------------|-----------|-------|
| 0.0 | 0.0100 | 0.2350 | -0.2250 |
| 0.3 | 0.0150 | 0.4900 | -0.4750 |
| 0.5 | 0.0150 | 0.7050 | -0.6900 |
| 0.7 | 0.0250 | 0.9950 | -0.9700 |
| 1.0 | 0.0900 | 1.0000 | -0.9100 |

Baseline (alpha=0.0): 0.0100

Logit lens (W_CE): 0.0726

**Hypothesis check**: Low logit lens suggests Qwen2-Math lm_head is not aligned with mod arithmetic tokens.

**Comparison**: Qwen2-Math (0.0900) << Phi-2 (1.0000).

## ⚠️ Tokenizer Barrier

The comparison is **fundamentally invalid**: Qwen2-Math's BPE tokenizer splits 87/97 numbers into subword tokens (e.g., "97" → [16, 23] for tokens "9" and "7"). Only single digits 0–9 encode to unique token IDs — **10 unique tokens for 97 classes**.

Consequences:
- **Logit lens capped at ~10%**: lm_head has only 10 output dimensions to distinguish 97 classes. W_CE val_acc=0.0726 reflects this ceiling, not absence of arithmetic structure.
- **Alpha sweep capped**: evaluation via `argmax over number_ids` compares logits among 10 tokens, making accurate 97-class identification impossible.
- **The hypothesis is untestable with this method**: even if Qwen2-Math encodes perfect arithmetic structure at L27, the lm_head's output space cannot express it.

To properly test the hypothesis, one would need:
1. A model whose tokenizer encodes each number 0–96 as a single token (like Phi-2's tokenizer does for 0–9).
2. Or: use a probe (logistic regression on hidden states), not the lm_head, to measure arithmetic structure — which `scan_models.py` already did: Qwen2-Math probe acc = 0.0276 (barely above random 0.0103), confirming no arithmetic structure regardless of tokenizer.