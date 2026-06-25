# Experiment A: Learned Projection — Summary

## Results

| Metric | Value |
|--------|-------|
| Small model → Phi-2 linear map | W: R¹²⁸ → R²⁵⁶⁰ |
| Test MSE | 20.42 |
| Cosine similarity (test) | **0.3264** |
| Steering delta | **+0.0000** (no effect) |

## Why W failed (cos_sim = 0.33 ≪ 0.85)

**root cause:** tokenizer mismatch.

- Small model input: raw token IDs `[0..96]` → learned embedding → residual stream
- Phi-2 input: text tokens `"0" "1"` etc. → BPE tokenizer → different embedding space

Small model activations capture structure of a learned 128-dim embedding space for numbers 0-96. Phi-2 activations capture structure of BPE tokens for string representations of numbers. These spaces are **linearly incommensurable** — a linear map cannot bridge them.

The low cosine similarity (0.33) confirms that W cannot align the two activation spaces.

## Why probe on W(small) = 0.9968 is misleading

W doesn't converge to target_acts. With MSE=20.4, the output still mostly preserves small_acts structure. The probe simply reads the original answer information that survived through the poor projection — not evidence of geometry transfer.

## Why steering = 0 effect

Steering vector computed from small model, projected through W, lands in a direction that corresponds to small model's residual stream, not Phi-2's. Since W maps into a space that doesn't align with Phi-2's internal representations, the hook addition has no meaningful effect on Phi-2's computation.

## Outcome

```
steering_delta = 0.0000 → C (гипотеза не подтверждена)
```

**Fundamental limitation:** tokenizer mismatch between models. The small model's token IDs (0-96 direct) and Phi-2's BPE tokenization of number strings create incompatible activation geometries.

For future work: train small model with same tokenizer as target model, or use continuous number embeddings (e.g., sin/cos encoding) instead of learned discrete embeddings.
