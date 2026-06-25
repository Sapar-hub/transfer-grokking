# Clean Experiment: Compatible Geometry Test

## Setup
| | Model A | Model B |
|---|---|---|
| d_model | 128 | 512 |
| n_layers | 2 | 6 |
| d_mlp | 512 | 2048 |
| Params | ~0.5M | ~8M |
| Vocabulary | 0-97 (raw IDs) | 0-97 (raw IDs) |
| Tokenizer | **identical** | **identical** |
| Grokking | ✅ (1.0) | ✅ (1.0) |

## Probe Accuracy

| Layer | Model A | Model B |
|-------|---------|---------|
| 0 | 0.0000 | 0.0000 |
| 1 | **1.0000** | 0.0083 |
| 2 | — | 0.2798 |
| 3 | — | 0.8264 |
| 4 | — | 0.9976 |
| 5 | — | **1.0000** |

Both models encode the algorithm perfectly in their final layers.

## Learned Projection W: R¹²⁸ → R⁵¹²

| Metric | Value | Threshold |
|--------|-------|-----------|
| Cosine similarity (test) | **0.2966** | ≥ 0.85 ✅ / < 0.50 ❌ |
| Test MSE | 0.2150 | — |
| Projected probe acc | **0.9362** | — |

## Steering

| Method | Mod Accuracy | Delta |
|--------|-------------|-------|
| Baseline (no steering) | 1.0000 | — |
| Learned W (α=0.1) | 1.0000 | +0.0000 |
| Learned W (α=0.5) | 1.0000 | +0.0000 |
| Learned W (α=1.0) | 1.0000 | +0.0000 |
| Learned W (α=2.0) | 1.0000 | +0.0000 |
| Learned W (α=5.0) | 1.0000 | +0.0000 |
| Learned W (α=10.0) | 0.9900 | -0.0100 |

## Analysis

### 1. Tokenizer gap eliminated — but result unchanged
With identical vocabulary and tokenizer, the linear map W still achieves only **cos_sim = 0.30** (vs 0.33 with Phi-2). The tokenizer mismatch was **not** the primary barrier.

### 2. Geometry scales non-linearly
Models of different sizes (128 vs 512 dim) learn activation geometries that are **linearly incommensurable**. Even though both individually represent the same algorithm (probe_acc=1.0), the coordinate systems differ.

### 3. Projected probe acc = 0.94 — misleading
W doesn't converge to B's space (MSE=0.215). The output still mostly resembles A's structure. The probe reads surviving A information, not evidence of geometry transfer.

### 4. Steering delta = 0 — ceiling effect
Both models already achieve 1.0 accuracy on the task. The steering vector cannot improve anything. A harder task is needed to measure steering effects.

## Root cause
The residual stream geometry changes fundamentally with model scale — even with identical vocabulary, tokenizer, and training objective. Linear maps between different-dimensional representations cannot bridge this gap.

## Files
```
artifacts/
├── small/best_model.pth
├── big/best_model.pth
├── activations/
│   ├── small_acts_test.npy      [2, 2823, 128]
│   └── big_acts_test.npy        [6, 2823, 512]
├── probes/
│   └── probe_comparison.png
├── projection/
│   ├── W.pth
│   ├── training_curve.png
│   ├── cos_sim_test.npy
│   └── projected_probe_acc.txt
└── steering/
    ├── steering_vec.npy
    ├── results_per_alpha.csv
    └── summary.md
```
