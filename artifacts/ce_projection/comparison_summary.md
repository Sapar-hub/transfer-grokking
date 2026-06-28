# CE Projection Experiment Summary

## Setup

| Parameter | Value |
|-----------|-------|
| Layer | 10 |
| D_small → D_phi2 | 128 → 2560 |
| Train / Test | 6586 / 2823 |
| W_CE loss | CE through frozen lm_head (no layernorm) |
| W_MSE loss | MSE + 0.01 × ortho |
| Epochs | 5000 |
| Optimizer | AdamW lr=0.001 |

## Logit lens & Probe

| Metric | W_MSE | W_CE | Delta |
|--------|-------|------|-------|
| Cos sim (test) | 0.1441 | 0.0000 | -0.1441 |
| Logit lens | 0.0117 | 1.0000 | +0.9883 |
| Probe on W(h) | 1.0000 | 1.0000 | +0.0000 |

## Alpha sweep (text accuracy at L10)

| Alpha | W_MSE | W_CE | Delta |
|-------|-------|------|-------|
| 0.0 | 0.2350 | 0.2350 | +0.0000 |
| 0.3 | 0.2900 | 0.2650 | -0.0250 |
| 0.5 | 0.3050 | 0.2600 | -0.0450 |
| 0.7 | 0.2800 | 0.2300 | -0.0500 |
| 1.0 | 0.0150 | 0.0400 | +0.0250 |

Baseline (no patch): 0.2350

**Logit lens verdict**: W_CE > 0.5 — MSE was the primary barrier.

**Alpha sweep verdict**: CE aligns with lm_head but text accuracy remains limited — context/geometry conflict persists.
