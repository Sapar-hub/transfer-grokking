# CE Projection Experiment Summary (seed=46, op=multiply)

## Setup

| Parameter | Value |
|-----------|-------|
| Layer | 10 |
| Operation | multiply mod 97 |
| Seed | 46 |
| D_small → D_phi2 | 128 → 2560 |
| Train / Test | 6586 / 2823 |
| W_CE loss | CE through frozen lm_head (no layernorm) |
| W_MSE loss | MSE + 0.01 × ortho |
| Epochs | 5000 |
| Optimizer | AdamW lr=0.001 |

## Logit lens & Probe

| Metric | W_MSE | W_CE | Delta |
|--------|-------|------|-------|
| Cos sim (test) | 0.0341 | 0.0005 | -0.0336 |
| Logit lens | 0.0262 | 1.0000 | +0.9738 |
| Probe on W(h) | 1.0000 | 1.0000 | +0.0000 |

## Alpha sweep (text accuracy at L10)

| Alpha | W_MSE | W_CE | Delta |
|-------|-------|------|-------|
| 0.0 | 0.0100 | 0.0100 | +0.0000 |
| 0.3 | 0.0100 | 0.0100 | +0.0000 |
| 0.5 | 0.0100 | 0.0050 | -0.0050 |
| 0.7 | 0.0050 | 0.0000 | -0.0050 |
| 1.0 | 0.0100 | 0.0200 | +0.0100 |

Baseline (no patch): 0.0100

---

_Seed 46, operation: multiply_
