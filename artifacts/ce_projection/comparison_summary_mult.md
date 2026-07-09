# CE Projection Experiment Summary (seed=42, op=mult)

## Setup

| Parameter | Value |
|-----------|-------|
| Layer | 10 |
| Operation | mult mod 97 |
| Seed | 42 |
| D_small → D_phi2 | 128 → 2560 |
| Train / Test | 6586 / 2823 |
| W_CE loss | CE through frozen lm_head + final_layernorm |
| W_MSE loss | MSE + 0.01 × ortho |
| Epochs | 5000 |
| Optimizer | AdamW lr=0.001 |

## Logit lens & Probe

| Metric | W_MSE | W_CE | Delta |
|--------|-------|------|-------|
| Cos sim (test) | 0.0343 | 0.0010 | -0.0333 |
| Logit lens | 0.0262 | 1.0000 | +0.9738 |
| Probe on W(h) | 1.0000 | 1.0000 | +0.0000 |

## Alpha sweep (text accuracy at L10)

| Alpha | W_MSE | W_CE | Delta |
|-------|-------|------|-------|
| 0.0 | 0.0300 | 0.0300 | +0.0000 |
| 0.3 | 0.0300 | 0.0300 | +0.0000 |
| 0.5 | 0.0350 | 0.0350 | +0.0000 |
| 0.7 | 0.0300 | 0.0300 | +0.0000 |
| 1.0 | 0.0150 | 0.0050 | -0.0100 |

Baseline (no patch): 0.0300

---

_Seed 42, operation: mult_
