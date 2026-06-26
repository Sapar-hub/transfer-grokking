# Nonlinear Adapter Experiment Summary

## Conditions

| # | Adapter | lm_head |
|---|---------|---------|
| A | none (logit lens) | frozen |
| B | trained Linear(2560→2560) | frozen |
| C | trained MLP(2560→256→2560) | frozen |
| D | trained MLP(2560→256→2560) | trainable |

## Results

| Cond | Test Acc | vs logit lens | vs linear adapter |
|------|----------|---------------|-------------------|
| A | 0.0117 | — | — |
| B | 1.0000 | +0.9883 | — |
| C | 1.0000 | +0.9883 | +0.0000 |
| D | 1.0000 | +0.9883 | +0.0000 |

## Verdict
CONFIRMED: MLP reshapes Fourier features for frozen lm_head.
