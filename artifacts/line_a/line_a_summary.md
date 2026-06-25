# Line A: Multi-layer Alignment Summary

## SVCCA Heatmap (k=20)

Existing W was trained on A[1] to B[5] (last layers).

| A\B | B[0] | B[1] | B[2] | B[3] | B[4] | B[5] |
|---|---|---|---|---|---|---|
| A[0] | 0.7340 | 0.6976 | 0.6388 | 0.5964 | 0.5729 | 0.1797 |
| A[1] | 0.4598 | 0.4429 | 0.4399 | 0.3867 | 0.3210 | 0.8347 |

Best aligned: **A[1] <-> B[5]** (CCA = 0.8347)

## Noise Injection

| sigma | steer_type | accuracy |
|---|------------|----------|
| 0.0 | none | 1.0000 |
| 0.0 | random | 1.0000 |
| 0.0 | learned | 1.0000 |
| 0.05 | none | 1.0000 |
| 0.05 | random | 1.0000 |
| 0.05 | learned | 1.0000 |
| 0.1 | none | 1.0000 |
| 0.1 | random | 1.0000 |
| 0.1 | learned | 0.9980 |
| 0.2 | none | 0.9280 |
| 0.2 | random | 0.9060 |
| 0.2 | learned | 0.9260 |
| 0.5 | none | 0.0880 |
| 0.5 | random | 0.0840 |
| 0.5 | learned | 0.0840 |

## Degradation (sigma=0)

| alpha | accuracy | delta |
|---|----------|---|
| 0.0 | 1.0000 | +0.0000 |
| 0.1 | 1.0000 | +0.0000 |
| 0.5 | 1.0000 | +0.0000 |
| 1.0 | 1.0000 | +0.0000 |
| 2.0 | 1.0000 | +0.0000 |
| 5.0 | 1.0000 | +0.0000 |
| 10.0 | 0.9820 | -0.0180 |

### Interpretation

- sigma=0.05: baseline=1.0000, random=1.0000, learned=1.0000, recovery=+0.0000, specificity=+0.0000
- sigma=0.1: baseline=1.0000, random=1.0000, learned=0.9980, recovery=-0.0020, specificity=-0.0020
- sigma=0.2: baseline=0.9280, random=0.9060, learned=0.9260, recovery=-0.0020, specificity=+0.0200
- sigma=0.5: baseline=0.0880, random=0.0840, learned=0.0840, recovery=-0.0040, specificity=+0.0000

- Max degradation from steering: 0.0180
  -> Steering is nearly lossless (aligned with model's solution).
