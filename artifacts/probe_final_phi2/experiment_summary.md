# Probe Final Phi-2: Linear on L31 (single template)

## Parameters

- Pairs collected: 2000
- Phi-2 text baseline: 0.2500
- Natural adapter best (L30, mixed templates): 0.0446

## Results

| Cond | Input | Adapter | Test Acc |
|------|-------|---------|----------|
| A: small_acts→97 | — | Linear→97 | 0.8600 |
| B: W(h_A)→97 | — | Linear→97 | 1.0000 |
| C: phi2_L31→97 | — | Linear→97 | 0.4133 |

## Verdict
Phi-2 partially encodes the answer at L31 (0.41), above lm_head's 0.25 but far from 1.0. Information is not perfectly linearly separable.

