# Residual Patch Experiment Summary

## Per-layer W training

| Layer | Cos Sim | Probe on W(h_A) |
|-------|---------|-----------------|
| 10 | 0.1441 | 1.0000 |
| 15 | 0.1641 | 1.0000 |
| 20 | 0.1836 | 1.0000 |
| 25 | 0.2122 | 1.0000 |
| 30 | 0.2325 | 1.0000 |

**By probe**: L*=10 (probe=1.0000)
**By cos_sim**: L*=30 (cos_sim=0.2325)
**Selected for alpha sweep**: L*=10

## Alpha sweep (text accuracy)

| Alpha | Accuracy |
|-------|----------|
| 0.0 | 0.2350 |
| 0.3 | 0.2900 |
| 0.5 | 0.3050 |
| 0.7 | 0.2800 |
| 1.0 | 0.0150 |

Baseline (alpha=0.0): 0.2350
Best (alpha=0.5): 0.3050
Delta: +0.0700

## Logit lens (direct decode from patched L)

| Alpha | Logit Lens Acc |
|-------|----------------|
| 0.0 | 0.0050 |
| 0.3 | 0.0050 |
| 0.5 | 0.0050 |
| 0.7 | 0.0050 |
| 1.0 | 0.0100 |

## Probe on L*+1

| Condition | Probe Acc |
|-----------|-----------|
| original | 0.0035 |
| patched | 1.0000 |

## Verdict
PATCH WORKS: context + correct state activates algorithm
Probe delta: +0.9965
