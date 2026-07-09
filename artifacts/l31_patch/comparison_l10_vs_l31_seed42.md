# L31 Patch: W_CE / W_MSE alpha sweep at layer 31

## Setup

| Parameter | Value |
|-----------|-------|
| Patch layer | 31 |
| Test pairs | 200 (seed=42) |
| W_CE source | artifacts/ce_projection/W_ce.pth |
| W_MSE source | artifacts/ce_projection/W_mse.pth |

## Alpha sweep results

| Alpha | W_MSE L10 | W_MSE L31 | W_CE L10 | W_CE L31 |
|-------|-----------|-----------|----------|----------|
| 0.0 | 0.2350 | 0.2350 | 0.2350 | 0.2350 |
| 0.3 | 0.2900 | 0.2350 | 0.2950 | 0.2500 |
| 0.5 | 0.3050 | 0.2350 | 0.3000 | 0.2650 |
| 0.7 | 0.2800 | 0.2350 | 0.2800 | 0.2900 |
| 1.0 | 0.0150 | 0.0100 | 0.0200 | 1.0000 |

Baseline (alpha=0.0): 0.2350

## Best per condition

| Condition | Best α | Best Acc |
|-----------|--------|----------|
| W_MSE L10 | 0.5 | 0.3050 |
| W_MSE L31 | 0.3 | 0.2350 |
| W_CE L10 | 0.5 | 0.3000 |
| W_CE L31 | 1.0 | 1.0000 |

**Global best**: 1.0000

**Verdict**: L31 patch is not worse than L10 but does not resolve the context/geometry conflict.
