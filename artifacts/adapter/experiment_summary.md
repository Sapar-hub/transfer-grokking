# Adapter Experiment Summary

| Alpha | nn.Linear Acc | LogisticRegression Acc |
|-------|---------------|------------------------|
| 0.0 | 0.0233 | 0.0122 |
| 0.5 | 0.1589 | 0.1922 |
| 1.0 | 0.9989 | 0.9989 |

**Patch layer:** L=10
**Collect layer:** L=11
**Train size:** 2100
**Test size:** 900

**Gap (alpha=1.0 - alpha=0.5):** +0.8400
Interpretation: gap = price of context interference on geometry

**Control (alpha=0.0, no patch):** 0.0233 (expected ~0.0035)
WARNING: unpatched adapter > random baseline — possible leak.
