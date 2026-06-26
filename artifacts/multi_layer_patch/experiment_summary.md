# Multi-Layer Patch Experiment Summary

## Design

- **Single-layer**: inject W_L10 at layer 10 only (baseline)
- **Per-layer W**: inject Ws[l] at layer l for l in [10, 15, 20, 25, 30]
- **Same W**: inject W_L10 at all layers [10, 15, 20, 25, 30]
- Injection layers: [10, 15, 20, 25, 30]

## Alpha Sweep (text accuracy)

| Alpha | Single-layer | Per-layer W | Same W |
|-------|-------------|-------------|--------|
| 0.0 | 0.2350 | 0.2350 | 0.2350 |
| 0.3 | 0.2900 | 0.1050 | 0.1050 |
| 0.5 | 0.3050 | 0.0100 | 0.0100 |
| 0.7 | 0.2800 | 0.0150 | 0.0100 |
| 1.0 | 0.0150 | 0.0100 | 0.0300 |

**Best**: Single=0.3050  Per-layer=0.1050  Same-W=0.1050
**Delta vs baseline**: Single=+0.0700  Per-layer=-0.1300  Same-W=-0.1300

## Logit Lens (decode from last patched layer)

| Alpha | Single-layer | Per-layer W | Same W |
|-------|-------------|-------------|--------|
| 0.0 | 0.0050 | 0.0300 | 0.0300 |
| 0.3 | 0.0050 | 0.0300 | 0.0300 |
| 0.5 | 0.0050 | 0.0150 | 0.0150 |
| 0.7 | 0.0050 | 0.0100 | 0.0150 |
| 1.0 | 0.0100 | 0.0150 | 0.0100 |

## Probe on L*+1 (layer after last injection)

| Condition | Single-layer | Per-layer W | Same W |
|-----------|-------------|-------------|--------|
| original | 0.0035 | 0.1267 | 0.1267 |
| patched  | 1.0000 | 0.9933 | 0.9933 |

## Verdict
Multi-layer injection HURTS: too many injections overwhelm Phi-2.
Per-layer W ≈ same-W: the specific projection per layer does not matter.
