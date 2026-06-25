# Experiment Summary: Grokking → Steering → Probing

## Hypothesis
Is there a compatible geometry between a small grokked transformer and a large LLM (Phi-2) such that algorithmic representations can be transferred via activation patching?

## Results

### Step 1 — Small Transformer Grokking
| Metric | Value |
|--------|-------|
| Architecture | d_model=128, n_layers=2, n_heads=4, d_mlp=512 |
| Task | (a + b) mod 97 |
| Grokking detected | Yes (epoch 13553) |
| Best val_acc | 0.9994 |

### Step 2 — Fourier Structure
| Metric | Value |
|--------|-------|
| PCA on diagonal pairs | Circular structure present |
| Probe acc (LogisticRegression, 97 classes) | **1.0000** |
| Status | ✅ Model has learned the algorithm |

### Step 3-4 — Phi-2 Probing
| Metric | Value |
|--------|-------|
| Baseline mod arithmetic | **0.2400** (24%, random=1.03%) |
| Best probe layer | **Layer 30** (out of 32) |
| Best probe acc | **0.4083** |
| Probe acc progression | 0.13 (early) → 0.41 (late) — clear increasing trend |
| Status | ✅ Structure exists in Phi-2, but weak |

### Step 5-6 — Steering
| Metric | Value |
|--------|-------|
| Steering vector source | blocks.1.hook_resid_post, pos 1 |
| Confidence range | 0.1866 — 0.9995 |
| Projection | Random orthogonal 128 → 2560 |
| Alpha sweep | 0.1, 0.5, 1.0, 2.0, 5.0 |

| Alpha | Mod Accuracy |
|-------|-------------|
| 0.0 (baseline) | 0.2400 |
| 0.1 | 0.2400 |
| 0.5 | 0.2450 |
| 1.0 | 0.2450 |
| 2.0 | 0.2450 |
| 5.0 | 0.2450 |

**Delta task = +0.005** (negligible)

### Step 7 — Degradation
Could not download lm_eval benchmarks (Hellaswag, Lambada, Winogrande, Boolq) due to network connectivity issues on this machine. Degradation assessment skipped.

## Outcome: C ✗

**Criteria checked:**
- `delta_task = 0.005 < 0.03` → Steering did not give significant effect
- Geometries are incompatible OR projection loses structure
- Hypothesis **not confirmed**

## Interpretation

1. **Small model learned the algorithm**: Probe acc = 1.0 confirms the grokked model perfectly represents modular arithmetic in its residual stream.
2. **Phi-2 has weak structure**: Probe acc rising from 0.13 to 0.41 across layers shows some encoding, but much weaker than the small model.
3. **Steering failed**: The random orthogonal projection (128 → 2560) likely destroys the geometric structure. The small model's 128-dim representation space cannot be faithfully embedded into Phi-2's 2560-dim space via a random projection without losing the algorithmic geometry.
4. **Possible fix**: Use a learned projection (e.g., train a linear map from small → Phi-2 activations on paired data) instead of random orthogonal projection.

## Artifacts

- `best_grokking_model.pth` — trained small model
- `train_val_curves.png` — grokking curves
- `pca_fourier_structure.png` — PCA on diagonal pairs
- `probe_accuracy_per_layer_phi2.png` — probe acc by layer
- `steering_results_per_alpha.csv` — accuracy per alpha
- `experiment_summary.md` — this file
