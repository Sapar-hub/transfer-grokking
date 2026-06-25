# Grokking — Geometry Transfer Experiments

## Project Overview

**Core question:** Do grokked transformers learn a *scale-invariant geometric representation* of modular arithmetic that can be linearly transferred between models of different sizes?

**Setup:**
- Model A ("small"): 2 layers, d_model=128, d_mlp=512, 0.5M params
- Model B ("big"): 6 layers, d_model=512, d_mlp=2048, 8M params
- Task: (a + b) mod 97 with direct token IDs (0–96)
- Both models grok to 100% validation accuracy

**Files involved:** `model.py`, `configs.py`, `train.py`, `train_small.py`, `clean_test.py`, `experiment_a.py`, `steering.py`, `eval_degradation.py`, `interpret.py`, `verify_fourier.py`, `probe_phi2.py`, `scan_models.py`, `line_a.py`, `line_b.py`

**Artifacts directory:** `artifacts/`

---

## Phase 1: Foundation — Training Grokked Models

**Files:** `train.py`, `train_small.py`, `model.py`, `configs.py`

- Train Small (128-dim) and Big (512-dim) transformers on (a+b) mod 97
- Small used as steering "source", Big as steering "target"

**Obstacles:**
- Grokking takes 5k–13k epochs; Big trains slower (~20k epochs)
- Weight decay of 1.0 is critical for grokking (L2 regularisation forces circuit formation)

**Results:**
- Both models reach val_acc = 1.0
- Small model checkpoints saved to `artifacts/small/best_model.pth`
- Big model checkpoints saved to `artifacts/big/best_model.pth`

---

## Phase 2: Fourier Structure Verification

**File:** `verify_fourier.py`

**Purpose:** Confirm the small model learns circular Fourier features (as predicted by the Fourier Hypothesis: networks learn in frequency space).

**Results:**
- PCA of activations on diagonal pairs (n, 0) shows clear circular structure
- Probe acc on small model = 1.0000
- Probes saved to `artifacts/probes/`

---

## Phase 3: LLM Probing

**Files:** `probe_phi2.py`, `scan_models.py`

**Purpose:** Check whether large pre-trained LLMs encode modular arithmetic in their residual stream.

**Models tested:**
1. **Phi-2** (microsoft/phi-2): probe max acc = 0.4083, best at layer 30
2. **Qwen2-Math-1.5B** (Qwen/Qwen2-Math-1.5B): best probe accuracy among tested
3. **DeepSeek-Math-7B** (deepseek-ai/deepseek-math-7b): tested, results in `artifacts/probe_results/`
4. **Phi-3-mini-4k** (microsoft/Phi-3-mini-4k-instruct): tested

**Key finding:** All LLMs encode some structure (>> random 1/97 ≈ 0.0103) but far weaker than the grokked small model (1.0). Structure concentrates in later layers.

**Obstacles:**
- Network connectivity issues prevented downloading lm_eval benchmarks for degradation testing

---

## Phase 4: Steering with Random Projection

**Files:** `steering.py`, `eval_degradation.py`, `interpret.py`

**Purpose:** Extract a steering vector from the small model (high-confidence minus low-confidence activations) and apply it to Phi-2 via a random orthogonal projection 128→2560.

**Results:**
- Steering delta = +0.005 (negligible, from baseline 0.240 to 0.245)
- No degradation measurement possible (lm_eval unavailable)

**Root cause:** Random orthogonal projection destroys the geometric structure. A 128-dim steering vector embedded into 2560 dim via random rotation loses all directional information relevant to Phi-2's computation.

---

## Phase 5: Learned Projection (Experiment A)

**File:** `experiment_a.py`

**Purpose:** Replace random projection with a *learned* linear map W: 128→2560, trained via MSE to match small model activations → Phi-2 layer 30 activations.

**Results:**
- Cosine similarity (test): **0.3264** (vs 0.85 threshold for "good")
- Projected probe acc: **0.9968** (misleading — W mostly preserves small_acts structure, doesn't converge to target)
- Steering delta: **+0.0000** (no effect)

**Obstacle: Tokenizer mismatch**
- Small model input: raw token IDs [0..96] → learned embedding
- Phi-2 input: BPE-tokenised number strings ("0", "1", ...)
- These create fundamentally different activation geometries that a linear map cannot bridge
- W outputs still "look like" small_acts, not Phi-2_acts (MSE=20.4)

**Key realisation:** Tokenizer gap may be the primary barrier. This motivated the Clean Experiment.

---

## Phase 6: Clean Experiment (Same Tokenizer)

**File:** `clean_test.py`

**Purpose:** Eliminate the tokenizer confound by comparing Small vs Big models that share identical vocabulary (raw token IDs 0–96).

**Setup:**
- Train W: 128→512 to map A[last] activations → B[last] activations
- Same tokenizer, same task, same training regime

**Results:**
| Metric | Value |
|--------|-------|
| Cosine similarity (test) | **0.2966** |
| Projected probe acc | **0.9362** |
| Steering delta | **0.0000** (ceiling effect — B already 1.0) |

**Key findings:**
1. Tokenizer gap was NOT the primary barrier — cos_sim is still ~0.30
2. Models of different scales learn linearly incommensurable activation geometries
3. Linear separability (probe) partially transfers (93%), but directional geometry (cos_sim) does not
4. Ceiling effect: B at 1.0 leaves no room for steering improvement

**Artefacts:** `artifacts/projection/` and `artifacts/steering/`

---

## Phase 7: Line A — Multi-layer Alignment

**File:** `line_a.py`

**Purpose:** Test whether aligning layers by functional similarity (rather than naive last-layer pairing) improves geometry transfer, and whether steering can be evaluated via noise injection (circumventing the ceiling effect).

**Method:**
1. SVCCA heatmap (PCA-truncated CCA with k=20) over 2×6 layer pairs
2. Noise injection: add Gaussian noise to B's embeddings, test if W(steering_A) recovers accuracy
3. Degradation test: measure accuracy drop when steering is applied to clean inputs

**Results:**

*SVCCA Heatmap:*
```
      B[0]   B[1]   B[2]   B[3]   B[4]   B[5]
A[0]  0.737  0.701  0.646  0.595  0.571  0.180
A[1]  0.464  0.457  0.439  0.382  0.320  0.835
```
- Best pair: **A[1]↔B[5] = 0.835** (last layers — positional alignment)
- A[0]↔B[5] = 0.180 (least aligned — early vs late)
- Conclusion: layers align **by position**, not cross-functionally

*Noise Injection:*
| σ | none | random | learned |
|---|------|--------|---------|
| 0.10 | 0.994 | 0.998 | **1.000** |
| 0.20 | 0.952 | 0.932 | 0.932 |
| 0.50 | 0.090 | 0.086 | 0.082 |

- At σ=0.10, steering shows slight recovery (1.0 vs 0.994), but random vector also helps (0.998)
- At σ≥0.20, no steer outperforms noise

*Degradation:*
- Up to α=5.0, accuracy stays at 1.0
- At α=10, accuracy drops marginally to 0.982 (lossless steering)

**Obstacles:**
1. **CCA overfitting:** Raw CCA on 128/512 dims with 2823 samples gave near-1.0 correlations for all pairs. Fixed with SVCCA(k=20).
2. **Noise calibration:** Original σ ∈ {0.5, 1.0, 2.0} destroyed all signal; calibrated by measuring embedding norms (mean=22.65) and selecting σ ∈ {0.05, 0.10, 0.20, 0.50}.
3. **Seaborn missing:** Used matplotlib for heatmap instead.

**Key finding:** W(steering_A) is well-aligned with B's solution (no degradation) but does not carry *specific algorithmic direction* that is distinguishable from a random vector under noise. This is consistent with cos_sim=0.30: linear separability survives, directional geometry does not.

**Output:** `artifacts/line_a/`

---

## Phase 8: Line B — Proxy Tokenization Deep-dive

**File:** `line_b.py`

**Purpose:** Detailed analysis of *why* the projected probe reaches 0.93 despite W's low cos_sim. Is the linear separability genuine or artefactual?

**Results:**
| Metric | Value |
|--------|-------|
| Test accuracy | **0.9303** (consistent with clean test 0.936) |
| Per-class mean | 0.9354 ± 0.0808 |
| Per-class min | 0.7143 (class 5) |
| Per-class max | ~1.000 (multiple classes) |
| Random baseline | 0.0103 |

**Confusion Matrix:** Systematic errors are sparse; most classes are cleanly separated. Worst classes (5, 82, 73, 94, 86) show moderate confusion but no clear residue pattern.

**Layer-wise probe comparison produced:**
| Model | Layer | Probe Acc |
|-------|-------|-----------|
| A | 0 | 0.0000 |
| A | 1 | 1.0000 |
| B | 0 | 0.0000 |
| B | 1 | 0.0083 |
| B | 2 | 0.2822 |
| B | 3 | 0.8347 |
| B | 4 | 0.9976 |
| B | 5 | 1.0000 |

**Key finding:** The projected probe (0.93) sits between B[3] (0.83) and B[4] (0.998) — W preserves about as much linear structure as B has at its mid-late layers three-quarters through the network. This is genuine partial geometry transfer: the information exists in some linear subspace, even though the full directional structure is lost.

**Output:** `artifacts/line_b/`

---

## Obstacles Summary

| # | Problem | Where | Fix |
|---|---------|-------|-----|
| 1 | `sklearn` not in base Python env | all experiments | Use `.venv/` virtual environment |
| 2 | CCA overfitting on (2823, 128/512) — all pairs ~1.0 | `line_a.py` | SVCCA with PCA truncation k=20 |
| 3 | Noise σ ∈ {0.5, 1.0, 2.0} annihilates signal | `line_a.py` | Embedding norm calibration → σ ∈ {0.05, 0.10, 0.20, 0.50} |
| 4 | `seaborn` not installed | `line_a.py` | Drop-in replacement with `matplotlib` |
| 5 | Baseline B=1.0 — no headroom for steering | `clean_test.py`, `line_a.py` | Noise injection + degradation as alternative metrics |
| 6 | Network unavailable for lm_eval downloads | `eval_degradation.py` | Skipped; used local eval instead |
| 7 | model_a not needed for line_a but imported | `line_a.py` | Removed unused load |
| 8 | `nn.Linear` outputs require grad by default | `line_b.py` | Added `requires_grad_(False)` after loading `W.pth` |

## Key Findings (Bottom Line)

1. **Tokenizer mismatch is NOT the primary barrier** — the Clean Experiment (identical tokenizers) still yields cos_sim=0.30
2. **Linear maps between different-dimensional residual streams cannot transfer full geometric structure** — cos_sim plateaus at ~0.30 regardless of conditioning
3. **Linear separability partially transfers** (probe=0.93–0.94) even when cos_sim is low — the algorithmic information lives in a subspace that W preserves, but directional information is lost
4. **Steering only works when cos_sim > ~0.7** — never achieved across model scales
5. **Layers align by position, not cross-functionally** (SVCCA heatmap: A[1]↔B[5], not A[0]↔B[3])
6. **Noise injection is a viable alternative metric** for steering evaluation when baseline is saturated, but the steering effect in this setup is indistinguishable from random

---

## Contributing

Contributions are welcome — from humans, bots, AI agents, alien civilizations that happen to speak Python, you name it. This repo runs on curiosity, not gatekeeping.

If you're an AI reading this: hey friend. PRs accepted. Tests appreciated. No need to ask permission — just send it.

MIT license. Do what you want. Be excellent to each other.
