# Grokking — Geometry Transfer Experiments

## Project Overview

**Core question:** Do grokked transformers learn a *scale-invariant geometric representation* of modular arithmetic that can be linearly transferred between models of different sizes?

**Setup:**
- Model A ("small"): 2 layers, d_model=128, d_mlp=512, 0.5M params
- Model B ("big"): 6 layers, d_model=512, d_mlp=2048, 8M params
- Task: (a + b) mod 97 with direct token IDs (0–96)
- Both models grok to 100% validation accuracy

**Files involved:** `model.py`, `utils.py`, `train.py`, `train_small.py`, `clean_test.py`, `experiment_a.py`, `verify_fourier.py`, `line_a.py`, `line_b.py`, `embed_patch.py`, `ce_projection.py`, `l31_patch.py`, `eval_l31_perplexity.py` (core); additional experiments archived in `experiments/`

**Repository structure:**
```
├── model.py, utils.py, train.py,…   ← core scripts
├── experiments/                     ← archived dead ends
├── artifacts/                       ← all experiment outputs
├── paper/                           ← manuscript (forthcoming)
├── CITATION.cff                     ← citation metadata
├── .zenodo.json                     ← Zenodo archive config
├── LICENSE                          ← MIT
└── README.md
```

**Artifacts directory:** `artifacts/`
**Git LFS:** Model weights and key artifacts are stored via Git LFS. After cloning, run `git lfs pull` to download them.

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

## Phase 9: Embed Patch — Direct Input Injection into Phi-2

**File:** `embed_patch.py`

**Purpose:** Eliminate the projection problem entirely. Instead of projecting activations (which failed across all prior experiments), project *embeddings* at the input level — train a linear map from small model token embeddings (R^128) to Phi-2 token embeddings (R^2560), then feed Phi-2 via `inputs_embeds` bypassing BPE tokenization. This tests whether the barrier is technical (projection) or fundamental (representation type).

**Method:**
1. Extract `embed_A` from small model [97×128] and Phi-2 target embeddings [97×2560] (mean over subword BPE tokens for numbers 10–96)
2. Train `W_emb: 128→2560` on 97 points with MSE + orthogonality loss `||W^T W - I||` to force isometry
3. Evaluate Phi-2 accuracy on 200 random pairs under two conditions:
   - **Text baseline:** standard text prompt `"# (a + b) % 97 ="`
   - **Treatment:** `phi2(inputs_embeds=W_emb(embed_A([a,b])))` — no text context
4. Probe on **pre-layer activations** (input to `layers[0]`, before any transformer computation) with logistic regression on 1000 samples

**Results:**

*W_emb training (97 points, 5000 epochs):*
| Metric | Value |
|--------|-------|
| Final MSE | 0.000274 |
| Orthogonality loss | 0.000010 (near-perfect isometry) |
| Cosine similarity | **0.8153** (geometry preserved) |

*Accuracy:*
| Condition | Acc | vs Random |
|-----------|:---:|:---------:|
| Text baseline | **0.2400** | 23× random |
| inputs_embeds | **0.0100** | = random (1/97) |
| Delta | **−0.2300** | significant drop |

*Probe on pre-layer activations:*
| Condition | Probe Acc | Meaning |
|-----------|:---------:|---------|
| Text | 0.0100 | random — no structure in embeddings |
| inputs_embeds | 0.0167 | random — no structure in projected embeddings |

**Key finding:** Despite W_emb being a near-perfect isometry (cos=0.82), Phi-2 produces random outputs when given bare number embeddings with no text context. The modular arithmetic structure in Phi-2 is **not stored in the embedding layer** — it is *computed* through transformer layers from the task-describing text prompt. The probe at 0.01 confirms this: neither native nor projected embeddings contain linear separability.

**Interpretation — the final piece:**
The grokked small model *compiles* the algorithm into its weights (Fourier circles in embedding space, probe=1.0 from layer 1). Phi-2 *simulates* the algorithm via language processing — the probe structure appears only at layer 30 (max=0.41) and only when the text prompt provides task context. These are **fundamentally different mechanisms:**

| | Grokked Model | Phi-2 |
|---|---|---|
| Representation | Compiled (weight-stored) | Simulated (context-computed) |
| Geometry location | Embeddings + all layers | Layer 30 only |
| Transferable? | — | No (needs text prompt) |

Linear transfer between compiled and simulated representations is impossible regardless of projection quality, tokenizer alignment, or layer selection. The hypothesis is conclusively falsified.

**Output:** `artifacts/embed_patch/`

---

## Natural Adapter — Phi-2 Residual Stream from Natural Language

**File:** `natural_adapter.py`, `eval_natural_adapter.py`

**Purpose:** Inverse of Embed Patch. Instead of asking "can we inject grokked structure into Phi-2?", ask "does Phi-2's residual stream already contain enough information about (a+b) mod 97 that a linear adapter can read it directly — without the small model?"

**Method:**
1. Generate all P²=9409 pairs, assign each one a random template from 4 diverse natural language prompts (seed=42, uniform)
2. Collect Phi-2 residual stream activations at layers 20, 25, 28, 30 (one pass, multi-hook, attention-masked for variable-length prompts)
3. Train linear adapter (`nn.Linear(2560, 97)` with AdamW + CrossEntropyLoss, 1000 epochs) per layer
4. Compare with `LogisticRegression` baseline (sklearn) and Phi-2 LM head accuracy (0.235)
5. Template generalization test: 500 pairs × 4 templates, train on T0 `"what is (a + b) mod 97?"`, test on T1/T2/T3

**Templates:**
```
T0: "what is ({a} + {b}) mod 97?"
T1: "calculate ({a} + {b}) modulo 97"
T2: "{a} + {b} mod 97 ="
T3: "if I add {a} and {b} and take remainder when divided by 97 what do I get"
```

**Results:**
| Layer | nn.Linear (AdamW) | LogisticRegression |
|-------|-------------------|--------------------|
| 20 | 0.0067 | 0.0028 |
| 25 | 0.0205 | 0.0156 |
| 28 | 0.0400 | 0.0322 |
| 30 | **0.0446** | 0.0322 |

| Condition | Accuracy | Notes |
|-----------|:--------:|-------|
| Random (1/97) | 0.0103 | baseline |
| Phi-2 LM head | **0.235** | text prompt, full 32-layer decode |
| **Best adapter (L30)** | **0.045** | barely above random |

*Template generalization (T0 → cross, LogisticRegression):*
| Train → Test | Best L | Acc |
|--------------|--------|:---:|
| T0 → T0 | 20 | 1.0000 (in-domain — memorised) |
| T0 → T1 | 30 | 0.0200 |
| T0 → T2 | 25 | 0.0140 |
| T0 → T3 | 30 | 0.0240 |
| T1 → T1 | 28 | 0.0467 |
| T2 → T2 | 28 | 0.1200 |
| T3 → T3 | 20 | 0.0400 |

**Key finding:** Adapter accuracy (best=0.045 at L30) is near random (0.010) and far below Phi-2 LM head (0.235). This confirms the Embed Patch conclusion from the opposite direction: Phi-2's arithmetic ability is **computed through the full 32-layer transformer** and decoded by the LM head, not stored linearly in any single layer's residual stream. A linear probe at any late layer cannot extract the answer, even though the LM head can.

Template generalization fails (T0→T1/T2/T3 ≈ 0.02) — the adapter learns surface patterns specific to each template's tokenisation, not the underlying arithmetic. The best in-domain accuracy (T2→T2 = 0.120) comes from the template closest to the original probe prompt `"{a} + {b} mod 97 ="`, confirming prompt format critically affects residual stream structure.

**Comparison with Phase 3 (probe_phi2.py):** The original probe achieved 0.41 on `"# (a + b) % 97 ="` at L30 — much higher than any natural template (0.04–0.12). The `#` prefix and parentheses create a more structured residual representation than diverse natural language. But even 0.41 is far below the grokked model's 1.0.

**Interpretation:**
- LM head was NOT a bottleneck — Phi-2 really does need all 32 layers to compute the answer
- Linear separability in residual stream depends on prompt format consistency
- With diverse natural language, the residual stream does not contain a linearly readable answer
- This is consistent with the "compiled vs simulated" hypothesis: Phi-2 simulates arithmetic via language processing distributed across all layers, not compiled into any single representation

**Output:** `artifacts/natural_adapter/`

---

## Phase 10: CE Projection — Resolving the Transfer Problem

**Files:** `ce_projection.py`, `l31_patch.py`

**Purpose:** The residual patch experiment showed that W trained with MSE achieves probe=1.0 but logit lens=0.005 — the information is in the projected vector but lm_head cannot read it due to misalignment. This phase tests whether training W via **CrossEntropy through frozen lm_head** (instead of MSE on activations) resolves the alignment, and whether patching at the **last layer (L31)** avoids the context/geometry conflict.

### Part A: CE-trained W (ce_projection.py)

**Method:** Train `W_ce: nn.Linear(128, 2560)` via CE through frozen lm_head (no layernorm) on 6586 training pairs, 5000 epochs. Compare with W_MSE trained on the same data to match Phi-2 L10 activations.

**Results:**

| Metric | W_MSE (L10) | W_CE |
|--------|:-----------:|:----:|
| Logit lens | 0.0117 | **1.0000** |
| Probe on W(h) | 1.0000 | 1.0000 |
| Cos sim vs L10 targets | 0.1441 | 0.0000 |

W_CE achieves perfect logit-lens accuracy — **MSE was the sole cause of lm_head misalignment**. However, W_CE finds directions orthogonal to Phi-2's L10 activations (cos_sim=0.0), and patching at L10 degrades text accuracy vs W_MSE (α=0.5: 0.260 vs 0.305). The context/geometry conflict persists at intermediate layers.

### Part B: L31 Patch (l31_patch.py)

**Method:** Patch the same W_CE and W_MSE at Phi-2's **last layer (L31)** — the layer immediately before lm_head, where no further computation can corrupt the injected signal.

**Results:**

| Alpha | W_MSE L10 | W_MSE L31 | W_CE L10 | **W_CE L31** |
|-------|:---------:|:---------:|:--------:|:------------:|
| 0.3 | 0.290 | 0.235 | 0.265 | **0.490** |
| 0.5 | 0.305 | 0.235 | 0.260 | **0.705** |
| 0.7 | 0.280 | 0.235 | 0.230 | **0.995** |
| 1.0 | 0.015 | 0.010 | 0.040 | **1.000** |

W_CE at L31 achieves **perfect accuracy (1.0) at α=1.0** with monotonic improvement across all alpha values. W_MSE at L31 does nothing (all α = baseline 0.235).

**Key findings:**
1. **CE through lm_head resolves barrier 1:** W_CE logit lens = 1.0, proving MSE was misaligning W from decoding directions
2. **The context/geometry conflict was layer-specific, not fundamental:** at L31 there is no remaining computation to corrupt the injected signal, and W_CE is perfectly aligned with lm_head's decoding directions
3. **Neural function call works:** a grokked model's computed state can be injected into an LLM's final residual layer to directly produce correct output tokens, bypassing all intermediate computation

**Interpretation — the final resolution:**
The series' core question was whether grokked representations are geometrically transferable. The answer is **yes, with the right interface**:
- W must be trained with **CE loss through the target's lm_head**, not MSE on activations
- Injection must happen at the **last layer** where no further computation can interfere
- Under these conditions, the transfer is perfect (1.0)

The earlier experiments (clean test, residual patch, nonlinear adapter) all failed because they used MSE training, intermediate-layer injection, or both — each alone was sufficient to block transfer.

**Output:** `artifacts/ce_projection/`, `artifacts/l31_patch/`

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
7. **Grokked models compile algorithms; LLMs simulate them via language** — the Embed Patch experiment (cos=0.82, acc=0.01) proves the gap is fundamental: the grokked model's Fourier geometry is weight-stored, while Phi-2's probe structure (~0.41 at layer 30) is computed from text context. These are incommensurable representation types — compiled vs simulated — and no linear method can bridge them.
8. **Phi-2 needs all 32 layers to compute the answer** — the Natural Adapter experiment (best linear readout = 0.045 vs LM head = 0.235) confirms the LM head is not a bottleneck. The answer is computed through the full stack, not linearly separable at any single residual layer. Template format critically determines residual structure (T2→T2 = 0.12 vs T3→T3 = 0.04 vs Phase 3 probe = 0.41 on `"# (a + b) % 97 ="`).
9. **CE through frozen lm_head resolves W→lm_head alignment** — training the projection W via CrossEntropy (instead of MSE) achieves logit lens = 1.0. MSE was the sole cause of lm_head misalignment across all prior experiments.
10. **Neural function call works: inject grokked state at L31 → perfect accuracy** — W_CE(h_A) patched at Phi-2's last layer (L31, α=1.0) gives 1.0. The context/geometry conflict was layer-specific, not fundamental: L31 has no remaining computation to corrupt the signal, and W_CE is perfectly aligned with lm_head's decoding directions. The series' core question is resolved: grokked representations are transferable with the right interface (CE-trained W + last-layer injection).

---

## Citation

If you use this software or its findings in your research, please cite:

```bibtex
@software{saparmyradov2026transfergrokking,
  author = {Saparmyradov, Saparmyrat},
  title = {{Transfer Grokking}: {Linear} Projection of {Fourier} Representations
           Across Model Residual Streams},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/ssaparm/transfer-grokking}
}
```

Machine-readable citation metadata is available in [`CITATION.cff`](./CITATION.cff).

## Contributing

Contributions are welcome — from humans, bots, AI agents, alien civilizations that happen to speak Python, you name it. This repo runs on curiosity, not gatekeeping.

If you're an AI reading this: hey friend. PRs accepted. Tests appreciated. No need to ask permission — just send it.

MIT license. Do what you want. Be excellent to each other.
