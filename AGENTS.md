# Grokking — Geometry Transfer Experiments

## Overview
Research repo: Do grokked transformers learn scale-invariant geometric representations of modular arithmetic?
- **Model A (small):** 2 layers, d_model=128, d_mlp=512
- **Model B (big):** 6 layers, d_model=512, d_mlp=2048
- **Task:** (a + b) mod 97 with direct token IDs (0–96)
- **Artifacts:** `artifacts/`
- **Experiments archive:** `experiments/` (dead ends and exploratory scripts)

## Setup
- No build system — pure Python, one script per experiment
- Virtual environment: `.venv/` — activate before running anything
- CPU-only (`DEVICE = torch.device("cpu")`)
- `matplotlib.use('Agg')` in all scripts (no display)
- No formatter/linter/typechecker config
- **Git LFS:** `artifacts/**/*.pth` and `artifacts/mod_arithmetic_labels.npy` are LFS-tracked. Run `git lfs pull` after cloning.

## Config
Configs live in `model.py`:
- `CFG_SMALL` (name="small"), `CFG_BIG` (name="big")
- `SmallTransformer()` returns `Transformer(CFG_SMALL)` for backward compat

## Entry Points
Every script is standalone (`if __name__ == "__main__": main()`):
| Script | Purpose | Phase |
|--------|---------|-------|
| `train_small.py` | Train small model (30% data, rolling window grokking detection) | aux |
| `train.py` | Train either small or big (70/30 split) | aux |
| `verify_fourier.py` | Confirm circular Fourier features | aux |
| `setup_phi2_cache.py` | Phi-2 model cache verification after manual download | aux |
| `main.py` | Original pipeline orchestrator (steps 1–8) | archive |
| `probe_phi2.py` | Probe Phi-2 layers for mod arithmetic structure | archive |
| `scan_models.py` | Probe Qwen2-Math, DeepSeek-Math, Phi-3 | archive |
| `experiment_a.py` | Learned projection 128→2560 (Small→Phi-2) | archive |
| `clean_test.py` | Clean experiment: Small→Big (same tokenizer) | archive |
| `experiments/line_a.py` | SVCCA heatmap + noise injection steering | archive |
| `experiments/line_b.py` | Projected probe deep-dive | archive |
| `steering.py` | Steering vector + random orthogonal projection | archive |
| `eval_degradation.py` | Downstream benchmark eval (needs lm_eval) | archive |
| `embed_patch.py` | inputs_embeds test: W_emb 128→2560, Phi-2 bypassing BPE | archive |
| `residual_patch.py` | Inject computed state (h_A) into Phi-2 residual stream via W + context prompt | archive |
| `multi_layer_patch.py` | Inject h_A at 5 layers simultaneously (per-layer W + same-W ablation) | archive |
| `nonlinear_adapter.py` | Train Linear/MLP adapter between W(h_A) and frozen lm_head — bridge Fourier→language | archive |
| `probe_final_phi2.py` | Train Linear(2560→97) on Phi-2 final layer (L31) activations from single template | archive |
| `cross_model_l31.py` | Cross-model L31: W_CE + alpha sweep for Qwen2-Math | archive |
| `analyze_baseline_asymmetry.py` | Phi-2 prediction distribution for add vs mult prompts | archive |
| `trivial_baselines.py` | Few-shot (5-shot) + LoRA baselines | archive |
| `ce_projection.py` | Train W: 128→2560 via CE through frozen lm_head; compare MSE vs CE | core (v2.0.0) |
| `l31_patch.py` | Patch W_CE/W_MSE at Phi-2 L31 — alpha sweep vs L10 | core (v2.0.0) |
| `eval_l31_perplexity.py` | Perplexity degradation: L31 patch on WikiText-2 | core (v2.0.0) |
| `cross_model_probe.py` | Cross-model probe: probe on W_CE-injected hidden states | core (v2.0.0) |
| `random_baseline.py` | Control: one-hot + random untrained network, L31 α-sweep | core (v2.0.0) |
| `self_projection.py` | Control: train W_self from Phi-2's own L31 acts | control |
| `margin_analysis.py` | Multi-seed crossing-α + per-class + perplexity fine grid (v3.0.0 correction) | correction (v3.0.0) |

## Run order
`self_projection.py` must run before `margin_analysis.py` — `margin_analysis.py` loads the control models (W_self, W_ce_scaled, W_ce_pca) that `self_projection.py` trains. `margin_analysis.py` supersedes `self_projection.py`'s coarse α sweep with fine-resolution crossing-α and per-class statistics, writing to `artifacts/margin_analysis/`.

## Commands
```bash
source .venv/bin/activate
python train_small.py               # Train small model
python train.py                     # Train both (70/30 split)
python verify_fourier.py            # Verify Fourier structure
python clean_test.py                # Run clean experiment
python experiments/line_a.py                    # SVCCA + noise injection
python experiments/line_b.py                    # Projected probe analysis
python experiment_a.py              # Learned projection Small→Phi-2
python scan_models.py               # Probe multiple LLMs
python embed_patch.py               # Embed patch: inputs_embeds via W_emb
python residual_patch.py            # Residual patch: inject computed state into Phi-2
python multi_layer_patch.py         # Multi-layer injection (5 layers simultaneously)
python nonlinear_adapter.py         # Linear/MLP adapter between W(h_A) and frozen lm_head
python probe_final_phi2.py          # Linear probe on Phi-2 L31 (single template)
python ce_projection.py             # CE-vs-MSE: train W through frozen lm_head
python l31_patch.py                 # Patch W_CE/W_MSE at Phi-2 L31 (alpha sweep vs L10)
python eval_l31_perplexity.py          # Perplexity degradation of L31 patch on WikiText-2
python cross_model_l31.py              # Cross-model validation (Qwen2-Math W_CE + L27 sweep)
  python self_projection.py           # Control: W_self from Phi-2 L31 acts, α-sweep vs W_ce |
python margin_analysis.py           # Fine-grid crossing-α + per-class + perplexity (supersedes self_projection.py's sweep)
python cross_model_probe.py            # Cross-model probe (bypasses BPE tokenizer barrier)
```

## Artifact Cache Map
Scripts skip computation if a cache file exists:
| Created By | File | Used By |
|-----------|------|---------|
| `train_small.py` / `train.py` | `artifacts/small/best_model.pth` | all downstream |
| `train.py` | `artifacts/big/best_model.pth` | `clean_test.py`, `experiments/line_a.py` |
| `clean_test.py` | `artifacts/activations/small_acts_test.npy` | `experiments/line_a.py`, `experiments/line_b.py` |
| `clean_test.py` | `artifacts/activations/big_acts_test.npy` | `experiments/line_a.py` |
| `clean_test.py` | `artifacts/activations/small_acts_train.npy` | `clean_test.py` (train_W) |
| `clean_test.py` | `artifacts/activations/big_acts_train.npy` | `clean_test.py` (train_W) |
| `clean_test.py` | `artifacts/activations/small_labels_train.npy` | `clean_test.py` (train_W) |
| `clean_test.py` | `artifacts/projection/W_seed{N}.pth` | `experiments/line_a.py`, `experiments/line_b.py` |
| `clean_test.py` | `artifacts/steering/steering_vec.npy` | `experiments/line_a.py` |
| `experiment_a.py` | `artifacts/experiment_a/projection_W.pth` | itself (cache) |
| `experiment_a.py` | `artifacts/experiment_a/phi2_layer30_activations.npy` | itself (cache) |
| `embed_patch.py` | `artifacts/embed_patch/W_emb.pth` | itself (cache) |
| `residual_patch.py` | `artifacts/residual_patch/phi2_activations.npz` | itself (cache) |
| `residual_patch.py` | `artifacts/residual_patch/W_layer*.pth` | `multi_layer_patch.py` |
| `multi_layer_patch.py` | `artifacts/multi_layer_patch/experiment_summary.md` | itself (cache) |
| `nonlinear_adapter.py` | `artifacts/nonlinear_adapter/mlp_adapter.pth` | itself (cache) |
| `nonlinear_adapter.py` | `artifacts/nonlinear_adapter/linear_adapter.pth` | itself (cache) |
| `probe_final_phi2.py` | `artifacts/probe_final_phi2/phi2_L31_acts.npy` | itself (cache) |
| `ce_projection.py` | `artifacts/ce_projection/phi2_L10_acts.npy` | itself (cache) |
| `ce_projection.py` | `artifacts/ce_projection/W_mse.pth` | itself (cache) |
| `ce_projection.py` | `artifacts/ce_projection/W_ce.pth` | itself (cache) |
| `l31_patch.py` | `artifacts/l31_patch/alpha_sweep_l31_seed{N}.csv` | itself (cache) |
| `l31_patch.py` | `artifacts/l31_patch/comparison_l10_vs_l31_seed{N}.md` | itself (cache) |
| `eval_l31_perplexity.py` | `artifacts/l31_patch/perplexity_sweep_seed{N}.csv` | itself (cache) |
| `cross_model_l31.py` | `artifacts/cross_model/W_ce_qwen2_math_1.5b.pth` | itself (cache) |
| `cross_model_l31.py` | `artifacts/cross_model/qwen2_math_1.5b_L27_acts.npy` | itself (cache) |
| `cross_model_l31.py` | `artifacts/cross_model/comparison_qwen2_math_1.5b_vs_phi2.md` | itself (cache) |
| `cross_model_probe.py` | `artifacts/cross_model/probe_comparison.md` | itself (cache) |
| `cross_model_probe.py` | `artifacts/cross_model/probe_comparison.png` | itself (cache) |
| `self_projection.py` | `artifacts/self_projection/phi2_L31_acts{_mult}.npy` | itself (cache) |
| `self_projection.py` | `artifacts/self_projection/seeds{_mult}/W_self_seed{N}.pth` | itself (cache) |
| `self_projection.py` | `artifacts/self_projection/seeds{_mult}/alpha_sweep_seed{N}.csv` | itself (cache) |
| `self_projection.py` | `artifacts/self_projection/comparison_summary{_mult}.md` | itself (cache) |
| `self_projection.py` | `artifacts/self_projection/comparison_seed{N}{_mult}.png` | itself (cache) |
| `self_projection.py` | `artifacts/self_projection/per_class_data.pkl` | itself (cache) |
| `self_projection.py` | `artifacts/self_projection/per_class_results.csv` | itself (cache) |
| `self_projection.py` | `artifacts/self_projection/per_class_comparison.png` | itself (cache) |
| `self_projection.py` | `artifacts/self_projection/per_class_dynamics.png` | itself (cache) |
| `margin_analysis.py` | `artifacts/margin_analysis/crossing_alpha_data.pkl` | itself (cache) |
| `margin_analysis.py` | `artifacts/margin_analysis/crossing_alpha_hist.png` | itself (cache) |
| `margin_analysis.py` | `artifacts/margin_analysis/per_class_crossing.csv` | itself (cache) |
| `margin_analysis.py` | `artifacts/margin_analysis/perplexity_fine_grid.csv` | itself (cache) |
| `margin_analysis.py` | `artifacts/margin_analysis/perplexity_fine_grid.png` | itself (cache) |

## Gotchas
- **Weight decay 1.0** is critical for grokking (L2 forces circuit formation)
- **SVCCA with k=20** required — raw CCA on 128/512 dim with N=2823 overfits to ~1.0
- **Noise calibration:** embedding norm ~22.65; use σ ∈ {0.05, 0.10, 0.20, 0.50}, not {0.5, 1.0, 2.0}
- **`seaborn` not installed** — use matplotlib for all plots
- **`nn.Linear` outputs require grad by default** — call `W.requires_grad_(False)` after loading W.pth
- **Ceiling effect:** B baseline = 1.0; use noise injection or degradation as alternative steering metrics
- **Proxy fallback:** `scan_models.py` tries SOCKS5 proxy first, falls back to direct connection
- **BPE splits numbers >9 into subword tokens** — for `phi2_targets` in `embed_patch.py`, take mean over all subword token embeddings per number, not just the first token
- **Qwen2-Math BPE splits all numbers >9**: Qwen2-Math's tokenizer maps 87/97 numbers to subword tokens (only digits 0–9 are single tokens). This makes lm_head-based evaluation of mod arithmetic impossible (10 unique tokens for 97 classes). For cross-model experiments, verify tokenizer first.
- **Prompt operator bug (CRITICAL):** 3 OP-aware scripts (`random_baseline.py`, `l31_patch.py`, `ce_projection.py`) hardcoded `+` in prompts. All now use `OP_SYMBOL[OP]` which is `{"add": "+", "mult": "*"}`. Addition-only scripts (`experiments/*`, `embed_patch.py`) remain unchanged. If extending OP support to a new script, use `OP_SYMBOL = {"add": "+", "mult": "*"}` instead of hardcoding.
- **Old multiplication baseline (0.010) was contaminated:** The wrong prompt (`+` instead of `*`) was used for all α values in the mult sweep. Corrected mult α=0.5=0.660 (was ~0.370). The claim "multiplication requires stronger injection" is weakened. When reporting multiplication results, always state which prompt was used.
- **Layernorm in W_CE training breaks α<1.0 patching (CRITICAL):** `ce_projection.py` trains W_CE through `final_layernorm` by default. This is faithful to Phi-2's architecture for logit-lens evaluation but makes W_CE dependent on layernorm statistics. During L31 patching at α<1.0, the mixed signal has different statistics → alignment degrades monotonically with lower α. If patching at α<1.0 is the goal, train W_CE without layernorm (remove `final_layernorm` kwarg). Logit lens accuracy is already 1.0 in either case.

## Key Findings
1. cos_sim between different-dim residual streams plateaus at ~0.30 regardless of conditioning
2. Linear separability partially transfers (probe = 0.93–0.94) even when cos_sim is low
3. Layers align by position, not cross-functionally (SVCCA: A[1]↔B[5] = 0.835)
4. Steering only distinguishable from random when cos_sim > ~0.7
5. Tokenizer mismatch is NOT the primary barrier (Clean Experiment confirms)
6. Grokked models compile algorithms; LLMs simulate them via language — fundamentally incommensurable (Embed Patch: cos=0.82, acc=0.01)
7. Residual patch partially works (+7% with alpha=0.5), but frozen W→logit lens gives ~0.005 — W trained with MSE doesn't align to lm_head (Residual Patch: probe=1.0, logit lens=0.005)
8. Multi-layer injection HURTS: injecting at 5 layers simultaneously degrades Phi-2 (alpha=0.3→0.105), while single-layer +7% holds. Per-layer W ≈ same W — layer-specific alignment irrelevant.
9. **Nonlinear adapter = old adapter, reformatted.** Trained Linear(2560→2560) between W(h_A) and frozen lm_head = 1.0, but this composes two linear layers → one Linear(W(h_A)→97). Identical to existing adapter=0.999. The claim "bottleneck is coordinate alignment" is not supported — lm_head is passive, gradient passes through it (Nonlinear Adapter: Linear=1.0, MLP=1.0, trainable lm_head=1.0).
10. **Single-template probe on L31 = 0.41 confirms syntactic pattern, not arithmetic encoding.** The jump from 0.04 (mixed templates) to 0.41 (single template) shows Phi-2 processes stable syntax → stable activation geometry. Different templates → different paths → structure disappears. Natural adapter conclusion (Phi-2 doesn't encode mod arithmetic linearly) stands. Template mixing was a measurement confound, not a conclusion confound (Probe Final: L31 Linear→97 = 0.41).
11. **CE-trained W resolves barrier 1: W_CE(h_A) → lm_head = 1.0.** Training W via CE through frozen lm_head + final_layernorm achieves perfect logit-lens accuracy, proving MSE was the sole cause of lm_head misalignment. However, barrier 2 persists: W_CE patched at L10 degrades text accuracy (α=0.5: 0.26 vs W_MSE 0.305) — the context/geometry incompatibility in a single residual stream remains unsolved. cos_sim=0.0 with L10 targets confirms W_CE finds directions orthogonal to Phi-2 activations (CE Projection: logit_lens=1.0, probe=1.0, cos_sim=0.0).
12. **L31 injection accuracy.** Injection at Phi-2's final layer (L31) achieves perfect task accuracy (1.0) at α=1.0 for both addition and multiplication, but intermediate α (≤0.7) is statistically indistinguishable from the unpatched baseline (0.235 add, 0.030 mult) — a discontinuous cliff, not the smooth interpolation reported pre-fix. Exact across all 5 seeds (42–46) post-layernorm-fix retraining. Mechanism: `final_layernorm`'s scale invariance lets weight decay collapse W_CE's output norm to ~0.7% of Phi-2's residual norm (‖W_CE(h)‖≈0.32 vs ‖h_Phi2‖≈48.3 for addition; ≈0.25 vs 48.3 for multiplication), making the injected signal negligible until it's the only term left (α=1.0).
13. **Perplexity at operating α (corrected by fine grid).** WikiText-2 last-token PPL: flat through α=0.95 (~80.7), then ramps smoothly: 0.98→81.1, 0.99→83.4, 0.995→99.8, 1.0→10¹⁰. The catastrophic collapse is real but preceded by a modest 25% rise at α=0.995. The original claim of a "flat-then-jump" from 63.3→10⁹ with no intermediate rise was a coarse-grid artifact — the same sampling resolution issue as the accuracy cliff. The fine grid shows continuous (but steep) degradation, not a discrete override.
14. **Cross-model comparison with Qwen2-Math invalidated by tokenizer mismatch.** Qwen2-Math's BPE splits 87/97 numbers into subword tokens → only 10 unique lm_head outputs for 97 classes. W_CE logit lens capped at 7.3% (not from absence of structure but from lm_head resolution). The hypothesis "math-pretrained LLMs resonate better with W_CE" is untestable via lm_head for models with number-splitting BPE. Per-layer probes confirm Qwen2-Math encodes no mod arithmetic structure (max=0.0276 vs random 0.0103, from scan_models.py).
15. **Cross-model probe (bypasses tokenizer): probe on W_CE-injected L_last converges for all models.** Using LogisticRegression probe on the injected hidden state (bypasses lm_head): Phi-2 L31 probe(α=0.0)=0.7067 → probe(α=0.5)=0.9996 → probe(α=1.0)=1.0000. Qwen2-Math-1.5B L27 probe(α=0.0)=0.3174 → probe(α=0.5)=0.9848 → probe(α=1.0)=0.9993. With the full mod arithmetic template, Qwen encodes partial structure (0.32 vs scan_models' 0.0276 with "a b" template — confirming finding #10 about template sensitivity). At α=0.5 both converge to ~0.99 — W_CE dominates and model-specific differences vanish. **Hypothesis not supported**: math pretraining does not improve W_CE resonance. Phi-3-mini-4k incomplete cache — not tested (Cross-model Probe: Phi-2=0.9996, Qwen=0.9848 at α=0.5).
16. **Control experiments: all linear controls FAIL** under the layernorm-inclusive CE objective. Multi-seed (5 main seeds × 5 random sub-seeds = 25 draws for random condition): addition controls = 0.0000 ± 0.000; multiplication controls = 0.015–0.019 (near chance 1.9%, which is the chance baseline given label-0 imbalance — see Finding #18). Grokked achieves 1.000 ± 0.000 on both ops. The onehot-mlp nonlinear control partially succeeds (add 0.700 ± 0.027, mult 0.449 ± 0.048) but remains well short of grokked. This closes the most critical critique: the 194→128 bottleneck genuinely cannot encode Fourier structure from unstructured inputs, and the layernorm-inclusive training eliminates any residual signal the old no-layernorm objective could barely extract (old pre-fix numbers: onehot 0.0188, random 0.0142). See `compile_random_baseline_summary.py` and `artifacts/random_baseline/results_seed*.csv`.
17. **Multiplication baseline bug (historical).** The OP_SYMBOL prompt-hardcoding bug caused multiplication's L31 α=0.5 accuracy to be measured as 0.370; correcting the prompt template raised this to 0.660 under the pre-layernorm-fix W_CE. This number is now superseded: under the layernorm-inclusive W_CE (Finding #12), multiplication α=0.5 accuracy is 0.030 (flat baseline), as the norm-mismatch mechanism dominates regardless of prompt correctness. Retained here only to document the OP_SYMBOL fix's validity at the time it was made.

18. **Multiplication label imbalance confound (CRITICAL for control interpretation).** For modular multiplication (a·b mod 97), label 0 appears 193× (all pairs where a=0 or b=0) vs 96× for labels 1–96. This raises the chance baseline from 1/97 ≈ 0.0103 to ~0.019 (always predict 0). The onehot and random controls achieving ~0.015–0.019 on multiplication is entirely explained by this imbalance — they learn "if a=0 or b=0, predict 0" and guess uniformly otherwise. This is not evidence of arithmetic structure. Always check the mult label distribution when interpreting control accuracy on multiplication; the ~0.019 baseline is structurally higher than addition's 0.000, but both represent complete failure.

19. **Self-projection controls.** Four-model comparison (grokked W_ce, rescaled W_ce (1370×), W_self from Phi-2 L31, PCA-128+W_ce from Phi-2 L31) across full α range on full test set (n=2823). Results fundamentally revise claims about the "discontinuous cliff":

    - **Finding #12 corrected — the "cliff" is a sampling artifact.** The original 5-point sweep {0.0, 0.3, 0.5, 0.7, 1.0} sampled on either side of the S-curve ramp (compressed into α∈[0.95, 1.0]) and missed it. At fine resolution, W_ce (raw) transitions smoothly: 0.281@0.7, 0.318@0.9, 0.496@0.98, 0.719@0.99, 0.983@0.995, 1.0@0.999. The curve is continuous — just steep. The paper's "discontinuous override" framing should be replaced with "an S-curve so steep that 5-point sampling made it look like a step."
    - **No model shows bimodal per-class accuracy** — but class-level σ was the wrong diagnostic. Per-class σ averages ~29 examples per class together, washing out within-class bimodality.
    - **Per-example crossing-α distributions show the real signal.** Each example gets a crossing-α (first α where prediction is correct). Grokked W_ce: the ~75% of non-baseline examples flip within Δα≈0.01 (P75–P95 = [0.985, 0.995]) — genuinely narrow per-example margins. W_self: spread over Δα≈0.65. PCA-128: spread over Δα≈0.74. The steepness gradient is directly explained: grokked produces near-uniform per-example crossing-α; non-grokked produces heterogeneous per-example crossing-α.
    - **Within each class, crossing-α is bimodal** — confirmed by within-class σ: W_ce (raw) has 79/97 classes with within-σ ≥ 0.3 (mean σ=0.373), because ~25% at α=0 (baseline) and ~75% at α≈0.99 produce a wide gap. W_ce (1370x) has all 97 classes with within-σ < 0.1 (mean σ=0.041) because the two groups are closer (α=0 vs α≈0.045). W_self and PCA-128: moderate within-σ (means 0.195, 0.211) with 93/97 and 87/97 classes in the 0.1–0.3 band respectively.
    - **The 18 σ<0.3 classes in W_ce (raw) have no competing mechanism** — they all have near-0% baseline accuracy (max frac_baseline=0.097), meaning only one mode exists (all examples flip at α≈0.99), eliminating the bimodality gap. This is a boundary effect, not a different mechanism.
    - **Multi-seed verification (seeds 42, 43, 44):** Seeds are genuinely different (different weight hashes, different initial training losses: 5.86, 5.72, 5.57; 293/2823 per-example crossing-α values differ between seeds 42 and 43 by Δα≈0.005). Summary stats match to 3 decimals because W_ce training converges to distributionally equivalent solutions, not from a cache bug. The per-example margin structure is a robust property of the grokked representation, not seed-dependent.
    - **Right-censoring:** W_ce (raw) reaches 100% of examples; W_self reaches 2475/2823 (12.3% unreachable); PCA-128 reaches 2002/2823 (29.1% unreachable). Survivorship bias understates σ for controls: censored means shift from 0.242→0.335 (W_self) and 0.225→0.451 (PCA-128).
    - **Scale rescaling** shifts the crossing-α left (Δα≈0.09 for 1370×) but preserves the tight distribution — the margin uniformity is in the geometry, not the norm.
    - **Ceilings: 1.0** (grokked) > **0.83** (W_self, self-referential limit) > **0.55** (PCA-128). The PCA-128 ceiling is notable: 93% variance retained but caps at 0.55 (860/2823 examples never reach ceiling), while the grokked source at 128 raw dims (far less overlap with Phi-2 variance) reaches 1.0. The discarded 7% of PCA variance contains class-discriminative structure that the CE objective requires. This is independent evidence that grokked geometry matters for the ceiling, separate from the margin-structure steepness.
    - See `artifacts/self_projection/crossing_alpha_hist.png` and `artifacts/self_projection/comparison_summary.md`.
