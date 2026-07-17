# Grokking — Geometry Transfer Experiments

## Overview
Research repo: Do grokked transformers learn scale-invariant geometric representations of modular arithmetic?
- **Model A (small):** 2 layers, d_model=128, d_mlp=512
- **Model B (big):** 6 layers, d_model=512, d_mlp=2048
- **Task:** (a + b) mod 97 with direct token IDs (0–96)
- **Pipeline:** `src/pipeline/` (7 active scripts + auxiliary)
- **Archive:** `archive/` (dead ends and superseded experiments)
- **Artifacts:** `artifacts/`
- **Paper:** `paper/main.tex`

## Setup
- No build system — pure Python, one script per experiment
- Virtual environment: `.venv/` — activate before running anything
- CPU-only (`DEVICE = torch.device("cpu")`)
- `matplotlib.use('Agg')` in all scripts (no display)
- No formatter/linter/typechecker config
- **Git LFS:** `artifacts/**/*.pth` and `artifacts/*labels*.npy` are LFS-tracked. Run `git lfs pull` after cloning.

## Config
Configs live in `src/model.py`:
- `CFG_SMALL` (name="small"), `CFG_BIG` (name="big")
- `SmallTransformer()` returns `Transformer(CFG_SMALL)` for backward compat

## Pipeline Orchestration
`run_pipeline.py` (repo root) runs all 7 steps for seeds 42–46, ops add+mult:
```bash
python run_pipeline.py
```
This replaces manual multi-seed orchestration. Steps can still be run individually.

## Entry Points
Every script is standalone (`if __name__ == "__main__": main()`). All scripts run from repo root.

### Active Pipeline (`src/pipeline/`, core paper results)
| Script | Purpose | Phase |
|--------|---------|-------|
| `cache_activations.py` | Run grokked model on all P² pairs; save post-block-1 activations + labels | core (step 1) |
| `train_small.py` | Train small model (add or mult, rolling window grokking detection) | auxiliary |
| `ce_projection.py` | Train W: 128→2560 via CE through frozen lm_head; compare MSE vs CE | core (step 2) |
| `l31_patch.py` | Patch W_CE/W_MSE at Phi-2 L31 — alpha sweep vs L10 | core (step 3) |
| `eval_l31_perplexity.py` | Perplexity degradation of L31 patch on WikiText-2 | core (step 4) |
| `cross_model_probe.py` | Cross-model probe: probe on W_CE-injected hidden states (bypasses BPE) | core (step 5) |
| `random_baseline.py` | Control: one-hot + random untrained network, L31 α-sweep | core (step 6) |
| `self_projection.py` | Control: train W_self from Phi-2's own L31 acts | control |
| `margin_analysis.py` | Multi-seed crossing-α + per-class + perplexity fine grid (v3.0.0 correction) | correction (step 7) |

### Utilities (`src/`)
| Script | Purpose |
|--------|---------|
| `model.py` | Transformer definitions (Attention, MLP, Transformer) |
| `utils.py` | Shared utilities (DEVICE, P, probes, data gen, plotting) |
| `verify_fourier.py` | Confirm circular Fourier features |

### Paper Figures
| Script | Purpose |
|--------|---------|
| `figures/generate_paper_figures.py` | Generate all 6 paper figure PNGs from `artifacts/final/` |

### Archive (`archive/` — dead ends, superseded, one-time utilities)
| Script | Purpose | Superseded By |
|--------|---------|---------------|
| `train.py` | Train small or big (70/30 split) | `train_small.py` |
| `clean_test.py` | Small→Big (same tokenizer), MSE-based | CE training (Finding #11) |
| `experiment_a.py` | Learned projection 128→2560 (MSE), Small→Phi-2 | CE training (Finding #11) |
| `embed_patch.py` | inputs_embeds via W_emb, bypass BPE | CE+L31 approach |
| `residual_patch.py` | Inject computed state into Phi-2 residual stream (MSE) | CE+L31 (Finding #7) |
| `multi_layer_patch.py` | Inject at 5 layers simultaneously | L31 single-layer (Finding #8) |
| `nonlinear_adapter.py` | Linear/MLP adapter between W(h_A) and frozen lm_head | Dead end (Finding #9) |
| `probe_final_phi2.py` | Linear probe on Phi-2 L31 (single template) | `cross_model_probe.py` (Finding #10) |
| `cross_model_l31.py` | Cross-model L31 for Qwen2-Math | Invalidated by BPE (Finding #14) |
| `scan_models.py` | Probe Qwen2-Math, DeepSeek-Math, Phi-3 | `cross_model_probe.py` (Finding #14) |
| `probe_phi2.py` | Probe Phi-2 layers for mod arithmetic structure | `cross_model_probe.py` |
| `steering.py` | Steering vector + random orthogonal projection | Dead end (Finding #4) |
| `eval_degradation.py` | Downstream benchmark eval (needs lm_eval) | Never succeeded (network) |
| `analyze_baseline_asymmetry.py` | Phi-2 prediction distribution for add vs mult | Exploratory only |
| `trivial_baselines.py` | Few-shot (5-shot) + LoRA baselines | Exploratory only |
| `setup_phi2_cache.py` | Phi-2 model cache verification after manual download | One-time utility |
| `archive/experiments/line_a.py` | SVCCA heatmap + noise injection steering | CE+L31 (Finding #3) |
| `archive/experiments/line_b.py` | Projected probe deep-dive | CE+L31 (Finding #2) |
| `archive/experiments/main.py` | Original pipeline orchestrator | `run_pipeline.py` |
| `compile_random_baseline_summary.py` | Random baseline table compiler | Superseded by `random_baseline.py` own output |
| `compile_results.py` | Result table compiler for README | Superseded by direct CSV reporting |

See `archive/INDEX.md` for the full one-line-per-file mapping to FINDINGS.md.

## Run order
1. `src/pipeline/cache_activations.py` → must run first (produces activations for steps 2+)
2. `src/pipeline/ce_projection.py` → trains W_CE/W_MSE
3. `src/pipeline/l31_patch.py` → uses W_CE from step 2
4. `src/pipeline/eval_l31_perplexity.py` → uses W_CE from step 2
5. `src/pipeline/cross_model_probe.py` → uses W_CE from step 2
6. `src/pipeline/random_baseline.py` → standalone (only needs small model weights)
7. `src/pipeline/self_projection.py` → must run before margin_analysis.py
8. `src/pipeline/margin_analysis.py` → loads W_self/W_ce_scaled/W_ce_pca from self_projection

`run_pipeline.py` automates all dependencies.

## Commands
```bash
source .venv/bin/activate

# Pipeline (7 steps, all seeds, both ops)
python run_pipeline.py

# Or individual steps:
python src/pipeline/cache_activations.py add
python src/pipeline/cache_activations.py mult
python src/pipeline/ce_projection.py 42 add
python src/pipeline/ce_projection.py 42 mult
python src/pipeline/l31_patch.py 42 add
python src/pipeline/l31_patch.py 42 mult
python src/pipeline/eval_l31_perplexity.py "42" add
python src/pipeline/eval_l31_perplexity.py "42" mult
python src/pipeline/cross_model_probe.py
python src/pipeline/random_baseline.py 42 add --partial
python src/pipeline/random_baseline.py 42 mult --partial
python src/pipeline/margin_analysis.py "42" add
python src/pipeline/margin_analysis.py "42" mult

# Train from scratch
python src/pipeline/train_small.py add
python src/pipeline/train_small.py mult

# Fourier verification
python src/verify_fourier.py

# Paper figures
python figures/generate_paper_figures.py

# Archive scripts
python archive/clean_test.py
python archive/experiments/line_a.py
python archive/experiments/line_b.py
```

## Artifact Cache Map
Scripts skip computation if a cache file exists. All paths relative to repo root.

| Created By | File | Used By |
|-----------|------|---------|
| `train_small.py` | `artifacts/small{_mult}/best_model.pth` | all downstream |
| `cache_activations.py` | `artifacts/small_model_activations{_mult}.npy` | `ce_projection.py` |
| `cache_activations.py` | `artifacts/mod_arithmetic_labels{_mult}.npy` | `ce_projection.py` |
| `ce_projection.py` | `artifacts/ce_projection/phi2_L10_acts{_mult}.npy` | itself (cache) |
| `ce_projection.py` | `artifacts/ce_projection/W_mse.pth` | itself (cache) |
| `ce_projection.py` | `artifacts/ce_projection/W_ce.pth` | itself (cache) |
| `ce_projection.py` | `artifacts/ce_projection/seeds{_mult}/W_ce_seed{N}.pth` | `l31_patch.py`, `eval_l31_perplexity.py`, `cross_model_probe.py`, `margin_analysis.py` |
| `ce_projection.py` | `artifacts/ce_projection/seeds{_mult}/W_mse_seed{N}.pth` | `l31_patch.py` |
| `l31_patch.py` | `artifacts/l31_patch{_mult}/alpha_sweep_l31_seed{N}.csv` | itself (cache) |
| `l31_patch.py` | `artifacts/l31_patch{_mult}/comparison_l10_vs_l31_seed{N}.md` | itself (cache) |
| `eval_l31_perplexity.py` | `artifacts/l31_patch{_mult}/perplexity_sweep_seed{N}.csv` | itself (cache) |
| `cross_model_probe.py` | `artifacts/cross_model/probe_comparison.md` | itself (cache) |
| `cross_model_probe.py` | `artifacts/cross_model/*_L*_acts.npy` | itself (cache) |
| `self_projection.py` | `artifacts/self_projection/phi2_L31_acts{_mult}.npy` | itself (cache) |
| `self_projection.py` | `artifacts/self_projection/seeds{_mult}/W_self_seed{N}.pth` | `margin_analysis.py` |
| `self_projection.py` | `artifacts/self_projection/seeds{_mult}/alpha_sweep_seed{N}.csv` | itself (cache) |
| `self_projection.py` | `artifacts/self_projection/*` | `margin_analysis.py` |
| `margin_analysis.py` | `artifacts/margin_analysis/crossing_alpha_data{_mult}.pkl` | itself (cache) |
| `margin_analysis.py` | `artifacts/margin_analysis/crossing_alpha_hist.png` | itself (cache) |
| `margin_analysis.py` | `artifacts/margin_analysis/per_class_crossing.csv` | itself (cache) |
| `margin_analysis.py` | `artifacts/margin_analysis/perplexity_fine_grid.csv` | itself (cache) |
| `margin_analysis.py` | `artifacts/margin_analysis/perplexity_fine_grid.png` | itself (cache) |
| `generate_paper_figures.py` | `artifacts/final/paper_figures/*.png` | itself (cache) |

**Note:** `{_mult}` = `""` for addition, `"_mult"` for multiplication. Suffixed variants live alongside their unsuffixed counterparts. `seeds{_mult}` → `seeds/` for add, `seeds_mult/` for mult. `l31_patch{_mult}` → `l31_patch/` for add, `l31_patch_mult/` for mult.

## OpenAI Grokking Paper (Power et al. 2022) — Hyperparameters

| Parameter | Paper Value | This Repo Value | Notes |
|-----------|------------|-----------------|-------|
| Layers | 2 | 2 | Same architecture (decoder-only transformer) |
| Heads | 4 | 4 | Same |
| d_model | 128 | 128 | Same |
| d_mlp | 512 (4×) | 512 (4×) | Same ReLU MLP |
| Dropout | 0.0 | 0.0 | Same |
| Optimizer | CustomAdamW | AdamW | Paper: betas=(0.9, 0.98), eps=1e-8 |
| Max LR | 1e-3 | 1e-4 | Paper uses 10× higher LR |
| LR schedule | Linear warmup(10) + cosine anneal to 1e-4 over 100k steps (optional) | Constant | Paper's `--anneal_lr` flag |
| Weight decay | Explored 0–1.0 (default 0) | **1.0** | Paper found higher WD helps grokking |
| Train data | 5–50% (equation strings) | 30% (direct token IDs) | Different tokenization |
| Training steps | 100k (step-based) | 30k–100k (epoch-based) | Paper uses 1 epoch = 1 full data pass |
| Task format | "a+b=c" with tokenizer | [a, b] direct token IDs | Paper: seq2seq; repo: classification |

**Key difference:** The paper treats modular arithmetic as an auto-regressive sequence task ("a+b=c" → tokenize each char), while this repo uses direct token IDs [a, b] → classify the result. The paper's higher LR (1e-3) with cosine annealing and lower WD default reflect the different task format. This repo's high WD (1.0) with low constant LR (1e-4) is tuned for the direct-token-ID formulation and empirically critical for grokking to occur within reasonable epochs.

## Gotchas
- **Weight decay 1.0** is critical for grokking (L2 forces circuit formation)
- **SVCCA with k=20** required — raw CCA on 128/512 dim with N=2823 overfits to ~1.0
- **Noise calibration:** embedding norm ~22.65; use σ ∈ {0.05, 0.10, 0.20, 0.50}, not {0.5, 1.0, 2.0}
- **`seaborn` not installed** — use matplotlib for all plots
- **`nn.Linear` outputs require grad by default** — call `W.requires_grad_(False)` after loading W.pth
- **Ceiling effect:** B baseline = 1.0; use noise injection or degradation as alternative steering metrics
- **Proxy fallback:** `archive/experiments/scan_models.py` tries SOCKS5 proxy first, falls back to direct connection
- **BPE splits numbers >9 into subword tokens** — for `phi2_targets` in `archive/embed_patch.py`, take mean over all subword token embeddings per number, not just the first token
- **Qwen2-Math BPE splits all numbers >9**: Qwen2-Math's tokenizer maps 87/97 numbers to subword tokens (only digits 0–9 are single tokens). This makes lm_head-based evaluation of mod arithmetic impossible (10 unique tokens for 97 classes). For cross-model experiments, verify tokenizer first.
- **Prompt operator bug (CRITICAL):** 3 OP-aware scripts (`random_baseline.py`, `l31_patch.py`, `ce_projection.py`) hardcoded `+` in prompts. All now use `OP_SYMBOL[OP]` which is `{"add": "+", "mult": "*"}`. Addition-only scripts (`archive/experiments/*`, `archive/embed_patch.py`) remain unchanged. If extending OP support to a new script, use `OP_SYMBOL = {"add": "+", "mult": "*"}` instead of hardcoding.
- **Old multiplication baseline (0.010) was contaminated:** The wrong prompt (`+` instead of `*`) was used for all α values in the mult sweep. Corrected mult α=0.5=0.660 (was ~0.370). The claim "multiplication requires stronger injection" is weakened. When reporting multiplication results, always state which prompt was used.
- **Layernorm in W_CE training breaks α<1.0 patching (CRITICAL):** `src/pipeline/ce_projection.py` trains W_CE through `final_layernorm` by default. This is faithful to Phi-2's architecture for logit-lens evaluation but makes W_CE dependent on layernorm statistics. During L31 patching at α<1.0, the mixed signal has different statistics → alignment degrades monotonically with lower α. If patching at α<1.0 is the goal, train W_CE without layernorm (remove `final_layernorm` kwarg). Logit lens accuracy is already 1.0 in either case.
- **Multiplication model dir**: `artifacts/small_mult/best_model.pth` (not `artifacts/small/`). The `{_mult}` suffix convention applies consistently.

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
12. **L31 injection accuracy.**
13. **Perplexity at operating α (corrected by fine grid).**
14. **Cross-model comparison with Qwen2-Math invalidated by tokenizer mismatch.**
15. **Cross-model probe (bypasses tokenizer): probe on W_CE-injected L_last converges for all models.**
16. **Control experiments: all linear controls FAIL** under the layernorm-inclusive CE objective.
17. **Multiplication baseline bug (historical).**
18. **Multiplication label imbalance confound (CRITICAL for control interpretation).**
19. **Self-projection controls.** See `docs/FINDINGS.md` for full findings 12–19.
