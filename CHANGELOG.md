# Changelog

## v3.0.0 (2026-07-15)

### Corrected
- **"Discontinuous override" → "steep but continuous S-curve".** The central claim of v2.0.0
  — that L31 injection accuracy jumps discontinuously from baseline to 1.0 — is
  corrected. Fine-resolution analysis (73 α points, n=2823) reveals a continuous
  S-curve compressed into Δα ≈ 0.05 (86% of non-baseline examples flip within
  [0.95, 1.0]), which the original 5-point grid under-sampled. The transition
  is steep but monotonic (0/20 non-monotonic in spot-check). All 11 occurrences
  of "discontinuous," "cliff," and "hard override" in the paper text have been
  rewritten to "sharp-threshold" or "steep but continuous" framing.

### Added
- **Margin analysis step (step 7)** in `run_pipeline.py`: fine-resolution
  crossing-α, per-class statistics, and perplexity fine grid. Supersedes
  `self_projection.py`'s coarse α sweep.
- **`artifacts/margin_analysis/`** output directory for crossing-α analysis,
  owned by `margin_analysis.py`.

### Changed
- `margin_analysis.py` output directory moved from `artifacts/self_projection/`
  to `artifacts/margin_analysis/` to clarify artifact ownership.
- `AGENTS.md` updated with run order: `self_projection.py` → `margin_analysis.py`.

### Validation
- Pipeline steps 1–5 fully wiped and regenerated from scratch (all 5 seeds,
  both operations). Every regenerated file matches independently
  cross-validated values.
- Step 6 (random baseline) regenerated post-fixes; unchanged since v2.0.0
  artifact generation. Not re-wiped in this pass.

### Notes
- v2.0.0's DOI'd Zenodo record should be annotated with a forward pointer to
  this release. See https://zenodo.org for metadata editing of published
  versions.

## v2.0.0 (2026-07-13)

### Added
- Self-projection controls (W_self, W_ce_scaled, W_ce_pca)
- Random baseline experiments (25 draws)
- Cross-model probe (bypasses BPE tokenizer)
- Per-class crossing-α analysis
- Perplexity fine grid evaluation

### Fixed
- OP_SYMBOL prompt-hardcoding bug (multiplication used "+" instead of "\*")
- Layernorm-inclusive CE training (final_layernorm included in forward path)
- Pipeline restructured from bash to Python (`run_pipeline.py`)
- Multi-seed sentinel bug (all 5 seeds properly trained and evaluated)

### Changed
- Paper's central claim updated post-layernorm-fix: norm-mismatch mechanism
  replaces earlier "continuous tradeoff" framing

## v1.0.0 (2026-07-01)

### Added
- Initial release: CE-vs-MSE projection comparison
- L10 and L31 patch experiments
- Cross-model validation (Qwen2-Math, Phi-3)
- "Discontinuous override" framing (corrected in v3.0.0)
