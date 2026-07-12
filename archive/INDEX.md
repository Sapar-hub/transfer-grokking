# Archive Index

One line per file pointing to the AGENTS.md Finding # or README Phase it produced / was superseded by.

## Root scripts

| File | Status |
|------|--------|
| `clean_test.py` | → Clean Experiment (same tokenizer), produced Finding #1, superseded by CE training (Finding #11) |
| `experiment_a.py` | → Learned projection 128→2560 (MSE-based), superseded by CE training (Finding #11) |
| `embed_patch.py` | → Embed patch (bypass BPE), produced Finding #6, not needed after CE+L31 approach |
| `analyze_baseline_asymmetry.py` | → Multiplication prediction distribution analysis, exploratory only |
| `trivial_baselines.py` | → Few-shot / LoRA baselines, exploratory only |
| `setup_phi2_cache.py` | → Phi-2 cache verification, one-time utility, not part of pipeline |
| `compile_results.py` | → Result table compiler for README, superseded by direct CSV reporting |
| `compile_random_baseline_summary.py` | → Random baseline summary compiler, superseded by `random_baseline.py`'s own output |
| `run_multi_seed_sweep.py` | → Multi-seed sweep orchestrator, superseded by `run_pipeline.sh` |
| `train.py` | → Multi-model trainer; only consumer was `clean_test.py` (archived); use `train_small.py` instead |
| `cross_model_l31.py` | → Cross-model L31 validation for Qwen2-Math; invalidated by Qwen2-Math BPE tokenizer mismatch (Finding #14); superseded by `cross_model_probe.py` (Finding #15) which bypasses the tokenizer barrier |

## `experiments/`

| File | Status |
|------|--------|
| `adapter.py` | → Natural language adapter (Phase 9), superseded by CE+L31 approach |
| `eval_degradation.py` | → Downstream benchmark eval via lm_eval (never succeeded — network unavailable) |
| `eval_natural_adapter.py` | → Natural adapter evaluation (Phase 9), superseded by CE+L31 |
| `interpret.py` | → Early interpretability exploration, not part of final pipeline |
| `line_a.py` | → SVCCA heatmap + noise injection (Phase 7), produced Finding #3, superseded by CE+L31 |
| `line_b.py` | → Projected probe deep-dive (Phase 8), produced Finding #2, superseded by CE+L31 |
| `main.py` | → Original pipeline orchestrator, superseded by numbered pipeline scripts |
| `multi_layer_patch.py` | → Multi-layer injection experiment, produced Finding #8, superseded by L31 single-layer approach |
| `natural_adapter.py` | → Natural adapter training (Phase 9), superseded by CE+L31 |
| `nonlinear_adapter.py` | → MLP adapter nonlinearity test, produced Finding #9, dead end |
| `plot_umap.py` | → UMAP visualization, exploratory only |
| `probe_final_phi2.py` | → Single-template L31 probe, produced Finding #10, superseded by cross_model_probe.py |
| `probe_phi2.py` | → Phi-2 layer probe (Phase 3), produced Finding #10 (precursor), superseded by CE approach |
| `residual_patch.py` | → Residual stream patch (MSE-based), produced Finding #7, superseded by CE+L31 |
| `scan_models.py` | → Multi-LLM probe scan (Phase 3), produced Finding #14 (Qwen2-Math tokenizer mismatch), superseded by cross_model_probe.py |
| `steering.py` | → Steering vector + random orthogonal projection (Phase 4), produced Finding #4, dead end |
| `wide_linear_multi_seed.py` | → Wide linear projection multi-seed test, exploratory only |
