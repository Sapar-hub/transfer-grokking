# Grokking — Geometry Transfer Experiments

## Overview
Research repo: Do grokked transformers learn scale-invariant geometric representations of modular arithmetic?
- **Model A (small):** 2 layers, d_model=128, d_mlp=512
- **Model B (big):** 6 layers, d_model=512, d_mlp=2048
- **Task:** (a + b) mod 97 with direct token IDs (0–96)
- **Artifacts:** `artifacts/`

## Setup
- No build system — pure Python, one script per experiment
- Virtual environment: `.venv/` — activate before running anything
- CPU-only (`DEVICE = torch.device("cpu")`)
- `matplotlib.use('Agg')` in all scripts (no display)
- No formatter/linter/typechecker config

## Config
Configs live in `model.py`:
- `CFG_SMALL` (name="small"), `CFG_BIG` (name="big")
- `SmallTransformer()` returns `Transformer(CFG_SMALL)` for backward compat

## Entry Points
Every script is standalone (`if __name__ == "__main__": main()`):
| Script | Purpose |
|--------|---------|
| `main.py` | Original pipeline orchestrator (steps 1–8) |
| `train_small.py` | Train small model (30% data, rolling window grokking detection) |
| `train.py` | Train either small or big (70/30 split) |
| `verify_fourier.py` | Confirm circular Fourier features |
| `probe_phi2.py` | Probe Phi-2 layers for mod arithmetic structure |
| `scan_models.py` | Probe Qwen2-Math, DeepSeek-Math, Phi-3 |
| `experiment_a.py` | Learned projection 128→2560 (Small→Phi-2) |
| `clean_test.py` | Clean experiment: Small→Big (same tokenizer) |
| `line_a.py` | SVCCA heatmap + noise injection steering |
| `line_b.py` | Projected probe deep-dive |
| `steering.py` | Steering vector + random orthogonal projection |
| `eval_degradation.py` | Downstream benchmark eval (needs lm_eval) |
| `embed_patch.py` | inputs_embeds test: W_emb 128→2560, Phi-2 bypassing BPE |
| `residual_patch.py` | Inject computed state (h_A) into Phi-2 residual stream via W + context prompt |
| `multi_layer_patch.py` | Inject h_A at 5 layers simultaneously (per-layer W + same-W ablation) |

## Commands
```bash
source .venv/bin/activate
python train_small.py               # Train small model
python train.py                     # Train both (70/30 split)
python verify_fourier.py            # Verify Fourier structure
python clean_test.py                # Run clean experiment
python line_a.py                    # SVCCA + noise injection
python line_b.py                    # Projected probe analysis
python experiment_a.py              # Learned projection Small→Phi-2
python scan_models.py               # Probe multiple LLMs
python embed_patch.py               # Embed patch: inputs_embeds via W_emb
python residual_patch.py            # Residual patch: inject computed state into Phi-2
python multi_layer_patch.py         # Multi-layer injection (5 layers simultaneously)
```

## Artifact Cache Map
Scripts skip computation if a cache file exists:
| Created By | File | Used By |
|-----------|------|---------|
| `train_small.py` / `train.py` | `artifacts/small/best_model.pth` | all downstream |
| `train.py` | `artifacts/big/best_model.pth` | `clean_test.py`, `line_a.py` |
| `clean_test.py` | `artifacts/activations/small_acts_test.npy` | `line_a.py`, `line_b.py` |
| `clean_test.py` | `artifacts/activations/big_acts_test.npy` | `line_a.py` |
| `clean_test.py` | `artifacts/projection/W.pth` | `line_a.py`, `line_b.py` |
| `clean_test.py` | `artifacts/steering/steering_vec.npy` | `line_a.py` |
| `experiment_a.py` | `artifacts/experiment_a/projection_W.pth` | itself (cache) |
| `experiment_a.py` | `artifacts/experiment_a/phi2_layer30_activations.npy` | itself (cache) |
| `embed_patch.py` | `artifacts/embed_patch/W_emb.pth` | itself (cache) |
| `residual_patch.py` | `artifacts/residual_patch/phi2_activations.npz` | itself (cache) |
| `residual_patch.py` | `artifacts/residual_patch/W_layer*.pth` | `multi_layer_patch.py` |
| `multi_layer_patch.py` | `artifacts/multi_layer_patch/experiment_summary.md` | itself (cache) |

## Gotchas
- **Weight decay 1.0** is critical for grokking (L2 forces circuit formation)
- **SVCCA with k=20** required — raw CCA on 128/512 dim with N=2823 overfits to ~1.0
- **Noise calibration:** embedding norm ~22.65; use σ ∈ {0.05, 0.10, 0.20, 0.50}, not {0.5, 1.0, 2.0}
- **`seaborn` not installed** — use matplotlib for all plots
- **`nn.Linear` outputs require grad by default** — call `W.requires_grad_(False)` after loading W.pth
- **Ceiling effect:** B baseline = 1.0; use noise injection or degradation as alternative steering metrics
- **Proxy fallback:** `scan_models.py` tries SOCKS5 proxy first, falls back to direct connection
- **BPE splits numbers >9 into subword tokens** — for `phi2_targets` in `embed_patch.py`, take mean over all subword token embeddings per number, not just the first token

## Key Findings
1. cos_sim between different-dim residual streams plateaus at ~0.30 regardless of conditioning
2. Linear separability partially transfers (probe = 0.93–0.94) even when cos_sim is low
3. Layers align by position, not cross-functionally (SVCCA: A[1]↔B[5] = 0.835)
4. Steering only distinguishable from random when cos_sim > ~0.7
5. Tokenizer mismatch is NOT the primary barrier (Clean Experiment confirms)
6. Grokked models compile algorithms; LLMs simulate them via language — fundamentally incommensurable (Embed Patch: cos=0.82, acc=0.01)
7. Residual patch partially works (+7% with alpha=0.5), but LM head cannot read Fourier representation (Residual Patch: probe=1.0, logit lens=0.005)
8. Multi-layer injection HURTS: injecting at 5 layers simultaneously degrades Phi-2 (alpha=0.3→0.105), while single-layer +7% holds. Per-layer W ≈ same W — layer-specific alignment irrelevant.
