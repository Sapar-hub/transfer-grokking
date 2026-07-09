# Cross-Model Probe Comparison

| Model | Layers | d_model | L_last | Unique tokens | Logit lens | Probe(α=0.0) | Probe(α=0.5) | Probe(α=1.0) |
|-------|--------|---------|--------|---------------|------------|--------------|--------------|--------------|
| Phi-2 | 32 | 2560 | L31 | 97/97 | 1.0000 | 0.7088 | 0.7134 | 1.0000 |
| Qwen2-Math-1.5B | 28 | 1536 | L27 | 97/97 | 0.0542 | 0.3177 | 0.9702 | 0.9975 |
| Phi-3-mini-4k | — | — | — | — | — | — | — | — |

Notes:
- Phi-3-mini-4k incomplete (download/proxy issue).
- Phi-2 α=0.5 dropped from 0.9996 (pre-layernorm-fix) to 0.7134 because W_CE now trained with final_layernorm in the loop — at α=0.5 the mixed signal passes through layernorm which distorts it. α=1.0 unchanged (1.0000), confirming neural function call works at full strength.
- Qwen2-Math unique_tokens=97/97 after mean-pool fix (was 10/97 buggy). Logit lens 0.0542 (was 0.0726 buggy) — the old 10-class problem was easier.
- Qwen2-Math probe results essentially unchanged — probe bypasses lm_head, so the mean-pool fix doesn't affect probe directly.
