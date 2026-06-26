# Embed Patch Summary

## Metrics

| Metric | Text Baseline | inputs_embeds | Delta |
|---|---|---|---|
| Mod Arithmetic Acc | 0.2400 | 0.0100 | -0.2300 |
| Probe Acc (pre-layer) | 0.0100 | 0.0167 | +0.0067 |
| Random baseline | 0.0103 | — | — |

## Verdict
FAILED: inputs_embeds accuracy dropped to random. W_emb preserves geometry
(cos=0.82) but Phi-2 cannot use it without the text prompt context.

