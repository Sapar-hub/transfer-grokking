# Natural Adapter Summary

## Baseline

| Condition | Accuracy |
|-----------|----------|
| Phi-2 LM head (text) | 0.2350 |
| Random (1/97) | 0.0103 |

## Per-layer adapter accuracy

| Layer | nn.Linear (AdamW) | LogisticRegression |
|-------|-------------------|--------------------|
| 20 | 0.0067 | 0.0028 |
| 25 | 0.0205 | 0.0156 |
| 28 | 0.0400 | 0.0322 |
| 30 | 0.0446 | 0.0322 |

**Best nn.Linear**: L30 acc=0.0446
**Best sklearn**: L28 acc=0.0322

## Template generalization (LogisticRegression, best per L)

| Train → Test | Best L | Acc |
|--------------|--------|-----|
| T0 → T0 | 20 | 1.0000 |
| T0 → T1 | 30 | 0.0200 |
| T0 → T2 | 25 | 0.0140 |
| T0 → T3 | 30 | 0.0240 |
| T1 → T1 | 28 | 0.0467 |
| T2 → T2 | 28 | 0.1200 |
| T3 → T3 | 20 | 0.0400 |

## Interpretation

**adapter acc ≈ random** → Phi-2 не кодирует ответ линейно в residual stream через естественный язык.

**Template generalization низкая** → adapter выучил поверхностный паттерн.
Не обобщается на новые формулировки.
