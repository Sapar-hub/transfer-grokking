# L31 Degradation: WikiText-2 last-token perplexity

| Alpha | Last-Token Loss | Last-Token PPL |
|-------|----------------|----------------|
| 0.0 | 4.1474 (+0.0000) | 63.2717 |
| 0.5 | 4.1878 (+0.0404) | 65.8769 |
| 1.0 | 34.0172 (+29.8698) | 593608459197375.1250 |

Baseline (no patch): loss=4.1474, ppl=63.2717

**Verdict**: α=1.0 degrades last-token prediction — patch corrupts general LM.