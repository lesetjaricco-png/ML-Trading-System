# v0.3_forward_atr_xgb_baseline_v1_directional_context_60f_v1

- Single directional feature-enrichment experiment
- Target: unchanged five-bar first-passage +/- 1 ATR[t]
- TEST rows/labels exposed: 0 / false

## Primary Validation Result
| Model | Direction AUC | Balanced accuracy | BUY recall | SELL recall | Direction macro F1 | Activity AUC |
|---|---:|---:|---:|---:|---:|---:|
| frozen first-passage 42f | 0.4994 | 0.5084 | 0.1707 | 0.8462 | 0.4363 | 0.7383 |
| enriched 60f | 0.4995 | 0.5017 | 0.2516 | 0.7518 | 0.4617 | 0.7475 |

## Conditional Direction Confusion Matrix
Rows are actual SELL/BUY; columns are predicted SELL/BUY.

```text
[1593, 526]
[1758, 591]
```

## Secondary Three-Class Context
- Accuracy: 0.4633
- Balanced accuracy: 0.4499
- Macro F1: 0.4301

## Decision
**INSUFFICIENT_DIRECTIONAL_INFORMATION**: Enriched BUY-vs-SELL AUC=0.4995, chance=0.5000, first-passage control=0.4994, previous best=0.5164, balanced accuracy=0.5017, BUY recall=0.2516, SELL recall=0.7518.

Stop adding feature complexity to this M15 OHLCV representation. Next research should change the information set or horizon, such as higher-timeframe context, order-flow/microstructure data, or another instrument, in a separately approved study.
