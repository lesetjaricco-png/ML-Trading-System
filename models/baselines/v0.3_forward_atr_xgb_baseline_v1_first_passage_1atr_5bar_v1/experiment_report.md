# v0.3_forward_atr_xgb_baseline_v1_first_passage_1atr_5bar_v1

- Target-only controlled experiment
- TEST rows/labels exposed: 0 / false

## Target
- Entry: current Close
- Horizon: five future M15 bars
- Barriers: entry +/- 1.0 x ATR[t]
- SAME_BAR_TIE rows excluded

## Distribution Before Tie Exclusion
| Split | Class | Count | Percentage |
|---|---|---:|---:|
| TRAIN | SELL | 20,214 | 28.11% |
| TRAIN | BUY | 21,284 | 29.60% |
| TRAIN | NO_TRADE | 30,386 | 42.25% |
| TRAIN | SAME_BAR_TIE | 33 | 0.05% |
| TRAIN | INCOMPLETE | 0 | 0.00% |
| VALIDATION | SELL | 2,119 | 26.52% |
| VALIDATION | BUY | 2,349 | 29.40% |
| VALIDATION | NO_TRADE | 3,516 | 44.00% |
| VALIDATION | SAME_BAR_TIE | 2 | 0.03% |
| VALIDATION | INCOMPLETE | 5 | 0.06% |

## Distribution After Tie Exclusion
| Split | Class | Count | Percentage |
|---|---|---:|---:|
| TRAIN | SELL | 20,214 | 28.12% |
| TRAIN | BUY | 21,284 | 29.61% |
| TRAIN | NO_TRADE | 30,386 | 42.27% |
| VALIDATION | SELL | 2,119 | 26.54% |
| VALIDATION | BUY | 2,349 | 29.42% |
| VALIDATION | NO_TRADE | 3,516 | 44.04% |

## Validation Comparison
| Model | Accuracy | Balanced acc. | Macro F1 | SELL P/R/F1 | BUY P/R/F1 | NO_TRADE P/R/F1 | Pred S/B/N | Direction AUC | Filter AUC | CV accuracy | CV macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| endpoint_42f | 0.5087 | 0.4064 | 0.3641 | 0.2560/0.4955/0.3376 | 0.3478/0.0423/0.0754 | 0.6772/0.6814/0.6793 | 37.98%/2.59%/59.43% | 0.5130 | 0.6690 | 0.5380 | not recorded |
| endpoint_25f | 0.5915 | 0.3890 | 0.3721 | 0.2973/0.0912/0.1396 | 0.3582/0.1638/0.2248 | 0.6396/0.9121/0.7519 | 6.02%/9.75%/84.23% | 0.5164 | 0.7012 | 0.5712 | 0.3913 |
| endpoint_18f | 0.5857 | 0.3604 | 0.3225 | 0.2694/0.0510/0.0858 | 0.3418/0.0869/0.1386 | 0.6131/0.9432/0.7432 | 3.72%/5.42%/90.86% | 0.5092 | 0.6645 | 0.5611 | 0.3525 |
| first_passage_42f | 0.4532 | 0.4457 | 0.4071 | 0.3200/0.6748/0.4341 | 0.4107/0.1205/0.1863 | 0.6741/0.5418/0.6008 | 55.97%/8.63%/35.40% | 0.4994 | 0.7383 | 0.4863 | 0.4493 |

## Confusion Matrix
Rows are actual SELL, BUY, NO_TRADE; columns are predicted SELL, BUY, NO_TRADE.

```text
[1430, 242, 447]
[1592, 283, 474]
[1447, 164, 1905]
```

## Five-fold TRAIN-only CV
| Fold | Accuracy | Macro F1 | Direction AUC | Filter AUC |
|---:|---:|---:|---:|---:|
| 1 | 0.4625 | 0.4488 | 0.5133 | 0.7752 |
| 2 | 0.5023 | 0.4775 | 0.5168 | 0.7878 |
| 3 | 0.4514 | 0.4172 | 0.5171 | 0.7932 |
| 4 | 0.5105 | 0.4375 | 0.4810 | 0.7901 |
| 5 | 0.5046 | 0.4654 | 0.5203 | 0.7817 |

## Verdict
**FILTER_ONLY_SIGNAL**: Validation BUY-vs-SELL AUC=0.4994, directional-vs-NO_TRADE AUC=0.7383, minimum BUY/SELL recall=0.1205, best endpoint minimum recall=0.0912.
