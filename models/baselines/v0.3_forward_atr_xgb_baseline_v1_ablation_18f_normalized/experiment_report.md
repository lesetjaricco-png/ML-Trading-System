# v0.3_forward_atr_xgb_baseline_v1_ablation_18f_normalized

- Status: completed
- Runtime: 31.89 seconds
- TEST labels read/evaluated: false / false

## Exact Ordered Features
1. `returns`
2. `returns_2`
3. `returns_5`
4. `returns_10`
5. `log_returns`
6. `high_low_ratio`
7. `close_open_ratio`
8. `rsi`
9. `bb_width`
10. `bb_pct`
11. `price_to_sma_10`
12. `price_to_sma_20`
13. `price_to_sma_50`
14. `price_to_sma_200`
15. `atr_pct`
16. `volatility_5`
17. `volatility_20`
18. `volume_ratio`

## TRAIN and VALIDATION
| Split | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| TRAIN | 0.6697 | 0.4838 | 0.5035 |
| VALIDATION | 0.5857 | 0.3604 | 0.3225 |

## Three-model Comparison
| Metric | Reference 42f | Ablation 25f | Normalized 18f | 18f-42f | 18f-25f |
|---|---:|---:|---:|---:|---:|
| Validation Accuracy | 0.5087 | 0.5915 | 0.5857 | +0.0770 | -0.0059 |
| Validation Balanced Accuracy | 0.4064 | 0.3890 | 0.3604 | -0.0460 | -0.0287 |
| Validation Macro F1 | 0.3641 | 0.3721 | 0.3225 | -0.0416 | -0.0496 |
| BUY Recall | 0.0423 | 0.1638 | 0.0869 | +0.0446 | -0.0769 |
| SELL Recall | 0.4955 | 0.0912 | 0.0510 | -0.4445 | -0.0402 |
| NO_TRADE Recall | 0.6814 | 0.9121 | 0.9432 | +0.2619 | +0.0311 |
| BUY Prediction % | 2.5904 | 9.7485 | 5.4186 | +2.8282 | -4.3299 |
| SELL Prediction % | 37.9802 | 6.0193 | 3.7167 | -34.2635 | -2.3026 |
| NO_TRADE Prediction % | 59.4294 | 84.2323 | 90.8647 | +31.4354 | +6.6325 |
| TRAIN-VALIDATION Accuracy Gap | 0.2309 | 0.1131 | 0.0841 | -0.1469 | -0.0290 |
| TRAIN-VALIDATION Balanced Accuracy Gap | 0.1987 | 0.1617 | 0.1234 | -0.0753 | -0.0383 |
| TRAIN-VALIDATION Macro F1 Gap | 0.2857 | 0.2139 | 0.1810 | -0.1047 | -0.0329 |
| CV Mean Accuracy | 0.5380 | 0.5712 | 0.5611 | +0.0230 | -0.0102 |
| CV Mean Macro F1 | 0.3861 | 0.3913 | 0.3525 | -0.0336 | -0.0388 |

## Per-class Validation
| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SELL | 0.2694 | 0.0510 | 0.0858 | 1,568 |
| BUY | 0.3418 | 0.0869 | 0.1386 | 1,703 |
| NO_TRADE | 0.6131 | 0.9432 | 0.7432 | 4,720 |

## Chronological TRAIN-only CV
| Fold | Accuracy | Balanced accuracy | Macro F1 | BUY recall |
|---:|---:|---:|---:|---:|
| 1 | 0.5320 | 0.3787 | 0.3725 | 0.1323 |
| 2 | 0.5461 | 0.3723 | 0.3553 | 0.1359 |
| 3 | 0.5674 | 0.3810 | 0.3696 | 0.1466 |
| 4 | 0.5864 | 0.3691 | 0.3430 | 0.1134 |
| 5 | 0.5733 | 0.3599 | 0.3221 | 0.0870 |

## Native Feature Importance
| Rank | Feature | Importance |
|---:|---|---:|
| 1 | volume_ratio | 0.184022 |
| 2 | atr_pct | 0.068403 |
| 3 | high_low_ratio | 0.066803 |
| 4 | volatility_20 | 0.053280 |
| 5 | price_to_sma_200 | 0.050245 |
| 6 | volatility_5 | 0.049424 |
| 7 | price_to_sma_50 | 0.049252 |
| 8 | price_to_sma_20 | 0.048643 |
| 9 | bb_width | 0.047523 |
| 10 | price_to_sma_10 | 0.045829 |
| 11 | rsi | 0.044916 |
| 12 | bb_pct | 0.044429 |
| 13 | returns_10 | 0.043799 |
| 14 | close_open_ratio | 0.043564 |
| 15 | returns_5 | 0.043318 |
| 16 | returns_2 | 0.039712 |
| 17 | log_returns | 0.038731 |
| 18 | returns | 0.038109 |

## Concentration
- 42f top-1/top-2/entropy: 0.4000 / 0.5101 / 0.7277
- 25f top-1/top-2/entropy: 0.6464 / 0.7361 / 0.5105
- 18f top-1/top-2/entropy: 0.1840 / 0.2524 / 0.9611
- Native importance is descriptive and does not establish causality.

## Verdict
**INCONCLUSIVE**: Normalized/relative features show stable above-majority CV signal, but weak BUY/SELL recall and lower class-balanced holdout metrics prevent a clear conclusion.
