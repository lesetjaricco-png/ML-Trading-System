# v0.3_forward_atr_xgb_baseline_v1_ablation_25f

- Status: completed
- Runtime: 35.39 seconds
- Features: 25
- TEST labels read/evaluated: false / false

## TRAIN and VALIDATION
| Split | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| TRAIN | 0.7046 | 0.5507 | 0.5860 |
| VALIDATION | 0.5915 | 0.3890 | 0.3721 |

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
19. `day_of_week`
20. `is_weekend`
21. `hour_of_day`
22. `is_market_open`
23. `is_asia_session`
24. `is_london_session`
25. `is_new_york_session`

## Baseline Comparison
| Metric | Baseline | 25-feature ablation | Difference |
|---|---:|---:|---:|
| Validation Accuracy | 0.5087 | 0.5915 | +0.0828 |
| Validation Balanced Accuracy | 0.4064 | 0.3890 | -0.0174 |
| Validation Macro F1 | 0.3641 | 0.3721 | +0.0080 |
| BUY Recall | 0.0423 | 0.1638 | +0.1216 |
| SELL Recall | 0.4955 | 0.0912 | -0.4043 |
| BUY Prediction % | 2.5904 | 9.7485 | +7.1581 |
| SELL Prediction % | 37.9802 | 6.0193 | -31.9610 |
| NO_TRADE Prediction % | 59.4294 | 84.2323 | +24.8029 |
| CV Mean Accuracy | 0.5380 | 0.5712 | +0.0332 |
| CV Mean Macro F1 | 0.3861 | 0.3913 | +0.0053 |
| TRAIN-VALIDATION Accuracy Gap | 0.2309 | 0.1131 | -0.1178 |
| TRAIN-VALIDATION Balanced Accuracy Gap | 0.1987 | 0.1617 | -0.0370 |
| TRAIN-VALIDATION Macro F1 Gap | 0.2857 | 0.2139 | -0.0718 |

## Per-class Validation
| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SELL | 0.2973 | 0.0912 | 0.1396 | 1,568 |
| BUY | 0.3582 | 0.1638 | 0.2248 | 1,703 |
| NO_TRADE | 0.6396 | 0.9121 | 0.7519 | 4,720 |

## Chronological TRAIN-only CV
| Fold | Accuracy | Balanced accuracy | Macro F1 | BUY recall |
|---:|---:|---:|---:|---:|
| 1 | 0.5523 | 0.4105 | 0.4102 | 0.1984 |
| 2 | 0.5587 | 0.4021 | 0.3955 | 0.2131 |
| 3 | 0.5748 | 0.4059 | 0.4017 | 0.1784 |
| 4 | 0.5905 | 0.3922 | 0.3794 | 0.1790 |
| 5 | 0.5798 | 0.3880 | 0.3698 | 0.1636 |

## Top Native Importance
| Rank | Feature | Importance |
|---:|---|---:|
| 1 | is_asia_session | 0.646383 |
| 2 | hour_of_day | 0.089761 |
| 3 | is_new_york_session | 0.023855 |
| 4 | is_london_session | 0.022197 |
| 5 | volume_ratio | 0.014997 |
| 6 | day_of_week | 0.014454 |
| 7 | price_to_sma_200 | 0.012876 |
| 8 | volatility_20 | 0.012484 |
| 9 | price_to_sma_50 | 0.012281 |
| 10 | atr_pct | 0.012220 |
| 11 | price_to_sma_20 | 0.012219 |
| 12 | bb_width | 0.011991 |
| 13 | rsi | 0.011399 |
| 14 | price_to_sma_10 | 0.011271 |
| 15 | volatility_5 | 0.011162 |

## Importance Concentration
- Baseline top-1/top-2 share: 0.4000 / 0.5101
- Ablation top-1/top-2 share: 0.6464 / 0.7361
- Baseline/ablation normalized entropy: 0.7277 / 0.5105
- Native gain importance is descriptive and does not establish causality.

## Verdict
**INCONCLUSIVE**: The measured changes are mixed or insufficiently consistent across validation, gaps, directional behavior, and CV.

Next experiment: Run one controlled session/time-only removal from the 25-feature set, retaining the 18 normalized/relative features and identical parameters, to measure whether stable calendar activity signal helps or masks directional generalization.
