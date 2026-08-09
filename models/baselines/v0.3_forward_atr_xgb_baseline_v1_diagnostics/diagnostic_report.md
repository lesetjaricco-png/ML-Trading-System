# v0.3_forward_atr_xgb_baseline_v1 Temporal Diagnostics

## Data Policy
- TRAIN rows: 71,917
- VALIDATION rows: 7,991
- TEST rows/labels exposed: 0 / false
- Reference baseline was loaded, not retrained.

## Target Shift (Validation minus Train)
- SELL: -0.74 pp
- BUY: -0.35 pp
- NO_TRADE: +1.09 pp

## Diagnostic Models
| Model | Split | Accuracy | Balanced accuracy | Macro F1 | BUY recall | SELL recall |
|---|---|---:|---:|---:|---:|---:|
| A_majority | train | 0.5797 | 0.3333 | 0.2447 | 0.0000 | 0.0000 |
| A_majority | validation | 0.5907 | 0.3333 | 0.2476 | 0.0000 | 0.0000 |
| B_session_time_only | train | 0.5929 | 0.3903 | 0.3623 | 0.1844 | 0.0503 |
| B_session_time_only | validation | 0.5999 | 0.3855 | 0.3562 | 0.1803 | 0.0383 |
| C_normalized_relative_only | train | 0.6697 | 0.4838 | 0.5035 | 0.2551 | 0.2101 |
| C_normalized_relative_only | validation | 0.5857 | 0.3604 | 0.3225 | 0.0869 | 0.0510 |
| D_full_42_reference | train | 0.7396 | 0.6051 | 0.6498 | 0.4474 | 0.3995 |
| D_full_42_reference | validation | 0.5087 | 0.4064 | 0.3641 | 0.0423 | 0.4955 |

## Strongest Feature Drift
| Feature | Family | KS | PSI | Std. Wasserstein |
|---|---|---:|---:|---:|
| sma_200 | price_level_or_scale_dependent | 0.8371 | 10.8274 | 1.8142 |
| sma_50 | price_level_or_scale_dependent | 0.8320 | 10.8531 | 1.8190 |
| bb_lower | price_level_or_scale_dependent | 0.8311 | 10.8754 | 1.8208 |
| ema_26 | price_level_or_scale_dependent | 0.8304 | 10.8531 | 1.8200 |
| sma_20 | price_level_or_scale_dependent | 0.8298 | 10.8516 | 1.8201 |
| ema_12 | price_level_or_scale_dependent | 0.8293 | 10.8537 | 1.8204 |
| sma_10 | price_level_or_scale_dependent | 0.8288 | 10.8560 | 1.8204 |
| bb_upper | price_level_or_scale_dependent | 0.8271 | 10.8422 | 1.8183 |
| obv | price_level_or_scale_dependent | 0.6036 | 8.3452 | 0.8608 |
| atr_pct | normalized_relative | 0.1783 | 0.3425 | 0.3492 |
| volatility_20 | normalized_relative | 0.1457 | 0.2111 | 0.2959 |
| volume_sma | price_level_or_scale_dependent | 0.1353 | 0.1446 | 0.2812 |
| volatility_5 | normalized_relative | 0.1287 | 0.0988 | 0.2351 |
| price_to_sma_200 | normalized_relative | 0.1239 | 0.2114 | 0.2479 |
| atr | price_level_or_scale_dependent | 0.1213 | 0.2861 | 0.1863 |

## Importance Stability
- is_asia_session: mean 0.3117, range 0.1550-0.4325, rank range 1-1
- hour_of_day: mean 0.0858, range 0.0629-0.0980, rank range 2-2

## Session and Regime Evidence
- TRAIN Asia NO_TRADE: 81.29%
- TRAIN London/New York NO_TRADE: 42.80% / 41.78%
- Largest hour-specific shift: UTC 20, 13.82 percentage points
- Every VALIDATION observation is in Q5 under TRAIN-defined Close quintiles.
- Best TRAIN-only BUY-vs-SELL univariate AUC: 0.5132 (macd_hist)

## Conclusions
1. Primary problem: a combination led by model overfitting and absolute-price regime shift. Full-model macro F1 falls 0.2857, whereas the session-only model is stable across TRAIN and VALIDATION.
2. Distribution drift is significant but concentrated: price-level/scale-dependent features have mean KS 0.4585 versus 0.0776 for normalized features and 0.0025 for session/time features.
3. Session/time explains directional activity more than direction. Asia is predominantly NO_TRADE (81.29% TRAIN), while London and New York are about 42.80% and 41.78% NO_TRADE. The session-only model reaches validation macro F1 0.3562, but BUY/SELL recall remains weak and asymmetric.
4. Absolute price-level features are unstable: all VALIDATION observations occupy Q5 under TRAIN-defined Close quintiles, and MA/Bollinger levels dominate the drift ranking. This makes tree thresholds learned at earlier index levels poor extrapolators.
5. BUY recall is not explained by aggregate class prevalence, which changes only slightly. TRAIN-only BUY-vs-SELL univariate separability peaks at AUC 0.5132 for macd_hist; the features distinguish directional activity from NO_TRADE far better than BUY from SELL. The full model's directional allocation then shifts from 11.86% BUY / 9.14% SELL on TRAIN to 2.59% BUY / 37.98% SELL on VALIDATION.
6. The 42 features contain some signal, but stable out-of-time information is limited: the full model improves validation macro F1 only 0.0079 over session/time alone while losing substantial accuracy; normalized-only performance is also modest (macro F1 0.3225).
7. Next experiment: a single controlled TRAIN/VALIDATION ablation using the union of the 18 normalized/relative and 7 session/time features, with the same XGBoost parameters and boundary. This directly tests whether scale-dependent absolute features cause the unstable directional allocation; it is an experiment, not a production feature-removal decision, and TEST must remain sealed.

Full measurements are in `diagnostic_report.json` and the six CSV files.
