# v0.3_forward_atr_xgb_baseline_v1_directional_asymmetric_buy_sell_v1

- Verdict: **DIRECTIONAL_ASYMMETRY_NOT_SUPPORTED**
- TEST rows/labels exposed: 0 / False
- Dataset SHA-256: `35995f3d673b914cd4631e95f7abc13ee78acec89c12129edbc1b1d353bb4634`

## Research question
Does separating BUY and SELL into independent success questions reveal directional
information that the frozen BUY-vs-SELL formulation failed to extract from the same
frozen OHLCV representation?

## Exact targets (locked before training)
- Horizon: 5 M15 bars
- BUY_success: `Close[t+5]/Close[t]-1 > +ATR%[t]` (frozen `target == 1`)
- SELL_success: `Close[t+5]/Close[t]-1 < -ATR%[t]` (frozen `target == 0`)
- NO_TRADE rows are negatives for both binary models

## Why this target
Frozen endpoint events are held fixed so the ablation isolates formulation change.
Literal first-passage path ordering was already tested as multiclass
(validation direction AUC = 0.49936062678464477).

## Feature set
- Frozen 42 OHLCV-derived features from `v0.3_forward_atr_xgb_baseline_v1`
- No external information families added

## Frozen splits
- TRAIN: 71917 rows (2022-05-16T04:45:00 → 2025-06-02T22:30:00)
- VALIDATION: 7991 rows (2025-06-02T22:45:00 → 2025-10-02T12:00:00)
- TEST: sealed / not accessed

## Label relationship (are B/C different from A?)
- TRAIN both-zero rate (NO_TRADE): 0.5797
- VALIDATION both-zero rate: 0.5907
- Strict complements? TRAIN=False VALIDATION=False
- Mutually exclusive? TRAIN=True VALIDATION=True
- Label correlation TRAIN/VAL: -0.2659 / -0.2571

## Models
- Control: frozen multiclass XGBoost 42f (`multi:softprob`)
- BUY model: binary XGBoost (`binary:logistic`) predicting BUY_success
- SELL model: binary XGBoost (`binary:logistic`) predicting SELL_success
- Combination: `P(BUY_success) / (P(BUY_success)+P(SELL_success))` on actual directional rows
- Hyperparameters: identical to frozen baseline; no search

## Results
| Treatment | Val ROC-AUC | Val PR-AUC | Balanced acc. | Precision | Recall | Prevalence |
|---|---:|---:|---:|---:|---:|---:|
| Control BUY-vs-SELL (directional rows) | 0.5130 | n/a | 0.5004 | 0.5255 | 0.0423 | n/a |
| BUY_success | 0.6116 | 0.2846 | 0.5018 | 0.3019 | 0.0094 | 0.2131 |
| SELL_success | 0.5754 | 0.2335 | 0.5449 | 0.2565 | 0.3074 | 0.1962 |
| Combination direction score | 0.5156 | 0.5275 | 0.5075 | 0.5384 | 0.2184 | n/a |

- Combination gain vs control: +0.0027
- Probability correlation (VAL): -0.0270
- Neither-dominates rate (both probs < 0.5): 0.7586

## Chronological CV (TRAIN only)
- BUY ROC-AUC mean/std: 0.6250 / 0.0093
- SELL ROC-AUC mean/std: 0.6098 / 0.0085
- Combination direction ROC-AUC mean/std: 0.5125 / 0.0195

## Null / permutation (VALIDATION label shuffles)
- BUY ROC-AUC p=0.0050 (null mean 0.4999)
- SELL ROC-AUC p=0.0050 (null mean 0.5005)
- Combination direction ROC-AUC p=0.0650 (null mean 0.4998)

## Overfitting assessment
- BUY train/val ROC-AUC: 0.9182 / 0.6116 (gap +0.3066)
- SELL train/val ROC-AUC: 0.9226 / 0.5754 (gap +0.3472)

## Leakage audit
- Forbidden feature columns: []
- TEST exposed: 0
- Train/validation chronological gap OK: True

## Final interpretation
Asymmetric labels include NO_TRADE as negatives, so the learning problems differ from BUY-vs-SELL on directional rows. Empirical usefulness is gated separately from that mathematical distinction.

**Verdict: DIRECTIONAL_ASYMMETRY_NOT_SUPPORTED**

## Recommended next experiment
Do not invest in asymmetric BUY/SELL model architecture on this representation. If path-ordered success remains scientifically interesting, treat it as a separate predeclared first-passage asymmetric study; otherwise move to a new information source.
