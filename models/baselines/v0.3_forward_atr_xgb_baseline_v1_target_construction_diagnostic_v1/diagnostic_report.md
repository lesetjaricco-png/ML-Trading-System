# v0.3_forward_atr_xgb_baseline_v1_target_construction_diagnostic_v1

- Mode: read-only; no target changes or retraining
- TEST rows/labels read: false / false

## Exact Construction
- Horizon: 5 M15 bars (75 minutes)
- Threshold: +/- 1.0 x ATR[t]
- BUY: Close[t+5]/Close[t]-1 is strictly greater than +ATR[t]/Close[t].
- SELL: Close[t+5]/Close[t]-1 is strictly less than -ATR[t]/Close[t].
- NO_TRADE: the endpoint return is inside or exactly on those bounds.
- TP/SL 100/20 are configured point counts; other target modes convert them to price distances using points x instrument point_size. V0.3 does not use them.
- There is no TP/SL tie in V0.3. Exact equality to either ATR boundary is NO_TRADE; configured conservative_sl tie handling is unused.
- max_bars and unresolved policy are also unused by V0.3.
- Spread, point size, tick value, commission, slippage, and intrahorizon High/Low path are not inputs to this target.

## Class Distribution
| Split | Class | Count | Percentage |
|---|---|---:|---:|
| TRAIN | SELL | 14,645 | 20.36% |
| TRAIN | BUY | 15,579 | 21.66% |
| TRAIN | NO_TRADE | 41,693 | 57.97% |
| VALIDATION | SELL | 1,568 | 19.62% |
| VALIDATION | BUY | 1,703 | 21.31% |
| VALIDATION | NO_TRADE | 4,720 | 59.07% |

## Boundary Fragility
| Split | Class | Within 0.05 ATR | Within 0.10 ATR | Within 0.25 ATR | Median boundary distance | Median move | Median threshold points |
|---|---|---:|---:|---:|---:|---:|---:|
| TRAIN | SELL | 3.97% | 7.62% | 18.60% | 0.878 ATR | 1.878 ATR | 33.62 |
| TRAIN | BUY | 4.46% | 8.85% | 20.59% | 0.789 ATR | 1.789 ATR | 34.21 |
| TRAIN | NO_TRADE | 3.34% | 6.77% | 18.51% | 0.581 ATR | 0.419 ATR | 36.07 |
| VALIDATION | SELL | 4.34% | 8.56% | 20.88% | 0.816 ATR | 1.816 ATR | 31.98 |
| VALIDATION | BUY | 4.76% | 9.16% | 22.08% | 0.760 ATR | 1.760 ATR | 33.13 |
| VALIDATION | NO_TRADE | 3.10% | 6.91% | 19.02% | 0.561 ATR | 0.439 ATR | 37.24 |

## Audit Integrity
- Reconstructed labels matching persisted labels: 100.000000%
- Audited rows: 79,903
- Intentionally unaudited validation tail: 5 rows; their t+5 endpoints are in sealed TEST.

## Economic Interpretation
The target expresses a volatility-scaled 75-minute endpoint move, so BUY and SELL have directional meaning. It does not express whether a realizable long or short trade hit TP before SL, survived adverse excursion, or remained profitable after spread and costs.

Close calls in this target are directional-vs-NO_TRADE boundary cases, not BUY-vs-SELL ties: the two directional classes are separated by a 2 ATR-wide NO_TRADE region.

The frozen raw cache contains OHLCV only, so spread and bid/ask effects cannot be measured retrospectively from this dataset.

## Conclusion
**TARGET_MECHANICALLY_SOUND_BUT_UNVALIDATED_ECONOMICALLY**: The labels are mechanically stable around their threshold, but economic validity remains unproven because the target ignores execution costs and intrahorizon path.

## Direct Answers
1. BUY requires the close exactly five bars later to exceed the entry close by strictly more than one entry-time ATR.
2. SELL requires that future close to fall by strictly more than one entry-time ATR.
3. NO_TRADE covers all endpoint moves from -1 ATR through +1 ATR, including exact boundary equality.
4. BUY/SELL are economically meaningful as large signed 75-minute market moves, but not as realizable trade outcomes because execution and path are absent.
5. The cutoff is not evidently noisy. The likely target bottleneck is endpoint/path misalignment; current evidence cannot separate that from intrinsically weak short-horizon direction predictability without one controlled relabel comparison.

Smallest controlled follow-up (not implemented): keep horizon=5 and barrier distance=1 ATR, but assign BUY/SELL by which High/Low barrier is reached first; assign unresolved rows NO_TRADE and drop same-bar ties. This isolates endpoint-vs-path semantics using the frozen OHLCV data.
