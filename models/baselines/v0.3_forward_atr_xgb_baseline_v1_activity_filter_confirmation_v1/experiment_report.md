# v0.3_forward_atr_xgb_baseline_v1_activity_filter_confirmation_v1

## Decision
**FAILED_CONFIRMATION**

## Untouched Confirmation
- Period: 2025-10-02T12:15:00 through 2026-08-07T22:30:00
- Rows: 19,977; final 5 rows excluded from outcomes
- Prior TRAIN/VALIDATION rows reused: 0 / 0

## Gross Results
| Strategy | Trades | BUY | SELL | Win rate | Avg return | Median return | Cumulative | Profit factor | Max drawdown | Avg MFE | Avg MAE | +1 ATR | -1 ATR | Permutation p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unfiltered MA10/20 | 19969 | 10337 | 9632 | 0.4972 | -0.000017 | 0.000000 | -0.342031 | 0.9741 | 0.6049 | 0.000921 | -0.000941 | 29.87% | 30.30% | - |
| MA10/20 + 0.70 | 11724 | 6595 | 5129 | 0.4921 | -0.000020 | -0.000020 | -0.233851 | 0.9751 | 0.5241 | 0.001111 | -0.001137 | 37.92% | 38.65% | 0.5868 |
| MA10/20 + 0.80 | 8995 | 5334 | 3661 | 0.4899 | -0.000019 | -0.000021 | -0.166805 | 0.9774 | 0.4318 | 0.001134 | -0.001163 | 39.78% | 40.54% | 0.5050 |

Gross outcomes are reported separately. Reliable execution-time spread/slippage history is unavailable, so no costs were invented.
