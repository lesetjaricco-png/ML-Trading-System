# v0.3_forward_atr_xgb_baseline_v1_activity_filter_strategy_experiment_v1

## Decision
**ACTIVITY_FILTER_WEAK_EVIDENCE**

## Validation
| Rule | Filter | Threshold | Trades | Win rate | Avg return | Profit factor | Permutation p |
|---|---|---:|---:|---:|---:|---:|---:|
| momentum_3 | unfiltered | - | 7937 | 0.4837 | -0.000039 | 0.9250 | - |
| momentum_3 | ml | 0.50 | 5582 | 0.4833 | -0.000041 | 0.9331 | 0.5529 |
| momentum_3 | random | 0.50 | 5582 | 0.4858 | -0.000037 | 0.9291 | - |
| momentum_3 | ml | 0.60 | 5129 | 0.4847 | -0.000046 | 0.9265 | 0.6926 |
| momentum_3 | random | 0.60 | 5129 | 0.4792 | -0.000052 | 0.8989 | - |
| momentum_3 | ml | 0.70 | 4547 | 0.4805 | -0.000066 | 0.8985 | 0.9681 |
| momentum_3 | random | 0.70 | 4547 | 0.4858 | -0.000036 | 0.9301 | - |
| momentum_3 | ml | 0.80 | 3312 | 0.4858 | -0.000050 | 0.9279 | 0.6727 |
| momentum_3 | random | 0.80 | 3312 | 0.4774 | -0.000044 | 0.9164 | - |
| ma_10_20 | unfiltered | - | 7985 | 0.4962 | 0.000017 | 1.0346 | - |
| ma_10_20 | ml | 0.50 | 5618 | 0.4927 | 0.000007 | 1.0126 | 0.8224 |
| ma_10_20 | random | 0.50 | 5618 | 0.4957 | 0.000008 | 1.0169 | - |
| ma_10_20 | ml | 0.60 | 5157 | 0.4945 | 0.000018 | 1.0299 | 0.4950 |
| ma_10_20 | random | 0.60 | 5157 | 0.5022 | 0.000031 | 1.0646 | - |
| ma_10_20 | ml | 0.70 | 4570 | 0.5020 | 0.000046 | 1.0762 | 0.0299 |
| ma_10_20 | random | 0.70 | 4570 | 0.5000 | 0.000011 | 1.0229 | - |
| ma_10_20 | ml | 0.80 | 3330 | 0.5015 | 0.000068 | 1.1072 | 0.0140 |
| ma_10_20 | random | 0.80 | 3330 | 0.4982 | 0.000042 | 1.0875 | - |
| breakout_20 | unfiltered | - | 1729 | 0.4639 | -0.000069 | 0.8846 | - |
| breakout_20 | ml | 0.50 | 1355 | 0.4686 | -0.000074 | 0.8896 | 0.6287 |
| breakout_20 | random | 0.50 | 1355 | 0.4627 | -0.000059 | 0.9008 | - |
| breakout_20 | ml | 0.60 | 1246 | 0.4631 | -0.000085 | 0.8770 | 0.6846 |
| breakout_20 | random | 0.60 | 1246 | 0.4639 | -0.000096 | 0.8444 | - |
| breakout_20 | ml | 0.70 | 1132 | 0.4647 | -0.000080 | 0.8876 | 0.6766 |
| breakout_20 | random | 0.70 | 1132 | 0.4647 | -0.000062 | 0.8943 | - |
| breakout_20 | ml | 0.80 | 878 | 0.4704 | -0.000048 | 0.9357 | 0.3114 |
| breakout_20 | random | 0.80 | 878 | 0.4442 | -0.000083 | 0.8670 | - |
| completed_bar_direction | unfiltered | - | 7879 | 0.4862 | -0.000024 | 0.9535 | - |
| completed_bar_direction | ml | 0.50 | 5555 | 0.4907 | -0.000018 | 0.9693 | 0.3613 |
| completed_bar_direction | random | 0.50 | 5555 | 0.4853 | -0.000032 | 0.9384 | - |
| completed_bar_direction | ml | 0.60 | 5102 | 0.4908 | -0.000024 | 0.9609 | 0.4990 |
| completed_bar_direction | random | 0.60 | 5102 | 0.4857 | -0.000030 | 0.9423 | - |
| completed_bar_direction | ml | 0.70 | 4522 | 0.4912 | -0.000025 | 0.9605 | 0.5449 |
| completed_bar_direction | random | 0.70 | 4522 | 0.4845 | -0.000015 | 0.9713 | - |
| completed_bar_direction | ml | 0.80 | 3299 | 0.4998 | -0.000011 | 0.9834 | 0.2675 |
| completed_bar_direction | random | 0.80 | 3299 | 0.4771 | -0.000075 | 0.8629 | - |

Gross outcomes only; no historical execution costs were invented.
