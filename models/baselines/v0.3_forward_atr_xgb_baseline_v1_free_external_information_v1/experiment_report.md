# v0.3_forward_atr_xgb_baseline_v1_free_external_information_v1

## Scientific Conclusion
**NO_DIRECTIONAL_SIGNAL_FROM_FREE_EXTERNAL_INFORMATION**

## Independent Directional Results
| Family | Features | Rows (train/validation) | Control AUC | Treatment AUC | Gain | CV control | CV treatment | CV gain | Permutation p | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| rates | 11 | 71,917/7,991 | 0.5130 | 0.5183 | +0.0053 | 0.5105 | 0.5179 | +0.0073 | 0.9200 | RETIRE_INFORMATION_FAMILY |
| market | 27 | 71,719/7,975 | 0.5118 | 0.5280 | +0.0162 | 0.5124 | 0.5138 | +0.0013 | 0.5300 | RETIRE_INFORMATION_FAMILY |
| volatility | 7 | 70,078/7,754 | 0.5087 | 0.5205 | +0.0118 | 0.5101 | 0.5174 | +0.0073 | 0.1650 | RETIRE_INFORMATION_FAMILY |
| calendar | 10 | 71,917/7,991 | 0.5130 | 0.5208 | +0.0078 | 0.5105 | 0.5120 | +0.0015 | 0.1650 | RETIRE_INFORMATION_FAMILY |

## Source Verdicts
- Macro: UNAVAILABLE_RELIABLE_HISTORY; no free initial-vintage/consensus path was used
- Rates: RETIRE_INFORMATION_FAMILY
- FX/commodities: RETIRE_INFORMATION_FAMILY
- Volatility: RETIRE_INFORMATION_FAMILY
- Calendar: RETIRE_INFORMATION_FAMILY

## Integrity
- TEST rows/labels exposed: 0 / false
- Protected artifacts unchanged: true
- Combined experiment run: false
- Acquired data disk: 39.96 MiB
