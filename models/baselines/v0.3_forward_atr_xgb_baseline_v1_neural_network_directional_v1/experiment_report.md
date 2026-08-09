# v0.3_forward_atr_xgb_baseline_v1_neural_network_directional_v1

## Verdict
**NEURAL_NO_DIRECTIONAL_SIGNAL**

## Directional Comparison
| Model | Validation AUC | Chronological evidence | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Frozen XGBoost 42f | 0.5130 | unavailable | 0.5004 | unavailable |
| Small MLP 42f | 0.4924 | internal holdout 0.4803 | 0.5073 | 0.4797 |

## Training
- Epochs / best epoch: 13 / 8
- TRAIN AUC: 0.5677
- Internal TRAIN-validation AUC: 0.4803
- External VALIDATION AUC: 0.4924
- TRAIN to external gap: 0.0753

## Integrity
- TEST rows/features/labels exposed: 0 / false / false
- Scaler TRAIN rows: 71,917
- Reload predictions identical: True
- Protected artifacts unchanged: true (102 checked)

Retire the neural-network branch; do not proceed automatically to LSTM.
