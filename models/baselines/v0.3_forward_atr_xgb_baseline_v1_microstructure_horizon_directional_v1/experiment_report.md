# v0.3_forward_atr_xgb_baseline_v1_microstructure_horizon_directional_v1

## Decision
**NO_DIRECTIONAL_SIGNAL**: best=m15_h1, validation AUC=0.5137, CV AUC=0.5141 +/- 0.0064.

## Validation Results
| Treatment | Horizon | AUC | CV mean | CV std | Classification |
|---|---:|---:|---:|---:|---|
| m15_h1 | 15m | 0.5137 | 0.5141 | 0.0064 | NO_DIRECTIONAL_SIGNAL |
| m15_h3 | 45m | 0.5100 | 0.5087 | 0.0138 | NO_DIRECTIONAL_SIGNAL |
| m1_h2 | 30m | 0.5063 | 0.5108 | 0.0095 | NO_DIRECTIONAL_SIGNAL |
| m15_h2 | 30m | 0.5035 | 0.5124 | 0.0039 | NO_DIRECTIONAL_SIGNAL |
| m15_h5 | 75m | 0.4997 | 0.5097 | 0.0145 | NO_DIRECTIONAL_SIGNAL |
| m1_h1 | 15m | 0.4992 | 0.5102 | 0.0128 | NO_DIRECTIONAL_SIGNAL |
| m1_h3 | 45m | 0.4977 | 0.5088 | 0.0075 | NO_DIRECTIONAL_SIGNAL |
| m1_h8 | 120m | 0.4962 | 0.5104 | 0.0159 | NO_DIRECTIONAL_SIGNAL |
| m1_spread_h5 | 75m | 0.4956 | 0.5076 | 0.0145 | NO_DIRECTIONAL_SIGNAL |
| m15_h4 | 60m | 0.4927 | 0.5075 | 0.0092 | NO_DIRECTIONAL_SIGNAL |
| m15_h8 | 120m | 0.4926 | 0.5109 | 0.0150 | NO_DIRECTIONAL_SIGNAL |
| m1_h5 | 75m | 0.4915 | 0.5061 | 0.0152 | NO_DIRECTIONAL_SIGNAL |
| m1_h4 | 60m | 0.4895 | 0.5081 | 0.0111 | NO_DIRECTIONAL_SIGNAL |
| spread_h5 | 75m | 0.4872 | 0.5052 | 0.0131 | NO_DIRECTIONAL_SIGNAL |

## Availability
- M1 OHLC, tick volume, and historical bar spread: available with verified MT5 provenance.
- Historical ticks and bid/ask imbalance: UNAVAILABLE_RELIABLE_HISTORY.
- Real volume is included in provenance and used only if nonzero; it is not a model feature.
