# v0.3_forward_atr_xgb_baseline_v1_broader_market_context_v1

## Decision
**FILTER_ONLY_SIGNAL_CONFIRMED**: Best validation directional AUC=0.5059; its five-fold mean directional AUC=0.5066; activity AUC=0.7429.

## Scientific Answer
Broader market context did not provide robust directional information absent from US30 M15 OHLCV; the reproducible signal remains primarily the likelihood of an active move.

## Validation Directional AUC
- higher_timeframe: 0.5059 (activity AUC 0.7429)
- cross_market: 0.4877 (activity AUC 0.7387)
- combined: 0.4940 (activity AUC 0.7443)
