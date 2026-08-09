# v0.3_forward_atr_xgb_baseline_v1_directional_signal_diagnostic_v1

- Mode: read-only diagnostic; no fitting or tuning
- TEST labels read/evaluated: false / false

## Class Distribution
| Split | Class | Count | Percentage |
|---|---|---:|---:|
| TRAIN | SELL | 14,645 | 20.36% |
| TRAIN | BUY | 15,579 | 21.66% |
| TRAIN | NO_TRADE | 41,693 | 57.97% |
| VALIDATION | SELL | 1,568 | 19.62% |
| VALIDATION | BUY | 1,703 | 21.31% |
| VALIDATION | NO_TRADE | 4,720 | 59.07% |

## Validation Model Comparison
| Model | Accuracy | Balanced acc. | Macro F1 | SELL P/R | BUY P/R | NO_TRADE P/R | Predictions S/B/N | Filter AUC | Direction AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| majority_no_trade | 0.5907 | 0.3333 | 0.2476 | 0.0000/0.0000 | 0.0000/0.0000 | 0.5907/1.0000 | 0.00%/0.00%/100.00% | n/a | n/a |
| reference_42f | 0.5087 | 0.4064 | 0.3641 | 0.2560/0.4955 | 0.3478/0.0423 | 0.6772/0.6814 | 37.98%/2.59%/59.43% | 0.6690 | 0.5130 |
| ablation_25f | 0.5915 | 0.3890 | 0.3721 | 0.2973/0.0912 | 0.3582/0.1638 | 0.6396/0.9121 | 6.02%/9.75%/84.23% | 0.7012 | 0.5164 |
| normalized_18f | 0.5857 | 0.3604 | 0.3225 | 0.2694/0.0510 | 0.3418/0.0869 | 0.6131/0.9432 | 3.72%/5.42%/90.86% | 0.6645 | 0.5092 |

## Feature Distribution Separation
| Split | Comparison | Median KS | Max KS | Features KS>=0.10 | Median robust effect |
|---|---|---:|---:|---:|---:|
| TRAIN | BUY_vs_NO_TRADE | 0.0504 | 0.2982 | 10 | 0.0353 |
| TRAIN | SELL_vs_NO_TRADE | 0.0496 | 0.2878 | 6 | 0.0381 |
| TRAIN | BUY_vs_SELL | 0.0150 | 0.0251 | 0 | 0.0088 |
| VALIDATION | BUY_vs_NO_TRADE | 0.0688 | 0.2466 | 8 | 0.0793 |
| VALIDATION | SELL_vs_NO_TRADE | 0.0397 | 0.2212 | 10 | 0.0380 |
| VALIDATION | BUY_vs_SELL | 0.0514 | 0.0986 | 0 | 0.0562 |

KS and robust median effects describe marginal distribution separation; they do not establish tradable causality.

## Conclusion
**FILTER_ONLY_SIGNAL**: NO_TRADE dominance is primarily a learnable activity filter; the current target/features do not provide sufficient BUY/SELL direction signal.

Best filter AUC=0.7012, best BUY-vs-SELL AUC=0.5164, best validation BUY/SELL recall=0.1638/0.4955, validation median BUY-vs-SELL feature KS=0.0514.

Do not proceed to another feature-removal experiment on the basis of NO_TRADE accuracy alone.
