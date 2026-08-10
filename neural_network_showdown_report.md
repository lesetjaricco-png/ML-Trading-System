# Neural Network Showdown - Frozen Directional Prediction Benchmark

## 1. Environment

- Windows 11, Python 3.14.6, selected interpreter `.venv/Scripts/python.exe`.
- PyTorch is not installed. TensorFlow/Keras is unavailable.
- Pylance found only the project Python 3.14 environment and the global Python 3.14 interpreter.
- A recursive interpreter check found no additional compatible isolated environment.
- No package, runtime, CUDA component, or global dependency was installed.

## 2. Hardware Constraints

- Physical RAM: 5.841 GiB.
- Free RAM at preflight: 0.953 GiB.
- CPU-only execution was required.
- Creating and populating a new Python/PyTorch runtime under this memory pressure was not considered safe.

The experiment therefore hit the predefined dependency/resource stop condition before implementation or training.

## 3. Dataset Identity

- Dataset: `data/processed/US30_2022-05-12_2026-08-09_M15_v0.3_forward_atr_v1.parquet`
- SHA-256: `35995f3d673b914cd4631e95f7abc13ee78acec89c12129edbc1b1d353bb4634`
- Metadata SHA-256: `f3a4667ffc2ca8b7dd1fddf0a44c842f0a871401b88f77c57e8e7aa25b79b042`
- TRAIN: 71,917 rows, 2022-05-16 04:45 through 2025-06-02 22:30.
- VALIDATION: 7,991 rows, 2025-06-02 22:45 through 2025-10-02 12:00.
- TEST boundary was read from metadata only. No TEST rows were materialized.

## 4. Feature Representation

The frozen ordered 42-feature US30 M15 representation was identified. No feature was added, removed, reordered, or recomputed.

## 5. Target Definition

The target remains the frozen directional subset: SELL = 0 and BUY = 1. The target, horizon, and construction were not changed.

## 6. Frozen XGBoost Control

- Model SHA-256: `6e30c834df78c6448a6987b9427e1a7ee0677eacaf15efd04e82a828d207f3b8`
- Evaluation report SHA-256: `f0590c40956a7b041103933b3386dad04c94b53818b08955213e9e921d655bcd`
- Validation directional AUC: 0.5129618201.
- Validation balanced accuracy: 0.5004.
- Existing verdict: `INSUFFICIENT_DIRECTIONAL_INFORMATION`.
- Authoritative apples-to-apples CV AUC and permutation results are absent from the frozen control artifacts. They are reported as unavailable; the control was not retrained.

## 7. MLP Results

The showdown MLP was not run because preflight stopped the benchmark. Prior independent sklearn evidence on the same frozen representation is retained as context, not presented as a new run:

- Architecture: 42 -> 32 -> 16 -> 1.
- Parameters: 1,921.
- Validation AUC: 0.4923667867.
- Internal chronological holdout AUC: 0.4802600374.
- Training time: 1.910 seconds.
- Reload predictions identical: yes.
- Prior verdict: `NO_NEURAL_INCREMENTAL_SIGNAL`.

## 8. CNN Results

Not run: no compatible CPU neural framework or existing isolated environment was available.

## 9. GRU Results

Not run: no compatible CPU neural framework or existing isolated environment was available. No sequences were constructed.

## 10. LSTM Results

Not run: the dependency/resource stop occurred before GRU, and the required architecture order was preserved.

## 11. Parameter Counts

Only the prior independent MLP count is available: 1,921 parameters. CNN, GRU, and LSTM parameter counts are unavailable because their framework-backed implementations were not created after the stop condition.

## 12. Training Times

No showdown model was trained. Prior independent MLP training took 1.910 seconds.

## 13. Memory Usage

Preflight observed 0.953 GiB free of 5.841 GiB physical RAM. No training-time or peak neural-framework memory measurement exists because training did not start.

## 14. Validation AUC Comparison

| Model | Validation AUC | Status |
|---|---:|---|
| Frozen XGBoost | 0.5129618201 | Frozen control |
| Prior independent sklearn MLP | 0.4923667867 | Negative context; not rerun |
| 1D CNN | unavailable | Not run |
| GRU | unavailable | Not run |
| LSTM | unavailable | Not run |

## 15. CV Comparison

No new chronological CV was run. Authoritative frozen-control CV metrics were unavailable, and no neural architecture entered training.

## 16. Permutation Results

No new permutation test was run. The frozen control artifact does not contain an authoritative apples-to-apples permutation result.

## 17. Overfitting Diagnostics

The prior MLP showed TRAIN AUC 0.5677, internal holdout AUC 0.4803, and external VALIDATION AUC 0.4924, which is negative evidence and indicates no stable directional generalization. No new architecture was trained.

## 18. Leakage and Integrity Results

- TEST rows/features/labels exposed: 0 / false / false.
- Dataset and frozen model identities verified before reporting.
- Feature, target, normalization, batching, and sequence code were not changed.
- No scaler was fitted and no sequence was constructed.
- No leakage was detected because no showdown preprocessing or training occurred.
- Protected artifacts remained unchanged.

Focused showdown tests were not added or run because the mandatory environment preflight stopped the work before implementation. Existing tests and the full repository suite were not claimed as post-experiment validation because no experiment ran.

## 19. Model Reload Determinism

No showdown model exists to reload. The prior independent MLP had bit-identical reloaded predictions.

## 20. Final Scientific Verdict

`NEURAL_EXPERIMENT_INCONCLUSIVE`

The requested four-architecture benchmark could not be completed reliably under the available dependency and memory conditions. Existing MLP evidence is negative, but it cannot establish results for CNN, GRU, or LSTM.

**Did neural networks uncover information that XGBoost could not?**

**INCONCLUSIVE.** The only completed prior neural model underperformed XGBoost, while the requested CNN, GRU, and LSTM comparisons could not safely run.