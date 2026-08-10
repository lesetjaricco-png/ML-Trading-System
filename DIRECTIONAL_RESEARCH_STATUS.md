# Directional Research Status

**As of:** 2026-08-10  
**Control:** `v0.3_forward_atr_xgb_baseline_v1` on frozen US30 M15 OHLCV (42 features), sealed TEST  
**Control directional BUY-vs-SELL validation AUC:** ≈ 0.513  
**Control / diagnostic filter (trade-vs-NO_TRADE) AUC:** ≈ 0.669 (best ablation filter AUC 0.7012)

This document freezes the scientific picture. It does not re-argue prior negatives.

---

## Bottom line

1. **Direction is not supported** on the frozen US30 M15 OHLCV representation, nor by the free external families already tested, nor by asymmetric BUY/SELL reformulation of the same endpoint labels.
2. **A real activity/filter signal exists** (trade-vs-NO_TRADE), but translating that filter into economic value on an external confirmation window **failed**.
3. **Still open for direction (unpaid):** point-in-time macro initial releases (ALFRED + official schedules) — never run; previously blocked as `UNAVAILABLE_RELIABLE_HISTORY`.
4. **Still open for direction (paid / quote):** CME YM/MYM and Treasury-futures order flow.
5. **Cloud runtime for new fitted experiments is blocked** here: frozen processed parquet and baseline `.joblib` are gitignored/local-only; ML training deps are absent and must not be auto-installed.

---

## Frozen control facts

| Item | Value |
|---|---|
| Dataset | `US30_..._M15_v0.3_forward_atr_v1` |
| Dataset SHA-256 | `35995f3d673b914cd4631e95f7abc13ee78acec89c12129edbc1b1d353bb4634` |
| Features | 42 OHLCV-derived; order frozen |
| Target | 5-bar forward close vs ±1×ATR%[t]; BUY / SELL / NO_TRADE |
| TRAIN | 71,917 rows (2022-05-16 → 2025-06-02) |
| VALIDATION | 7,991 rows (2025-06-02 → 2025-10-02) |
| TEST | 19,977 rows; sealed for directional selection/exploration |
| Direction AUC (VAL) | 0.5130 |
| Filter AUC (VAL) | 0.6690 |
| Prior verdict | `INSUFFICIENT_DIRECTIONAL_INFORMATION` / `FILTER_ONLY_SIGNAL` |

Protected binaries (parquet, `.joblib`) are **not** in this cloud clone. Metadata and evaluation JSON are present.

---

## Ruled out (do not repeat)

| Family / experiment | Verdict | Key evidence |
|---|---|---|
| OHLCV feature churn / ablations (18f, 25f, 60f context) | No directional recovery | Direction AUC stays ~0.50–0.52; filter improves |
| First-passage multiclass retarget | No directional recovery | VAL direction AUC ≈ 0.499 |
| Broader US500 / USTEC / HTF context | `FILTER_ONLY_SIGNAL_CONFIRMED` | Best direction AUC 0.5059; activity AUC ~0.74 |
| Broker M1 microstructure / horizon sweep | `NO_DIRECTIONAL_SIGNAL` | Best VAL AUC 0.5137 |
| Free rates / FX-commodities / daily VIX / calendar | `NO_DIRECTIONAL_SIGNAL_FROM_FREE_EXTERNAL_INFORMATION` | All families `RETIRE_INFORMATION_FAMILY` |
| Asymmetric BUY/SELL on same endpoint labels | `DIRECTIONAL_ASYMMETRY_NOT_SUPPORTED` | Combination direction AUC 0.5156 (+0.0027); one-vs-rest AUCs behave like filter |
| Neural architectures as direction fix | `NEURAL_NO_DIRECTIONAL_SIGNAL` / showdown `NEURAL_EXPERIMENT_INCONCLUSIVE` | Prior MLP below XGB; CNN/GRU/LSTM not safely runnable |

Also unavailable as reliable history in this program: broker VIX/DXY continuous identities, historical ticks/DOM, free consensus/forecast surprises, exchange breadth without licensed membership history.

---

## Positive signal that remains

### Classification: activity / filter

- BUY-vs-NO_TRADE and SELL-vs-NO_TRADE show measurable marginal separation; BUY-vs-SELL does not (VAL median KS ≈ 0.05, 0 features with KS ≥ 0.10).
- Filter AUCs ~0.67–0.70 (and higher with broader context features) reproduce across diagnostics.
- Scientific status: **`FILTER_ONLY_SIGNAL`** — the model mainly ranks whether a ±1 ATR move is likely, not its sign.

### Economic translation: not confirmed

| Stage | Verdict |
|---|---|
| Validation strategy probe (MA10/20 + ML activity probs) | `ACTIVITY_FILTER_WEAK_EVIDENCE` (threshold-dependent; some p ≤ 0.05 on VAL only) |
| Untouched confirmation (TEST timestamps, one-shot, no labels for training) | `FAILED_CONFIRMATION` |

Therefore: **do not deploy** an activity filter as a proven trading edge. The classification signal is real; the economic claim is not.

---

## Still scientifically open

### A. Highest unpaid directional EV — ALFRED / official macro PIT

- Inventory rank #1; free-external pilot **did not** acquire or test it (`macro.rows = 0`, reason: no `FRED_API_KEY`, BLS schedule blocked).
- Hypothesis: discrete release shocks can resolve sign in event windows in a way that continuous price transforms cannot.
- Requirements: free FRED/ALFRED key, official release timestamps, initial vintages only (`output_type=4` where valid), causal `available_time_utc`, TRAIN/VALIDATION only, frozen 42f control intact.
- Stop if: no reconstructible initial releases with audited release times; event-window n too small; VAL direction AUC gain fails predeclared threshold + permutation; leakage/hash fail.

### B. Low-cost directional pilots still unused

- Bounded GDELT headline volume/tone (TRAIN-first feasibility).
- These are lower EV than ALFRED and must not expand into paid news subscriptions without a separate decision.

### C. Highest novelty paid directional path

- CME DataMine sample quote for YM/MYM (+ optional ZT/ZF/ZN/ZB) trades/quotes — not purchased; design only.

### D. Non-directional product research (if abandoning direction)

- Controlled **filter-only** work that never scores BUY-vs-SELL as success.
- Must not reopen falsified directional claims.
- Must not re-use the already-burned confirmation partition for selection.
- See `NEXT_EXPERIMENT_BRIEF.md` §2.

---

## Integrity constraints (still binding)

- Do not modify frozen dataset or frozen XGBoost baseline artifacts.
- Do not access TEST for training, selection, or exploratory directional work.
- The activity-filter confirmation already consumed TEST once as a sealed confirmation window; treat that result as spent evidence, not a tunable holdout.
- Do not auto-install missing ML dependencies in constrained environments; record `EXPERIMENT_INCONCLUSIVE` / blocked runtime instead.
- Do not manufacture metrics when parquet/`.joblib` are absent.

---

## Recommended scientific next step

1. **Primary (direction):** run the ALFRED point-in-time macro experiment locally per `NEXT_EXPERIMENT_BRIEF.md` §1 — only unpaid inventory item with both novelty and prior non-execution.
2. **If ALFRED fails or remains unavailable:** stop unpaid directional fishing on this endpoint; either quote CME order-flow **or** switch explicitly to filter-only research under §2 (no direction claims).
3. **Do not** deploy MA/filter strategies, re-churn OHLCV features, re-test retired free external families, or re-run asymmetric BUY/SELL on the same labels.
