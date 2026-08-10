# Next Experiment Brief

**Date:** 2026-08-10  
**Branch context:** cloud continuation after `b7ec376`  
**Cloud runtime status:** `EXPERIMENT_INCONCLUSIVE` — blocked before fitting

---

## Why this cloud environment cannot run the experiment

| Blocker | Detail |
|---|---|
| Missing frozen dataset | `data/processed/US30_2022-05-12_2026-08-09_M15_v0.3_forward_atr_v1.parquet` is gitignored and absent (expected SHA-256 `35995f3d673b914cd4631e95f7abc13ee78acec89c12129edbc1b1d353bb4634`) |
| Missing frozen model binary | Baseline / activity `.joblib` artifacts are gitignored and absent |
| Missing ML deps | `xgboost`, `sklearn`, `pandas`, `pyarrow`, `joblib` not installed; **must not** auto-install per program rules |
| Missing FRED key | `FRED_API_KEY` unset; ALFRED vintage API returns HTTP 400 without a key |
| Macro already marked unavailable in free-external acquire | Manifest: no rows; BLS ICS previously 403 from that environment |

No metrics were manufactured. Metadata/evaluation JSON alone are insufficient to train or score.

Local machine that already holds the frozen parquet + `.joblib` and a registered free FRED API key is the intended runtime.

---

## §1 Primary next experiment (highest unpaid directional EV)

### Experiment ID

`v0.3_forward_atr_xgb_baseline_v1_alfred_macro_pit_v1`

### Scientific question

Do **point-in-time initial macro releases** (ALFRED vintages + official release schedules), joined causally onto frozen US30 M15 bars, improve **BUY-vs-SELL** validation directional AUC beyond the frozen 42-feature control in a way that survives chronological CV and label permutation?

This is **not** a repeat of free-external rates/FX/VIX/calendar. Those families are retired. Macro initial releases were never acquired or fitted.

### Why this is highest EV among unpaid opens

- Inventory priority #1; introduces discrete information shocks, not another transform of prices.
- Free-external explicitly left macro at `UNAVAILABLE_RELIABLE_HISTORY` (0 rows).
- Asymmetry / OHLCV / US500-USTEC / microstructure / daily rates-VIX-calendar already negative for direction.

### Predeclared design (lock before any fit)

**Control**

- Frozen 42 features only.
- Same XGBoost hyperparameters as `v0.3_forward_atr_xgb_baseline_v1`.
- Directional metric: ROC-AUC on actual BUY/SELL rows only (`target ∈ {0,1}`).
- Reference control VAL direction AUC: **0.5129618201**.

**Treatment**

- Control features **plus** a small audited macro feature block only.
- Series candidates (approve only after vintage + release-time audit): CPI/core CPI, Employment Situation (payrolls, unemployment, AHE), PPI, retail sales, GDP, initial claims, FOMC decision/policy rate.
- Features (examples; finalize after acquisition QA):
  - hours since last eligible release (capped)
  - event-window flags (pre/post N hours around scheduled release; schedule known before T)
  - signed initial-release change vs prior initial release (revision-safe)
  - missingness / stale indicators
- **Forbidden:** consensus/forecast surprises unless a vendor proves pre-release snapshot history; current ALFRED “latest” values as if they were initial; midnight-of-observation as `available_time`.

**Causal join**

- Prediction time `T` = M15 bar open + 15 minutes.
- Use observation only if `available_time_utc <= T`.
- `available_time_utc` from official release schedule / documented publication instant, not FRED vintage calendar midnight alone.
- Max age per series declared in advance; beyond max age → missing.
- Forward-fill only already-public observations; retain age.

**Splits**

- TRAIN / VALIDATION only from frozen metadata.
- TEST sealed: zero rows, features, or labels.

**Evaluation gates (all required for `CREDIBLE_MACRO_DIRECTIONAL_SIGNAL`)**

1. Protected hashes unchanged (dataset, metadata, frozen baseline report/model if present).
2. Zero TEST exposure.
3. VAL direction AUC gain vs control ≥ **+0.02** absolute (predeclared; smaller moves are noise).
4. TRAIN-only chronological CV mean gain ≥ **+0.01** with same sign as VAL.
5. One-sided VAL label-shuffle permutation p ≤ **0.05** (n ≥ 200 shuffles).
6. Event-window subsample analysis reported separately; if all gain is outside event windows, treat as suspicious / likely non-macro.
7. If acquisition cannot produce audited initial releases for ≥ 3 core families covering TRAIN+VAL, stop with `MACRO_HISTORY_UNAVAILABLE` — do not substitute revised series.

**Negative / retire verdicts**

| Outcome | Verdict |
|---|---|
| Acquisition gates fail | `MACRO_HISTORY_UNAVAILABLE` |
| Fit runs but gates 3–5 fail | `RETIRE_INFORMATION_FAMILY` (macro PIT) |
| Leakage / hash / TEST touch | `EXPERIMENT_INVALID` |
| Deps/artifacts missing | `EXPERIMENT_INCONCLUSIVE` |

**Do not** combine macro with retired free-external families in the first fit. No hyperparameter search. No TEST peek.

### Local run checklist

```text
1. Confirm frozen parquet SHA-256 == 35995f3d673b914cd4631e95f7abc13ee78acec89c12129edbc1b1d353bb4634
2. export FRED_API_KEY=...   # free key; never commit
3. Acquire ALFRED vintages (output_type=4 where valid) + official schedules through VALIDATION last available time only
   last_allowed_available_time_utc = 2025-10-02T12:15:00+00:00
4. Persist under data/information_sources/alfred_macro_pit_v1/ with provenance + SHA-256
5. Build M15-aligned features with causal as-of join; write acquisition QA report
6. Fit control vs treatment XGB; VAL + TRAIN chronological CV + permutation
7. Write models/baselines/v0.3_forward_atr_xgb_baseline_v1_alfred_macro_pit_v1/experiment_report.md
8. If CREDIBLE: only then consider a predeclared combined follow-up; else retire macro PIT for this endpoint
```

Suggested acquisition probes (from `data_acquisition_plan.md`):

```text
fred/series/vintagedates?series_id=CPIAUCSL
fred/series/observations?series_id=CPIAUCSL&output_type=4&observation_start=2022-01-01
```

Map each series to an official release timestamp source before joining.

### Resource budget

- Chunk API pages; write raw batches before next request.
- Keep ≤ one aligned M15 frame in memory.
- Expected event count is small → report effective n in event windows; do not overclaim.

---

## §2 Controlled activity-filter next step (non-directional)

Use this **only if** abandoning directional recovery for now, or after ALFRED retires.

### Why not “just deploy the filter”

- Classification filter AUC ~0.67–0.70 is real (`FILTER_ONLY_SIGNAL`).
- Validation strategy probe was only `ACTIVITY_FILTER_WEAK_EVIDENCE`.
- Untouched confirmation on the sealed post-validation window was **`FAILED_CONFIRMATION`**.
- That confirmation partition is **spent** — do not retune thresholds on it.

### Experiment ID

`v0.3_forward_atr_xgb_baseline_v1_activity_filter_product_v1`

### Scientific question (explicitly non-directional)

Can a **frozen** activity probability `P(active) = P(BUY)+P(SELL)` improve **risk/participation metrics** versus unfiltered participation under predeclared costs and **without** using model sign?

Success metrics must **not** include BUY-vs-SELL AUC or directional accuracy.

### Allowed claims if gates pass

- “Filter reduces participation in low-activity regimes” / improves gross or cost-adjusted participation economics under stated assumptions.

### Forbidden claims

- Any recovery of directional edge.
- Reinterpretation of asymmetric one-vs-rest AUCs as direction.
- Threshold search on the burned confirmation window.

### Predeclared design sketch

1. **Freeze** the activity model and threshold set on TRAIN+VALIDATION **before** any new holdout (or use only VAL for a single predeclared threshold; no grid on future data).
2. Evaluate on a **new** forward window collected after the prior confirmation end (`> 2026-08-07T22:30:00`), or stop and wait — do not recycle TEST.
3. Direction rule: none from the ML model. If a mechanical rule is used, it is a separate factor; report factorial ablation (rule-only / filter-only / both).
4. Primary metrics: trade frequency, average absolute move / ATR, hit rate of |move|≥1 ATR, drawdown contribution when filtered out, and — only if reliable spread/slippage history exists — cost-adjusted EV. If costs unavailable, label outcomes `GROSS_ONLY` and do not invent costs.
5. Null: random filters matched for trade count; permutation of filter scores.
6. Pass rule: predeclared improvement on ≥2 primary metrics with permutation p ≤ 0.05, reproduced on the new window, with no directional metric in the decision function.

### Stop conditions

- No new untouched window → `WAIT_FOR_NEW_HOLDOUT` (do not reopen TEST).
- Filter helps activity ranking but not economics → keep research label `FILTER_ONLY_SIGNAL`; **no deployment**.
- Any analysis that optimizes directional PnL with model sign → out of scope / invalid for this brief.

---

## §3 What not to run next

- Another OHLCV indicator expansion
- US500/USTEC or HTF context rerun
- Free daily rates / VIX / FX / calendar directional rerun
- Asymmetric BUY/SELL on the same endpoint labels
- Neural showdown without an isolated env + memory budget already proven safe
- Paid news subscriptions before CME sample quote decision

---

## Cloud deliverable for this turn

| Artifact | Role |
|---|---|
| `DIRECTIONAL_RESEARCH_STATUS.md` | Frozen scientific status |
| `NEXT_EXPERIMENT_BRIEF.md` | This brief: blocked runtime + exact local next steps |

No fitted experiment report directory was created under `models/baselines/` because no honest run was possible.
