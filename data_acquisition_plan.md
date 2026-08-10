# Causal Multi-Source Data Acquisition Plan

**Research date:** 2026-08-10  
**Authorization:** design and read-only probes only  
**Forbidden in this phase:** model training, TEST access, baseline/artifact modification, paid purchase, synthetic substitution, and heavyweight infrastructure

## Scope and Stop Conditions

The acquisition boundary is the last frozen VALIDATION prediction instant. Every script must derive that boundary from frozen metadata and fail if a requested or returned timestamp enters TEST. Do not hard-code a later date merely because a provider makes it available.

Stop a source before acquisition if any of these fails:

1. Stable instrument/series identity.
2. Historical coverage across the required TRAIN and VALIDATION interval.
3. A live or near-live path using the same definition.
4. Explicit source and availability timestamps.
5. Licensing suitable for private research and eventual personal live trading.
6. Incremental retrieval without loading a full archive into RAM.
7. No overwrite of an existing raw version.

## Lightweight Architecture

```text
provider or MT5
  -> metadata/coverage probe
  -> immutable raw batches (JSON/CSV/parquet)
  -> provenance sidecar + SHA-256
  -> UTC normalization with source_time and available_time
  -> causal backward as-of alignment
  -> shared historical/live feature functions
  -> compact M15 feature parquet
```

Use pandas, requests/standard-library HTTP, pyarrow already present, and small parquet files. No database, Docker, message broker, local language model, or GPU service is required.

## Directory and File Contract

Proposed new versioned paths, created only in the implementation phase:

```text
data/information_sources/<source>/<dataset_version>/raw/
data/information_sources/<source>/<dataset_version>/normalized/
data/information_sources/<source>/<dataset_version>/provenance.json
data/information_sources/manifests/<experiment_version>.json
```

Each raw filename includes source, instrument/series, UTC start/end, resolution, and acquisition batch. Files are immutable. A changed response creates a new version; it never overwrites prior data.

Every provenance record must contain:

- source, provider, endpoint/URL, source version, and licensing notes
- acquisition UTC timestamp and API parameters with secrets redacted
- instrument/series identity, timeframe, timezone, timestamp semantics
- requested and actual date range, row count, columns, batches
- missing expected rows, missing values, duplicate timestamps
- minimum/maximum `source_time` and `available_time`
- data and metadata SHA-256
- revision/vintage policy and forward-fill policy
- historical/live definition identifier
- TEST-boundary assertion result

## Phase 0: Read-Only Preconditions

### Frozen boundary and hashes

Do not read target values from TEST. Read only split metadata and protected file hashes using the repository's existing helper patterns.

```powershell
& ".venv\Scripts\python.exe" -c "import json; from pathlib import Path; p=Path('data/processed/US30_2022-05-12_2026-08-09_M15_v0.3_forward_atr_v1.metadata.json'); m=json.loads(p.read_text()); print(json.dumps(m.get('splits', m), indent=2))"
```

Implementation code must fail closed unless TRAIN/VALIDATION boundaries and protected hashes match the frozen baseline report.

### Resource budget

- Process one instrument and one date chunk at a time.
- Default market-data chunk: 30-45 days.
- Default API page: <= 10,000 rows or provider maximum.
- Write each batch before requesting the next.
- Keep no more than one raw batch and one aligned M15 frame in memory.
- Aggregate tick/order-book data to one-minute or M15 features during a bounded sample; do not retain multi-year data in RAM.

## Phase 1: Free Selected Pilot

### A. MT5 symbol identity and coverage probes

Probe metadata first. Do not accept symbol-name resemblance as identity proof.

```python
import MetaTrader5 as mt5

assert mt5.initialize(), mt5.last_error()
try:
    for query in (
        "*EURUSD*", "*GBPUSD*", "*USDJPY*", "*USDCHF*",
        "*XAUUSD*", "*XAGUSD*", "*US2000*", "*IWM*",
        "*BRENT*", "*WTI*", "*XTI*", "*XBR*",
    ):
        for item in mt5.symbols_get(group=query) or ():
            print(item.name, item.description, item.path, item.expiration_time)
finally:
    mt5.shutdown()
```

For each candidate passing identity review, run a count-only M15 probe at three points: TRAIN start, middle, and VALIDATION end. Then request at most one week to verify schema and UTC semantics.

```python
rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start_utc, end_utc)
# Inspect only len(rates), first/last time, dtype names, and MT5 last_error.
```

Acceptance:

- spot or explicitly continuous identity; no silent expiring contract chain
- first usable bar no later than seven days after TRAIN start
- last usable closed bar covers VALIDATION end
- same named symbol available live
- UTC bar-open timestamp and fields documented
- no fallback provider

Reject DXY unless a stable licensed index identity passes all checks. Otherwise record exactly `UNAVAILABLE_RELIABLE_HISTORY`. Do not build an undocumented proxy.

Acquisition, after approval, should copy the existing `acquire_directional_context_data.py` pattern: 30-45 day chunks, M15 bars, temporary file then atomic rename, provenance and hashes, no TEST timestamps, and refusal to overwrite.

### B. Official macro schedule snapshots

Start with the BLS ICS feed and official Fed/BEA/Census calendars. The exact first request is:

```powershell
Invoke-WebRequest -Uri "https://www.bls.gov/schedule/news_release/bls.ics" -OutFile "$env:TEMP\bls.ics"
Get-FileHash "$env:TEMP\bls.ics" -Algorithm SHA256
```

Equivalent official endpoints/pages to inventory before implementation:

- BLS: `https://www.bls.gov/schedule/news_release/bls.ics`
- Federal Reserve FOMC calendars: `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm`
- BEA release schedule/API: `https://apps.bea.gov/api/` and official release schedule
- Census economic indicators: `https://www.census.gov/economic-indicators/calendar-listview.html`
- Department of Labor claims: official weekly release/archive

Persist the raw schedule snapshot and `acquired_at_utc`. Calendar changes are new snapshots. Normalize to:

```text
event_id, event_family, release_name, country, importance_source,
scheduled_time_utc, schedule_snapshot_time_utc, source_url
```

Historical schedule acceptance requires archived official timestamps or an agency archive. If a historical time cannot be established, do not infer it from the observation date.

### C. Initial macro values and vintages

Use ALFRED/FRED only for series where initial-release reconstruction is verified. Register a free FRED API key and keep it in `FRED_API_KEY`; never commit it.

Example discovery calls:

```powershell
$key = $env:FRED_API_KEY
Invoke-RestMethod "https://api.stlouisfed.org/fred/series/vintagedates?series_id=CPIAUCSL&api_key=$key&file_type=json"
Invoke-RestMethod "https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key=$key&file_type=json&output_type=4&observation_start=2022-01-01"
```

Candidate series must be mapped and independently checked against agency releases. Typical discovery identifiers include CPI/headline and core CPI, payroll employment, unemployment, average hourly earnings, PPI, retail sales, GDP, claims, and policy rates; identifiers are not approved merely because they exist in FRED.

Store:

```text
series_id, observation_period, value, vintage_date,
release_time_utc, available_time_utc, revision_number, source_url
```

`available_time_utc` comes from the official release, not the FRED vintage date at midnight. Consensus/forecast is omitted unless a vendor later proves pre-release snapshot history.

### D. Treasury yields

Use official Treasury daily par yield curve files/pages or FRED series `DGS2`, `DGS5`, `DGS10`, and `DGS30` as a convenience mirror. Treasury states the underlying indicative quotes are obtained at approximately 3:30 p.m. ET.

Discovery request:

```powershell
Invoke-WebRequest -Uri "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml" -Method Head
```

Because Treasury endpoint formats have changed, implementation must resolve the current official XML/CSV link from the Interest Rate Statistics page rather than rely on an old hard-coded path.

Normalize levels, one-day changes, rolling past-only volatility, and `2Y - 10Y`. Set `available_time_utc` to a conservative observed publication time. Before that time on date `D`, use the latest already-published prior value. Record age in hours.

### E. Cboe daily volatility indices

Probe official Cboe CSV downloads for `VIX`, `VIX9D`, `VIX3M`, and `VVIX`. Known official URL pattern to verify before acquisition:

```text
https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv
https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv
https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv
```

```powershell
Invoke-WebRequest -Uri "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv" -OutFile "$env:TEMP\VIX_History.csv"
Get-FileHash "$env:TEMP\VIX_History.csv" -Algorithm SHA256
```

Acceptance requires official Cboe host, documented columns, complete date coverage, and a conservative availability time. Public daily files are daily state only. Never forward-fill date `D`'s close into date `D` intraday predictions before publication.

### F. Session and calendar features

Source official NYSE/Nasdaq holiday and early-close calendars. Convert all rules through `zoneinfo.ZoneInfo("America/New_York")`; never apply a fixed UTC offset across daylight-saving changes.

Definitions:

- cash core: 09:30-16:00 ET, adjusted for official early close
- pre-market/post-market: explicit exchange definitions
- minutes since/until open/close: based on that day's official schedule
- holiday adjacency: based on actual prior/next trading day
- month/quarter/year end: final official trading session, not final calendar day
- opening range/session high/low/return: completed bars only

Options/futures expiry and FOMC/CPI/NFP week flags require official calendars stored as snapshots. Do not encode folklore rules when an official date differs.

## Phase 2: Optional News Pilot

GDELT is the only selected no-cost pilot. Query bounded windows and aggregate remotely; do not download raw yearly archives (GDELT notes that a year can exceed 2.5 TB).

Example 15-minute/short-window query pattern:

```text
https://api.gdeltproject.org/api/v2/doc/doc?query=(stocks OR "Federal Reserve" OR inflation)&mode=timelinevolraw&format=json&startdatetime=YYYYMMDDHHMMSS&enddatetime=YYYYMMDDHHMMSS
```

Acquire one TRAIN-only month first. Measure duplicates, coverage, timestamp lag, source composition, and query stability. Candidate outputs are count, normalized volume, average tone, negative-tail share, and topic counts over the previous 15/30/60 minutes.

Reject the source if rerunning a fixed historical query materially rewrites results without version information, if timestamps cannot be interpreted causally, or if coverage gaps cluster around important events.

Do not call NewsAPI in live/research production on its free Developer plan; it is development-only, delayed 24 hours, and limited to one month. Do not start its $449/month plan without separate approval. Alpha Vantage's 25-request/day free allowance does not include its premium intraday FX endpoint; its premium plans currently start at $49.99/month.

## Phase 3: Paid Sources, Quote and Sample Only

No purchase is authorized.

### A. CME YM/MYM and Treasury futures

Request a written quote and a free/sample day for:

- YM and MYM trades and best bid/offer; MBO only if affordable
- ZT, ZF, ZN, and ZB trades and best bid/offer
- 2022-05 through VALIDATION end historical coverage
- matching live non-display personal-use entitlement
- timestamps, sequence numbers, aggressor/trade conditions, depth schema
- retention and derived-data rights

DataMine catalog: `https://datamine.new.cmegroup.com/catalog`

Ask for costs separately:

```text
historical one-time cost
live monthly exchange fee
vendor/platform monthly fee
non-display/personal algorithmic use fee
derived data/storage restrictions
```

Sample acceptance:

- exchange event timestamps and sequence are monotonic/reconcilable
- trades and quotes cover regular and overnight sessions
- contract roll can be handled explicitly by volume/open-interest rule fixed in advance
- one day can be streamed and aggregated under the RAM limit
- live feed reproduces identical event fields

First sample features: signed trade imbalance, cumulative delta, top-of-book imbalance, spread, depth depletion, cancellation intensity, and order-flow acceleration over past-only 1/5/15-minute windows. Label them `EXCHANGE_DATA`; keep MT5 fields labeled `BROKER_DATA`.

### B. Economic forecast history

Request sample rows and quote from Trading Economics or another calendar vendor. Require:

- actual, forecast, previous, revised previous
- event and release timestamp with timezone
- forecast snapshot timestamp or proof that historical forecast is the pre-release consensus
- revision/vintage history
- historical depth through 2022-05
- same live event identifier and schema
- private algorithmic-use and retention rights

Reject any feed that exposes only the latest revised values or cannot prove when a forecast was known. Finnhub's currently published Enterprise price is $3,500/month billed annually and is rejected at this phase.

### C. Breadth and options

Ask Cboe/NYSE/Nasdaq or a vendor for a sample and quote for intraday advances/declines, highs/lows, constituent breadth, VIX term structure, put/call, skew, and options trades/open interest. Require point-in-time index membership and live parity. Do not reconstruct historical breadth from today's constituents.

## Shared Normalization Schema

All source adapters emit a long-form normalized table before features:

```text
source, dataset_version, instrument_or_event_id, field,
value, source_time_utc, available_time_utc, received_time_utc,
revision_id, quality_flags, raw_batch_sha256
```

Market bars additionally emit `bar_open_utc` and `bar_close_utc`. Events additionally emit `scheduled_time_utc`, `release_time_utc`, `forecast_snapshot_time_utc`, and revision fields where present.

Historical and live adapters call the same pure feature functions. Live replay tests feed historical normalized rows in timestamp order and compare vectors byte-for-byte or within declared floating tolerance.

## Causal Alignment Procedure

For each US30 prediction timestamp `T`:

1. Filter source rows to `available_time_utc <= T`.
2. For bar closes, also require `bar_close_utc <= T`.
3. Select the latest eligible row by `available_time_utc`, not observation-period label.
4. Apply the declared maximum source age; otherwise leave missing.
5. Record selected `source_time_utc`, `available_time_utc`, and age.
6. Compute rolling features from the already-aligned historical sequence only.
7. Assert `max_available_time_utc <= T` for every output row.

Missing values remain missing or use a fixed missing indicator. No backfill. Forward fill only an already-public observation and retain its age.

## Validation Before Any Modeling

Acquisition is complete only when these data tests pass:

- schema, identity, timezone, monotonicity, duplicate and null checks
- expected-session coverage and gap report
- raw/provenance hash verification
- no returned timestamp beyond the sealed boundary
- no feature source available after prediction time
- stale observations rejected at source-specific limits
- DST, holidays, early closes, and event-release boundary tests
- revised macro value unavailable before revision timestamp
- source bar close unavailable before close
- historical/live replay feature equivalence
- artifact overwrite refusal

Recommended focused command once acquisition tests exist:

```powershell
& ".venv\Scripts\python.exe" -m pytest -q tests/test_information_source_acquisition.py tests/test_causal_information_alignment.py
```

Do not run model training from an acquisition script.

## Later Experiment Matrix

Only after the feature store is frozen:

| Treatment | Increment over frozen 42 features |
|---|---|
| Control | None |
| A | New equity/Russell context; existing US500/USTEC retained as controls |
| B | FX/USD breadth |
| C | Rates |
| D | Commodities |
| E | Macro calendar/initial releases |
| F | News volume/tone |
| G | Broker or exchange microstructure, never mixed without source labels |
| H | Breadth/volatility |
| I | All families that independently passed acquisition and causality gates |

Use existing XGBoost first. Preserve the sealed TEST set. Report validation and chronological CV directional AUC, balanced accuracy, macro F1, precision/recall, calibration, folds, permutation importance, family ablation, label shuffle, and threshold robustness. Predeclare a material improvement threshold before fitting; an AUC movement such as 0.500 to 0.507 is not evidence.

## Implementation Order

1. Build a read-only MT5 identity/coverage audit; approve exact symbols.
2. Implement common provenance and normalized timestamp schema.
3. Acquire selected MT5 M15 families through VALIDATION only.
4. Add official session calendars and macro schedule snapshots.
5. Add audited ALFRED initial values, daily Treasury rates, and Cboe daily volatility state.
6. Freeze and audit the feature store; do not train yet.
7. Optionally run a one-month GDELT feasibility pilot.
8. Request CME and calendar-vendor samples/quotes only if the free store passes all gates.

The first paid decision should be CME order flow, not a broad news subscription: it is the largest change in information type and has a clearer historical/live equivalence path. Purchase only after sample-size, file-size, licensing, and aggregation checks are documented.