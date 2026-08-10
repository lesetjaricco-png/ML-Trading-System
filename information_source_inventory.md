# Causal Multi-Source Information Inventory

**Research date:** 2026-08-10  
**Phase:** discovery and acquisition design only  
**Target:** frozen US30 M15 information available at prediction time  
**Current data boundary:** TRAIN and VALIDATION only; TEST remains sealed

## Decision Summary

The previous experiments rule out another broad technical-indicator expansion. The existing MT5 US500/USTEC treatment produced validation directional AUC 0.4877, while M1 price/tick-volume/spread treatments remained near chance. New work should prioritize information that is not merely another transformation of US30 or correlated index OHLC:

1. Point-in-time macro releases and their market reactions.
2. Rates and volatility state, with publication times explicitly modeled.
3. Broker-consistent FX and commodity prices after symbol identity and coverage checks.
4. Genuine exchange trades/quotes/order-book data only after a sample and price quote.
5. Constituent-level breadth only if historical membership and live reproduction can be licensed.

The recommended free pilot is deliberately small: verified MT5 FX/commodities, official event schedules, ALFRED initial-release values where reconstructable, daily Treasury rates, official Cboe end-of-day volatility indices, and deterministic US session/calendar features. It does **not** claim that daily rates or event flags contain M15 direction; the experiment must determine that.

## Existing Controls and Facts

- The frozen 42-feature US30 baseline is unchanged.
- Its data source is IC Markets MT5, M15, UTC bar-open timestamps, 2022-05-12 onward.
- Coarse hour, weekday, Asia/London/New York session flags already exist. New temporal features must add exchange-calendar precision, holidays, early closes, and event timing rather than duplicate them.
- Verified MT5 US500 and USTEC M15 bars are available and already tested. Broader-market validation directional AUCs were 0.5059 (higher timeframe), 0.4877 (cross-market), and 0.4940 (combined).
- US30 MT5 M1 bars contain broker OHLC, tick volume, spread, and zero real volume. Historical tick and bid/ask queries returned no rows at multiple checkpoints.
- MT5 currently exposes live DOM metadata for US30 (`ticks_bookdepth=10`), but this does not create historical DOM.
- Broker VIX and DXY identities observed previously were current-expiry futures CFDs, not stable spot or continuous series. Both remain `UNAVAILABLE_RELIABLE_HISTORY` for this strict experiment.

## Source Comparison

Costs are public prices observed on 2026-08-10 or `QUOTE REQUIRED`. No purchase is authorized by this document.

| Source | Historical | Live / near-live | Resolution | Cost | API | Timestamp quality | Point-in-time safe | Status |
|---|---|---|---|---:|---|---|---|---|
| IC Markets MT5 | Broker chart history | Yes | Tick where retained; M1/M15 bars | Existing account | MetaTrader5 Python API | High for UTC market timestamps; broker retention varies | Yes after bar-close rule | **SELECTED FOR COVERAGE PROBES** |
| FRED / ALFRED | Long history and vintages | Updates after source publication | Daily to annual | Free | FRED REST | Date/vintage, generally not an intraday publication instant | Conditional | **SELECTED, SLOW DATA ONLY** |
| US Treasury | Daily par yields | Daily after publication | Daily close indication | Free | Official XML/CSV pages | Underlying quotes about 3:30 p.m. ET; publication lag must be logged | Yes only after observed publication | **SELECTED, SLOW DATA ONLY** |
| BLS | Published CPI/PPI/labor series | Official releases | Monthly/weekly; scheduled to minute | Free | BLS REST + release ICS | High for scheduled release time; API has published values, not consensus | Conditional | **SELECTED FOR SCHEDULES/INITIAL VALUES** |
| BEA / Census / Fed | Official GDP, retail, durable goods, FOMC releases | Official releases | Event-level | Free | Agency APIs/pages | Good schedules; revision handling differs by agency | Conditional | **SECOND OFFICIAL MACRO PHASE** |
| Trading Economics calendar | Historical actual/forecast/previous | Yes | Event-level | Enterprise only; quote required | REST/WebSocket | Intended for release timestamps and forecasts | Must verify forecast snapshots/revisions | **QUOTE BEFORE USE** |
| Finnhub calendar/news | Historical surprises only on Enterprise | Yes | Event/news/tick | $3,500/month, billed annually ($42,000/year) | REST/WebSocket | Event time and Unix news time | Conditional | **REJECT AT CURRENT COST** |
| Alpha Vantage | FX and news history | Near-live by entitlement | 1/5/15 min, article time | Free: 25 requests/day; intraday premium starts $49.99/month | REST | Market timestamps good; news publication semantics need audit | Conditional | **BACKUP ONLY** |
| Twelve Data | Multi-asset history | WebSocket/live by entitlement | 1 min+ | Basic free; Grow $79/month or $790/year | REST/WebSocket | Good market timestamps; source/delay varies | Conditional | **BACKUP AFTER MT5** |
| Tiingo | Equities/ETF history and IEX intraday | REST/WebSocket | 1 min+ / top of book | Current individual price not verified; check before use | REST/WebSocket | Strong; full IEX TOPS requires exchange agreement | Conditional | **ETF/BREADTH BACKUP** |
| Massive (Polygon) | US stock history | Yes by plan | Tick/second/minute | $29/month delayed aggregates; $79 delayed trades; $199 real-time trades/quotes | REST/WebSocket/flat files | Strong exchange timestamps | Conditional | **BACKUP; ENTITLEMENT REVIEW REQUIRED** |
| NewsAPI | Up to 5 years on paid plan | Real-time paid | Article | $449/month Business; $1,749/month Advanced | REST | UTC `publishedAt`; not guaranteed first-seen time | Conditional | **REJECT FOR FIRST PILOT** |
| GDELT 2.0 | 15-minute archives; very large | Updates every 15 minutes | 15 min aggregate/article | Free | DOC API/raw files/BigQuery | Processing time is usable; publisher time quality varies | Conditional | **SMALL PILOT ONLY** |
| Cboe public indices | Daily VIX-family history | Website; licensed live feeds | Daily public | Free historical | CSV downloads | Daily date only | Yes after close/publication | **SELECTED AS DAILY STATE** |
| Cboe DataShop | Options trades/quotes/summaries | Separate licensed live source | Tick/minute/daily | Quote/product pricing required | Download/API | Exchange-grade | Yes | **SAMPLE/QUOTE REQUIRED** |
| CME DataMine | Futures/options MBO, trades, quotes, settlements | Licensed CME feed/vendor | Tick/order event+ | Per-product quote/order | DataMine files/API | Exchange-grade | Yes | **MOST PROMISING PAID PILOT** |
| NYSE/Nasdaq proprietary feeds | Trades, quotes, depth, imbalance | Yes | Event/1-second | Quote and agreements required | Direct/vendor feed | Exchange-grade | Yes | **NOT FIRST PILOT** |
| NOAA/NWS | Long weather/climate history | Alerts/observations | Event/hour/day | Free | REST | Good observation/issue times | Yes | **LOW-PRIORITY NEGATIVE CONTROL** |

## Candidate Inventory

### 1. US30 Target Instrument

| Item | Assessment |
|---|---|
| Why directional | Control representation only; it defines incremental value. |
| Historical/live | Existing IC Markets MT5 M15 pipeline. |
| Point-in-time rule | A bar close is available at `bar_open + 15 minutes`. |
| Recommendation | Freeze all 42 features and hashes. Do not reacquire, alter, or enrich in this phase. |

### 2. Cross-Market Equity Indices

| Candidate | Historical / live source | Availability and resolution | Causal safety | Recommendation |
|---|---|---|---|---|
| US500, USTEC | Existing IC Markets MT5 | Verified M15 historical + live | High after close | Keep only as already-tested controls; do not repeat the same treatment. |
| US2000 / Russell | MT5 stable spot CFD if available; IWM ETF via MT5/Tiingo backup | Must probe identity and 2022-05 coverage | High after close | Probe. Add only if stable identity and full TRAIN/VALIDATION coverage. |
| Dow / US30 | Existing target | Verified | High | No duplicate source. |
| VIX | Cboe official daily history; paid licensed intraday alternative | Daily free, intraday licensed | Daily state safe after publication | Include daily lagged state; do not call it intraday VIX. |
| Cross-market breadth | Constituent bars plus point-in-time membership, or licensed exchange breadth | Not reliably public at M15 | Conditional | Defer unless vendor supplies historical and live parity with membership history. |

Prior negative results make additional US500/USTEC indicators low expected value. Russell may add small-cap/risk appetite information that was absent, but it is still synchronous price context and should be a low-cost ablation, not the headline experiment.

### 3. FX and USD

| Candidate | Source | Availability | Point-in-time safety | Recommendation |
|---|---|---|---|---|
| EURUSD, GBPUSD, USDJPY, USDCHF | IC Markets MT5 first; Twelve Data backup | Expected M15/live; exact coverage not yet approved/probed | High after source bar close | **Selected for MT5 coverage probe**. |
| DXY | Broker identity is expiry-based; ICE data is licensed | No verified continuous broker history | Unsafe to splice contracts silently | **UNAVAILABLE_RELIABLE_HISTORY**. |
| Synthetic dollar basket | Could be computed from FX | Reproducible but is not DXY | Safe if explicitly named | Do not substitute in strict DXY slot. A separate `fx_usd_breadth` feature may be tested and documented as a derived basket. |

Possible features are lagged returns, realized volatility, USD breadth, and US30/FX divergence. These are plausible because currency moves can encode rate and global-risk repricing before or alongside cash equities, but usefulness is unproven.

### 4. Interest Rates and Treasury Market

| Candidate | Historical source | Live source | Resolution | Point-in-time status | Recommendation |
|---|---|---|---|---|---|
| 2Y/5Y/10Y/30Y par yields | Treasury/FRED (`DGS2`, `DGS5`, `DGS10`, `DGS30`) | Same daily publication | Daily | Safe only after observed publication; never at midnight of observation date | Include lagged daily level/change/slope as regime context. |
| 2s10s | Derived from causally available 2Y and 10Y | Same | Daily | Safe after both legs available | Include. |
| Intraday rate reaction | CME ZT/ZF/ZN/ZB futures or BrokerTec | Licensed historical/live | Tick/M1/M15 | High | **Promising paid source; sample/quote first**. |

Daily yields cannot explain within-day M15 direction by changing every bar. Their role is regime conditioning. Intraday Treasury futures are more promising because equity direction frequently reflects real-time discount-rate repricing, but this needs licensed market data.

### 5. Commodities

| Candidate | Historical/live source | Availability | Recommendation |
|---|---|---|---|
| XAUUSD, XAGUSD | IC Markets MT5 | Expected broker M15 + live; probe required | Selected for coverage probe. |
| WTI, Brent | IC Markets MT5 stable spot/continuous identity if present | Identity/roll/coverage must be verified | Probe; reject expiry chains without explicit roll metadata. |

Gold and oil may encode real-rate, inflation, geopolitical, and risk sentiment shocks. Use only source returns/volatility and explicit relative moves; no additional generic indicator library.

### 6. Market Microstructure

| Data | IC Markets/MT5 finding | Classification | Recommendation |
|---|---|---|---|
| M1 OHLC, spread, tick volume | 1,200,410 verified rows through validation boundary | `BROKER TICK/BAR DATA` | Already tested; retain as negative evidence. |
| Real trade volume | Zero in verified US30 M1 history | Not available | Never treat tick volume as exchange volume. |
| Historical bid/ask ticks | Zero rows at multiple checkpoints | `UNAVAILABLE_RELIABLE_HISTORY` | Reject for historical treatment. |
| Live DOM | API supports subscription and US30 metadata reports depth 10 | `BROKER DOM`, not exchange L2 | May log prospectively, but no backtest until enough untouched history accumulates. |
| Exchange trades/L2/order flow | CME YM/MYM DataMine or licensed feed | `EXCHANGE TRADE/ORDER-BOOK DATA` | **Highest novelty, quote/sample required**. |

Genuine YM/MYM trade-sign imbalance, depth imbalance, cancellations, and cumulative delta could contain directional information because they observe aggressive trading and liquidity supply rather than only completed prices. This is the most plausible paid experiment, but only a small date-range sample should be purchased first to measure file size, schema, coverage, and aggregation feasibility.

### 7. Economic Calendar and Macro Events

| Component | Source | Point-in-time finding | Recommendation |
|---|---|---|---|
| Scheduled timestamps | BLS, BEA, Census, Federal Reserve official calendars | Public before event; times usually ET | Selected. Persist every calendar snapshot with acquisition time. |
| Actual/previous initial values | Agency releases + ALFRED `output_type=4` where supported | Vintages can reconstruct initial releases; exact intraday join needs official release schedule | Selected only after event-by-event audit. |
| Forecast/consensus | Trading Economics/Finnhub or specialist calendar vendor | Not supplied by official agencies/FRED | `UNAVAILABLE_RELIABLE_HISTORY` in the free plan. |
| Revisions | ALFRED/agency release archives | Available for many, not all series | Store initial, revised, and revision timestamps separately. |

Strict surprise features require a forecast snapshot that existed before release. A current database field called “forecast” is not enough unless the vendor proves snapshot history. Until then, omit `actual - forecast`; do not backfill it from current pages.

Selected event families: CPI/core CPI, PPI, Employment Situation (NFP, unemployment, average hourly earnings), GDP, retail sales, ISM manufacturing/services, PMI where licensing permits, initial claims, FOMC decisions/minutes, consumer confidence, durable goods, and housing. Agency-by-agency source validation is required before inclusion.

### 8. News and Sentiment

| Source | Strength | Limitation | Recommendation |
|---|---|---|---|
| GDELT | Free, global, 15-minute updates, tone/topics | Noisy; publisher timestamps and historical corpus are large | Run a bounded TRAIN-only pilot of counts/tone; no raw multi-terabyte download. |
| Alpha Vantage | Financial topic filters and sentiment, up to 1,000 results/request | Free limit 25/day; coverage/first-seen semantics need audit | Backup pilot only. |
| Finnhub | Real-time market/company news | Full long history and news sentiment are paid/Enterprise; data deletion/redistribution restrictions | Reject at current cost. |
| NewsAPI | Broad source search | $449/month for five-year live-capable plan; `publishedAt` is not guaranteed first-seen | Reject first pilot. |

Use headline count, source count, topic count, and precomputed source tone only after deduplication. Do not run local Transformers. A lightweight lexicon score may be compared with GDELT tone, but headline volume is the cleaner first feature.

### 9. Market Breadth

Desired NYSE/Nasdaq advances, declines, new highs/lows, percent above moving averages, and sector breadth are attractive because they observe the distribution under an index rather than its aggregate price. However, no reliable free M15 historical plus live API was verified. Exchange feeds and constituent reconstruction are licensed and operationally harder; reconstruction also needs point-in-time constituent membership to avoid survivorship bias.

**Status:** `UNAVAILABLE_RELIABLE_HISTORY` for the free strict experiment. Request vendor samples/pricing later. Do not scrape chart websites.

### 10. Volatility and Options

| Candidate | Availability | Cost | Recommendation |
|---|---|---:|---|
| VIX, VIX9D, VIX3M, VVIX | Official Cboe daily CSV history; licensed live/intraday | Free daily | Include one-day-lagged or post-publication daily state. |
| VIX term structure | Derived from causally available indices/futures | Depends on source | Daily free approximation acceptable if clearly defined; intraday requires licensed feed. |
| Put/call ratio | Cboe daily historical files where available | Free daily | Probe and include only with stable documented publication timing. |
| SPX/DJX options trades, IV, skew, OI | Cboe DataShop/vendor | Quote required | Sample/quote after free pilot. |

Options information can be forward-looking, but daily end-of-day files cannot be injected into earlier bars. Intraday skew and dealer-position proxies are potentially valuable but expensive and methodologically complex.

### 11. Session and Liquidity Regime

Use official NYSE/Nasdaq calendars and `America/New_York` timezone rules for cash open/close, early close, pre/post-market, holiday adjacency, and minutes since/until session boundaries. Opening range, session return, and session high/low must use only bars completed by prediction time. Known future calendar facts such as a scheduled close are allowed; future prices are not.

### 12. Calendar and Seasonal Information

Add week-of-month, month, quarter, month/quarter/year-end flags, official options/futures expiry dates, and known macro-event week flags. These are low-cost controls, not presumed alpha. Fit any normalization or target encoding inside each chronological training fold only.

### 13. Weather

NOAA/NWS offers free historical observations and live alerts. Plausible channels are hurricanes, severe storms, energy disruption, and broad operational shocks, not ordinary temperature. A small event flag can serve as an intentionally low-priority alternative-information ablation.

**Recommendation:** reject routine weather features. Consider only national severe-weather/hurricane flags after the core information set, with zero paid spend.

## Leakage and Timestamp Contract

For prediction instant `T`:

- Market bar features require `bar_close_time <= T`.
- Tick/quote/order-book events require exchange or broker event time `<= T` and a recorded receive time in live operation.
- Scheduled-event flags may use the schedule snapshot acquired before `T`.
- Event actuals require `release_time <= T`; forecast requires a snapshot captured before release.
- Daily Treasury/Cboe values become available at observed publication time, never at the date label's midnight.
- A backward as-of join must specify a source-specific maximum age. Beyond it, the feature is missing.
- Forward filling is permitted only from an observation already public, with `source_time`, `available_time`, and `age_seconds` retained.
- Revised values are separate observations. Historical rows use the value vintage available at `T`.
- All rolling statistics are computed after causal alignment and fit only on past/current rows.

## Ranked Priority

| Priority | Information | Expected usefulness | Historical source | Live source | Cost | Difficulty | Causal safety |
|---:|---|---|---|---|---:|---|---|
| 1 | Point-in-time macro schedule + initial releases | High around event windows; genuinely new | BLS/BEA/Census/Fed + ALFRED | Official agencies | Free without consensus | High | Medium; event-specific audit required |
| 2 | MT5 FX + gold/oil + Russell coverage-approved bars | Medium; new cross-asset channels, easy ablations | IC Markets MT5 | Same MT5 | Existing | Low-Medium | High after bar close |
| 3 | Intraday Treasury futures | High if equity direction is rate-shock driven | CME DataMine/vendor | CME licensed feed/vendor | Quote required | Medium | High |
| 4 | YM/MYM exchange trades and order book | High novelty; observes pressure/liquidity directly | CME DataMine MBO/trades/quotes | CME licensed feed/vendor | Quote required | High | High |
| 5 | Cboe volatility state/term structure | Medium regime information | Cboe daily; licensed intraday | Cboe/vendor | Free daily; quote intraday | Low-High | High if publication lagged |
| 6 | Market breadth | Medium-High in theory; distributed equity information | Licensed exchange/vendor or survivorship-safe reconstruction | Same | Quote required | High | Medium-High |
| 7 | GDELT headline volume/tone | Low-Medium; event shock proxy | GDELT | GDELT 15-min updates | Free | Medium | Medium |
| 8 | Detailed session/calendar | Low-Medium control value | Official exchange calendars | Deterministic | Free | Low | High |
| 9 | Options skew/put-call/order flow | Potentially high but expensive/complex | Cboe DataShop/vendor | Licensed feed | Quote required | High | High |
| 10 | Weather extremes | Low; negative-control value | NOAA | NWS/NOAA | Free | Low | High |

## Smallest High-Value Information Set

Implement first, subject to the probes in `data_acquisition_plan.md`:

1. MT5 EURUSD, GBPUSD, USDJPY, USDCHF, XAUUSD, XAGUSD, stable WTI/Brent identities, and US2000/IWM where full coverage exists.
2. Official macro event timestamps plus causally reconstructed initial values for a small core: CPI/core CPI, Employment Situation, PPI, retail sales, GDP, initial claims, and FOMC decisions.
3. Daily 2Y/5Y/10Y/30Y Treasury state and 2s10s, available only after official publication.
4. Daily VIX/VIX9D/VIX3M/VVIX state, available only after close/publication.
5. Exact NYSE session, holiday, early-close, month/quarter-end, and event-window features.

This set is free, compact, reproducible live, and materially different from adding indicators. The particularly promising component is **macro release information**, because it introduces discrete information shocks that can resolve sign, unlike another lag of a price series. Its weakness is small event sample size and the absence of free historical consensus. The most promising paid follow-up is a **small CME YM/MYM plus Treasury-futures order-flow sample**, requested only after the free pilot and only after written pricing/licensing review.

## Primary Documentation

- MetaTrader5 Python: <https://www.mql5.com/en/docs/python_metatrader5>
- FRED/ALFRED API: <https://fred.stlouisfed.org/docs/api/fred/>
- US Treasury interest rates: <https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics>
- BLS API and release calendar: <https://www.bls.gov/developers/> and <https://www.bls.gov/schedule/news_release/>
- NYSE hours/calendars: <https://www.nyse.com/markets/hours-calendars>
- Cboe VIX history/DataShop: <https://www.cboe.com/tradable_products/vix/vix_historical_data/> and <https://datashop.cboe.com/>
- CME DataMine: <https://www.cmegroup.com/market-data/datamine-historical-data.html>
- GDELT: <https://www.gdeltproject.org/data.html>
- NOAA/NWS APIs: <https://www.ncei.noaa.gov/cdo-web/webservices/v2> and <https://www.weather.gov/documentation/services-web-api>
- Vendor terms/pricing must be rechecked immediately before any paid action.