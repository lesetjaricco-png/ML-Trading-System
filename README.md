# ML Trading System

XGBoost-based trading and backtesting pipeline with data ingestion, feature engineering, model training, signal generation, and event-driven backtesting.

## Baseline Version

This repository is prepared as a reproducible baseline:

- Version label: v0.1-baseline
- Baseline metrics (local reference):
  - Total trades: 263
  - Win rate: 40.3%
  - Profit factor: 0.47
  - Average win: 26.45
  - Average loss: -38.08
  - Max drawdown: 5.74%
  - Calmar ratio: -0.23
  - Final portfolio value: 94,290.62

## Project Layout

```
ML-Trading-System/
├── config/
│   └── config.yaml
├── data/
│   └── raw/
├── logs/
├── models/
├── src/
├── tests/
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

## Important MT5 Note

MetaTrader5 live terminal integration is platform-specific and usually not available inside a generic Linux Docker container.

For reproducible Docker baseline runs, this project uses the cached MT5 dataset file:

- data/raw/US30_2026-01-01_2026-08-08_15m_mt5.parquet

As long as this file is present, `main.py` runs in Docker without requiring live MT5 terminal connectivity.

## Install Docker

1. Install Docker Desktop from https://docs.docker.com/desktop/
2. Open a terminal in the repository root.
3. Verify installation:

```bash
docker --version
docker compose version
```

## Docker Quick Start

Build image:

```bash
docker compose build
```

Run baseline backtest pipeline:

```bash
docker compose run --rm backtest
```

Run tests inside Docker:

```bash
docker compose run --rm tests
```

Run the Docker smoke test locally:

```bash
pytest tests/test_docker_smoke.py -v
```

## What Persists Between Runs

`docker-compose.yml` mounts host directories into the container:

- `./config` -> `/app/config`
- `./data` -> `/app/data`
- `./models` -> `/app/models`
- `./logs` -> `/app/logs`

This keeps configuration, cached data, trained models, and logs on your machine after container exit.

## Rebuild After Code Changes

```bash
docker compose build
docker compose run --rm backtest
```

Use this whenever dependencies or Docker-related files change.

## Run Without Docker (Optional)

```bash
pip install -r requirements.txt
python main.py
pytest tests/ -v
```

## Baseline Verification Guidance

After running in Docker, compare key metrics against baseline:

- Trades: about 263
- Win rate: about 40.3%
- Profit factor: about 0.47
- Max drawdown: about 5.74%
- Calmar: about -0.23
- Final portfolio value: about 94,290.62

Small differences can occur due to platform/library numerical differences, but major deviations should be investigated.
