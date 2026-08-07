# ML Trading System

An **XGBoost-based** Intelligent Trading Decisions Engine that predicts buy signals from historical OHLCV data, manages risk, and back-tests strategies.

---

## Architecture

```
ML-Trading-System/
├── config/
│   └── config.yaml          # All hyper-parameters and settings
├── src/
│   ├── data_ingestion.py    # Download / cache OHLCV data (yfinance)
│   ├── feature_engineering.py  # Technical indicator features + target labels
│   ├── model.py             # XGBClassifier wrapper (train / predict / persist)
│   ├── risk_management.py   # Position sizing, stop-loss, take-profit, drawdown guard
│   ├── backtesting.py       # Event-driven backtest engine + performance metrics
│   └── utils.py             # Config loading, logging helpers
├── tests/                   # pytest test suite (36 tests)
├── data/                    # Raw & processed data (auto-created)
├── models/                  # Saved model artefacts (auto-created)
├── logs/                    # Log files (auto-created)
├── main.py                  # Pipeline entry point
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (downloads AAPL 2018-2024 by default)
python main.py

# 3. Use a custom ticker / date range
python main.py --ticker MSFT --start 2020-01-01 --end 2024-01-01

# 4. Run tests
pytest tests/ -v
```

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `DataIngestion` | Downloads OHLCV data via yfinance and caches as Parquet |
| 2 | `FeatureEngineer` | Computes RSI, MACD, Bollinger Bands, ATR, SMAs, EMAs, OBV, candlestick features; generates binary target labels |
| 3 | `XGBoostModel` | Time-series split training with early stopping; outputs accuracy, precision, recall, F1, ROC-AUC |
| 4 | Signal generation | Model predicts buy-probability per bar; thresholds to BUY/NO-BUY |
| 5 | `BacktestEngine` | Simulates long-only trading with commission & slippage; computes Sharpe, Sortino, Calmar, win-rate, profit factor |

---

## Configuration

Edit `config/config.yaml` to change any parameter:

```yaml
data:
  ticker: "AAPL"
  start_date: "2018-01-01"
  end_date:   "2024-01-01"

model:
  n_estimators: 500
  learning_rate: 0.05
  ...

risk:
  stop_loss: 0.02
  take_profit: 0.04
  max_drawdown: 0.15
```

---

## Features

- **25+ technical indicators**: RSI, MACD, Bollinger Bands, ATR, SMA/EMA, OBV, volume ratio, candlestick patterns
- **XGBoost classifier** with early stopping and walk-forward cross-validation
- **ATR-based position sizing** with maximum drawdown circuit-breaker
- **Event-driven backtester** supporting stop-loss, take-profit, and end-of-bar exits
- **Full performance report**: total/annual return, Sharpe/Sortino/Calmar ratios, win rate, profit factor
