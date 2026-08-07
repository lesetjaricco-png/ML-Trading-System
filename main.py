"""ML Trading System — main pipeline entry point.

Usage
-----
    python main.py                          # run full pipeline with config.yaml
    python main.py --config my_config.yaml  # custom config
    python main.py --ticker MSFT --start 2020-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import logging

from src.backtesting import BacktestEngine
from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineer
from src.model import XGBoostModel
from src.risk_management import RiskManager
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost ML Trading System")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def run_pipeline(cfg: dict, use_cache: bool = True) -> dict:
    """Execute the full training + backtest pipeline.

    Parameters
    ----------
    cfg:
        Configuration dictionary (see ``config/config.yaml``).
    use_cache:
        Pass *False* to force re-download of market data.

    Returns
    -------
    dict
        Performance metrics from the backtest.
    """
    # ------------------------------------------------------------------ #
    # 1. Data ingestion                                                    #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 1: Data Ingestion ===")
    ingestion = DataIngestion(data_dir="data/raw")
    df_raw = ingestion.fetch(
        ticker=cfg["data"]["ticker"],
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
        interval=cfg["data"]["interval"],
        use_cache=use_cache,
    )
    logger.info("Downloaded %d rows for %s", len(df_raw), cfg["data"]["ticker"])

    # ------------------------------------------------------------------ #
    # 2. Feature engineering                                               #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 2: Feature Engineering ===")
    fe = FeatureEngineer(
        rsi_period=cfg["features"]["rsi_period"],
        macd_fast=cfg["features"]["macd_fast"],
        macd_slow=cfg["features"]["macd_slow"],
        macd_signal=cfg["features"]["macd_signal"],
        bb_period=cfg["features"]["bb_period"],
        bb_std=cfg["features"]["bb_std"],
        sma_periods=cfg["features"]["sma_periods"],
        ema_periods=cfg["features"]["ema_periods"],
        atr_period=cfg["features"]["atr_period"],
        volume_sma_period=cfg["features"]["volume_sma_period"],
        lookahead_days=cfg["signal"]["lookahead_days"],
        return_threshold=cfg["signal"]["return_threshold"],
    )
    df_features = fe.transform(df_raw)
    logger.info("Feature matrix: %s", df_features.shape)

    # ------------------------------------------------------------------ #
    # 3. Model training                                                    #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 3: Model Training ===")
    model = XGBoostModel(
        n_estimators=cfg["model"]["n_estimators"],
        max_depth=cfg["model"]["max_depth"],
        learning_rate=cfg["model"]["learning_rate"],
        subsample=cfg["model"]["subsample"],
        colsample_bytree=cfg["model"]["colsample_bytree"],
        min_child_weight=cfg["model"]["min_child_weight"],
        gamma=cfg["model"]["gamma"],
        reg_alpha=cfg["model"]["reg_alpha"],
        reg_lambda=cfg["model"]["reg_lambda"],
        random_state=cfg["model"]["random_state"],
        eval_metric=cfg["model"]["eval_metric"],
        early_stopping_rounds=cfg["model"]["early_stopping_rounds"],
        prediction_threshold=cfg["signal"]["prediction_threshold"],
    )
    metrics = model.train(
        df_features,
        feature_columns=fe.feature_columns,
        test_size=cfg["model"]["test_size"],
        validation_size=cfg["model"]["validation_size"],
    )
    logger.info("Training metrics: %s", metrics)
    model.save()

    # ------------------------------------------------------------------ #
    # 4. Signal generation                                                 #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 4: Signal Generation ===")
    df_signals = model.generate_signals(df_features)
    buy_signals = df_signals["signal"].sum()
    logger.info("Total BUY signals: %d / %d bars", buy_signals, len(df_signals))

    # ------------------------------------------------------------------ #
    # 5. Backtesting                                                       #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 5: Backtesting ===")
    risk = RiskManager(
        max_position_size=cfg["risk"]["max_position_size"],
        stop_loss=cfg["risk"]["stop_loss"],
        take_profit=cfg["risk"]["take_profit"],
        max_drawdown=cfg["risk"]["max_drawdown"],
        risk_per_trade=cfg["risk"]["risk_per_trade"],
    )
    engine = BacktestEngine(
        initial_capital=cfg["backtest"]["initial_capital"],
        commission=cfg["backtest"]["commission"],
        slippage=cfg["backtest"]["slippage"],
        risk_manager=risk,
    )
    equity = engine.run(df_signals)
    perf = engine.performance_metrics(equity)

    logger.info("=== Backtest Results ===")
    for k, v in perf.items():
        logger.info("  %-30s %s", k, v)

    return perf


def main() -> None:
    setup_logging(log_file="trading_system.log")
    args = parse_args()
    cfg = load_config(args.config)

    if args.ticker:
        cfg["data"]["ticker"] = args.ticker
    if args.start:
        cfg["data"]["start_date"] = args.start
    if args.end:
        cfg["data"]["end_date"] = args.end

    run_pipeline(cfg, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
