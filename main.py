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
from datetime import datetime

import pandas as pd

from src.backtesting import BacktestEngine
from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineer
from src.instruments import resolve_instrument_spec
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


def compute_split_indices(n: int, test_size: float = 0.2, validation_size: float = 0.1) -> tuple[int, int]:
    """Return the validation and test split indices for a chronological split."""
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - validation_size))
    return val_idx, test_idx


def select_backtest_subset(df: pd.DataFrame, test_size: float = 0.2, validation_size: float = 0.1) -> pd.DataFrame:
    """Return the holdout subset that corresponds to the model test window."""
    n = len(df)
    if n == 0:
        return df.copy()

    _, test_idx = compute_split_indices(n, test_size=test_size, validation_size=validation_size)
    return df.iloc[test_idx:].copy()


def build_diagnostic_report(
    *,
    df_raw: pd.DataFrame,
    df_features: pd.DataFrame,
    split_summary: dict,
    df_signals: pd.DataFrame,
    equity: pd.DataFrame,
    perf: dict,
) -> dict:
    """Build a concise diagnostic summary for the pipeline run."""
    return {
        "data": {
            "rows": len(df_raw),
            "start": df_raw.index[0] if len(df_raw) else None,
            "end": df_raw.index[-1] if len(df_raw) else None,
        },
        "features": {
            "feature_rows": len(df_features),
            "target_counts": df_features["target"].value_counts(dropna=False).to_dict() if "target" in df_features.columns else {},
            "split_summary": split_summary,
        },
        "signals": {
            "buy_signals": int(df_signals["signal"].sum()) if "signal" in df_signals.columns else 0,
            "buy_signal_ratio": round(float(df_signals["signal"].mean()) if "signal" in df_signals.columns else 0.0, 4),
        },
        "backtest": {
            "rows": len(equity),
            "total_trades": perf.get("total_trades", 0),
            "final_portfolio_value": perf.get("final_portfolio_value", None),
            "max_drawdown": perf.get("max_drawdown", None),
        },
    }


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
        source=cfg["data"].get("source", "mt5"),
        use_cache=use_cache,
    )
    logger.info("Downloaded %d rows for %s", len(df_raw), cfg["data"]["ticker"])

    # ------------------------------------------------------------------ #
    # 2. Feature engineering                                               #
    # ------------------------------------------------------------------ #
    logger.info("=== Step 2: Feature Engineering ===")
    instrument_name = cfg["data"].get("ticker")
    instrument_spec = resolve_instrument_spec(instrument_name, fallback_point_size=0.01)
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
        timeframe=cfg["data"].get("interval", "15m"),
        take_profit_points=cfg["target"].get("take_profit_points", 100),
        stop_loss_points=cfg["target"].get("stop_loss_points", 20),
        max_bars=cfg["target"].get("max_bars", 40),
        same_bar_rule=cfg["target"].get("same_bar_rule", "drop"),
        unresolved_policy=cfg["target"].get("unresolved_policy", "drop"),
        instrument_config=cfg.get("instruments", {}),
        instrument_spec=instrument_spec,
    )
    df_features = fe.transform(df_raw, instrument_name=instrument_name)
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
    val_idx, test_idx = compute_split_indices(
        len(df_features),
        test_size=cfg["model"]["test_size"],
        validation_size=cfg["model"]["validation_size"],
    )
    split_summary = {
        "train_rows": val_idx,
        "val_rows": test_idx - val_idx,
        "test_rows": len(df_features) - test_idx,
    }
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
    test_df = df_signals.iloc[test_idx:].copy()
    backtest_df = select_backtest_subset(
        df_signals,
        test_size=cfg["model"]["test_size"],
        validation_size=cfg["model"]["validation_size"],
    )
    assert len(backtest_df) == len(test_df), f"Backtest rows {len(backtest_df)} != test rows {len(test_df)}"
    assert backtest_df.index.equals(test_df.index), "Backtest subset should match the holdout/test index exactly"
    assert backtest_df.index[0] >= test_df.index[0], "Backtest must start on or after the first test timestamp"
    assert backtest_df.index[-1] <= test_df.index[-1], "Backtest must end on or before the last test timestamp"
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
        instrument_spec=instrument_spec,
        take_profit_points=cfg["target"].get("take_profit_points", 100),
        stop_loss_points=cfg["target"].get("stop_loss_points", 20),
        same_bar_rule=cfg["target"].get("same_bar_rule", "drop"),
    )
    equity = engine.run(backtest_df)
    perf = engine.performance_metrics(equity)

    logger.info("=== Backtest Results ===")
    for k, v in perf.items():
        logger.info("  %-30s %s", k, v)

    diagnostic_report = build_diagnostic_report(
        df_raw=df_raw,
        df_features=df_features,
        split_summary=split_summary,
        df_signals=df_signals,
        equity=equity,
        perf=perf,
    )
    logger.info("Diagnostic report: %s", diagnostic_report)

    return perf


def resolve_date(value: str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"current", "today", "now"}:
        return datetime.now().strftime("%Y-%m-%d")
    return value


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

    cfg["data"]["start_date"] = resolve_date(cfg["data"].get("start_date"))
    cfg["data"]["end_date"] = resolve_date(cfg["data"].get("end_date"))

    run_pipeline(cfg, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
