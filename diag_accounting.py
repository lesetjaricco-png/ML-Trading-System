import sys
from pathlib import Path
import yaml

sys.path.insert(0, 'c:/Users/micca/personal_projects/Trading/ML-Trading-System')

from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineer
from src.instruments import resolve_instrument_spec
from src.model import XGBoostModel
from src.backtesting import BacktestEngine
from src.risk_management import RiskManager
from main import resolve_date

cfg = yaml.safe_load(Path('config/config.yaml').read_text())
start_date = resolve_date(cfg['data'].get('start_date'))
end_date = resolve_date(cfg['data'].get('end_date'))
ing = DataIngestion(data_dir='data/raw')
df_raw = ing.fetch(ticker=cfg['data']['ticker'], start_date=start_date, end_date=end_date, interval=cfg['data']['interval'], source=cfg['data'].get('source', 'mt5'), use_cache=True)
instrument_spec = resolve_instrument_spec(cfg['data']['ticker'], fallback_point_size=0.01)
fe = FeatureEngineer(
    rsi_period=cfg['features']['rsi_period'],
    macd_fast=cfg['features']['macd_fast'],
    macd_slow=cfg['features']['macd_slow'],
    macd_signal=cfg['features']['macd_signal'],
    bb_period=cfg['features']['bb_period'],
    bb_std=cfg['features']['bb_std'],
    sma_periods=cfg['features']['sma_periods'],
    ema_periods=cfg['features']['ema_periods'],
    atr_period=cfg['features']['atr_period'],
    volume_sma_period=cfg['features']['volume_sma_period'],
    timeframe=cfg['data'].get('interval', '15m'),
    take_profit_points=cfg['target'].get('take_profit_points', 100),
    stop_loss_points=cfg['target'].get('stop_loss_points', 20),
    max_bars=cfg['target'].get('max_bars', 40),
    same_bar_rule=cfg['target'].get('same_bar_rule', 'drop'),
    unresolved_policy=cfg['target'].get('unresolved_policy', 'drop'),
    instrument_config=cfg.get('instruments', {}),
    instrument_spec=instrument_spec,
    target_mode=cfg.get('experiment', {}).get('target_mode', 'v0.1_tp_before_sl'),
)
df_features = fe.transform(df_raw, instrument_name=cfg['data']['ticker'])
model = XGBoostModel(
    n_estimators=cfg['model']['n_estimators'],
    max_depth=cfg['model']['max_depth'],
    learning_rate=cfg['model']['learning_rate'],
    subsample=cfg['model']['subsample'],
    colsample_bytree=cfg['model']['colsample_bytree'],
    min_child_weight=cfg['model']['min_child_weight'],
    gamma=cfg['model']['gamma'],
    reg_alpha=cfg['model']['reg_alpha'],
    reg_lambda=cfg['model']['reg_lambda'],
    random_state=cfg['model']['random_state'],
    eval_metric=cfg['model']['eval_metric'],
    early_stopping_rounds=cfg['model']['early_stopping_rounds'],
    prediction_threshold=cfg['signal'].get('prediction_threshold', 0.55),
    buy_threshold=cfg['signal'].get('buy_threshold', 0.70),
    sell_threshold=cfg['signal'].get('sell_threshold', 0.70),
    signal_mode=cfg.get('experiment', {}).get('signal_mode', 'v0.1_binary'),
)
try:
    model.load()
except Exception:
    model.train(df_features, feature_columns=fe.feature_columns, test_size=cfg['model']['test_size'], validation_size=cfg['model']['validation_size'])
df_signals = model.generate_signals(df_features)
val_idx, test_idx = int(len(df_signals) * (1 - cfg['model']['test_size'])) if False else (int(len(df_signals) * (1 - cfg['model']['test_size']) * (1 - cfg['model']['validation_size'])), int(len(df_signals) * (1 - cfg['model']['test_size'])))
backtest_df = df_signals.iloc[test_idx:].copy()
risk = RiskManager(max_position_size=cfg['risk']['max_position_size'], stop_loss=cfg['risk']['stop_loss'], take_profit=cfg['risk']['take_profit'], max_drawdown=cfg['risk']['max_drawdown'], risk_per_trade=cfg['risk']['risk_per_trade'])
engine = BacktestEngine(initial_capital=cfg['backtest']['initial_capital'], commission=cfg['backtest']['commission'], slippage=cfg['backtest']['slippage'], risk_manager=risk, instrument_spec=instrument_spec, take_profit_points=cfg['target'].get('take_profit_points', 100), stop_loss_points=cfg['target'].get('stop_loss_points', 20), same_bar_rule=cfg['target'].get('same_bar_rule', 'drop'))
equity = engine.run(backtest_df)
perf = engine.performance_metrics(equity)
print('trades', len(engine.trades))
print('sum pnl', sum(t.pnl for t in engine.trades))
print('cash', engine.cash)
print('final portfolio', perf['final_portfolio_value'])
print('initial + sum pnl', cfg['backtest']['initial_capital'] + sum(t.pnl for t in engine.trades))
print('equity final', equity['portfolio_value'].iloc[-1])
print('last trade', engine.trades[-1].exit_reason if engine.trades else None)
