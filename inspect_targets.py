import pandas as pd
from src.feature_engineering import FeatureEngineer

dates = pd.date_range('2024-01-01', periods=6, freq='15min')
df = pd.DataFrame({
    'Open':[100,101,102,103,104,105],
    'High':[100,105,102,103,104,105],
    'Low':[100,99,95,97,102,103],
    'Close':[100,102,98,99,103,104],
    'Volume':[1000,1000,1000,1000,1000,1000],
}, index=dates)
fe = FeatureEngineer(timeframe='15m', take_profit_points=5, stop_loss_points=2, max_bars=1, same_bar_rule='drop', unresolved_policy='drop')
result = fe.transform(df)
print(result[['Close','target']])
print(result['target'].unique())


dates2 = pd.date_range('2024-01-01', periods=2, freq='15min')
df2 = pd.DataFrame({
    'Open':[100,101],
    'High':[100,104],
    'Low':[100,100],
    'Close':[100,101],
    'Volume':[1000,1000],
}, index=dates2)
fe2 = FeatureEngineer(timeframe='15m', take_profit_points=100, stop_loss_points=20, max_bars=1, same_bar_rule='drop', unresolved_policy='drop', instrument_config={'US30': {'take_profit_points': 4, 'stop_loss_points': 1}})
result2 = fe2.transform(df2, instrument_name='US30')
print(result2[['Close','target']])
print(result2['target'].unique())
