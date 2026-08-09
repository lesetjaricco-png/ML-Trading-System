import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineer
from src.instruments import resolve_instrument_spec
from src.model import XGBoostModel
from src.utils import load_config

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent


def build_dataset(cfg_path: str):
    cfg = load_config(cfg_path)
    ingestion = DataIngestion(data_dir=str(ROOT / 'data' / 'raw'))
    df_raw = ingestion.fetch(
        ticker=cfg['data']['ticker'],
        start_date=cfg['data'].get('start_date'),
        end_date=cfg['data'].get('end_date'),
        interval=cfg['data']['interval'],
        source=cfg['data'].get('source', 'mt5'),
        use_cache=True,
    )
    instrument_name = cfg['data']['ticker']
    instrument_spec = resolve_instrument_spec(instrument_name, fallback_point_size=0.01)
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
    df = fe.transform(df_raw, instrument_name=instrument_name)
    test_idx = int(len(df) * (1 - cfg['model']['test_size']))
    val_idx = int(test_idx * (1 - cfg['model']['validation_size']))
    return cfg, fe, df, instrument_name, test_idx, val_idx


def prepare_frame(df, test_idx):
    test_df = df.iloc[test_idx:].copy()
    train_df = df.iloc[:test_idx].copy()
    return train_df, test_df


def quantile_target_rates(df, feature, target_col='target', q=4):
    x = df[feature].astype(float)
    bins = pd.qcut(x.rank(method='first'), q=q, labels=False, duplicates='drop')
    out = []
    for b in sorted(set(bins)):
        mask = bins == b
        if mask.sum() == 0:
            continue
        if np.issubdtype(df[target_col].dtype, np.number):
            if pd.api.types.is_integer_dtype(df[target_col]) or pd.api.types.is_float_dtype(df[target_col]):
                if len(np.unique(df.loc[mask, target_col])) <= 2:
                    rate = float(df.loc[mask, target_col].mean())
                else:
                    rate = float((df.loc[mask, target_col] == 1).mean())
            else:
                rate = float((df.loc[mask, target_col] == 1).mean())
        else:
            rate = float((df.loc[mask, target_col] == 1).mean())
        out.append({'bin': int(b), 'count': int(mask.sum()), 'target_rate': rate})
    return out


def single_feature_diagnostics(df, feature_columns, target_col='target'):
    test_idx = int(len(df) * 0.8)
    train_df = df.iloc[:test_idx]
    test_df = df.iloc[test_idx:]
    rows = []
    for feature in feature_columns:
        if feature not in df.columns:
            continue
        X_train = train_df[[feature]].astype(float)
        y_train = train_df[target_col]
        X_test = test_df[[feature]].astype(float)
        y_test = test_df[target_col]
        try:
            model = LogisticRegression(max_iter=2000, random_state=42)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            pred = (probs >= 0.5).astype(int)
        except Exception:
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = None
        rows.append({'feature': feature, 'accuracy': float(acc), 'roc_auc': None if auc is None else float(auc)})
    return rows


def resolution_analysis(df, fe, instrument_name):
    bins = {'0_bars': 0, '1_bar': 0, '2_bars': 0, '3_bars': 0, '4_5_bars': 0, '6_10_bars': 0, '11_20_bars': 0, '21_40_bars': 0, 'unresolved': 0}
    tp_first = 0
    sl_first = 0
    for idx in range(len(df)):
        entry_price = float(df.iloc[idx]['Close'])
        tp_price, sl_price = fe._resolve_tp_sl_levels(entry_price, instrument_name)
        first_tp = None
        first_sl = None
        for offset in range(1, fe.max_bars + 1):
            if idx + offset >= len(df):
                break
            row = df.iloc[idx + offset]
            if first_tp is None and row['High'] >= tp_price:
                first_tp = offset
            if first_sl is None and row['Low'] <= sl_price:
                first_sl = offset
            if first_tp is not None and first_sl is not None:
                break
        if first_tp is None and first_sl is None:
            bins['unresolved'] += 1
        elif first_tp is not None and first_sl is None:
            if first_tp == 1:
                bins['1_bar'] += 1
            elif first_tp == 2:
                bins['2_bars'] += 1
            elif 3 <= first_tp <= 5:
                bins['4_5_bars'] += 1
            elif 6 <= first_tp <= 10:
                bins['6_10_bars'] += 1
            elif 11 <= first_tp <= 20:
                bins['11_20_bars'] += 1
            else:
                bins['21_40_bars'] += 1
            tp_first += 1
        elif first_tp is None and first_sl is not None:
            if first_sl == 1:
                bins['1_bar'] += 1
            elif first_sl == 2:
                bins['2_bars'] += 1
            elif 3 <= first_sl <= 5:
                bins['4_5_bars'] += 1
            elif 6 <= first_sl <= 10:
                bins['6_10_bars'] += 1
            elif 11 <= first_sl <= 20:
                bins['11_20_bars'] += 1
            else:
                bins['21_40_bars'] += 1
            sl_first += 1
        else:
            # same-bar hit of both; treat as immediate resolution
            bins['0_bars'] += 1
            if first_tp is not None and first_sl is not None:
                tp_first += 0
                sl_first += 0
    total = sum(bins.values())
    return {k: {'count': int(v), 'share': float(v / total)} for k, v in bins.items()}, {'tp_first': tp_first, 'sl_first': sl_first}


def conditional_rates(df, target_col='target'):
    out = {}
    bullish = df['Close'] > df['Open']
    bearish = df['Close'] < df['Open']
    out['bullish_target_rate'] = float(df.loc[bullish, target_col].mean())
    out['bearish_target_rate'] = float(df.loc[bearish, target_col].mean())
    out['bullish_buy_rate'] = float((df.loc[bullish, target_col] == 1).mean())
    out['bearish_buy_rate'] = float((df.loc[bearish, target_col] == 1).mean())
    out['bullish_sell_rate'] = float((df.loc[bullish, target_col] == 0).mean())
    out['bearish_sell_rate'] = float((df.loc[bearish, target_col] == 0).mean())
    for name, cond in [('positive_return', df['returns'] > 0), ('negative_return', df['returns'] < 0), ('body_gt_range', (df['Close'] - df['Open']).abs() > (df['High'] - df['Low']) / 2), ('body_lt_range', (df['Close'] - df['Open']).abs() <= (df['High'] - df['Low']) / 2), ('prev_pos', df['returns'].shift(1) > 0), ('prev_neg', df['returns'].shift(1) < 0), ('last3_pos', df['returns'].rolling(3).sum() > 0), ('last5_pos', df['returns'].rolling(5).sum() > 0)]:
        out[f'{name}_target_rate'] = float(df.loc[cond, target_col].mean())
    return out


def future_return_relationship(df, target_col='target'):
    out = {}
    for horizon in [1,2,3,5,10,20,40]:
        future_ret = df['Close'].shift(-horizon) / df['Close'] - 1.0
        corr = float(np.corrcoef(df[target_col].astype(float), future_ret.fillna(np.nan))[0, 1])
        buckets = pd.qcut(future_ret.rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'])
        bucket_stats = []
        for b in ['Q1','Q2','Q3','Q4']:
            mask = buckets == b
            bucket_stats.append({'bucket': b, 'share': float(mask.mean()), 'target_rate': float(df.loc[mask, target_col].mean())})
        out[horizon] = {'corr': corr, 'bucket_stats': bucket_stats}
    return out


def regime_structure(df, target_col='target'):
    out = {}
    vol_q = pd.qcut(df['volatility_5'].fillna(0).rank(method='first'), q=4, labels=['Q1','Q2','Q3','Q4'])
    trend = np.where(df['returns_5'] > 0, 'up', 'down')
    session = np.where(df['hour_of_day'].between(0, 7), 'asia', np.where(df['hour_of_day'].between(8, 15), 'london', 'ny'))
    direction = np.where(df['Close'] >= df['Open'], 'bullish', 'bearish')
    for name, values in [('volatility', vol_q), ('trend', trend), ('session', session), ('direction', direction)]:
        stats = []
        for lev in sorted(set(values)):
            mask = values == lev
            if mask.sum() == 0:
                continue
            stats.append({'level': str(lev), 'target_rate': float(df.loc[mask, target_col].mean()), 'count': int(mask.sum())})
        out[name] = stats
    return out


for cfg_path, name in [('config/config.yaml', 'V0.1'), ('config/config_v02.yaml', 'V0.2')]:
    cfg, fe, df, instrument_name, test_idx, val_idx = build_dataset(cfg_path)
    train_df = df.iloc[:test_idx].copy()
    test_df = df.iloc[test_idx:].copy()
    feature_columns = fe.feature_columns

    if name == 'V0.1':
        target_col = 'target'
    else:
        target_col = 'target'

    # Top features from the prior model for the same split
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
        random_state=cfg['model'].get('random_state', 42),
        eval_metric=cfg['model']['eval_metric'],
        early_stopping_rounds=cfg['model']['early_stopping_rounds'],
        prediction_threshold=cfg['signal'].get('prediction_threshold', 0.55),
        buy_threshold=cfg['signal'].get('buy_threshold', 0.70),
        sell_threshold=cfg['signal'].get('sell_threshold', 0.70),
        signal_mode=cfg.get('experiment', {}).get('signal_mode', 'v0.1_binary'),
    )
    model.train(df, feature_columns=feature_columns, test_size=cfg['model']['test_size'], validation_size=cfg['model']['validation_size'])
    top_features = model.feature_importance().head(15)['feature'].tolist()

    quantile_results = {}
    for feature in top_features:
        quantile_results[feature] = quantile_target_rates(test_df, feature, target_col=target_col)

    single_feature = single_feature_diagnostics(df, top_features, target_col=target_col)
    resolution, counts = resolution_analysis(df, fe, instrument_name)
    cond_rates = conditional_rates(test_df, target_col=target_col)
    future = future_return_relationship(test_df, target_col=target_col)
    regimes = regime_structure(test_df, target_col=target_col)

    out = {
        'target_dist': dict(Counter(df[target_col].dropna())),
        'split_sizes': {'train': int(len(train_df)), 'test': int(len(test_df))},
        'top_features': top_features,
        'quantile_target_rates': quantile_results,
        'single_feature_diagnostics': single_feature,
        'resolution': resolution,
        'resolution_counts': counts,
        'conditional_rates': cond_rates,
        'future_return_relationship': future,
        'regime_structure': regimes,
    }
    (ROOT / f'forensic_{name.lower()}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(name, json.dumps({'top_features': top_features[:5], 'single_feature_top': single_feature[:5], 'resolution': resolution, 'conditional_rates': cond_rates, 'future_return_relationship': future}, indent=2))
