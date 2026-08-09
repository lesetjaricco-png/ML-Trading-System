import json
import math
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

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
    return cfg, fe, df, instrument_name, instrument_spec, test_idx, val_idx


def metrics(y_true, y_pred, probs=None, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = labels if labels is not None else sorted(np.unique(np.concatenate([y_true, y_pred])))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=0) if len(labels) == 2 else precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='binary', zero_division=0) if len(labels) == 2 else recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0) if len(labels) == 2 else f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    auc = None
    if probs is not None:
        try:
            if len(labels) == 2:
                auc = roc_auc_score(y_true, probs)
            else:
                y_true_bin = label_binarize(y_true, classes=labels)
                auc = roc_auc_score(y_true_bin, probs, average='macro')
        except Exception:
            auc = None
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'roc_auc': None if auc is None else float(auc),
    }


def fit_and_eval(df, feature_columns, cfg, test_idx, val_idx, seed=42):
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
        random_state=seed,
        eval_metric=cfg['model']['eval_metric'],
        early_stopping_rounds=cfg['model']['early_stopping_rounds'],
        prediction_threshold=cfg['signal'].get('prediction_threshold', 0.55),
        buy_threshold=cfg['signal'].get('buy_threshold', 0.70),
        sell_threshold=cfg['signal'].get('sell_threshold', 0.70),
        signal_mode=cfg.get('experiment', {}).get('signal_mode', 'v0.1_binary'),
    )
    model.train(df, feature_columns=feature_columns, test_size=cfg['model']['test_size'], validation_size=cfg['model']['validation_size'])
    X_test = df[feature_columns].iloc[test_idx:].values
    probs = model.predict_proba(X_test)
    y_true = df['target'].iloc[test_idx:].to_numpy()
    if probs.ndim == 1:
        y_pred = (probs >= 0.5).astype(int)
        labels = [0, 1]
    else:
        y_pred = probs.argmax(axis=1)
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    if probs.ndim == 2 and len(np.unique(y_true)) > 2:
        classes = getattr(model.model, 'classes_', None)
        if classes is not None:
            y_pred = np.array([classes[i] for i in y_pred])
            labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    m = metrics(y_true, y_pred, probs if probs.ndim == 1 else None, labels=labels)
    if probs.ndim == 2 and len(np.unique(y_true)) > 2:
        y_true_bin = label_binarize(y_true, classes=labels)
        m['roc_auc'] = float(roc_auc_score(y_true_bin, probs, average='macro'))
    return model, m, probs, y_true, y_pred


def majority_metrics(y_true):
    majority = Counter(y_true).most_common(1)[0][0]
    y_pred = np.full(len(y_true), majority)
    return metrics(y_true, y_pred, None, labels=sorted(np.unique(y_true)))


def random_metrics(y_true, n_runs=20, seed=42):
    labels = sorted(np.unique(y_true))
    p = np.array([np.mean(y_true == label) for label in labels])
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_runs):
        y_pred = rng.choice(labels, size=len(y_true), p=p)
        rows.append(metrics(y_true, y_pred, None, labels=labels))
    out = {}
    for key in rows[0].keys():
        if key == 'confusion_matrix':
            continue
        vals = [r[key] for r in rows if r[key] is not None]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def prob_summary(probs):
    vals = np.asarray(probs).ravel()
    return {
        'min': float(np.min(vals)),
        'p10': float(np.percentile(vals, 10)),
        'p25': float(np.percentile(vals, 25)),
        'median': float(np.median(vals)),
        'p75': float(np.percentile(vals, 75)),
        'p90': float(np.percentile(vals, 90)),
        'max': float(np.max(vals)),
    }


def bucket_counts(probs):
    vals = np.asarray(probs).ravel()
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.00)]
    out = {}
    for lo, hi in bins:
        mask = (vals >= lo) & (vals < hi)
        out[f'{lo:.2f}-{hi:.2f}'] = int(mask.sum())
    return out


def calibration_summary(probs, y_true):
    vals = np.asarray(probs).ravel()
    y_true = np.asarray(y_true).ravel()
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]
    out = []
    for lo, hi in bins:
        mask = (vals >= lo) & (vals < hi)
        if mask.sum() == 0:
            out.append({'bucket': f'{lo:.2f}-{hi:.2f}', 'n': 0, 'avg_pred': None, 'actual_positive_rate': None})
        else:
            out.append({'bucket': f'{lo:.2f}-{hi:.2f}', 'n': int(mask.sum()), 'avg_pred': float(np.mean(vals[mask])), 'actual_positive_rate': float(np.mean(y_true[mask] == 1))})
    return out


def target_difficulty(df, test_idx, fe, instrument_name):
    out = {'tp_first_candle': 0, 'tp_2_candles': 0, 'tp_3_5': 0, 'tp_6_10': 0, 'tp_11_20': 0, 'tp_21_40': 0, 'sl': 0, 'unresolved': 0}
    for idx in range(test_idx, len(df)):
        entry_price = float(df.iloc[idx]['Close'])
        tp_price, sl_price = fe._resolve_tp_sl_levels(entry_price, instrument_name)
        first_tp = None
        first_sl = None
        for offset in range(1, fe.max_bars + 1):
            if idx + offset >= len(df):
                break
            nxt = df.iloc[idx + offset]
            if first_tp is None and nxt['High'] >= tp_price:
                first_tp = offset
            if first_sl is None and nxt['Low'] <= sl_price:
                first_sl = offset
            if first_tp is not None and first_sl is not None:
                break
        if first_tp is None and first_sl is None:
            out['unresolved'] += 1
        elif first_tp is not None and first_sl is None:
            if first_tp == 1:
                out['tp_first_candle'] += 1
            elif first_tp == 2:
                out['tp_2_candles'] += 1
            elif 3 <= first_tp <= 5:
                out['tp_3_5'] += 1
            elif 6 <= first_tp <= 10:
                out['tp_6_10'] += 1
            elif 11 <= first_tp <= 20:
                out['tp_11_20'] += 1
            else:
                out['tp_21_40'] += 1
        else:
            out['sl'] += 1
    total = sum(out.values())
    return {k: float(v / total) for k, v in out.items()}


def conditional_probabilities(df, feature_columns, y_true, top_n=5):
    out = []
    for col in feature_columns[:top_n]:
        if col not in df.columns:
            continue
        s = pd.Series(df[col]).astype(float)
        if not np.issubdtype(s.dtype, np.number):
            continue
        try:
            q = pd.qcut(s.rank(method='first'), q=10, labels=False, duplicates='drop')
        except Exception:
            q = pd.qcut(s, q=10, labels=False, duplicates='drop')
        buckets = []
        for b in sorted(np.unique(q)):
            mask = q == b
            if mask.sum() == 0:
                continue
            buckets.append({'bucket': int(b), 'win_rate': float(np.mean(y_true[mask] == 1))})
        out.append({'feature': col, 'buckets': buckets})
    return out


def ablation_metrics(df, feature_columns, cfg, test_idx, val_idx, subset_name, subset):
    if len(subset) == 0:
        return None
    _, m, _, _, _ = fit_and_eval(df, subset, cfg, test_idx, val_idx, seed=cfg['model'].get('random_state', 42))
    return {'roc_auc': m['roc_auc'], 'f1': m['f1'], 'accuracy': m['accuracy'], 'precision': m['precision'], 'recall': m['recall']}


def to_plain_counter(counter):
    return {int(k): int(v) for k, v in Counter(counter).items()}


results = {}
for cfg_path, name in [('config/config.yaml', 'V0.1'), ('config/config_v02.yaml', 'V0.2')]:
    cfg, fe, df, instrument_name, instrument_spec, test_idx, val_idx = build_dataset(cfg_path)
    feature_columns = fe.feature_columns
    y_full = df['target'].to_numpy()
    y_train = y_full[:val_idx]
    y_val = y_full[val_idx:test_idx]
    y_test = y_full[test_idx:]

    majority = majority_metrics(y_test)
    random = random_metrics(y_test)
    model, normal_metrics, probs, y_true_test, y_pred_test = fit_and_eval(df, feature_columns, cfg, test_idx, val_idx, seed=cfg['model'].get('random_state', 42))
    imp = model.feature_importance()
    top_features = imp.head(15).to_dict(orient='records')

    df_shuf = df.copy()
    train_target_vals = df_shuf['target'].iloc[:val_idx].to_numpy(copy=True)
    rng = np.random.default_rng(42)
    rng.shuffle(train_target_vals)
    df_shuf.loc[df_shuf.index[:val_idx], 'target'] = train_target_vals
    _, shuffled_metrics, _, _, _ = fit_and_eval(df_shuf, feature_columns, cfg, test_idx, val_idx, seed=cfg['model'].get('random_state', 42))

    df_fshuf = df.copy()
    train_feature_rows = df_fshuf.loc[df_fshuf.index[:val_idx], feature_columns].copy()
    perm = np.random.RandomState(42).permutation(len(train_feature_rows))
    df_fshuf.loc[df_fshuf.index[:val_idx], feature_columns] = train_feature_rows.iloc[perm].values
    _, feature_shuffle_metrics, _, _, _ = fit_and_eval(df_fshuf, feature_columns, cfg, test_idx, val_idx, seed=cfg['model'].get('random_state', 42))

    prob_values = probs if probs.ndim == 1 else probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    ablations = {
        'all_current_features': ablation_metrics(df, feature_columns, cfg, test_idx, val_idx, 'all', feature_columns),
        'price_technical_only': ablation_metrics(df, feature_columns, cfg, test_idx, val_idx, 'price_technical_only', [c for c in feature_columns if c not in {'volatility_5', 'volatility_20', 'volume_sma', 'volume_ratio', 'obv'}]),
        'remove_volatility': ablation_metrics(df, feature_columns, cfg, test_idx, val_idx, 'remove_volatility', [c for c in feature_columns if c not in {'volatility_5', 'volatility_20'}]),
        'remove_volume': ablation_metrics(df, feature_columns, cfg, test_idx, val_idx, 'remove_volume', [c for c in feature_columns if c not in {'volume_sma', 'volume_ratio', 'obv'}]),
        'remove_future_like': ablation_metrics(df, feature_columns, cfg, test_idx, val_idx, 'remove_future_like', [c for c in feature_columns if c not in {'returns', 'returns_2', 'returns_5', 'returns_10', 'log_returns', 'high_low_ratio', 'close_open_ratio', 'volatility_5', 'volatility_20', 'atr', 'atr_pct', 'body_size', 'upper_shadow', 'lower_shadow'}]),
    }

    results[name] = {
        'target_dist': {
            'train': to_plain_counter(y_train),
            'val': to_plain_counter(y_val),
            'test': to_plain_counter(y_test),
        },
        'majority_baseline': majority,
        'random_baseline': random,
        'normal_model': normal_metrics,
        'shuffled_target_model': shuffled_metrics,
        'feature_shuffled_model': feature_shuffle_metrics,
        'top_features': top_features,
        'conditional_probabilities': conditional_probabilities(df.iloc[test_idx:].copy(), [f['feature'] for f in top_features], y_test, top_n=5),
        'probability_summary': prob_summary(prob_values),
        'probability_buckets': bucket_counts(prob_values),
        'calibration': calibration_summary(prob_values, (y_test == 1).astype(int)),
        'target_difficulty': target_difficulty(df, test_idx, fe, instrument_name),
        'ablations': {k: v for k, v in ablations.items() if v is not None},
    }

out_path = ROOT / 'audit_model_info_results.json'
with out_path.open('w', encoding='utf-8') as fh:
    json.dump(results, fh, indent=2)
print(f'wrote {out_path}')
print(json.dumps(results, indent=2))
