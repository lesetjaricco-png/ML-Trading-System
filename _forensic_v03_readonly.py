"""Read-only V0.3 forensic audit — do not import in production."""
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle as skshuffle

warnings.filterwarnings("ignore")

from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineer
from src.instruments import resolve_instrument_spec
from src.model import XGBoostModel
from src.utils import load_config
from main import resolve_date, compute_split_indices

ROOT = Path(__file__).resolve().parent
cfg = load_config("config/config_v03.yaml")
ingestion = DataIngestion(data_dir=str(ROOT / "data" / "raw"))
df_raw = ingestion.fetch(
    ticker=cfg["data"]["ticker"],
    start_date=resolve_date(cfg["data"].get("start_date")),
    end_date=resolve_date(cfg["data"].get("end_date")),
    interval=cfg["data"]["interval"],
    source=cfg["data"].get("source", "mt5"),
    use_cache=True,
)
instrument_name = cfg["data"]["ticker"]
instrument_spec = resolve_instrument_spec(instrument_name, fallback_point_size=0.01)


def make_fe():
    return FeatureEngineer(
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
        target_mode="v0.3_forward_atr",
        forward_horizon=cfg.get("target", {}).get("forward_horizon", 5),
        atr_threshold_multiplier=cfg.get("target", {}).get("atr_threshold_multiplier", 1.0),
    )


fe = make_fe()
df_features = fe.transform(df_raw.copy(), instrument_name=instrument_name)
feature_cols = [c for c in fe.feature_columns if c in df_features.columns]
n = len(df_features)
val_idx, test_idx = compute_split_indices(n, cfg["model"]["test_size"], cfg["model"]["validation_size"])
report = {}


def perturbation_test(df_raw, probe_ts, delta=777.7):
    fe_local = make_fe()
    original_full = fe_local.transform(df_raw.copy(), instrument_name=instrument_name)
    modified = df_raw.copy()
    future_mask = modified.index > probe_ts
    first_future_ts = modified.index[future_mask][0]
    modified.loc[future_mask, ["Open", "High", "Low", "Close"]] += delta
    perturbed_full = fe_local.transform(modified, instrument_name=instrument_name)
    result = {
        "probe_ts": str(probe_ts),
        "first_modified_future_ts": str(first_future_ts),
        "delta": delta,
        "changed_features": [],
    }
    orig_row = original_full.loc[probe_ts]
    pert_row = perturbed_full.loc[probe_ts]
    for col in feature_cols:
        o, p = orig_row[col], pert_row[col]
        if pd.isna(o) and pd.isna(p):
            continue
        if pd.isna(o) or pd.isna(p) or not np.isclose(float(o), float(p), rtol=1e-9, atol=1e-12):
            result["changed_features"].append(
                {"feature": col, "original": float(o), "perturbed": float(p), "abs_diff": float(abs(p - o))}
            )
    ot, pt = orig_row["target"], pert_row["target"]
    result["target_changed"] = bool(ot != pt)
    result["target_original"] = float(ot)
    result["target_perturbed"] = float(pt)
    return result


def feature_only_at_ts(df_raw, probe_ts, delta=777.7):
    fe_local = make_fe()
    df_orig = df_raw.copy().sort_index()
    df_mod = df_orig.copy()
    df_mod.loc[df_mod.index > probe_ts, ["Open", "High", "Low", "Close"]] += delta

    def compute_features_no_target(df):
        d = df.copy()
        d = fe_local._add_price_features(d)
        d = fe_local._add_rsi(d)
        d = fe_local._add_macd(d)
        d = fe_local._add_bollinger_bands(d)
        d = fe_local._add_moving_averages(d)
        d = fe_local._add_atr(d)
        d = fe_local._add_volatility_features(d)
        d = fe_local._add_volume_features(d)
        d = fe_local._add_candlestick_features(d)
        d = fe_local._add_time_features(d)
        return d

    o = compute_features_no_target(df_orig)
    p = compute_features_no_target(df_mod)
    changed = []
    for col in feature_cols:
        ov, pv = o.at[probe_ts, col], p.at[probe_ts, col]
        if pd.isna(ov) and pd.isna(pv):
            continue
        if pd.isna(ov) or pd.isna(pv) or not np.isclose(float(ov), float(pv), rtol=1e-12, atol=1e-15):
            changed.append({"feature": col, "original": float(ov), "perturbed": float(pv)})
    return {"probe_ts": str(probe_ts), "changed_features": changed}


probe_indices = {
    "early": df_raw.index[250],
    "middle": df_raw.index[len(df_raw) // 2],
    "validation_region": df_features.index[val_idx + 100],
    "test_region": df_features.index[test_idx + 100],
}
report["perturbation_test_full_transform"] = {
    k: perturbation_test(df_raw, ts) for k, ts in probe_indices.items()
}
report["perturbation_test_feature_only"] = {k: feature_only_at_ts(df_raw, ts) for k, ts in probe_indices.items()}

emp_changed = set(
    x["feature"] for x in report["perturbation_test_feature_only"]["middle"]["changed_features"]
)
report["feature_classification"] = {
    col: ("C_confirmed_non_causal" if col in emp_changed else "A_causal") for col in feature_cols
}

report["model_input"] = {
    "feature_columns": fe.feature_columns,
    "target_in_features": "target" in fe.feature_columns,
    "invalid_in_features": [
        c
        for c in fe.feature_columns
        if c in {"target"} or c.startswith("future_") or c.endswith("_outcome")
    ],
}

train_part = df_features.iloc[:val_idx]
test_part = df_features.iloc[test_idx:]
train_keys = set(map(tuple, train_part[feature_cols].round(10).values))
cross_dup = cross_same = cross_diff = 0
for i, row in enumerate(test_part[feature_cols].round(10).values):
    if tuple(row) in train_keys:
        cross_dup += 1
        train_targets = train_part.loc[(train_part[feature_cols].round(10).values == row).all(axis=1), "target"].values
        if test_part.iloc[i]["target"] in train_targets:
            cross_same += 1
        else:
            cross_diff += 1
report["duplicates"] = {
    "test_duplicate_feature_rows": int(test_part[feature_cols].duplicated().sum()),
    "test_rows_with_identical_train_features": cross_dup,
    "identical_features_same_target": cross_same,
    "identical_features_diff_target": cross_diff,
    "test_rows": len(test_part),
}

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
    signal_mode="v0.3_forward_atr",
)
report["full_model_metrics"] = model.train(
    df_features,
    feature_columns=fe.feature_columns,
    test_size=cfg["model"]["test_size"],
    validation_size=cfg["model"]["validation_size"],
)

single_feat = {}
for feature in ["rsi", "returns_2", "returns_5", "returns", "returns_10", "macd", "atr_pct"]:
    X_train = df_features.iloc[:val_idx][[feature]].astype(float)
    y_train = df_features.iloc[:val_idx]["target"]
    X_test = df_features.iloc[test_idx:][[feature]].astype(float)
    y_test = df_features.iloc[test_idx:]["target"]
    try:
        m = LogisticRegression(max_iter=3000, random_state=42)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
    except Exception:
        m = DecisionTreeClassifier(max_depth=5, random_state=42)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
    single_feat[feature] = float(accuracy_score(y_test, pred))
report["single_feature_test_accuracy"] = single_feat

horizon = cfg["target"]["forward_horizon"]
alignment_checks = []
for i in range(test_idx, min(test_idx + 20, n)):
    row = df_features.iloc[i]
    ts = df_features.index[i]
    raw_idx = df_raw.index.get_loc(ts)
    entry_close = float(row["Close"])
    future_close = float(df_raw.iloc[raw_idx + horizon]["Close"])
    fwd_ret = (future_close / entry_close) - 1.0
    threshold = abs(cfg["target"]["atr_threshold_multiplier"] * float(row["atr_pct"]))
    expected = 1 if fwd_ret > threshold else (0 if fwd_ret < -threshold else 2)
    alignment_checks.append({"match": int(row["target"]) == expected})
report["target_alignment_all_match"] = all(c["match"] for c in alignment_checks)

dist_by_mult = {}
for mult in [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]:
    fe_m = make_fe()
    fe_m.atr_threshold_multiplier = mult
    targets = fe_m._build_v03_forward_atr_target_series(df_features, instrument_name)
    c = Counter(targets)
    total = sum(c.values())
    dist_by_mult[str(mult)] = {
        "BUY_%": round(100 * c.get(1, 0) / total, 2),
        "SELL_%": round(100 * c.get(0, 0) / total, 2),
        "NO_TRADE_%": round(100 * c.get(2, 0) / total, 2),
    }
report["target_distribution_by_atr_multiplier"] = dist_by_mult

c = Counter(df_features["target"].astype(int))
total = len(df_features)
report["current_target_distribution"] = {
    "BUY_%": round(100 * c.get(1, 0) / total, 2),
    "SELL_%": round(100 * c.get(0, 0) / total, 2),
    "NO_TRADE_%": round(100 * c.get(2, 0) / total, 2),
    "test_counts": dict(Counter(test_part["target"].astype(int))),
}

r5_test, fwd_test = [], []
for ts in df_features.index[test_idx:]:
    loc = df_raw.index.get_loc(ts)
    if loc + 5 >= len(df_raw):
        continue
    r5_test.append(df_features.loc[ts, "returns_5"])
    fwd_test.append(df_raw.iloc[loc + 5]["Close"] / df_raw.iloc[loc]["Close"] - 1)
report["returns_5_vs_forward5_corr_test"] = float(np.corrcoef(r5_test, fwd_test)[0, 1])

df_shuf = df_features.copy()
df_shuf.iloc[:val_idx, df_shuf.columns.get_loc("target")] = skshuffle(
    df_features.iloc[:val_idx]["target"].values, random_state=42
)
model_shuf = XGBoostModel(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    early_stopping_rounds=50,
    signal_mode="v0.3_forward_atr",
)
report["shuffled_target_test_accuracy"] = model_shuf.train(
    df_shuf,
    feature_columns=fe.feature_columns,
    test_size=cfg["model"]["test_size"],
    validation_size=cfg["model"]["validation_size"],
)["accuracy"]

report["data_summary"] = {
    "raw_rows": len(df_raw),
    "feature_rows": n,
    "val_idx": val_idx,
    "test_idx": test_idx,
    "test_rows": n - test_idx,
}

out = ROOT / "forensic_v03_audit_output.json"
out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
print(json.dumps(report, indent=2, default=str))
