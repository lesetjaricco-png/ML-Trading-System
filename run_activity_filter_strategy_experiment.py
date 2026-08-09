"""Evaluate the frozen ML activity score with independent directional rules."""

from __future__ import annotations

import json
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diagnose_baseline import load_train_validation_only, sha256
from run_ablation_25f import (
    BASELINE_DIR,
    BASELINE_ID,
    BASELINE_MODEL_PATH,
    BASELINE_REPORT_PATH,
    DATASET_PATH,
    METADATA_PATH,
    ensure_output_available,
)
from run_first_passage_target_experiment import (
    OHLC_RECONSTRUCTION_COLUMNS,
    OUTPUT_DIR as FIRST_PASSAGE_DIR,
    reconstruct_ohlc,
)
from src.model import XGBoostModel


EXPERIMENT_ID = f"{BASELINE_ID}_activity_filter_strategy_experiment_v1"
OUTPUT_DIR = BASELINE_DIR / EXPERIMENT_ID
FROZEN_MODEL_ID = f"{BASELINE_ID}_first_passage_1atr_5bar_v1"
FROZEN_MODEL_PATH = FIRST_PASSAGE_DIR / f"{FROZEN_MODEL_ID}.joblib"
FROZEN_REPORT_PATH = FIRST_PASSAGE_DIR / "experiment_report.json"
HORIZON = 5
THRESHOLDS = (0.50, 0.60, 0.70, 0.80)
PERMUTATIONS = 500
RANDOM_SEED = 42
RULES = (
    "momentum_3",
    "ma_10_20",
    "breakout_20",
    "completed_bar_direction",
)


def protected_hashes() -> dict[str, str]:
    paths = {
        "dataset": DATASET_PATH,
        "metadata": METADATA_PATH,
        "endpoint_model": BASELINE_MODEL_PATH,
        "endpoint_report": BASELINE_REPORT_PATH,
        "frozen_activity_model": FROZEN_MODEL_PATH,
        "frozen_activity_report": FROZEN_REPORT_PATH,
    }
    return {f"{name}_sha256": sha256(path) for name, path in paths.items()}


def previous_artifact_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(BASELINE_DIR)).replace("\\", "/"): sha256(path)
        for path in sorted(BASELINE_DIR.rglob("*"))
        if path.is_file() and OUTPUT_DIR not in path.parents
    }


def activity_probability(model: XGBoostModel, frame: pd.DataFrame) -> pd.Series:
    """Return P(SELL)+P(BUY); relative direction probabilities are discarded."""
    probabilities = model.model.predict_proba(
        model.scaler.transform(frame[model.feature_columns].to_numpy())
    )
    class_positions = {
        int(label): position for position, label in enumerate(model.model.classes_)
    }
    if set(class_positions) != {0, 1, 2}:
        raise RuntimeError("Frozen model classes must be SELL, BUY, NO_TRADE")
    values = probabilities[:, class_positions[0]] + probabilities[:, class_positions[1]]
    return pd.Series(values, index=frame.index, name="activity_probability")


def build_directional_rules(frame: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed causal rule set without model probabilities or future values."""
    if not frame.index.equals(ohlc.index):
        raise ValueError("Feature and OHLC indices must match")
    rules = pd.DataFrame(index=frame.index)
    rules["momentum_3"] = np.sign(ohlc["Close"].pct_change(3)).astype(float)
    rules["ma_10_20"] = np.sign(frame["sma_10"] - frame["sma_20"]).astype(float)
    prior_high = ohlc["High"].rolling(20, min_periods=20).max().shift(1)
    prior_low = ohlc["Low"].rolling(20, min_periods=20).min().shift(1)
    rules["breakout_20"] = np.select(
        [ohlc["Close"] >= prior_high, ohlc["Close"] <= prior_low],
        [1.0, -1.0],
        default=0.0,
    )
    rules["completed_bar_direction"] = np.sign(
        ohlc["Close"] - ohlc["Open"]
    ).astype(float)
    if list(rules.columns) != list(RULES):
        raise RuntimeError("Directional rule order differs from the predeclared manifest")
    return rules


def build_rule_outcomes(
    ohlc: pd.DataFrame,
    atr: pd.Series,
    directions: pd.Series,
    *,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Measure future outcomes after directions have already been fixed."""
    if not ohlc.index.equals(atr.index) or not ohlc.index.equals(directions.index):
        raise ValueError("Outcome inputs must share one chronological index")
    rows: list[dict[str, Any]] = []
    for position in range(len(ohlc) - horizon):
        direction_value = directions.iloc[position]
        if not np.isfinite(direction_value):
            continue
        direction = int(direction_value)
        if direction not in {-1, 1}:
            continue
        entry = float(ohlc["Close"].iloc[position])
        entry_atr = float(atr.iloc[position])
        future = ohlc.iloc[position + 1 : position + horizon + 1]
        terminal_return = direction * (float(future["Close"].iloc[-1]) - entry) / entry
        if direction > 0:
            mfe = (float(future["High"].max()) - entry) / entry
            mae = (float(future["Low"].min()) - entry) / entry
            favorable = future["High"].to_numpy() >= entry + entry_atr
            adverse = future["Low"].to_numpy() <= entry - entry_atr
        else:
            mfe = (entry - float(future["Low"].min())) / entry
            mae = (entry - float(future["High"].max())) / entry
            favorable = future["Low"].to_numpy() <= entry - entry_atr
            adverse = future["High"].to_numpy() >= entry + entry_atr
        first_passage = "UNRESOLVED"
        for favorable_hit, adverse_hit in zip(favorable, adverse):
            if favorable_hit and adverse_hit:
                first_passage = "SAME_BAR_TIE"
                break
            if favorable_hit:
                first_passage = "FAVORABLE"
                break
            if adverse_hit:
                first_passage = "ADVERSE"
                break
        rows.append(
            {
                "timestamp": ohlc.index[position],
                "direction": direction,
                "terminal_return": terminal_return,
                "mfe": mfe,
                "mae": mae,
                "reached_plus_1atr": bool(favorable.any()),
                "reached_minus_1atr": bool(adverse.any()),
                "first_passage": first_passage,
                "maximum_outcome_timestamp": future.index[-1],
            }
        )
    return pd.DataFrame(rows).set_index("timestamp")


def strategy_metrics(outcomes: pd.DataFrame, selected: pd.Series) -> dict[str, Any]:
    selected = selected.reindex(outcomes.index, fill_value=False).astype(bool)
    trades = outcomes.loc[selected]
    returns = trades["terminal_return"].to_numpy(dtype=float)
    positive = returns[returns > 0]
    negative = returns[returns < 0]
    if len(returns):
        equity = 1.0 + np.cumsum(returns)
        peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
        max_drawdown = float(np.max((peaks - equity) / np.maximum(peaks, 1e-12)))
    else:
        max_drawdown = 0.0
    return {
        "signals": len(outcomes),
        "trades": len(trades),
        "trade_frequency": float(len(trades) / len(outcomes)) if len(outcomes) else 0.0,
        "buy_trades": int((trades["direction"] == 1).sum()),
        "sell_trades": int((trades["direction"] == -1).sum()),
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "average_return": float(returns.mean()) if len(returns) else 0.0,
        "median_return": float(np.median(returns)) if len(returns) else 0.0,
        "cumulative_gross_return": float(returns.sum()),
        "profit_factor": float(positive.sum() / abs(negative.sum())) if len(negative) else None,
        "maximum_drawdown": max_drawdown,
        "average_mfe": float(trades["mfe"].mean()) if len(trades) else 0.0,
        "average_mae": float(trades["mae"].mean()) if len(trades) else 0.0,
        "reached_plus_1atr_pct": float(100 * trades["reached_plus_1atr"].mean()) if len(trades) else 0.0,
        "reached_minus_1atr_pct": float(100 * trades["reached_minus_1atr"].mean()) if len(trades) else 0.0,
        "first_passage_favorable_pct": float(100 * (trades["first_passage"] == "FAVORABLE").mean()) if len(trades) else 0.0,
        "first_passage_adverse_pct": float(100 * (trades["first_passage"] == "ADVERSE").mean()) if len(trades) else 0.0,
        "first_passage_tie_pct": float(100 * (trades["first_passage"] == "SAME_BAR_TIE").mean()) if len(trades) else 0.0,
    }


def _seed(split: str, rule: str, threshold: float) -> int:
    key = f"{RANDOM_SEED}|{split}|{rule}|{threshold:.2f}".encode("ascii")
    return zlib.crc32(key) & 0xFFFFFFFF


def evaluate_filter_controls(
    outcomes: pd.DataFrame,
    scores: pd.Series,
    *,
    split: str,
    rule: str,
    threshold: float,
    permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    aligned_scores = scores.reindex(outcomes.index)
    if aligned_scores.isna().any():
        raise RuntimeError("Activity scores do not cover all eligible rule timestamps")
    selected = aligned_scores >= threshold
    ml_metrics = strategy_metrics(outcomes, selected)
    rng = np.random.default_rng(_seed(split, rule, threshold))
    permuted_average_returns = np.empty(permutations, dtype=float)
    first_random_metrics: dict[str, Any] | None = None
    score_values = aligned_scores.to_numpy(copy=True)
    for iteration in range(permutations):
        random_selected = rng.permutation(score_values) >= threshold
        if int(random_selected.sum()) != int(selected.sum()):
            raise RuntimeError("Frequency-matched random control changed trade count")
        random_metrics = strategy_metrics(
            outcomes, pd.Series(random_selected, index=outcomes.index)
        )
        if first_random_metrics is None:
            first_random_metrics = random_metrics
        permuted_average_returns[iteration] = random_metrics["average_return"]
    observed = ml_metrics["average_return"]
    p_value = float(
        (1 + np.count_nonzero(permuted_average_returns >= observed))
        / (permutations + 1)
    )
    return {
        "threshold": threshold,
        "ml_filter": ml_metrics,
        "frequency_matched_random_filter": first_random_metrics,
        "permutation": {
            "iterations": permutations,
            "statistic": "average_terminal_return",
            "observed": observed,
            "random_mean": float(permuted_average_returns.mean()),
            "random_std": float(permuted_average_returns.std(ddof=0)),
            "one_sided_p_value": p_value,
            "trade_count_preserved_each_iteration": True,
        },
    }


def classify_result(results: dict[str, Any]) -> dict[str, Any]:
    validation_support: dict[str, list[float]] = {}
    train_support: dict[str, list[float]] = {}
    unfiltered_edges: list[str] = []
    for rule in RULES:
        validation_control = results["validation"][rule]["unfiltered"]
        if validation_control["average_return"] > 0 and validation_control["profit_factor"] is not None and validation_control["profit_factor"] > 1:
            unfiltered_edges.append(rule)
        for split, destination in (("train", train_support), ("validation", validation_support)):
            control = results[split][rule]["unfiltered"]
            supported = []
            for treatment in results[split][rule]["thresholds"]:
                ml = treatment["ml_filter"]
                random = treatment["frequency_matched_random_filter"]
                permutation = treatment["permutation"]
                if (
                    ml["average_return"] > control["average_return"]
                    and ml["average_return"] > random["average_return"]
                    and ml["win_rate"] > control["win_rate"]
                    and ml["win_rate"] > random["win_rate"]
                    and permutation["one_sided_p_value"] <= 0.05
                ):
                    supported.append(treatment["threshold"])
            destination[rule] = supported
    robust_rules = [
        rule for rule in RULES
        if len(validation_support[rule]) >= 3 and len(train_support[rule]) >= 2
    ]
    weak_rules = [rule for rule in RULES if len(validation_support[rule]) >= 1]
    if robust_rules:
        verdict = "ACTIVITY_FILTER_HAS_ECONOMIC_VALUE"
    elif weak_rules:
        verdict = "ACTIVITY_FILTER_WEAK_EVIDENCE"
    elif unfiltered_edges:
        verdict = "DIRECTIONAL_RULE_DOMINATES"
    else:
        verdict = "ACTIVITY_FILTER_NO_ECONOMIC_VALUE"
    return {
        "verdict": verdict,
        "validation_supported_thresholds": validation_support,
        "train_supported_thresholds": train_support,
        "robust_rules": robust_rules,
        "weak_rules": weak_rules,
        "unfiltered_positive_profit_factor_rules": unfiltered_edges,
    }


def _evaluate_split(
    split: str,
    frame: pd.DataFrame,
    ohlc: pd.DataFrame,
    scores: pd.Series,
    rules: pd.DataFrame,
) -> tuple[dict[str, Any], list[pd.DataFrame]]:
    split_results: dict[str, Any] = {}
    audits: list[pd.DataFrame] = []
    for rule in RULES:
        outcomes = build_rule_outcomes(ohlc, frame["atr"], rules[rule])
        unfiltered = strategy_metrics(
            outcomes, pd.Series(True, index=outcomes.index)
        )
        treatments = [
            evaluate_filter_controls(
                outcomes,
                scores,
                split=split,
                rule=rule,
                threshold=threshold,
            )
            for threshold in THRESHOLDS
        ]
        split_results[rule] = {
            "unfiltered": unfiltered,
            "thresholds": treatments,
        }
        audit = outcomes[["maximum_outcome_timestamp"]].copy()
        audit.insert(0, "split", split)
        audit.insert(1, "rule", rule)
        audit["information_timestamp"] = audit.index + pd.Timedelta(minutes=15)
        audit["entry_timestamp"] = audit["information_timestamp"]
        audit["information_not_after_entry"] = True
        audits.append(audit)
    return split_results, audits


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    ensure_output_available(output_dir)
    started = time.perf_counter()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    baseline_report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    hashes_before = protected_hashes()
    prior_before = previous_artifact_hashes()
    features = list(baseline_report["features"]["names_in_order"])
    if features != list(metadata["feature_names"]) or len(features) != 42:
        raise RuntimeError("Frozen feature order mismatch")
    requested = list(dict.fromkeys([*features, *OHLC_RECONSTRUCTION_COLUMNS]))
    train, validation, access = load_train_validation_only(
        DATASET_PATH, metadata, requested
    )
    if access["test_rows_exposed"] != 0 or access["test_labels_exposed"]:
        raise RuntimeError("Sealed loader exposed TEST")
    model = XGBoostModel(models_dir=str(FIRST_PASSAGE_DIR))
    model.load(FROZEN_MODEL_ID)
    if model.feature_columns != features:
        raise RuntimeError("Frozen model feature order mismatch")
    split_frames = {"train": train, "validation": validation}
    exposed = pd.concat([train, validation])
    exposed_ohlc = reconstruct_ohlc(exposed)
    exposed_rules = build_directional_rules(exposed, exposed_ohlc)
    results: dict[str, Any] = {}
    audit_frames: list[pd.DataFrame] = []
    reproducibility: dict[str, bool] = {}
    for split, frame in split_frames.items():
        ohlc = exposed_ohlc.loc[frame.index]
        scores = activity_probability(model, frame)
        repeat_scores = activity_probability(model, frame)
        reproducibility[split] = bool(np.array_equal(scores.to_numpy(), repeat_scores.to_numpy()))
        rules = exposed_rules.loc[frame.index]
        split_result, audits = _evaluate_split(split, frame, ohlc, scores, rules)
        results[split] = split_result
        audit_frames.extend(audits)
    decision = classify_result(results)
    hashes_after = protected_hashes()
    prior_after = {relative: sha256(BASELINE_DIR / relative) for relative in prior_before}
    if hashes_before != hashes_after or prior_before != prior_after:
        raise RuntimeError("Protected artifact changed")
    if not all(reproducibility.values()):
        raise RuntimeError("Frozen activity probabilities are not deterministic")
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "scientific_question": "Does frozen ML activity probability improve independently directed trades?",
        "predeclared_design": {
            "activity_probability": "P(SELL) + P(BUY)",
            "direction_from_model": False,
            "directional_rules": list(RULES),
            "thresholds": list(THRESHOLDS),
            "horizon_bars": HORIZON,
            "permutations": PERMUTATIONS,
            "random_seed": RANDOM_SEED,
            "cost_policy": "Gross returns only; no historical bid/ask or slippage invented.",
        },
        "data_access": access,
        "results": results,
        "decision": decision,
        "integrity": {
            "protected_hashes_before": hashes_before,
            "protected_hashes_after": hashes_after,
            "protected_artifacts_checked": len(prior_before),
            "protected_artifacts_unchanged": True,
            "frozen_model_loaded_not_retrained": True,
            "frozen_feature_order_verified": True,
            "test_rows_exposed": 0,
            "test_features_read": False,
            "test_labels_read": False,
            "final_validation_tail_excluded": HORIZON,
            "activity_probability_reproducible": reproducibility,
            "all_information_timestamps_not_after_entry": True,
            "future_values_used_only_for_outcomes": True,
        },
    }
    output_dir.mkdir(parents=True)
    rows = []
    for split in ("train", "validation"):
        for rule in RULES:
            rows.append({"split": split, "rule": rule, "filter": "unfiltered", "threshold": None, **results[split][rule]["unfiltered"]})
            for treatment in results[split][rule]["thresholds"]:
                rows.append({"split": split, "rule": rule, "filter": "ml", "threshold": treatment["threshold"], **treatment["ml_filter"], "permutation_p_value": treatment["permutation"]["one_sided_p_value"]})
                rows.append({"split": split, "rule": rule, "filter": "random", "threshold": treatment["threshold"], **treatment["frequency_matched_random_filter"]})
    pd.DataFrame(rows).to_csv(output_dir / "strategy_comparison.csv", index=False)
    pd.concat(audit_frames).to_parquet(output_dir / "timestamp_outcome_audit.parquet")
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        f"# {EXPERIMENT_ID}", "", "## Decision",
        f"**{decision['verdict']}**", "", "## Validation",
        "| Rule | Filter | Threshold | Trades | Win rate | Avg return | Profit factor | Permutation p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["split"] != "validation":
            continue
        threshold = "-" if row["threshold"] is None else f"{row['threshold']:.2f}"
        profit_factor = "n/a" if row["profit_factor"] is None else f"{row['profit_factor']:.4f}"
        p_value = row.get("permutation_p_value")
        p_text = "-" if p_value is None or pd.isna(p_value) else f"{p_value:.4f}"
        lines.append(
            f"| {row['rule']} | {row['filter']} | {threshold} | {row['trades']} | "
            f"{row['win_rate']:.4f} | {row['average_return']:.6f} | {profit_factor} | {p_text} |"
        )
    lines.extend(["", "Gross outcomes only; no historical execution costs were invented."])
    (output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_experiment()["decision"], indent=2))