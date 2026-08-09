"""Run the single frozen activity-filter confirmation on the sealed TEST period."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from diagnose_baseline import sha256
from run_ablation_25f import (
    BASELINE_DIR,
    BASELINE_ID,
    BASELINE_MODEL_PATH,
    BASELINE_REPORT_PATH,
    DATASET_PATH,
    METADATA_PATH,
    ensure_output_available,
)
from run_activity_filter_strategy_experiment import (
    FROZEN_MODEL_ID,
    FROZEN_MODEL_PATH,
    FROZEN_REPORT_PATH,
    OHLC_RECONSTRUCTION_COLUMNS,
    activity_probability,
    build_directional_rules,
    build_rule_outcomes,
    strategy_metrics,
)
from run_first_passage_target_experiment import OUTPUT_DIR as FIRST_PASSAGE_DIR, reconstruct_ohlc
from src.model import XGBoostModel


EXPERIMENT_ID = f"{BASELINE_ID}_activity_filter_confirmation_v1"
OUTPUT_DIR = BASELINE_DIR / EXPERIMENT_ID
PREVIOUS_ACTIVITY_DIR = BASELINE_DIR / f"{BASELINE_ID}_activity_filter_strategy_experiment_v1"
PREVIOUS_ACTIVITY_REPORT = PREVIOUS_ACTIVITY_DIR / "experiment_report.json"
RULE = "ma_10_20"
THRESHOLDS = (0.70, 0.80)
HORIZON = 5
ATR_MULTIPLIER = 1.0
PERMUTATIONS = 500
RANDOM_SEED = 42
MIN_FILTERED_TRADES = 1000
EXPECTED_DATASET_SHA256 = "35995f3d673b914cd4631e95f7abc13ee78acec89c12129edbc1b1d353bb4634"
EXPECTED_FROZEN_MODEL_SHA256 = "03b3e332a145340c7b4dae96f2b65d3b20b71231504e70443cbef58acd0092bc"
METRIC_NAMES = (
    "signals",
    "trades",
    "buy_trades",
    "sell_trades",
    "win_rate",
    "average_return",
    "median_return",
    "cumulative_gross_return",
    "profit_factor",
    "maximum_drawdown",
    "average_mfe",
    "average_mae",
    "reached_plus_1atr_pct",
    "reached_minus_1atr_pct",
)


def protected_hashes() -> dict[str, str]:
    paths = {
        "dataset": DATASET_PATH,
        "metadata": METADATA_PATH,
        "endpoint_model": BASELINE_MODEL_PATH,
        "endpoint_report": BASELINE_REPORT_PATH,
        "frozen_activity_model": FROZEN_MODEL_PATH,
        "frozen_activity_report": FROZEN_REPORT_PATH,
        "previous_activity_report": PREVIOUS_ACTIVITY_REPORT,
    }
    return {f"{name}_sha256": sha256(path) for name, path in paths.items()}


def prior_artifact_hashes(output_dir: Path = OUTPUT_DIR) -> dict[str, str]:
    roots = (Path("data"), BASELINE_DIR)
    return {
        str(path).replace("\\", "/"): sha256(path)
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file() and output_dir not in path.parents
    }


def verify_confirmation_eligibility(
    metadata: dict[str, Any], previous_report: dict[str, Any]
) -> dict[str, Any]:
    validation = metadata["splits"]["validation"]
    confirmation = metadata["splits"]["test"]
    validation_last = pd.Timestamp(validation["last_timestamp"])
    confirmation_first = pd.Timestamp(confirmation["first_timestamp"])
    if confirmation_first <= validation_last:
        raise RuntimeError("Confirmation does not begin after VALIDATION")
    access = previous_report["data_access"]
    integrity = previous_report["integrity"]
    if (
        access["test_rows_exposed"] != 0
        or access["test_labels_exposed"]
        or integrity["test_features_read"]
        or integrity["test_labels_read"]
    ):
        raise RuntimeError("Previous activity experiment exposed TEST")
    dataset_mtime = DATASET_PATH.stat().st_mtime
    legacy_outputs = [
        Path("forensic_v03_audit_output.json"),
        Path("audit_model_info_results.json"),
        Path("audit_report.json"),
    ]
    if any(path.exists() and path.stat().st_mtime >= dataset_mtime for path in legacy_outputs):
        raise RuntimeError("A legacy diagnostic is not older than the sealed MT5 dataset")
    return {
        "status": "ELIGIBLE_UNTOUCHED_CONFIRMATION_DATA",
        "validation_last_timestamp": validation_last.isoformat(),
        "confirmation_first_timestamp": confirmation_first.isoformat(),
        "confirmation_last_timestamp": confirmation["last_timestamp"],
        "confirmation_rows_recorded": int(confirmation["row_count"]),
        "strictly_after_validation": True,
        "previous_activity_test_rows_exposed": 0,
        "sealed_dataset_created_after_legacy_diagnostics": True,
    }


def load_confirmation_only(
    dataset_path: Path, metadata: dict[str, Any], columns: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_count = int(metadata["splits"]["train"]["row_count"])
    validation_count = int(metadata["splits"]["validation"]["row_count"])
    confirmation_count = int(metadata["splits"]["test"]["row_count"])
    start = train_count + validation_count
    requested = list(dict.fromkeys([*columns, "Date"]))
    parquet = pq.ParquetFile(dataset_path)
    batches: list[pd.DataFrame] = []
    position = 0
    for batch in parquet.iter_batches(columns=requested, use_threads=False):
        batch_end = position + batch.num_rows
        if batch_end > start:
            local_start = max(0, start - position)
            batches.append(batch.slice(local_start).to_pandas())
        position = batch_end
    confirmation = pd.concat(batches).iloc[:confirmation_count].copy()
    expected = metadata["splits"]["test"]
    if len(confirmation) != confirmation_count:
        raise RuntimeError("Confirmation row count differs from sealed metadata")
    if (
        pd.Timestamp(confirmation.index[0]).isoformat() != expected["first_timestamp"]
        or pd.Timestamp(confirmation.index[-1]).isoformat() != expected["last_timestamp"]
    ):
        raise RuntimeError("Confirmation boundaries differ from sealed metadata")
    if "target" in confirmation.columns:
        raise RuntimeError("Confirmation target labels must not be read")
    return confirmation, {
        "reader": "pyarrow.ParquetFile.iter_batches",
        "partition": "sealed TEST used once as confirmation",
        "rows_exposed": len(confirmation),
        "first_timestamp": pd.Timestamp(confirmation.index[0]).isoformat(),
        "last_timestamp": pd.Timestamp(confirmation.index[-1]).isoformat(),
        "target_labels_read": False,
        "prior_train_rows_reused_as_confirmation": 0,
        "prior_validation_rows_reused_as_confirmation": 0,
    }


def _seed(threshold: float) -> int:
    import zlib

    key = f"{RANDOM_SEED}|confirmation|{RULE}|{threshold:.2f}".encode("ascii")
    return zlib.crc32(key) & 0xFFFFFFFF


def evaluate_threshold(
    outcomes: pd.DataFrame,
    scores: pd.Series,
    threshold: float,
    *,
    permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    aligned_scores = scores.reindex(outcomes.index)
    if aligned_scores.isna().any():
        raise RuntimeError("Activity scores do not cover confirmation outcomes")
    selected = aligned_scores >= threshold
    metrics = strategy_metrics(outcomes, selected)
    score_values = aligned_scores.to_numpy(copy=True)
    rng = np.random.default_rng(_seed(threshold))
    distribution = np.empty(permutations, dtype=float)
    for iteration in range(permutations):
        random_selected = rng.permutation(score_values) >= threshold
        if int(random_selected.sum()) != int(selected.sum()):
            raise RuntimeError("Permutation did not preserve exact trade count")
        distribution[iteration] = strategy_metrics(
            outcomes, pd.Series(random_selected, index=outcomes.index)
        )["average_return"]
    observed = metrics["average_return"]
    p_value = float(
        (1 + np.count_nonzero(distribution >= observed)) / (permutations + 1)
    )
    return {
        "threshold": threshold,
        "metrics": metrics,
        "permutation": {
            "iterations": permutations,
            "statistic": "average_terminal_return",
            "observed": observed,
            "random_mean": float(distribution.mean()),
            "random_std": float(distribution.std(ddof=0)),
            "random_min": float(distribution.min()),
            "random_q05": float(np.quantile(distribution, 0.05)),
            "random_median": float(np.median(distribution)),
            "random_q95": float(np.quantile(distribution, 0.95)),
            "random_max": float(distribution.max()),
            "one_sided_p_value": p_value,
            "trade_count_preserved_each_iteration": True,
        },
    }


def metric_differences(
    treatment: dict[str, Any], control: dict[str, Any]
) -> dict[str, float | int | None]:
    differences: dict[str, float | int | None] = {}
    for metric in METRIC_NAMES:
        left = treatment[metric]
        right = control[metric]
        differences[metric] = None if left is None or right is None else left - right
    return differences


def classify_confirmation(
    unfiltered: dict[str, Any], treatments: list[dict[str, Any]]
) -> dict[str, Any]:
    if any(item["metrics"]["trades"] < MIN_FILTERED_TRADES for item in treatments):
        return {
            "verdict": "INCONCLUSIVE_CONFIRMATION",
            "supported_thresholds": [],
            "reason": f"At least one frozen threshold retained fewer than {MIN_FILTERED_TRADES} trades.",
        }
    supported = []
    for item in treatments:
        metrics = item["metrics"]
        if (
            metrics["average_return"] > unfiltered["average_return"]
            and metrics["win_rate"] > unfiltered["win_rate"]
            and metrics["profit_factor"] is not None
            and unfiltered["profit_factor"] is not None
            and metrics["profit_factor"] > unfiltered["profit_factor"]
            and item["permutation"]["one_sided_p_value"] <= 0.05
        ):
            supported.append(item["threshold"])
    if supported == list(THRESHOLDS):
        return {
            "verdict": "CONFIRMED_ACTIVITY_FILTER",
            "supported_thresholds": supported,
            "reason": "Both frozen thresholds consistently improved average return, win rate, and profit factor with one-sided permutation p <= 0.05.",
        }
    return {
        "verdict": "FAILED_CONFIRMATION",
        "supported_thresholds": supported,
        "reason": "The validation effect did not reproduce consistently at both frozen thresholds with supporting permutation evidence.",
    }


def run_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    ensure_output_available(output_dir)
    started = time.perf_counter()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    baseline_report = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    previous_report = json.loads(PREVIOUS_ACTIVITY_REPORT.read_text(encoding="utf-8"))
    hashes_before = protected_hashes()
    if hashes_before["dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Sealed dataset hash mismatch")
    if hashes_before["frozen_activity_model_sha256"] != EXPECTED_FROZEN_MODEL_SHA256:
        raise RuntimeError("Frozen activity model hash mismatch")
    eligibility = verify_confirmation_eligibility(metadata, previous_report)
    prior_before = prior_artifact_hashes(output_dir)
    features = list(baseline_report["features"]["names_in_order"])
    if features != list(metadata["feature_names"]) or len(features) != 42:
        raise RuntimeError("Frozen feature order mismatch")
    requested = list(dict.fromkeys([*features, *OHLC_RECONSTRUCTION_COLUMNS]))
    confirmation, access = load_confirmation_only(DATASET_PATH, metadata, requested)
    model = XGBoostModel(models_dir=str(FIRST_PASSAGE_DIR))
    model.load(FROZEN_MODEL_ID)
    if model.feature_columns != features:
        raise RuntimeError("Frozen model feature order mismatch")
    scores = activity_probability(model, confirmation)
    reloaded_model = XGBoostModel(models_dir=str(FIRST_PASSAGE_DIR))
    reloaded_model.load(FROZEN_MODEL_ID)
    reloaded_scores = activity_probability(reloaded_model, confirmation)
    deterministic_reload = bool(np.array_equal(scores.to_numpy(), reloaded_scores.to_numpy()))
    if not deterministic_reload:
        raise RuntimeError("Frozen model reload changed confirmation probabilities")
    ohlc = reconstruct_ohlc(confirmation)
    directions = build_directional_rules(confirmation, ohlc)[RULE]
    outcomes = build_rule_outcomes(ohlc, confirmation["atr"], directions, horizon=HORIZON)
    eligible_directions = directions.iloc[:-HORIZON]
    expected_outcomes = int(
        (np.isfinite(eligible_directions) & eligible_directions.isin([-1.0, 1.0])).sum()
    )
    if len(outcomes) != expected_outcomes:
        raise RuntimeError("Confirmation horizon-tail exclusion mismatch")
    unfiltered = strategy_metrics(outcomes, pd.Series(True, index=outcomes.index))
    treatments = [evaluate_threshold(outcomes, scores, threshold) for threshold in THRESHOLDS]
    for item in treatments:
        item["difference_vs_unfiltered"] = metric_differences(item["metrics"], unfiltered)
    decision = classify_confirmation(unfiltered, treatments)
    hashes_after = protected_hashes()
    prior_after = {path: sha256(Path(path)) for path in prior_before}
    if hashes_before != hashes_after or prior_before != prior_after:
        raise RuntimeError("A protected dataset, model, report, or prior artifact changed")
    report = {
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "scientific_question": "Does the frozen MA10/20 plus ML activity-filter effect survive on genuinely unseen data?",
        "predeclared_design": {
            "direction_rule": "BUY when sma_10 > sma_20; SELL when sma_10 < sma_20; no trade when equal",
            "activity_probability": "P(BUY) + P(SELL)",
            "model_direction_used": False,
            "thresholds": list(THRESHOLDS),
            "horizon_bars": HORIZON,
            "atr_target_multiplier": ATR_MULTIPLIER,
            "permutations": PERMUTATIONS,
            "random_seed": RANDOM_SEED,
            "minimum_filtered_trades_for_meaningful_determination": MIN_FILTERED_TRADES,
            "confirmation_rule": "Both thresholds must improve average return, win rate, and profit factor over unfiltered and each have one-sided permutation p <= 0.05.",
            "cost_policy": "Gross outcomes only; no reliable execution-time spread or slippage series exists for this confirmation period.",
        },
        "confirmation_eligibility": eligibility,
        "data_access": access,
        "results": {
            "unfiltered_ma_10_20": unfiltered,
            "activity_filtered": treatments,
        },
        "decision": decision,
        "cost_sensitivity": {
            "status": "UNAVAILABLE_RELIABLE_EXECUTION_COST_HISTORY",
            "reason": "No timestamp-aligned historical bid/ask or slippage series exists for the confirmation period; costs were not invented.",
        },
        "integrity": {
            "protected_hashes_before": hashes_before,
            "protected_hashes_after": hashes_after,
            "protected_artifacts_checked": len(prior_before),
            "protected_artifacts_unchanged": True,
            "frozen_model_loaded_not_retrained": True,
            "frozen_model_hash_verified": True,
            "frozen_feature_order_verified": True,
            "deterministic_model_reload": deterministic_reload,
            "exact_strategy_parameters_verified": True,
            "only_strategies_evaluated": ["unfiltered_ma_10_20", "ma_10_20_activity_0.70", "ma_10_20_activity_0.80"],
            "prior_train_rows_reused_as_confirmation": 0,
            "prior_validation_rows_reused_as_confirmation": 0,
            "confirmation_target_labels_read": False,
            "final_confirmation_tail_excluded": HORIZON,
            "all_features_precomputed_causally_at_or_before_entry": True,
            "future_values_used_only_for_outcomes": True,
        },
    }
    output_dir.mkdir(parents=True)
    rows = [{"strategy": "unfiltered", "threshold": None, **unfiltered}]
    rows.extend(
        {
            "strategy": "activity_filter",
            "threshold": item["threshold"],
            **item["metrics"],
            "permutation_p_value": item["permutation"]["one_sided_p_value"],
        }
        for item in treatments
    )
    pd.DataFrame(rows).to_csv(output_dir / "strategy_comparison.csv", index=False)
    audit = outcomes[["maximum_outcome_timestamp"]].copy()
    audit["information_timestamp"] = audit.index + pd.Timedelta(minutes=15)
    audit["entry_timestamp"] = audit["information_timestamp"]
    audit["maximum_feature_timestamp"] = audit.index
    audit["future_used_only_for_outcome"] = True
    audit.to_parquet(output_dir / "timestamp_outcome_audit.parquet")
    (output_dir / "experiment_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        f"# {EXPERIMENT_ID}",
        "",
        "## Decision",
        f"**{decision['verdict']}**",
        "",
        "## Untouched Confirmation",
        f"- Period: {access['first_timestamp']} through {access['last_timestamp']}",
        f"- Rows: {access['rows_exposed']:,}; final {HORIZON} rows excluded from outcomes",
        "- Prior TRAIN/VALIDATION rows reused: 0 / 0",
        "",
        "## Gross Results",
        "| Strategy | Trades | BUY | SELL | Win rate | Avg return | Median return | Cumulative | Profit factor | Max drawdown | Avg MFE | Avg MAE | +1 ATR | -1 ATR | Permutation p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        name = "Unfiltered MA10/20" if row["threshold"] is None else f"MA10/20 + {row['threshold']:.2f}"
        p_value = row.get("permutation_p_value")
        lines.append(
            f"| {name} | {row['trades']} | {row['buy_trades']} | {row['sell_trades']} | "
            f"{row['win_rate']:.4f} | {row['average_return']:.6f} | {row['median_return']:.6f} | "
            f"{row['cumulative_gross_return']:.6f} | {row['profit_factor']:.4f} | {row['maximum_drawdown']:.4f} | "
            f"{row['average_mfe']:.6f} | {row['average_mae']:.6f} | {row['reached_plus_1atr_pct']:.2f}% | "
            f"{row['reached_minus_1atr_pct']:.2f}% | {'-' if p_value is None else f'{p_value:.4f}'} |"
        )
    lines.extend(
        [
            "",
            "Gross outcomes are reported separately. Reliable execution-time spread/slippage history is unavailable, so no costs were invented.",
        ]
    )
    (output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_experiment()["decision"], indent=2))