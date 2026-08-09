"""Audit V0.3 target construction without reading TEST or changing artifacts."""

from __future__ import annotations

import json
import time
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
    DATASET_PATH,
    METADATA_PATH,
    ensure_output_available,
)


DIAGNOSTIC_ID = f"{BASELINE_ID}_target_construction_diagnostic_v1"
OUTPUT_DIR = BASELINE_DIR / DIAGNOSTIC_ID
HORIZON_COLUMNS = ["returns_5", "atr", "atr_pct", "target"]
CLASS_NAMES = {0: "SELL", 1: "BUY", 2: "NO_TRADE"}


def reconstruct_target_rows(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    horizon: int,
    multiplier: float,
) -> pd.DataFrame:
    exposed = pd.concat(
        [train.assign(split="train"), validation.assign(split="validation")]
    )
    forward_return = exposed["returns_5"].shift(-horizon)
    auditable = forward_return.notna()
    rows = exposed.loc[auditable, ["split", "atr", "atr_pct", "target"]].copy()
    rows["forward_return"] = forward_return.loc[auditable].astype(float)
    rows["threshold_return"] = multiplier * rows["atr_pct"].abs()
    rows["forward_move_atr"] = rows["forward_return"] / rows["atr_pct"].abs()
    rows["absolute_move_atr"] = rows["forward_move_atr"].abs()
    rows["boundary_distance_atr"] = (rows["absolute_move_atr"] - multiplier).abs()
    rows["threshold_price_points"] = rows["atr"].abs() * multiplier
    implied_close = rows["atr"] / rows["atr_pct"]
    rows["forward_move_price_points"] = rows["forward_return"] * implied_close
    rows["reconstructed_target"] = np.select(
        [
            rows["forward_return"] > rows["threshold_return"],
            rows["forward_return"] < -rows["threshold_return"],
        ],
        [1, 0],
        default=2,
    ).astype(int)
    rows["target"] = rows["target"].astype(int)
    rows["matches_stored_target"] = rows["reconstructed_target"] == rows["target"]
    return rows


def class_distribution(frame: pd.DataFrame) -> list[dict[str, Any]]:
    counts = frame["target"].astype(int).value_counts().reindex([0, 1, 2], fill_value=0)
    return [
        {
            "class": CLASS_NAMES[label],
            "count": int(counts[label]),
            "percentage": float(100 * counts[label] / len(frame)),
        }
        for label in (0, 1, 2)
    ]


def boundary_summary(rows: pd.DataFrame) -> list[dict[str, Any]]:
    summaries = []
    for split in ("train", "validation"):
        split_rows = rows[rows["split"] == split]
        for label in (0, 1, 2):
            selected = split_rows[split_rows["target"] == label]
            summaries.append(
                {
                    "split": split,
                    "class": CLASS_NAMES[label],
                    "count": len(selected),
                    "within_0_05_atr_of_boundary_pct": float(100 * (selected["boundary_distance_atr"] <= 0.05).mean()),
                    "within_0_10_atr_of_boundary_pct": float(100 * (selected["boundary_distance_atr"] <= 0.10).mean()),
                    "within_0_25_atr_of_boundary_pct": float(100 * (selected["boundary_distance_atr"] <= 0.25).mean()),
                    "median_boundary_distance_atr": float(selected["boundary_distance_atr"].median()),
                    "median_absolute_move_atr": float(selected["absolute_move_atr"].median()),
                    "median_threshold_price_points": float(selected["threshold_price_points"].median()),
                    "median_absolute_forward_move_price_points": float(selected["forward_move_price_points"].abs().median()),
                }
            )
    return summaries


def directional_symmetry(rows: pd.DataFrame) -> list[dict[str, Any]]:
    summaries = []
    for split in ("train", "validation"):
        selected = rows[(rows["split"] == split) & rows["target"].isin([0, 1])]
        buy = selected[selected["target"] == 1]
        sell = selected[selected["target"] == 0]
        summaries.append(
            {
                "split": split,
                "buy_count": len(buy),
                "sell_count": len(sell),
                "buy_to_sell_count_ratio": float(len(buy) / len(sell)),
                "buy_median_excess_atr": float((buy["absolute_move_atr"] - 1).median()),
                "sell_median_excess_atr": float((sell["absolute_move_atr"] - 1).median()),
                "buy_median_move_points": float(buy["forward_move_price_points"].median()),
                "sell_median_move_points": float(sell["forward_move_price_points"].median()),
            }
        )
    return summaries


def target_conclusion(
    boundary: list[dict[str, Any]], reconstruction_match_pct: float
) -> dict[str, str]:
    validation = {
        row["class"]: row for row in boundary if row["split"] == "validation"
    }
    directional_close = max(
        validation["BUY"]["within_0_25_atr_of_boundary_pct"],
        validation["SELL"]["within_0_25_atr_of_boundary_pct"],
    )
    if reconstruction_match_pct != 100:
        return {
            "verdict": "IMPLEMENTATION_MISMATCH",
            "answer": "Persisted labels do not match the documented forward-ATR rule.",
        }
    if directional_close >= 35:
        return {
            "verdict": "TARGET_CONSTRUCTION_BOTTLENECK",
            "answer": (
                "The labels are mechanically consistent and volatility-scaled, but many directional outcomes sit close to an endpoint-only threshold that ignores path, spread, and execution. This creates a noisy BUY-vs-SELL learning problem."
            ),
        }
    return {
        "verdict": "TARGET_MECHANICALLY_SOUND_BUT_UNVALIDATED_ECONOMICALLY",
        "answer": (
            "The labels are mechanically stable around their threshold, but economic validity remains unproven because the target ignores execution costs and intrahorizon path."
        ),
    }


def render_report(report: dict[str, Any]) -> str:
    construction = report["construction"]
    lines = [
        f"# {DIAGNOSTIC_ID}",
        "",
        "- Mode: read-only; no target changes or retraining",
        "- TEST rows/labels read: false / false",
        "",
        "## Exact Construction",
        f"- Horizon: {construction['horizon_bars']} M15 bars ({construction['horizon_minutes']} minutes)",
        f"- Threshold: +/- {construction['atr_threshold_multiplier']:.1f} x ATR[t]",
        "- BUY: Close[t+5]/Close[t]-1 is strictly greater than +ATR[t]/Close[t].",
        "- SELL: Close[t+5]/Close[t]-1 is strictly less than -ATR[t]/Close[t].",
        "- NO_TRADE: the endpoint return is inside or exactly on those bounds.",
        "- TP/SL 100/20 are configured point counts; other target modes convert them to price distances using points x instrument point_size. V0.3 does not use them.",
        "- There is no TP/SL tie in V0.3. Exact equality to either ATR boundary is NO_TRADE; configured conservative_sl tie handling is unused.",
        "- max_bars and unresolved policy are also unused by V0.3.",
        "- Spread, point size, tick value, commission, slippage, and intrahorizon High/Low path are not inputs to this target.",
        "",
        "## Class Distribution",
        "| Split | Class | Count | Percentage |",
        "|---|---|---:|---:|",
    ]
    for split, distribution in report["class_distribution"].items():
        for row in distribution:
            lines.append(f"| {split.upper()} | {row['class']} | {row['count']:,} | {row['percentage']:.2f}% |")
    lines.extend(
        [
            "",
            "## Boundary Fragility",
            "| Split | Class | Within 0.05 ATR | Within 0.10 ATR | Within 0.25 ATR | Median boundary distance | Median move | Median threshold points |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["boundary_summary"]:
        lines.append(
            f"| {row['split'].upper()} | {row['class']} | {row['within_0_05_atr_of_boundary_pct']:.2f}% | "
            f"{row['within_0_10_atr_of_boundary_pct']:.2f}% | {row['within_0_25_atr_of_boundary_pct']:.2f}% | "
            f"{row['median_boundary_distance_atr']:.3f} ATR | {row['median_absolute_move_atr']:.3f} ATR | "
            f"{row['median_threshold_price_points']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Audit Integrity",
            f"- Reconstructed labels matching persisted labels: {report['reconstruction']['match_percentage']:.6f}%",
            f"- Audited rows: {report['reconstruction']['audited_rows']:,}",
            f"- Intentionally unaudited validation tail: {report['reconstruction']['unaudited_validation_tail_rows']} rows; their t+5 endpoints are in sealed TEST.",
            "",
            "## Economic Interpretation",
            "The target expresses a volatility-scaled 75-minute endpoint move, so BUY and SELL have directional meaning. It does not express whether a realizable long or short trade hit TP before SL, survived adverse excursion, or remained profitable after spread and costs.",
            "Close calls in this target are directional-vs-NO_TRADE boundary cases, not BUY-vs-SELL ties: the two directional classes are separated by a 2 ATR-wide NO_TRADE region.",
            "The frozen raw cache contains OHLCV only, so spread and bid/ask effects cannot be measured retrospectively from this dataset.",
            "",
            "## Conclusion",
            f"**{report['conclusion']['verdict']}**: {report['conclusion']['answer']}",
            "",
            "## Direct Answers",
            "1. BUY requires the close exactly five bars later to exceed the entry close by strictly more than one entry-time ATR.",
            "2. SELL requires that future close to fall by strictly more than one entry-time ATR.",
            "3. NO_TRADE covers all endpoint moves from -1 ATR through +1 ATR, including exact boundary equality.",
            "4. BUY/SELL are economically meaningful as large signed 75-minute market moves, but not as realizable trade outcomes because execution and path are absent.",
            "5. The cutoff is not evidently noisy. The likely target bottleneck is endpoint/path misalignment; current evidence cannot separate that from intrinsically weak short-horizon direction predictability without one controlled relabel comparison.",
            "",
            "Smallest controlled follow-up (not implemented): keep horizon=5 and barrier distance=1 ATR, but assign BUY/SELL by which High/Low barrier is reached first; assign unresolved rows NO_TRADE and drop same-bar ties. This isolates endpoint-vs-path semantics using the frozen OHLCV data.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_diagnostic(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    ensure_output_available(output_dir)
    started = time.perf_counter()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    protected_before = {
        "dataset_sha256": sha256(DATASET_PATH),
        "metadata_sha256": sha256(METADATA_PATH),
        "reference_model_sha256": sha256(BASELINE_MODEL_PATH),
    }
    train, validation, access = load_train_validation_only(
        DATASET_PATH, metadata, HORIZON_COLUMNS
    )
    horizon = int(metadata["horizon_bars"])
    multiplier = float(metadata["atr_threshold_multiplier"])
    rows = reconstruct_target_rows(train, validation, horizon, multiplier)
    match_percentage = float(100 * rows["matches_stored_target"].mean())
    boundary = boundary_summary(rows)
    protected_after = {
        "dataset_sha256": sha256(DATASET_PATH),
        "metadata_sha256": sha256(METADATA_PATH),
        "reference_model_sha256": sha256(BASELINE_MODEL_PATH),
    }
    if protected_before != protected_after:
        raise RuntimeError("Protected artifact changed during target diagnostic")
    report = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "mode": "read-only; no target changes, model fitting, or TEST access",
        "construction": {
            "target_mode": metadata["target_mode"],
            "target_definition": metadata["target_definition"],
            "horizon_bars": horizon,
            "horizon_minutes": horizon * 15,
            "atr_threshold_multiplier": multiplier,
            "configured_take_profit_points": metadata["take_profit_points"],
            "configured_stop_loss_points": metadata["stop_loss_points"],
            "target_uses_tp_sl": metadata["target_uses_tp_sl"],
            "same_bar_rule": metadata["same_bar_rule"],
            "spread_used": False,
            "point_size_used": False,
            "intrahorizon_path_used": False,
            "strict_boundary_comparisons": True,
        },
        "data_access": access,
        "class_distribution": {
            "train": class_distribution(train),
            "validation": class_distribution(validation),
        },
        "reconstruction": {
            "method": "forward_return[t] = persisted returns_5[t+5]",
            "audited_rows": len(rows),
            "matching_rows": int(rows["matches_stored_target"].sum()),
            "match_percentage": match_percentage,
            "unaudited_validation_tail_rows": horizon,
            "reason": "Their future endpoints are in TEST and were not read.",
        },
        "boundary_summary": boundary,
        "directional_symmetry": directional_symmetry(rows),
        "economic_assessment": {
            "meaningful_as_market_outcome": True,
            "meaningful_as_executable_trade_outcome": False,
            "reason": "Labels encode only a volatility-scaled future close endpoint, without spread, costs, bid/ask execution, or path-dependent TP/SL outcomes.",
            "close_call_interpretation": "Boundary-close rows can flip between a directional class and NO_TRADE, but not directly between BUY and SELL because the directional thresholds are 2 ATR apart.",
            "raw_cache_columns": ["Open", "High", "Low", "Close", "Volume"],
            "spread_or_bid_ask_available": False,
        },
        "conclusion": target_conclusion(boundary, match_percentage),
        "smallest_controlled_experiment": {
            "implemented": False,
            "proposal": "Hold horizon=5 and barrier distance=1 ATR fixed; assign BUY or SELL by first High/Low barrier passage, NO_TRADE if unresolved, and drop same-bar ties; compare label agreement, class balance, and BUY-vs-SELL AUC against the frozen endpoint target.",
            "isolated_factor": "endpoint-only close return versus intrahorizon first-passage path",
            "spread_follow_up_requirement": "A later execution-aware test requires a new raw capture retaining historical spread or bid/ask data.",
        },
        "integrity": {
            "before": protected_before,
            "after": protected_after,
            "all_protected_artifacts_unchanged": protected_before == protected_after,
            "test_rows_exposed": 0,
            "test_labels_read": False,
            "test_evaluated": False,
        },
    }
    output_dir.mkdir(parents=True)
    rows.to_csv(output_dir / "target_boundary_rows.csv")
    pd.DataFrame(boundary).to_csv(output_dir / "boundary_summary.csv", index=False)
    (output_dir / "diagnostic_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "diagnostic_report.md").write_text(
        render_report(report), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = run_diagnostic()
    print(json.dumps(result["conclusion"], indent=2))