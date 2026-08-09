"""Focused tests for the read-only V0.3 target-construction diagnostic."""

from pathlib import Path

import pandas as pd
import pytest

from diagnose_target_construction import (
    boundary_summary,
    ensure_output_available,
    reconstruct_target_rows,
    target_conclusion,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2026-01-01", periods=8, freq="15min")
    frame = pd.DataFrame(
        {
            "returns_5": [0.0, 0.0, 0.0, 0.0, 0.0, 0.011, -0.012, 0.010],
            "atr": [100.0] * 8,
            "atr_pct": [0.010] * 8,
            "target": [1, 0, 2, 2, 2, 2, 2, 2],
        },
        index=index,
    )
    return frame.iloc[:2], frame.iloc[2:]


def test_reconstruction_uses_strict_five_bar_endpoint_rule():
    train, validation = _frames()

    rows = reconstruct_target_rows(train, validation, horizon=5, multiplier=1.0)

    assert rows["reconstructed_target"].tolist() == [1, 0, 2]
    assert rows["matches_stored_target"].all()
    assert rows.iloc[2]["forward_return"] == pytest.approx(0.010)
    assert rows.iloc[2]["boundary_distance_atr"] == pytest.approx(0.0)


def test_boundary_summary_identifies_close_directional_labels():
    train, validation = _frames()
    rows = reconstruct_target_rows(train, validation, horizon=5, multiplier=1.0)

    summary = boundary_summary(rows)
    train_buy = next(
        row for row in summary if row["split"] == "train" and row["class"] == "BUY"
    )

    assert train_buy["within_0_10_atr_of_boundary_pct"] == pytest.approx(100.0)
    assert train_buy["median_absolute_move_atr"] == pytest.approx(1.1)


def test_close_directional_outcomes_flag_target_bottleneck():
    boundary = [
        {"split": "validation", "class": "BUY", "within_0_25_atr_of_boundary_pct": 40.0},
        {"split": "validation", "class": "SELL", "within_0_25_atr_of_boundary_pct": 38.0},
        {"split": "validation", "class": "NO_TRADE", "within_0_25_atr_of_boundary_pct": 20.0},
    ]

    conclusion = target_conclusion(boundary, reconstruction_match_pct=100.0)

    assert conclusion["verdict"] == "TARGET_CONSTRUCTION_BOTTLENECK"


def test_reconstruction_mismatch_has_priority():
    boundary = [
        {"split": "validation", "class": class_name, "within_0_25_atr_of_boundary_pct": 0.0}
        for class_name in ("BUY", "SELL", "NO_TRADE")
    ]

    conclusion = target_conclusion(boundary, reconstruction_match_pct=99.0)

    assert conclusion["verdict"] == "IMPLEMENTATION_MISMATCH"


def test_target_diagnostic_refuses_existing_output(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        ensure_output_available(output)