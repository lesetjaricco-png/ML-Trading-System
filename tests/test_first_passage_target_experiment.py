"""Pre-training controls for the first-passage target experiment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from diagnose_baseline import load_train_validation_only
from run_first_passage_target_experiment import (
    INCOMPLETE_LABEL,
    TIE_LABEL,
    build_first_passage_target,
    ensure_output_available,
    independently_reconstruct_target,
    reconstruct_ohlc,
)


def _market(
    highs: list[float], lows: list[float], closes: list[float] | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    count = len(highs)
    index = pd.date_range("2026-01-01", periods=count, freq="15min")
    close_values = closes or [100.0] * count
    ohlc = pd.DataFrame(
        {"Open": close_values, "High": highs, "Low": lows, "Close": close_values},
        index=index,
    )
    return ohlc, pd.Series([10.0] * count, index=index)


def test_exact_five_bar_horizon_includes_fifth_excludes_sixth():
    fifth_bar_ohlc, fifth_bar_atr = _market(
        [100, 100, 100, 100, 100, 111, 100],
        [100] * 7,
    )
    sixth_bar_ohlc, sixth_bar_atr = _market(
        [100, 100, 100, 100, 100, 100, 111],
        [100] * 7,
    )

    fifth_bar_target = build_first_passage_target(fifth_bar_ohlc, fifth_bar_atr)
    sixth_bar_target = build_first_passage_target(sixth_bar_ohlc, sixth_bar_atr)

    assert fifth_bar_target.iloc[0]["first_passage_target"] == 1
    assert fifth_bar_target.iloc[0]["first_hit_bar"] == 5
    assert sixth_bar_target.iloc[0]["first_passage_target"] == 2


def test_uses_atr_from_entry_not_future_bar():
    ohlc, atr = _market([100, 106, 100, 100, 100, 100], [100] * 6)
    atr.iloc[0] = 5.0
    atr.iloc[1] = 50.0

    target = build_first_passage_target(ohlc, atr)

    assert target.iloc[0]["upper_barrier"] == pytest.approx(105.0)
    assert target.iloc[0]["first_passage_target"] == 1


def test_detects_buy_barrier_first():
    ohlc, atr = _market([100, 111, 100, 100, 100, 100], [100, 100, 89, 100, 100, 100])

    assert build_first_passage_target(ohlc, atr).iloc[0]["first_passage_target"] == 1


def test_detects_sell_barrier_first():
    ohlc, atr = _market([100, 100, 111, 100, 100, 100], [100, 89, 100, 100, 100, 100])

    assert build_first_passage_target(ohlc, atr).iloc[0]["first_passage_target"] == 0


def test_unresolved_horizon_is_no_trade():
    ohlc, atr = _market([105] * 6, [95] * 6)

    assert build_first_passage_target(ohlc, atr).iloc[0]["first_passage_target"] == 2


def test_same_bar_tie_is_explicit_and_excludable():
    ohlc, atr = _market([100, 111, 100, 100, 100, 100], [100, 89, 100, 100, 100, 100])

    target = build_first_passage_target(ohlc, atr)

    assert target.iloc[0]["first_passage_target"] == TIE_LABEL
    assert target.iloc[0]["first_hit_bar"] == 1


def test_target_uses_only_next_five_bars_and_no_feature_leakage():
    ohlc, atr = _market([100] * 8, [100] * 8)
    original = build_first_passage_target(ohlc, atr).iloc[0]["first_passage_target"]
    changed = ohlc.copy()
    changed.iloc[6, changed.columns.get_loc("High")] = 1000

    modified = build_first_passage_target(changed, atr).iloc[0]["first_passage_target"]

    assert original == modified == 2


def test_independent_reconstruction_matches_primary_builder():
    ohlc, atr = _market(
        [100, 111, 100, 100, 111, 105, 100, 100],
        [100, 100, 89, 100, 89, 95, 100, 100],
    )

    primary = build_first_passage_target(ohlc, atr)["first_passage_target"].to_numpy()
    independent = independently_reconstruct_target(ohlc, atr)

    np.testing.assert_array_equal(primary, independent)
    assert np.count_nonzero(primary == INCOMPLETE_LABEL) == 5


def test_ohlc_reconstruction_uses_only_current_causal_columns():
    index = pd.date_range("2026-01-01", periods=1, freq="15min")
    frame = pd.DataFrame(
        {
            "atr": [10.0],
            "atr_pct": [0.1],
            "close_open_ratio": [0.25],
            "upper_shadow": [3.0],
            "lower_shadow": [2.0],
        },
        index=index,
    )

    ohlc = reconstruct_ohlc(frame)

    assert ohlc.iloc[0].to_dict() == pytest.approx(
        {"Open": 80.0, "High": 103.0, "Low": 78.0, "Close": 100.0}
    )


def test_bounded_loader_keeps_test_inaccessible(tmp_path: Path):
    index = pd.date_range("2026-01-01", periods=12, freq="15min", name="Date")
    frame = pd.DataFrame(
        {"feature": np.arange(12, dtype=float), "target": [0, 1, 2] * 4}, index=index
    )
    dataset = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pandas(frame), dataset)
    metadata = {
        "splits": {
            "train": {"row_count": 5, "first_timestamp": index[0].isoformat(), "last_timestamp": index[4].isoformat()},
            "validation": {"row_count": 2, "first_timestamp": index[5].isoformat(), "last_timestamp": index[6].isoformat()},
            "test": {"row_count": 5, "first_timestamp": index[7].isoformat(), "last_timestamp": index[11].isoformat()},
        }
    }

    train, validation, access = load_train_validation_only(dataset, metadata, ["feature", "target"])

    assert len(train) + len(validation) == 7
    assert validation.index[-1] == index[6]
    assert access["test_rows_exposed"] == 0
    assert access["test_labels_exposed"] is False


def test_experiment_refuses_overwrite(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        ensure_output_available(output)