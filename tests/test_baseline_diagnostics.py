"""Focused tests for sealed baseline diagnostics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from diagnose_baseline import (
    feature_family,
    load_train_validation_only,
    population_stability_index,
    session_table,
    target_summary,
)


def test_bounded_loader_exposes_only_train_and_validation(tmp_path: Path):
    index = pd.date_range("2026-01-01", periods=10, freq="15min", name="Date")
    frame = pd.DataFrame(
        {"feature": np.arange(10, dtype=float), "target": [0, 1, 2, 0, 1] * 2},
        index=index,
    )
    path = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pandas(frame), path)
    metadata = {
        "splits": {
            "train": {"row_count": 6, "first_timestamp": index[0].isoformat(), "last_timestamp": index[5].isoformat()},
            "validation": {"row_count": 2, "first_timestamp": index[6].isoformat(), "last_timestamp": index[7].isoformat()},
            "test": {"row_count": 2, "first_timestamp": index[8].isoformat(), "last_timestamp": index[9].isoformat()},
        }
    }

    train, validation, access = load_train_validation_only(
        path, metadata, ["feature", "target"]
    )

    assert len(train) == 6
    assert len(validation) == 2
    assert validation.index[-1] == index[7]
    assert access["test_rows_exposed"] == 0
    assert access["test_labels_exposed"] is False


def test_target_summary_and_psi_are_deterministic():
    frame = pd.DataFrame({"target": [0, 1, 2, 2]})
    assert target_summary(frame) == {
        "row_count": 4,
        "sell_pct": 25.0,
        "buy_pct": 25.0,
        "no_trade_pct": 50.0,
    }
    assert population_stability_index(np.arange(100), np.arange(100)) == 0.0


def test_feature_families_cover_requested_groups():
    assert feature_family("sma_200") == "price_level_or_scale_dependent"
    assert feature_family("atr_pct") == "normalized_relative"
    assert feature_family("hour_of_day") == "session_time"


def test_session_table_allows_hour_present_in_only_one_split():
    columns = {
        "target": [0, 1],
        "is_asia_session": [1, 1],
        "is_london_session": [0, 0],
        "is_new_york_session": [0, 0],
    }
    train = pd.DataFrame(
        columns,
        index=pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 01:00"]),
    )
    validation = pd.DataFrame(
        columns,
        index=pd.DatetimeIndex(["2026-02-01 00:00", "2026-02-01 02:00"]),
    )

    result = session_table(train, validation)
    train_only_hour = result[
        (result["dimension"] == "utc_hour")
        & (result["group"] == "1")
        & (result["split"] == "train")
    ].iloc[0]

    assert np.isnan(train_only_hour["validation_minus_train_buy_pct_pp"])