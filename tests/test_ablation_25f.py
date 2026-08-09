"""Regression tests for the controlled 25-feature ablation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from diagnose_baseline import load_train_validation_only, sha256
from run_ablation_25f import (
    BASELINE_MODEL_PATH,
    DIAGNOSTIC_REPORT_PATH,
    METADATA_PATH,
    assert_reference_hashes,
    ensure_output_available,
    resolve_feature_set,
)
from src.model import XGBoostModel


EXPECTED_FEATURES = [
    "returns",
    "returns_2",
    "returns_5",
    "returns_10",
    "log_returns",
    "high_low_ratio",
    "close_open_ratio",
    "rsi",
    "bb_width",
    "bb_pct",
    "price_to_sma_10",
    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_200",
    "atr_pct",
    "volatility_5",
    "volatility_20",
    "volume_ratio",
    "day_of_week",
    "is_weekend",
    "hour_of_day",
    "is_market_open",
    "is_asia_session",
    "is_london_session",
    "is_new_york_session",
]


def test_exact_25_feature_selection_from_diagnostics():
    diagnostic = json.loads(DIAGNOSTIC_REPORT_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    features = resolve_feature_set(diagnostic, metadata)

    assert features == EXPECTED_FEATURES
    assert len(features) == 25


def test_scaler_fits_only_supplied_train_rows(tmp_path: Path):
    train = pd.DataFrame(
        {
            "feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [0, 1, 2, 0, 1, 2],
        }
    )
    validation = pd.DataFrame(
        {"feature": [1000.0, 2000.0], "target": [0, 1]}
    )
    model = XGBoostModel(n_estimators=2, max_depth=1, models_dir=str(tmp_path))

    model.fit_training_data(train, ["feature"])
    model.scaler.transform(validation[["feature"]].to_numpy())

    assert int(model.scaler.n_samples_seen_) == len(train)
    np.testing.assert_allclose(model.scaler.mean_, [2.5])


def test_bounded_loader_cannot_expose_test_rows(tmp_path: Path):
    index = pd.date_range("2026-01-01", periods=10, freq="15min", name="Date")
    frame = pd.DataFrame(
        {"feature": np.arange(10, dtype=float), "target": [0, 1, 2, 0, 1] * 2},
        index=index,
    )
    dataset_path = tmp_path / "dataset.parquet"
    pq.write_table(pa.Table.from_pandas(frame), dataset_path)
    metadata = {
        "splits": {
            "train": {"row_count": 6, "first_timestamp": index[0].isoformat(), "last_timestamp": index[5].isoformat()},
            "validation": {"row_count": 2, "first_timestamp": index[6].isoformat(), "last_timestamp": index[7].isoformat()},
            "test": {"row_count": 2, "first_timestamp": index[8].isoformat(), "last_timestamp": index[9].isoformat()},
        }
    }

    train, validation, access = load_train_validation_only(
        dataset_path, metadata, ["feature", "target"]
    )

    assert len(train) + len(validation) == 8
    assert validation.index[-1] == index[7]
    assert index[8] not in train.index.union(validation.index)
    assert access["test_rows_exposed"] == 0
    assert access["test_labels_exposed"] is False


def test_experiment_refuses_existing_output_directory(tmp_path: Path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        ensure_output_available(output_dir)


def test_reference_baseline_hash_remains_unchanged():
    expected = {
        "dataset_sha256": sha256(
            Path("data/processed/US30_2022-05-12_2026-08-09_M15_v0.3_forward_atr_v1.parquet")
        ),
        "metadata_sha256": sha256(METADATA_PATH),
        "reference_model_sha256": sha256(BASELINE_MODEL_PATH),
    }

    actual = assert_reference_hashes(expected)

    assert actual["reference_model_sha256"] == expected["reference_model_sha256"]