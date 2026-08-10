"""Independent free-information treatment and integrity tests."""

from __future__ import annotations

import json

import pandas as pd

from acquire_free_external_information import validation_end_utc
from run_free_external_information_experiment import (
    MANIFEST_PATH,
    build_calendar_features,
    build_market_features,
    build_rates_features,
    build_volatility_features,
    classify_treatment,
    protected_hashes,
)


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_acquired_normalized_rows_never_cross_validation_boundary():
    manifest = _manifest()
    paths = [
        manifest["sources"]["treasury"]["data_path"],
        manifest["sources"]["volatility"]["data_path"],
        manifest["sources"]["calendar"]["data_path"],
        *[item["data_path"] for item in manifest["sources"]["mt5"]["accepted"]],
    ]

    for path in paths:
        available = pd.read_parquet(path, columns=["available_time_utc"])["available_time_utc"]
        assert available.max() <= validation_end_utc()


def test_external_feature_budgets_and_timestamp_audits():
    manifest = _manifest()
    predictions = pd.date_range("2025-09-01", periods=100, freq="15min", tz="UTC")
    expected = {
        "rates": (build_rates_features, 11),
        "market": (build_market_features, 27),
        "volatility": (build_volatility_features, 7),
        "calendar": (build_calendar_features, 10),
    }

    for _, (builder, count) in expected.items():
        features, audit = builder(predictions, manifest)
        assert features.shape == (100, count)
        assert audit.get("future_observations", 0) == 0
        assert not any("target" in column.lower() for column in features)


def test_mt5_incomplete_bars_were_not_persisted():
    manifest = _manifest()
    cutoff = validation_end_utc()

    for instrument in manifest["sources"]["mt5"]["accepted"]:
        frame = pd.read_parquet(
            instrument["data_path"], columns=["source_time_utc", "available_time_utc"]
        )
        assert (frame["available_time_utc"] == frame["source_time_utc"] + pd.Timedelta(minutes=15)).all()
        assert frame["available_time_utc"].max() <= cutoff


def test_decision_requires_validation_cv_and_permutation_evidence():
    control = {"roc_auc": 0.513}
    strong_validation = {"roc_auc": 0.535}
    strong_folds = [
        {"auc_gain": gain}
        for gain in (0.012, 0.009, 0.011, -0.001, 0.010)
    ]
    weak_folds = [{"auc_gain": 0.001}] * 5

    credible = classify_treatment(
        strong_validation,
        control,
        strong_folds,
        {"one_sided_p_value": 0.03},
    )
    weak = classify_treatment(
        strong_validation,
        control,
        weak_folds,
        {"one_sided_p_value": 0.20},
    )

    assert credible["verdict"] == "CREDIBLE_DIRECTIONAL_IMPROVEMENT"
    assert weak["verdict"] == "RETIRE_INFORMATION_FAMILY"


def test_loading_and_feature_construction_do_not_change_protected_artifacts():
    before = protected_hashes()
    manifest = _manifest()
    predictions = pd.date_range("2025-09-01", periods=10, freq="15min", tz="UTC")

    build_rates_features(predictions, manifest)
    build_volatility_features(predictions, manifest)

    assert protected_hashes() == before