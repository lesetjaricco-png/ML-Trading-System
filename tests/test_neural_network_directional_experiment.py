"""Preflight tests for the single resource-constrained MLP experiment."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from run_ablation_25f import ensure_output_available
from run_neural_network_directional_experiment import (
    ARRAY_MEMORY_HEADROOM_MULTIPLIER,
    BATCH_SIZE,
    EXPECTED_PARAMETER_COUNT,
    HIDDEN_LAYERS,
    MAX_EPOCHS,
    PATIENCE,
    RANDOM_SEED,
    binary_metrics,
    build_model,
    chronological_internal_split,
    classify_result,
    parameter_count,
    validate_manifest,
)


def test_architecture_and_resource_manifest_are_frozen():
    model = build_model()

    assert HIDDEN_LAYERS == (32, 16)
    assert parameter_count(42) == EXPECTED_PARAMETER_COUNT == 1921
    assert ARRAY_MEMORY_HEADROOM_MULTIPLIER == 8
    assert BATCH_SIZE == 256
    assert MAX_EPOCHS == 50
    assert PATIENCE == 5
    assert model.hidden_layer_sizes == (32, 16)
    assert model.activation == "relu"
    assert model.solver == "adam"
    assert model.learning_rate_init == 0.001
    assert model.shuffle is False
    assert model.early_stopping is False


def test_feature_manifest_rejects_target_and_future_columns():
    valid = [f"feature_{index}" for index in range(42)]
    validate_manifest(valid)

    with pytest.raises(RuntimeError, match="Invalid frozen feature"):
        validate_manifest([*valid[:-1], "target"])
    with pytest.raises(RuntimeError, match="Invalid frozen feature"):
        validate_manifest([*valid[:-1], "future_close"])


def test_internal_validation_is_strictly_chronological():
    index = pd.date_range("2026-01-01", periods=100, freq="15min")
    frame = pd.DataFrame({"target": [0, 1] * 50}, index=index)

    fit, holdout = chronological_internal_split(frame)

    assert len(fit) == 90
    assert len(holdout) == 10
    assert fit.index[-1] < holdout.index[0]


def test_scaler_fit_count_excludes_validation_rows():
    train = np.arange(60, dtype=np.float32).reshape(20, 3)
    validation = np.full((5, 3), 1000, dtype=np.float32)
    scaler = StandardScaler().fit(train)
    scaler.transform(validation)

    assert int(scaler.n_samples_seen_) == len(train)
    np.testing.assert_allclose(scaler.mean_, train.mean(axis=0))


def test_model_seed_and_single_epoch_api_are_deterministic_configuration():
    model = build_model()

    assert model.random_state == RANDOM_SEED == 42
    assert model.max_iter == 1
    assert hasattr(model, "partial_fit")
    assert model.warm_start is False


def test_binary_metrics_use_fixed_half_threshold():
    actual = np.asarray([0, 0, 1, 1])
    probability = np.asarray([0.1, 0.6, 0.4, 0.9])

    metrics = binary_metrics(actual, probability)

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["prediction_distribution"]["SELL"]["count"] == 2


def _metrics(auc: float, std: float = 0.1, sell_pct: float = 50.0) -> dict:
    return {
        "roc_auc": auc,
        "probability_std": std,
        "prediction_distribution": {
            "SELL": {"percentage": sell_pct},
            "BUY": {"percentage": 100 - sell_pct},
        },
    }


def test_decision_rule_rejects_tiny_or_inconsistent_gain():
    train = _metrics(0.60)
    internal = _metrics(0.53)

    assert classify_result(train, internal, _metrics(0.515))["verdict"] == "NEURAL_NO_DIRECTIONAL_SIGNAL"
    assert classify_result(train, _metrics(0.51), _metrics(0.55))["verdict"] == "NEURAL_NO_DIRECTIONAL_SIGNAL"
    assert classify_result(train, internal, _metrics(0.55))["verdict"] == "NEURAL_DIRECTIONAL_SIGNAL"
    assert classify_result(train, _metrics(0.70), _metrics(0.55))["flags"]["cross_period_instability"] is True


def test_output_directory_cannot_be_overwritten(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        ensure_output_available(output)