"""Shared point-in-time contracts for external information sources."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCHEMA_COLUMNS = [
    "source",
    "dataset_version",
    "instrument_or_event_id",
    "field",
    "value",
    "source_time_utc",
    "available_time_utc",
    "received_time_utc",
    "revision_id",
    "quality_flags",
    "raw_batch_sha256",
]
TIMESTAMP_COLUMNS = [
    "source_time_utc",
    "available_time_utc",
    "received_time_utc",
]
FORBIDDEN_FEATURE_TOKENS = ("target", "future", "label", "outcome", "lead")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_feature_names(names: Iterable[str]) -> None:
    rejected = [
        name
        for name in names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if rejected:
        raise RuntimeError(f"Forbidden target/future-derived features: {rejected}")


def normalize_observations(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SCHEMA_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Normalized observations are missing columns: {missing}")
    normalized = frame.loc[:, SCHEMA_COLUMNS].copy()
    for column in TIMESTAMP_COLUMNS:
        normalized[column] = pd.to_datetime(normalized[column], utc=True, errors="raise")
    if normalized["available_time_utc"].isna().any():
        raise ValueError("available_time_utc cannot be missing")
    if (normalized["available_time_utc"] < normalized["source_time_utc"]).any():
        raise ValueError("An observation cannot be available before its source time")
    if normalized["raw_batch_sha256"].str.fullmatch(r"[0-9a-f]{64}").ne(True).any():
        raise ValueError("Every normalized row requires a lowercase SHA-256 hash")
    return normalized.sort_values(
        ["instrument_or_event_id", "field", "available_time_utc", "revision_id"],
        kind="stable",
    ).reset_index(drop=True)


def causal_asof_join(
    prediction_times: pd.DatetimeIndex,
    observations: pd.DataFrame,
    *,
    feature_name: str,
    maximum_age: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_feature_names([feature_name])
    if prediction_times.tz is None:
        raise RuntimeError("Prediction timestamps must be timezone-aware UTC")
    if str(prediction_times.tz) != "UTC":
        raise RuntimeError(f"Prediction timezone must be UTC, got {prediction_times.tz}")
    if not prediction_times.is_monotonic_increasing or not prediction_times.is_unique:
        raise RuntimeError("Prediction timestamps must be unique and chronological")
    normalized = normalize_observations(observations)
    if normalized["instrument_or_event_id"].nunique() != 1 or normalized["field"].nunique() != 1:
        raise ValueError("causal_asof_join accepts exactly one instrument/event field")

    source = normalized[["value", "source_time_utc", "available_time_utc", "revision_id"]]
    source = source.sort_values("available_time_utc", kind="stable")
    left = pd.DataFrame({"prediction_time_utc": prediction_times})
    joined = pd.merge_asof(
        left,
        source,
        left_on="prediction_time_utc",
        right_on="available_time_utc",
        direction="backward",
        tolerance=maximum_age,
        allow_exact_matches=True,
    )
    valid = joined["available_time_utc"].notna()
    if (joined.loc[valid, "available_time_utc"] > joined.loc[valid, "prediction_time_utc"]).any():
        raise RuntimeError("Future observation passed causal alignment")
    age = joined["prediction_time_utc"] - joined["available_time_utc"]
    features = pd.DataFrame(
        {feature_name: joined["value"].to_numpy()}, index=prediction_times
    )
    audit = joined.assign(age_seconds=age.dt.total_seconds()).set_index(
        "prediction_time_utc"
    )
    return features, audit


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable metadata: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def estimate_frame_mib(rows: int, numeric_columns: int, text_columns: int = 0) -> float:
    estimated_bytes = rows * (numeric_columns * 8 + text_columns * 64 + 32)
    return estimated_bytes / 2**20