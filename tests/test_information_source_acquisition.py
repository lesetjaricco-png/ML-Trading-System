"""Free external-information acquisition gate tests."""

from __future__ import annotations

from datetime import date

import pandas as pd

from acquire_free_external_information import (
    EARLY_CLOSES,
    acquire_calendar,
    exchange_holidays,
    macro_status,
    resource_preflight,
    validation_end_utc,
)


def test_resource_preflight_is_bounded_for_six_gib_machine():
    resources = resource_preflight()

    assert resources["decision"] == "SAFE"
    assert resources["estimated_peak_ram_mib"] <= 300
    assert resources["estimated_disk_mib"] <= 150
    assert resources["market_chunk_days"] <= 31


def test_validation_boundary_is_last_validation_bar_close():
    assert validation_end_utc() == pd.Timestamp("2025-10-02T12:15:00Z")


def test_exchange_calendar_contains_known_closures():
    assert date(2024, 3, 29) in exchange_holidays(2024)
    assert date(2024, 6, 19) in exchange_holidays(2024)
    assert date(2025, 1, 9) in exchange_holidays(2025)
    assert date(2024, 7, 3) in EARLY_CLOSES
    assert date(2024, 11, 29) in EARLY_CLOSES


def test_calendar_acquisition_is_normalized_and_capped(tmp_path):
    result = acquire_calendar(tmp_path)
    frame = pd.read_parquet(result["data_path"])

    assert result["status"] == "ACQUIRED"
    assert frame["available_time_utc"].max() <= validation_end_utc()
    assert set(frame["field"]) == {
        "is_trading_day",
        "holiday",
        "early_close",
        "day_before_holiday",
        "first_trading_day",
        "last_trading_day",
        "month_end",
        "quarter_end",
        "year_end",
    }
    assert frame["raw_batch_sha256"].str.len().eq(64).all()


def test_macro_fails_closed_without_point_in_time_vintages():
    status = macro_status()

    assert status["status"] == "UNAVAILABLE_RELIABLE_HISTORY"
    assert status["consensus"] == "UNAVAILABLE_RELIABLE_HISTORY"
    assert status["rows"] == 0