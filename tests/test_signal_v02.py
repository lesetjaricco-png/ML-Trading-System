from __future__ import annotations

import pandas as pd

from src.feature_engineering import FeatureEngineer
from src.signal_decision import SignalDecisionEngine


def test_v02_target_uses_three_classes(sample_ohlcv):
    fe = FeatureEngineer(target_mode="v0.2_directional")
    result = fe.transform(sample_ohlcv, instrument_name="US30")
    labels = set(result["target"].dropna().unique())
    assert labels.issubset({0, 1, 2})
    assert labels == {0, 1, 2}


def test_signal_decision_buy_threshold():
    engine = SignalDecisionEngine(buy_threshold=0.70, sell_threshold=0.70)
    assert engine.decide(0.72, 0.20) == 1


def test_signal_decision_sell_threshold():
    engine = SignalDecisionEngine(buy_threshold=0.70, sell_threshold=0.70)
    assert engine.decide(0.20, 0.70) == -1


def test_signal_decision_no_trade_when_both_weak():
    engine = SignalDecisionEngine(buy_threshold=0.70, sell_threshold=0.70)
    assert engine.decide(0.69, 0.40) == 0


def test_signal_decision_prefers_higher_probability_when_both_high():
    engine = SignalDecisionEngine(buy_threshold=0.70, sell_threshold=0.70)
    assert engine.decide(0.72, 0.71) == 1
    assert engine.decide(0.71, 0.72) == -1


def test_signal_decision_respects_configured_thresholds():
    engine = SignalDecisionEngine(buy_threshold=0.65, sell_threshold=0.65)
    assert engine.decide(0.66, 0.30) == 1
