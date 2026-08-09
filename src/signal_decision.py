from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SignalDecisionEngine:
    """Convert directional probabilities into BUY / SELL / NO TRADE decisions."""

    buy_threshold: float = 0.70
    sell_threshold: float = 0.70
    minimum_probability_edge: float = 0.00

    def decide(self, buy_prob: float, sell_prob: float) -> int:
        if buy_prob >= self.buy_threshold and sell_prob >= self.sell_threshold:
            if buy_prob > sell_prob + self.minimum_probability_edge:
                return 1
            if sell_prob > buy_prob + self.minimum_probability_edge:
                return -1
            return 1

        if buy_prob >= self.buy_threshold:
            return 1

        if sell_prob >= self.sell_threshold:
            return -1

        return 0
