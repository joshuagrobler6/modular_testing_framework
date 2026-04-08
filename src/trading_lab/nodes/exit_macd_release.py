from __future__ import annotations

from dataclasses import dataclass

from trading_lab.contracts import ActionRequest, DecisionContext, NodeContract, NodeSpec
from trading_lab.nodes.exit_shared import (
    action_request_exit,
    closes_from_ctx,
    macd_histogram_series,
    position_direction,
)
from trading_lab.registry import register_exit

DEFAULT_EXIT_MACD_RELEASE_NAME = "exit_macd_release"
_ALLOWED_RELEASE_KINDS = {"histogram_cross", "histogram_slope", "macd_signal_cross"}


def build_exit_macd_release_contract(
    *,
    name: str = DEFAULT_EXIT_MACD_RELEASE_NAME,
    fast_lookback: int = 12,
    slow_lookback: int = 26,
    signal_lookback: int = 9,
    release_kind: str = "histogram_cross",
    confirm_bars: int = 1,
) -> NodeContract:
    required_history = slow_lookback + signal_lookback + confirm_bars
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind="exit",
            version="1.0.0",
            required_capabilities=("single_position_per_symbol",),
            emitted_action_types=("close", "hold"),
            required_history=required_history,
            description="MACD momentum release exit using completed bars only.",
        ),
        manifest={
            "family": "macd_release",
            "module": __name__,
            "class": "MacdReleaseExitNode",
            "uses_completed_bars_only": True,
            "parameters": {
                "fast_lookback": fast_lookback,
                "slow_lookback": slow_lookback,
                "signal_lookback": signal_lookback,
                "release_kind": release_kind,
                "confirm_bars": confirm_bars,
            },
        },
    )


@dataclass(frozen=True, slots=True)
class MacdReleaseExitNode:
    fast_lookback: int = 12
    slow_lookback: int = 26
    signal_lookback: int = 9
    release_kind: str = "histogram_cross"
    confirm_bars: int = 1

    def __post_init__(self) -> None:
        if self.fast_lookback <= 0:
            raise ValueError("fast_lookback must be positive.")
        if self.slow_lookback <= self.fast_lookback:
            raise ValueError("slow_lookback must be greater than fast_lookback.")
        if self.signal_lookback <= 0:
            raise ValueError("signal_lookback must be positive.")
        if self.release_kind not in _ALLOWED_RELEASE_KINDS:
            raise ValueError(f"release_kind must be one of {_ALLOWED_RELEASE_KINDS}.")
        if self.confirm_bars <= 0:
            raise ValueError("confirm_bars must be positive.")

    def __call__(self, ctx: DecisionContext) -> ActionRequest:
        direction = position_direction(ctx)
        if direction is None:
            return ActionRequest()

        closes = closes_from_ctx(ctx)
        macd_line, signal_line, histogram = macd_histogram_series(
            closes,
            fast_lookback=self.fast_lookback,
            slow_lookback=self.slow_lookback,
            signal_lookback=self.signal_lookback,
        )
        if not self._should_exit(direction, macd_line, signal_line, histogram):
            return ActionRequest()

        return action_request_exit(
            direction,
            (
                f"macd_release release_kind={self.release_kind} "
                f"confirm_bars={self.confirm_bars}"
            ),
        )

    def _should_exit(
        self,
        direction: str,
        macd_line: list[float | None],
        signal_line: list[float | None],
        histogram: list[float | None],
    ) -> bool:
        if self.release_kind == "histogram_cross":
            recent = histogram[-self.confirm_bars:]
            if len(recent) < self.confirm_bars:
                return False
            if direction == "long":
                return all(value is not None and value < 0.0 for value in recent)
            return all(value is not None and value > 0.0 for value in recent)

        if self.release_kind == "histogram_slope":
            slopes = []
            for index in range(len(histogram) - self.confirm_bars, len(histogram)):
                if index <= 0 or histogram[index] is None or histogram[index - 1] is None:
                    return False
                slopes.append(histogram[index] - histogram[index - 1])
            if direction == "long":
                return all(value < 0.0 for value in slopes)
            return all(value > 0.0 for value in slopes)

        recent_macd = macd_line[-self.confirm_bars:]
        recent_signal = signal_line[-self.confirm_bars:]
        if len(recent_macd) < self.confirm_bars or len(recent_signal) < self.confirm_bars:
            return False
        comparisons = [
            (macd_value, signal_value)
            for macd_value, signal_value in zip(recent_macd, recent_signal, strict=False)
        ]
        if direction == "long":
            return all(
                macd_value is not None and signal_value is not None and macd_value < signal_value
                for macd_value, signal_value in comparisons
            )
        return all(
            macd_value is not None and signal_value is not None and macd_value > signal_value
            for macd_value, signal_value in comparisons
        )


DEFAULT_EXIT_MACD_RELEASE = MacdReleaseExitNode()
DEFAULT_EXIT_MACD_RELEASE_CONTRACT = build_exit_macd_release_contract()

register_exit(
    DEFAULT_EXIT_MACD_RELEASE_NAME,
    DEFAULT_EXIT_MACD_RELEASE,
    DEFAULT_EXIT_MACD_RELEASE_CONTRACT,
)
