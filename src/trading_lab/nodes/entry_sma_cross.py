from __future__ import annotations

from dataclasses import dataclass

from trading_lab.contracts import ActionRequest, DecisionContext, NodeContract, NodeSpec
from trading_lab.registry import register_entry

DEFAULT_ENTRY_SMA_CROSS_NAME = "entry_sma_cross"


def _window_mean(values: list[float], window: int) -> float:
    return sum(values[-window:]) / window


def build_entry_sma_cross_contract(
    *,
    name: str = DEFAULT_ENTRY_SMA_CROSS_NAME,
    fast_window: int = 2,
    slow_window: int = 3,
    allow_short_signals: bool = True,
) -> NodeContract:
    emitted_action_types: tuple[str, ...] = (
        ("enter_long", "enter_short", "hold")
        if allow_short_signals
        else ("enter_long", "hold")
    )
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind="entry",
            version="1.0.0",
            required_capabilities=("single_position_per_symbol",),
            emitted_action_types=emitted_action_types,
            required_history=slow_window + 1,
            description="Simple moving-average crossover entry using completed bars only.",
        ),
        manifest={
            "family": "baseline",
            "module": __name__,
            "class": "SMACrossEntryNode",
            "uses_completed_bars_only": True,
            "parameters": {
                "fast_window": fast_window,
                "slow_window": slow_window,
                "allow_short_signals": allow_short_signals,
            },
        },
    )


@dataclass(frozen=True, slots=True)
class SMACrossEntryNode:
    fast_window: int = 2
    slow_window: int = 3
    allow_short_signals: bool = True

    def __post_init__(self) -> None:
        if self.fast_window <= 0:
            raise ValueError("fast_window must be positive.")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be greater than fast_window.")

    def __call__(self, ctx: DecisionContext) -> ActionRequest:
        if not ctx.position.is_flat:
            return ActionRequest()

        closes = [float(bar.close) for bar in ctx.history]
        if len(closes) < self.slow_window + 1:
            return ActionRequest()

        previous_closes = closes[:-1]
        current_fast = _window_mean(closes, self.fast_window)
        current_slow = _window_mean(closes, self.slow_window)
        previous_fast = _window_mean(previous_closes, self.fast_window)
        previous_slow = _window_mean(previous_closes, self.slow_window)

        if previous_fast <= previous_slow and current_fast > current_slow:
            return ActionRequest(
                action_type="enter_long",
                reason=(
                    f"sma_cross_up fast={self.fast_window} slow={self.slow_window}"
                ),
            )

        if (
            self.allow_short_signals
            and previous_fast >= previous_slow
            and current_fast < current_slow
        ):
            return ActionRequest(
                action_type="enter_short",
                reason=(
                    f"sma_cross_down fast={self.fast_window} slow={self.slow_window}"
                ),
            )

        return ActionRequest()


DEFAULT_ENTRY_SMA_CROSS = SMACrossEntryNode()
DEFAULT_ENTRY_SMA_CROSS_CONTRACT = build_entry_sma_cross_contract()

register_entry(
    DEFAULT_ENTRY_SMA_CROSS_NAME,
    DEFAULT_ENTRY_SMA_CROSS,
    DEFAULT_ENTRY_SMA_CROSS_CONTRACT,
)


__all__ = [
    "DEFAULT_ENTRY_SMA_CROSS",
    "DEFAULT_ENTRY_SMA_CROSS_CONTRACT",
    "DEFAULT_ENTRY_SMA_CROSS_NAME",
    "SMACrossEntryNode",
    "build_entry_sma_cross_contract",
]
