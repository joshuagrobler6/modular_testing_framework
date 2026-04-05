from __future__ import annotations

from dataclasses import dataclass

from trading_lab.contracts import ActionRequest, DecisionContext, NodeContract, NodeSpec
from trading_lab.registry import register_exit

DEFAULT_EXIT_TIME_STOP_NAME = "exit_time_stop"


def _bars_in_trade(ctx: DecisionContext) -> int:
    if ctx.position.is_flat or ctx.position.entry_time is None:
        return 0

    for index, bar in enumerate(ctx.history):
        if bar.timestamp == ctx.position.entry_time:
            return len(ctx.history) - index

    raise ValueError("position.entry_time was not found in ctx.history.")


def build_exit_time_stop_contract(
    *,
    name: str = DEFAULT_EXIT_TIME_STOP_NAME,
    hold_bars: int = 2,
) -> NodeContract:
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind="exit",
            version="1.0.0",
            required_capabilities=("single_position_per_symbol",),
            emitted_action_types=("close", "hold"),
            required_history=max(1, hold_bars),
            description="Time-based exit that closes after a configured number of bars.",
        ),
        manifest={
            "family": "baseline",
            "module": __name__,
            "class": "TimeStopExitNode",
            "uses_completed_bars_only": True,
            "parameters": {"hold_bars": hold_bars},
        },
    )


@dataclass(frozen=True, slots=True)
class TimeStopExitNode:
    hold_bars: int = 2

    def __post_init__(self) -> None:
        if self.hold_bars <= 0:
            raise ValueError("hold_bars must be positive.")

    def __call__(self, ctx: DecisionContext) -> ActionRequest:
        if ctx.position.is_flat:
            return ActionRequest()

        bars_in_trade = _bars_in_trade(ctx)
        if bars_in_trade >= self.hold_bars:
            return ActionRequest(
                action_type="close",
                reason=f"time_stop hold_bars={self.hold_bars}",
            )
        return ActionRequest()


DEFAULT_EXIT_TIME_STOP = TimeStopExitNode()
DEFAULT_EXIT_TIME_STOP_CONTRACT = build_exit_time_stop_contract()

register_exit(
    DEFAULT_EXIT_TIME_STOP_NAME,
    DEFAULT_EXIT_TIME_STOP,
    DEFAULT_EXIT_TIME_STOP_CONTRACT,
)


__all__ = [
    "DEFAULT_EXIT_TIME_STOP",
    "DEFAULT_EXIT_TIME_STOP_CONTRACT",
    "DEFAULT_EXIT_TIME_STOP_NAME",
    "TimeStopExitNode",
    "build_exit_time_stop_contract",
]
