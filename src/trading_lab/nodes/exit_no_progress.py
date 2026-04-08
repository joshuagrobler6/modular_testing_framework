from __future__ import annotations

from dataclasses import dataclass

from trading_lab.contracts import ActionRequest, DecisionContext, NodeContract, NodeSpec
from trading_lab.nodes.exit_shared import (
    action_request_exit,
    bars_held_from_ctx,
    entry_price_from_ctx,
    open_profit_points,
    position_direction,
)
from trading_lab.registry import register_exit

DEFAULT_EXIT_NO_PROGRESS_NAME = "exit_no_progress"


def build_exit_no_progress_contract(
    *,
    name: str = DEFAULT_EXIT_NO_PROGRESS_NAME,
    evaluation_bars: int = 5,
    min_open_profit_points: float = 0.0,
) -> NodeContract:
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind="exit",
            version="1.0.0",
            required_capabilities=("single_position_per_symbol",),
            emitted_action_types=("close", "hold"),
            required_history=1,
            description="No-progress timeout exit using completed bars only.",
        ),
        manifest={
            "family": "no_progress",
            "module": __name__,
            "class": "NoProgressExitNode",
            "uses_completed_bars_only": True,
            "parameters": {
                "evaluation_bars": evaluation_bars,
                "min_open_profit_points": min_open_profit_points,
            },
        },
    )


@dataclass(frozen=True, slots=True)
class NoProgressExitNode:
    evaluation_bars: int = 5
    min_open_profit_points: float = 0.0

    def __post_init__(self) -> None:
        if self.evaluation_bars <= 0:
            raise ValueError("evaluation_bars must be positive.")
        if self.min_open_profit_points < 0.0:
            raise ValueError("min_open_profit_points must be non-negative.")

    def __call__(self, ctx: DecisionContext) -> ActionRequest:
        direction = position_direction(ctx)
        if direction is None:
            return ActionRequest()

        bars_held = bars_held_from_ctx(ctx)
        entry_price = entry_price_from_ctx(ctx)
        open_profit = open_profit_points(ctx)
        if bars_held is None or entry_price is None or open_profit is None:
            return ActionRequest()
        if bars_held < self.evaluation_bars:
            return ActionRequest()
        if open_profit > self.min_open_profit_points:
            return ActionRequest()

        return action_request_exit(
            direction,
            (
                "no_progress "
                f"evaluation_bars={self.evaluation_bars} "
                f"min_open_profit_points={self.min_open_profit_points:.6f} "
                f"open_profit_points={open_profit:.6f}"
            ),
        )


DEFAULT_EXIT_NO_PROGRESS = NoProgressExitNode()
DEFAULT_EXIT_NO_PROGRESS_CONTRACT = build_exit_no_progress_contract()

register_exit(
    DEFAULT_EXIT_NO_PROGRESS_NAME,
    DEFAULT_EXIT_NO_PROGRESS,
    DEFAULT_EXIT_NO_PROGRESS_CONTRACT,
)
