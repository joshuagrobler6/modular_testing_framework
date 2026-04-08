from __future__ import annotations

from dataclasses import dataclass

from trading_lab.contracts import ActionRequest, DecisionContext, NodeContract, NodeSpec
from trading_lab.nodes.exit_shared import (
    action_request_exit,
    bars_held_from_ctx,
    entry_index_from_ctx,
    entry_price_from_ctx,
    full_price_series_from_ctx,
    history_end_index,
    position_direction,
    slice_between_indices,
)
from trading_lab.registry import register_exit

DEFAULT_EXIT_MFE_GIVEBACK_NAME = "exit_mfe_giveback"
_ALLOWED_GIVEBACK_KINDS = {"fraction", "points"}


def build_exit_mfe_giveback_contract(
    *,
    name: str = DEFAULT_EXIT_MFE_GIVEBACK_NAME,
    activation_profit_points: float = 1.0,
    giveback_kind: str = "fraction",
    giveback_fraction: float = 0.35,
    giveback_points: float = 1.0,
) -> NodeContract:
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind="exit",
            version="1.0.0",
            required_capabilities=("single_position_per_symbol",),
            emitted_action_types=("close", "hold"),
            required_history=1,
            description="MFE giveback exit using completed bars only.",
        ),
        manifest={
            "family": "mfe_giveback",
            "module": __name__,
            "class": "MfeGivebackExitNode",
            "uses_completed_bars_only": True,
            "parameters": {
                "activation_profit_points": activation_profit_points,
                "giveback_kind": giveback_kind,
                "giveback_fraction": giveback_fraction,
                "giveback_points": giveback_points,
            },
        },
    )


@dataclass(frozen=True, slots=True)
class MfeGivebackExitNode:
    activation_profit_points: float = 1.0
    giveback_kind: str = "fraction"
    giveback_fraction: float = 0.35
    giveback_points: float = 1.0

    def __post_init__(self) -> None:
        if self.activation_profit_points < 0.0:
            raise ValueError("activation_profit_points must be non-negative.")
        if self.giveback_kind not in _ALLOWED_GIVEBACK_KINDS:
            raise ValueError(f"giveback_kind must be one of {_ALLOWED_GIVEBACK_KINDS}.")
        if not 0.0 < self.giveback_fraction < 1.0:
            raise ValueError("giveback_fraction must be between 0 and 1.")
        if self.giveback_points <= 0.0:
            raise ValueError("giveback_points must be positive.")

    def __call__(self, ctx: DecisionContext) -> ActionRequest:
        direction = position_direction(ctx)
        bars_held = bars_held_from_ctx(ctx)
        entry_price = entry_price_from_ctx(ctx)
        if direction is None or bars_held is None or entry_price is None:
            return ActionRequest()

        entry_index = entry_index_from_ctx(ctx)
        if entry_index is None:
            return ActionRequest()

        closes, highs, lows = full_price_series_from_ctx(ctx)
        current_index = history_end_index(ctx)
        current_close_price = closes[current_index]
        if direction == "long":
            mfe_price = max(
                slice_between_indices(
                    highs,
                    start_index=entry_index,
                    end_index=current_index,
                )
            )
            current_profit = current_close_price - entry_price
            peak_profit = mfe_price - entry_price
        else:
            mfe_price = min(
                slice_between_indices(
                    lows,
                    start_index=entry_index,
                    end_index=current_index,
                )
            )
            current_profit = entry_price - current_close_price
            peak_profit = entry_price - mfe_price

        if peak_profit < self.activation_profit_points:
            return ActionRequest()

        if self.giveback_kind == "fraction":
            allowed_drawdown = self.giveback_fraction * peak_profit
        else:
            allowed_drawdown = self.giveback_points
        giveback = peak_profit - current_profit
        if giveback < allowed_drawdown:
            return ActionRequest()

        return action_request_exit(
            direction,
            (
                f"mfe_giveback giveback_kind={self.giveback_kind} "
                f"peak_profit={peak_profit:.6f} current_profit={current_profit:.6f} giveback={giveback:.6f}"
            ),
        )


DEFAULT_EXIT_MFE_GIVEBACK = MfeGivebackExitNode()
DEFAULT_EXIT_MFE_GIVEBACK_CONTRACT = build_exit_mfe_giveback_contract()

register_exit(
    DEFAULT_EXIT_MFE_GIVEBACK_NAME,
    DEFAULT_EXIT_MFE_GIVEBACK,
    DEFAULT_EXIT_MFE_GIVEBACK_CONTRACT,
)
