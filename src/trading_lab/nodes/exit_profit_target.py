from __future__ import annotations

from dataclasses import dataclass

from trading_lab.contracts import ActionRequest, DecisionContext, NodeContract, NodeSpec
from trading_lab.nodes.exit_shared import (
    action_request_exit,
    atr_series_from_ctx,
    current_close,
    entry_price_from_ctx,
    history_end_index,
    position_direction,
)
from trading_lab.registry import register_exit

DEFAULT_EXIT_PROFIT_TARGET_NAME = "exit_profit_target"
_ALLOWED_TARGET_KINDS = {"percent", "atr_from_entry"}


def build_exit_profit_target_contract(
    *,
    name: str = DEFAULT_EXIT_PROFIT_TARGET_NAME,
    target_kind: str = "atr_from_entry",
    target_percent: float = 0.02,
    atr_lookback: int = 20,
    atr_multiple: float = 2.0,
) -> NodeContract:
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind="exit",
            version="1.0.0",
            required_capabilities=("single_position_per_symbol",),
            emitted_action_types=("close", "hold"),
            required_history=atr_lookback + 1 if target_kind == "atr_from_entry" else 1,
            description="Full take-profit exit using completed bars only.",
        ),
        manifest={
            "family": "profit_target",
            "module": __name__,
            "class": "ProfitTargetExitNode",
            "uses_completed_bars_only": True,
            "parameters": {
                "target_kind": target_kind,
                "target_percent": target_percent,
                "atr_lookback": atr_lookback,
                "atr_multiple": atr_multiple,
            },
        },
    )


@dataclass(frozen=True, slots=True)
class ProfitTargetExitNode:
    target_kind: str = "atr_from_entry"
    target_percent: float = 0.02
    atr_lookback: int = 20
    atr_multiple: float = 2.0

    def __post_init__(self) -> None:
        if self.target_kind not in _ALLOWED_TARGET_KINDS:
            raise ValueError(f"target_kind must be one of {_ALLOWED_TARGET_KINDS}.")
        if self.target_percent <= 0.0:
            raise ValueError("target_percent must be positive.")
        if self.atr_lookback <= 0:
            raise ValueError("atr_lookback must be positive.")
        if self.atr_multiple <= 0.0:
            raise ValueError("atr_multiple must be positive.")

    def __call__(self, ctx: DecisionContext) -> ActionRequest:
        direction = position_direction(ctx)
        entry_price = entry_price_from_ctx(ctx)
        if direction is None or entry_price is None:
            return ActionRequest()

        close = current_close(ctx)
        target_price = self._target_price(ctx, direction, entry_price)
        if target_price is None:
            return ActionRequest()

        hit = close >= target_price if direction == "long" else close <= target_price
        if not hit:
            return ActionRequest()

        return action_request_exit(
            direction,
            (
                f"profit_target target_kind={self.target_kind} "
                f"target_price={target_price:.6f} close={close:.6f}"
            ),
        )

    def _target_price(self, ctx: DecisionContext, direction: str, entry_price: float) -> float | None:
        if self.target_kind == "percent":
            multiplier = 1.0 + self.target_percent if direction == "long" else 1.0 - self.target_percent
            return entry_price * multiplier

        atr_values = atr_series_from_ctx(ctx, self.atr_lookback)
        atr = atr_values[history_end_index(ctx)]
        if atr is None or atr <= 0.0:
            return None
        distance = self.atr_multiple * atr
        return entry_price + distance if direction == "long" else entry_price - distance


DEFAULT_EXIT_PROFIT_TARGET = ProfitTargetExitNode()
DEFAULT_EXIT_PROFIT_TARGET_CONTRACT = build_exit_profit_target_contract()

register_exit(
    DEFAULT_EXIT_PROFIT_TARGET_NAME,
    DEFAULT_EXIT_PROFIT_TARGET,
    DEFAULT_EXIT_PROFIT_TARGET_CONTRACT,
)
