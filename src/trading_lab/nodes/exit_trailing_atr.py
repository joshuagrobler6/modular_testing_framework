from __future__ import annotations

from dataclasses import dataclass

from trading_lab.contracts import ActionRequest, DecisionContext, NodeContract, NodeSpec
from trading_lab.nodes.exit_shared import (
    action_request_exit,
    atr_series,
    bars_held_from_ctx,
    closes_from_ctx,
    highs_from_ctx,
    lows_from_ctx,
    position_direction,
    slice_since_entry,
)
from trading_lab.registry import register_exit

DEFAULT_EXIT_TRAILING_ATR_NAME = "exit_trailing_atr"
_ALLOWED_REFERENCE_KINDS = {"highest_high", "highest_close", "lowest_low", "lowest_close"}


def build_exit_trailing_atr_contract(
    *,
    name: str = DEFAULT_EXIT_TRAILING_ATR_NAME,
    atr_lookback: int = 20,
    atr_multiple: float = 3.0,
    reference_kind: str = "highest_high",
    activation_bars: int = 1,
) -> NodeContract:
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind="exit",
            version="1.0.0",
            required_capabilities=("single_position_per_symbol",),
            emitted_action_types=("close", "hold"),
            required_history=atr_lookback + 1,
            description="ATR trailing exit using completed bars only.",
        ),
        manifest={
            "family": "trailing_atr",
            "module": __name__,
            "class": "TrailingAtrExitNode",
            "uses_completed_bars_only": True,
            "parameters": {
                "atr_lookback": atr_lookback,
                "atr_multiple": atr_multiple,
                "reference_kind": reference_kind,
                "activation_bars": activation_bars,
            },
        },
    )


@dataclass(frozen=True, slots=True)
class TrailingAtrExitNode:
    atr_lookback: int = 20
    atr_multiple: float = 3.0
    reference_kind: str = "highest_high"
    activation_bars: int = 1

    def __post_init__(self) -> None:
        if self.atr_lookback <= 0:
            raise ValueError("atr_lookback must be positive.")
        if self.atr_multiple <= 0.0:
            raise ValueError("atr_multiple must be positive.")
        if self.reference_kind not in _ALLOWED_REFERENCE_KINDS:
            raise ValueError(f"reference_kind must be one of {_ALLOWED_REFERENCE_KINDS}.")
        if self.activation_bars <= 0:
            raise ValueError("activation_bars must be positive.")

    def __call__(self, ctx: DecisionContext) -> ActionRequest:
        direction = position_direction(ctx)
        bars_held = bars_held_from_ctx(ctx)
        if direction is None or bars_held is None or bars_held < self.activation_bars:
            return ActionRequest()

        closes = closes_from_ctx(ctx)
        highs = highs_from_ctx(ctx)
        lows = lows_from_ctx(ctx)
        atr = atr_series(highs, lows, closes, self.atr_lookback)[-1]
        if atr is None or atr <= 0.0:
            return ActionRequest()

        trail_stop = self._trail_stop(direction, closes, highs, lows, bars_held, atr)
        close = closes[-1]
        crossed = close <= trail_stop if direction == "long" else close >= trail_stop
        if not crossed:
            return ActionRequest()

        return action_request_exit(
            direction,
            (
                f"trailing_atr reference_kind={self.reference_kind} "
                f"atr_multiple={self.atr_multiple:.4f} stop={trail_stop:.6f} close={close:.6f}"
            ),
        )

    def _trail_stop(
        self,
        direction: str,
        closes: list[float],
        highs: list[float],
        lows: list[float],
        bars_held: int,
        atr: float,
    ) -> float:
        if direction == "long":
            reference_series = highs if self.reference_kind == "highest_high" else closes
            reference = max(slice_since_entry(reference_series, bars_held))
            return reference - self.atr_multiple * atr

        reference_series = lows if self.reference_kind == "lowest_low" else closes
        reference = min(slice_since_entry(reference_series, bars_held))
        return reference + self.atr_multiple * atr


DEFAULT_EXIT_TRAILING_ATR = TrailingAtrExitNode()
DEFAULT_EXIT_TRAILING_ATR_CONTRACT = build_exit_trailing_atr_contract()

register_exit(
    DEFAULT_EXIT_TRAILING_ATR_NAME,
    DEFAULT_EXIT_TRAILING_ATR,
    DEFAULT_EXIT_TRAILING_ATR_CONTRACT,
)
