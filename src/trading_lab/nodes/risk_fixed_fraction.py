from __future__ import annotations

import math
from dataclasses import dataclass

from trading_lab.contracts import (
    ActionBatch,
    DecisionContext,
    NodeContract,
    NodeSpec,
    RiskDecision,
)
from trading_lab.registry import register_risk

DEFAULT_RISK_FIXED_FRACTION_NAME = "risk_fixed_fraction"


def _round_down_to_increment(quantity: float, increment: float) -> float:
    steps = math.floor(quantity / increment)
    return steps * increment


def build_risk_fixed_fraction_contract(
    *,
    name: str = DEFAULT_RISK_FIXED_FRACTION_NAME,
    capital_fraction: float = 0.1,
    stop_loss_pct: float | None = None,
    max_holding_bars: int | None = None,
) -> NodeContract:
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind="risk",
            version="1.0.0",
            required_capabilities=("portfolio_view", "single_position_per_symbol"),
            emitted_action_types=("hold",),
            required_history=1,
            requires_portfolio_view=True,
            description="Fixed-fraction sizing with optional stop-loss metadata and bar limits.",
        ),
        manifest={
            "family": "baseline",
            "module": __name__,
            "class": "FixedFractionRiskNode",
            "parameters": {
                "capital_fraction": capital_fraction,
                "stop_loss_pct": stop_loss_pct,
                "max_holding_bars": max_holding_bars,
            },
            "sizing_mode": "fixed_fraction",
            "stop_loss_is_advisory_only": stop_loss_pct is not None,
        },
    )


@dataclass(frozen=True, slots=True)
class FixedFractionRiskNode:
    capital_fraction: float = 0.1
    stop_loss_pct: float | None = None
    max_holding_bars: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.capital_fraction <= 1.0:
            raise ValueError("capital_fraction must be in the interval (0, 1].")
        if self.stop_loss_pct is not None and self.stop_loss_pct <= 0.0:
            raise ValueError("stop_loss_pct must be positive when provided.")
        if self.max_holding_bars is not None and self.max_holding_bars <= 0:
            raise ValueError("max_holding_bars must be positive when provided.")

    def __call__(
        self,
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        allow_exit = True

        if not entry_intent.is_active or not ctx.position.is_flat:
            return RiskDecision(
                allow_exit=allow_exit,
                reason=self._risk_reason(ctx, entry_intent, exit_intent),
            )

        if self.max_holding_bars is not None:
            remaining_bars = ctx.session.bars_total - ctx.session.bar_index - 1
            if remaining_bars < self.max_holding_bars:
                return RiskDecision(
                    allow_exit=allow_exit,
                    reason=(
                        "insufficient_remaining_bars "
                        f"max_holding_bars={self.max_holding_bars}"
                    ),
                )

        unit_notional = ctx.bar.close * ctx.instrument.contract_multiplier
        allocation = ctx.portfolio.equity * self.capital_fraction
        raw_quantity = allocation / unit_notional
        quantity = _round_down_to_increment(
            raw_quantity,
            ctx.instrument.quantity_increment,
        )

        if quantity <= 0.0:
            return RiskDecision(
                allow_exit=allow_exit,
                reason="insufficient_capital_for_minimum_quantity",
            )

        return RiskDecision(
            allow_entry=True,
            entry_quantity=quantity,
            allow_exit=allow_exit,
            reason=self._risk_reason(ctx, entry_intent, exit_intent, quantity=quantity),
        )

    def _risk_reason(
        self,
        ctx: DecisionContext,
        entry_intent: EntryIntent,
        exit_intent: ExitIntent,
        *,
        quantity: float | None = None,
    ) -> str:
        parts = [f"capital_fraction={self.capital_fraction:.4f}"]
        if quantity is not None:
            parts.append(f"quantity={quantity:g}")

        if self.stop_loss_pct is not None and entry_intent.is_active:
            if entry_intent.action_type == "enter_long":
                stop_price = ctx.bar.close * (1.0 - self.stop_loss_pct)
            else:
                stop_price = ctx.bar.close * (1.0 + self.stop_loss_pct)
            parts.append(f"stop_loss_pct={self.stop_loss_pct:.4f}")
            parts.append(f"stop_reference={stop_price:.6f}")

        if self.max_holding_bars is not None:
            parts.append(f"max_holding_bars={self.max_holding_bars}")

        if exit_intent.is_active:
            parts.append("exit_pending")

        return " ".join(parts)


DEFAULT_RISK_FIXED_FRACTION = FixedFractionRiskNode()
DEFAULT_RISK_FIXED_FRACTION_CONTRACT = build_risk_fixed_fraction_contract()

register_risk(
    DEFAULT_RISK_FIXED_FRACTION_NAME,
    DEFAULT_RISK_FIXED_FRACTION,
    DEFAULT_RISK_FIXED_FRACTION_CONTRACT,
)


__all__ = [
    "DEFAULT_RISK_FIXED_FRACTION",
    "DEFAULT_RISK_FIXED_FRACTION_CONTRACT",
    "DEFAULT_RISK_FIXED_FRACTION_NAME",
    "FixedFractionRiskNode",
    "build_risk_fixed_fraction_contract",
]
