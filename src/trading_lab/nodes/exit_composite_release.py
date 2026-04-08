from __future__ import annotations

from dataclasses import dataclass

from trading_lab.contracts import ActionRequest, DecisionContext, NodeContract, NodeSpec
from trading_lab.nodes.exit_shared import (
    action_request_exit,
    atr_series,
    bars_held_from_ctx,
    baseline_series,
    closes_from_ctx,
    current_close,
    diff_series,
    entry_price_from_ctx,
    highs_from_ctx,
    lows_from_ctx,
    macd_histogram_series,
    open_profit_points,
    position_direction,
    scale_series,
    slice_since_entry,
    zscore_series,
)
from trading_lab.registry import register_exit

DEFAULT_EXIT_COMPOSITE_RELEASE_NAME = "exit_composite_release"
_ALLOWED_BASELINES = {"ema", "sma", "median"}
_ALLOWED_SCALES = {"atr", "residual_std", "residual_mad"}


def build_exit_composite_release_contract(
    *,
    name: str = DEFAULT_EXIT_COMPOSITE_RELEASE_NAME,
    max_hold_bars: int | None = None,
    no_progress_bars: int | None = None,
    no_progress_min_profit_points: float = 0.0,
    profit_target_percent: float | None = None,
    profit_target_atr_multiple: float | None = None,
    atr_target_lookback: int = 20,
    trailing_atr_lookback: int | None = None,
    trailing_atr_multiple: float = 3.0,
    trailing_activation_bars: int = 1,
    macd_fast_lookback: int | None = None,
    macd_slow_lookback: int = 26,
    macd_signal_lookback: int = 9,
    macd_confirm_bars: int = 1,
    z_baseline_kind: str = "ema",
    z_baseline_lookback: int = 50,
    z_scale_kind: str = "atr",
    z_scale_lookback: int = 20,
    z_exit_threshold: float | None = None,
    z_gradient_horizon: int = 3,
    z_gradient_threshold: float | None = None,
    z_acceleration_horizon: int = 1,
    z_acceleration_threshold: float | None = None,
    z_confirm_bars: int = 1,
    mfe_activation_profit_points: float | None = None,
    mfe_giveback_fraction: float = 0.35,
) -> NodeContract:
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind="exit",
            version="1.0.0",
            required_capabilities=("single_position_per_symbol",),
            emitted_action_types=("close", "hold"),
            required_history=_required_history(
                atr_target_lookback=atr_target_lookback,
                trailing_atr_lookback=trailing_atr_lookback,
                macd_fast_lookback=macd_fast_lookback,
                macd_slow_lookback=macd_slow_lookback,
                macd_signal_lookback=macd_signal_lookback,
                macd_confirm_bars=macd_confirm_bars,
                z_baseline_lookback=z_baseline_lookback,
                z_scale_kind=z_scale_kind,
                z_scale_lookback=z_scale_lookback,
                z_gradient_horizon=z_gradient_horizon,
                z_acceleration_horizon=z_acceleration_horizon,
                z_confirm_bars=z_confirm_bars,
            ),
            description="Composite full-exit release node using completed bars only.",
        ),
        manifest={
            "family": "composite_release",
            "module": __name__,
            "class": "CompositeReleaseExitNode",
            "uses_completed_bars_only": True,
            "parameters": {
                "max_hold_bars": max_hold_bars,
                "no_progress_bars": no_progress_bars,
                "no_progress_min_profit_points": no_progress_min_profit_points,
                "profit_target_percent": profit_target_percent,
                "profit_target_atr_multiple": profit_target_atr_multiple,
                "atr_target_lookback": atr_target_lookback,
                "trailing_atr_lookback": trailing_atr_lookback,
                "trailing_atr_multiple": trailing_atr_multiple,
                "trailing_activation_bars": trailing_activation_bars,
                "macd_fast_lookback": macd_fast_lookback,
                "macd_slow_lookback": macd_slow_lookback,
                "macd_signal_lookback": macd_signal_lookback,
                "macd_confirm_bars": macd_confirm_bars,
                "z_baseline_kind": z_baseline_kind,
                "z_baseline_lookback": z_baseline_lookback,
                "z_scale_kind": z_scale_kind,
                "z_scale_lookback": z_scale_lookback,
                "z_exit_threshold": z_exit_threshold,
                "z_gradient_horizon": z_gradient_horizon,
                "z_gradient_threshold": z_gradient_threshold,
                "z_acceleration_horizon": z_acceleration_horizon,
                "z_acceleration_threshold": z_acceleration_threshold,
                "z_confirm_bars": z_confirm_bars,
                "mfe_activation_profit_points": mfe_activation_profit_points,
                "mfe_giveback_fraction": mfe_giveback_fraction,
            },
        },
    )


@dataclass(frozen=True, slots=True)
class CompositeReleaseExitNode:
    max_hold_bars: int | None = None
    no_progress_bars: int | None = None
    no_progress_min_profit_points: float = 0.0
    profit_target_percent: float | None = None
    profit_target_atr_multiple: float | None = None
    atr_target_lookback: int = 20
    trailing_atr_lookback: int | None = None
    trailing_atr_multiple: float = 3.0
    trailing_activation_bars: int = 1
    macd_fast_lookback: int | None = None
    macd_slow_lookback: int = 26
    macd_signal_lookback: int = 9
    macd_confirm_bars: int = 1
    z_baseline_kind: str = "ema"
    z_baseline_lookback: int = 50
    z_scale_kind: str = "atr"
    z_scale_lookback: int = 20
    z_exit_threshold: float | None = None
    z_gradient_horizon: int = 3
    z_gradient_threshold: float | None = None
    z_acceleration_horizon: int = 1
    z_acceleration_threshold: float | None = None
    z_confirm_bars: int = 1
    mfe_activation_profit_points: float | None = None
    mfe_giveback_fraction: float = 0.35

    def __post_init__(self) -> None:
        if self.max_hold_bars is not None and self.max_hold_bars <= 0:
            raise ValueError("max_hold_bars must be positive when provided.")
        if self.no_progress_bars is not None and self.no_progress_bars <= 0:
            raise ValueError("no_progress_bars must be positive when provided.")
        if self.no_progress_min_profit_points < 0.0:
            raise ValueError("no_progress_min_profit_points must be non-negative.")
        if self.profit_target_percent is not None and self.profit_target_percent <= 0.0:
            raise ValueError("profit_target_percent must be positive when provided.")
        if self.profit_target_atr_multiple is not None and self.profit_target_atr_multiple <= 0.0:
            raise ValueError("profit_target_atr_multiple must be positive when provided.")
        if self.atr_target_lookback <= 0:
            raise ValueError("atr_target_lookback must be positive.")
        if self.trailing_atr_lookback is not None and self.trailing_atr_lookback <= 0:
            raise ValueError("trailing_atr_lookback must be positive when provided.")
        if self.trailing_atr_multiple <= 0.0:
            raise ValueError("trailing_atr_multiple must be positive.")
        if self.trailing_activation_bars <= 0:
            raise ValueError("trailing_activation_bars must be positive.")
        if self.macd_fast_lookback is not None and self.macd_fast_lookback <= 0:
            raise ValueError("macd_fast_lookback must be positive when provided.")
        if self.macd_slow_lookback <= 0:
            raise ValueError("macd_slow_lookback must be positive.")
        if self.macd_signal_lookback <= 0:
            raise ValueError("macd_signal_lookback must be positive.")
        if self.macd_confirm_bars <= 0:
            raise ValueError("macd_confirm_bars must be positive.")
        if self.z_baseline_kind not in _ALLOWED_BASELINES:
            raise ValueError(f"z_baseline_kind must be one of {_ALLOWED_BASELINES}.")
        if self.z_scale_kind not in _ALLOWED_SCALES:
            raise ValueError(f"z_scale_kind must be one of {_ALLOWED_SCALES}.")
        if self.z_baseline_lookback <= 0:
            raise ValueError("z_baseline_lookback must be positive.")
        if self.z_scale_lookback <= 0:
            raise ValueError("z_scale_lookback must be positive.")
        if self.z_gradient_horizon <= 0:
            raise ValueError("z_gradient_horizon must be positive.")
        if self.z_acceleration_horizon <= 0:
            raise ValueError("z_acceleration_horizon must be positive.")
        if self.z_confirm_bars <= 0:
            raise ValueError("z_confirm_bars must be positive.")
        if self.z_exit_threshold is not None and self.z_exit_threshold < 0.0:
            raise ValueError("z_exit_threshold must be non-negative when provided.")
        if self.z_gradient_threshold is not None and self.z_gradient_threshold < 0.0:
            raise ValueError("z_gradient_threshold must be non-negative when provided.")
        if self.z_acceleration_threshold is not None and self.z_acceleration_threshold < 0.0:
            raise ValueError("z_acceleration_threshold must be non-negative when provided.")
        if self.mfe_activation_profit_points is not None and self.mfe_activation_profit_points < 0.0:
            raise ValueError("mfe_activation_profit_points must be non-negative when provided.")
        if not 0.0 < self.mfe_giveback_fraction < 1.0:
            raise ValueError("mfe_giveback_fraction must be between 0 and 1.")

    def __call__(self, ctx: DecisionContext) -> ActionRequest:
        direction = position_direction(ctx)
        if direction is None:
            return ActionRequest()

        reason = self._first_trigger_reason(ctx, direction)
        if reason is None:
            return ActionRequest()
        return action_request_exit(direction, reason)

    def _first_trigger_reason(self, ctx: DecisionContext, direction: str) -> str | None:
        bars_held = bars_held_from_ctx(ctx)
        entry_price = entry_price_from_ctx(ctx)
        if self.max_hold_bars is not None and bars_held is not None and bars_held >= self.max_hold_bars:
            return f"composite time_stop max_hold_bars={self.max_hold_bars} bars_held={bars_held}"

        open_profit = open_profit_points(ctx)
        if (
            self.no_progress_bars is not None
            and bars_held is not None
            and open_profit is not None
            and bars_held >= self.no_progress_bars
            and open_profit <= self.no_progress_min_profit_points
        ):
            return (
                "composite no_progress "
                f"no_progress_bars={self.no_progress_bars} "
                f"open_profit_points={open_profit:.6f}"
            )

        closes = closes_from_ctx(ctx)
        highs = highs_from_ctx(ctx)
        lows = lows_from_ctx(ctx)
        close = current_close(ctx)

        if entry_price is not None and self.profit_target_percent is not None:
            target_price = (
                entry_price * (1.0 + self.profit_target_percent)
                if direction == "long"
                else entry_price * (1.0 - self.profit_target_percent)
            )
            hit = close >= target_price if direction == "long" else close <= target_price
            if hit:
                return f"composite profit_target_percent={self.profit_target_percent:.6f}"

        if entry_price is not None and self.profit_target_atr_multiple is not None:
            atr_target = atr_series(highs, lows, closes, self.atr_target_lookback)[-1]
            if atr_target is not None and atr_target > 0.0:
                target_distance = self.profit_target_atr_multiple * atr_target
                target_price = entry_price + target_distance if direction == "long" else entry_price - target_distance
                hit = close >= target_price if direction == "long" else close <= target_price
                if hit:
                    return f"composite profit_target_atr_multiple={self.profit_target_atr_multiple:.6f}"

        if self.trailing_atr_lookback is not None and bars_held is not None and bars_held >= self.trailing_activation_bars:
            atr_trail = atr_series(highs, lows, closes, self.trailing_atr_lookback)[-1]
            if atr_trail is not None and atr_trail > 0.0:
                if direction == "long":
                    reference = max(slice_since_entry(highs, bars_held))
                    stop = reference - self.trailing_atr_multiple * atr_trail
                    if close <= stop:
                        return f"composite trailing_atr stop={stop:.6f}"
                else:
                    reference = min(slice_since_entry(lows, bars_held))
                    stop = reference + self.trailing_atr_multiple * atr_trail
                    if close >= stop:
                        return f"composite trailing_atr stop={stop:.6f}"

        if self.macd_fast_lookback is not None:
            macd_line, signal_line, histogram = macd_histogram_series(
                closes,
                fast_lookback=self.macd_fast_lookback,
                slow_lookback=self.macd_slow_lookback,
                signal_lookback=self.macd_signal_lookback,
            )
            recent_hist = histogram[-self.macd_confirm_bars:]
            recent_macd = macd_line[-self.macd_confirm_bars:]
            recent_signal = signal_line[-self.macd_confirm_bars:]
            if direction == "long":
                if all(value is not None and value < 0.0 for value in recent_hist):
                    return "composite macd_histogram_cross"
                if all(
                    macd_value is not None and signal_value is not None and macd_value < signal_value
                    for macd_value, signal_value in zip(recent_macd, recent_signal, strict=False)
                ):
                    return "composite macd_signal_cross"
            else:
                if all(value is not None and value > 0.0 for value in recent_hist):
                    return "composite macd_histogram_cross"
                if all(
                    macd_value is not None and signal_value is not None and macd_value > signal_value
                    for macd_value, signal_value in zip(recent_macd, recent_signal, strict=False)
                ):
                    return "composite macd_signal_cross"

        if any(
            value is not None
            for value in (self.z_exit_threshold, self.z_gradient_threshold, self.z_acceleration_threshold)
        ):
            baseline = baseline_series(closes, self.z_baseline_kind, self.z_baseline_lookback)
            scale = scale_series(
                closes=closes,
                highs=highs,
                lows=lows,
                baseline=baseline,
                scale_kind=self.z_scale_kind,
                scale_lookback=self.z_scale_lookback,
            )
            z_values = zscore_series(closes, baseline, scale)
            z_gradients = diff_series(z_values, self.z_gradient_horizon, divide_by_horizon=True)
            z_accelerations = diff_series(z_gradients, self.z_acceleration_horizon, divide_by_horizon=False)
            if self.z_exit_threshold is not None:
                recent = z_values[-self.z_confirm_bars:]
                if direction == "long" and all(value is not None and value <= self.z_exit_threshold for value in recent):
                    return "composite z_threshold_cross"
                if direction == "short" and all(value is not None and value >= -self.z_exit_threshold for value in recent):
                    return "composite z_threshold_cross"
            if self.z_gradient_threshold is not None:
                recent = z_gradients[-self.z_confirm_bars:]
                if direction == "long" and all(value is not None and value <= -self.z_gradient_threshold for value in recent):
                    return "composite z_gradient_flip"
                if direction == "short" and all(value is not None and value >= self.z_gradient_threshold for value in recent):
                    return "composite z_gradient_flip"
            if self.z_acceleration_threshold is not None:
                recent = z_accelerations[-self.z_confirm_bars:]
                if direction == "long" and all(value is not None and value <= -self.z_acceleration_threshold for value in recent):
                    return "composite z_acceleration_flip"
                if direction == "short" and all(value is not None and value >= self.z_acceleration_threshold for value in recent):
                    return "composite z_acceleration_flip"

        if self.mfe_activation_profit_points is not None and bars_held is not None and entry_price is not None:
            if direction == "long":
                peak_profit = max(slice_since_entry(highs, bars_held)) - entry_price
                current_profit = close - entry_price
            else:
                peak_profit = entry_price - min(slice_since_entry(lows, bars_held))
                current_profit = entry_price - close
            if peak_profit >= self.mfe_activation_profit_points:
                giveback = peak_profit - current_profit
                if giveback >= self.mfe_giveback_fraction * peak_profit:
                    return "composite mfe_giveback"

        return None



def _required_history(
    *,
    atr_target_lookback: int,
    trailing_atr_lookback: int | None,
    macd_fast_lookback: int | None,
    macd_slow_lookback: int,
    macd_signal_lookback: int,
    macd_confirm_bars: int,
    z_baseline_lookback: int,
    z_scale_kind: str,
    z_scale_lookback: int,
    z_gradient_horizon: int,
    z_acceleration_horizon: int,
    z_confirm_bars: int,
) -> int:
    histories = [1, atr_target_lookback + 1]
    if trailing_atr_lookback is not None:
        histories.append(trailing_atr_lookback + 1)
    if macd_fast_lookback is not None:
        histories.append(macd_slow_lookback + macd_signal_lookback + macd_confirm_bars)

    z_scale_history = z_scale_lookback + 1 if z_scale_kind == "atr" else z_baseline_lookback + z_scale_lookback - 1
    histories.append(max(z_baseline_lookback, z_scale_history) + z_gradient_horizon + z_acceleration_horizon + z_confirm_bars)
    return max(histories)


DEFAULT_EXIT_COMPOSITE_RELEASE = CompositeReleaseExitNode()
DEFAULT_EXIT_COMPOSITE_RELEASE_CONTRACT = build_exit_composite_release_contract()

register_exit(
    DEFAULT_EXIT_COMPOSITE_RELEASE_NAME,
    DEFAULT_EXIT_COMPOSITE_RELEASE,
    DEFAULT_EXIT_COMPOSITE_RELEASE_CONTRACT,
)
