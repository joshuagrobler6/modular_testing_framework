from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Iterable, Sequence

from trading_lab.contracts import (
    ActionRequest,
    Bar,
    BarHistorySeries,
    BarHistoryWindow,
    DecisionContext,
    NodeContract,
    NodeSpec,
)
from trading_lab.registry import register_entry

DEFAULT_ENTRY_ZSCORE_STATE_NAME = "entry_zscore_state"

_ALLOWED_BASELINES = {"ema", "sma", "median"}
_ALLOWED_SCALES = {"atr", "residual_std", "residual_mad"}


def build_entry_zscore_state_contract(
    *,
    name: str = DEFAULT_ENTRY_ZSCORE_STATE_NAME,
    baseline_kind: str = "ema",
    baseline_lookback: int = 50,
    scale_kind: str = "atr",
    scale_lookback: int = 20,
    z_threshold: float = 0.75,
    gradient_threshold: float | None = None,
    gradient_horizon: int = 3,
    acceleration_threshold: float | None = None,
    acceleration_horizon: int = 1,
    persistence_bars: int = 1,
    require_baseline_slope: bool = False,
    baseline_slope_horizon: int = 5,
    allow_short_signals: bool = True,
) -> NodeContract:
    emitted = (
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
            emitted_action_types=emitted,
            required_history=_required_history(
                baseline_kind=baseline_kind,
                baseline_lookback=baseline_lookback,
                scale_kind=scale_kind,
                scale_lookback=scale_lookback,
                gradient_horizon=gradient_horizon,
                acceleration_horizon=acceleration_horizon,
                persistence_bars=persistence_bars,
                require_baseline_slope=require_baseline_slope,
                baseline_slope_horizon=baseline_slope_horizon,
            ),
            description=(
                "Z-score state entry using completed bars only. "
                "Supports threshold, gradient, acceleration, persistence, and baseline slope gating."
            ),
        ),
        manifest={
            "family": "zscore_state",
            "module": __name__,
            "class": "ZScoreStateEntryNode",
            "uses_completed_bars_only": True,
            "parameters": {
                "baseline_kind": baseline_kind,
                "baseline_lookback": baseline_lookback,
                "scale_kind": scale_kind,
                "scale_lookback": scale_lookback,
                "z_threshold": z_threshold,
                "gradient_threshold": gradient_threshold,
                "gradient_horizon": gradient_horizon,
                "acceleration_threshold": acceleration_threshold,
                "acceleration_horizon": acceleration_horizon,
                "persistence_bars": persistence_bars,
                "require_baseline_slope": require_baseline_slope,
                "baseline_slope_horizon": baseline_slope_horizon,
                "allow_short_signals": allow_short_signals,
            },
        },
    )


@dataclass(frozen=True, slots=True)
class ZScoreStateEntryNode:
    baseline_kind: str = "ema"
    baseline_lookback: int = 50
    scale_kind: str = "atr"
    scale_lookback: int = 20
    z_threshold: float = 0.75
    gradient_threshold: float | None = None
    gradient_horizon: int = 3
    acceleration_threshold: float | None = None
    acceleration_horizon: int = 1
    persistence_bars: int = 1
    require_baseline_slope: bool = False
    baseline_slope_horizon: int = 5
    allow_short_signals: bool = True

    def __post_init__(self) -> None:
        if self.baseline_kind not in _ALLOWED_BASELINES:
            raise ValueError(f"baseline_kind must be one of {_ALLOWED_BASELINES}.")
        if self.scale_kind not in _ALLOWED_SCALES:
            raise ValueError(f"scale_kind must be one of {_ALLOWED_SCALES}.")
        if self.baseline_lookback <= 0:
            raise ValueError("baseline_lookback must be positive.")
        if self.scale_lookback <= 0:
            raise ValueError("scale_lookback must be positive.")
        if self.gradient_horizon <= 0:
            raise ValueError("gradient_horizon must be positive.")
        if self.acceleration_horizon <= 0:
            raise ValueError("acceleration_horizon must be positive.")
        if self.persistence_bars <= 0:
            raise ValueError("persistence_bars must be positive.")
        if self.baseline_slope_horizon <= 0:
            raise ValueError("baseline_slope_horizon must be positive.")
        if self.z_threshold < 0.0:
            raise ValueError("z_threshold must be non-negative.")
        if self.gradient_threshold is not None and self.gradient_threshold < 0.0:
            raise ValueError("gradient_threshold must be non-negative when provided.")
        if self.acceleration_threshold is not None and self.acceleration_threshold < 0.0:
            raise ValueError("acceleration_threshold must be non-negative when provided.")

    def __call__(self, ctx: DecisionContext) -> ActionRequest:
        if not ctx.position.is_flat:
            return ActionRequest()

        if isinstance(ctx.history, BarHistoryWindow):
            signal_plan = _cached_signal_plan(
                ctx.history.series,
                self.baseline_kind,
                self.baseline_lookback,
                self.scale_kind,
                self.scale_lookback,
                self.z_threshold,
                self.gradient_threshold,
                self.gradient_horizon,
                self.acceleration_threshold,
                self.acceleration_horizon,
                self.persistence_bars,
                self.require_baseline_slope,
                self.baseline_slope_horizon,
                self.allow_short_signals,
            )
            return signal_plan[ctx.history.end_index]

        signal_plan = _compute_flat_signal_plan(
            tuple(ctx.history),
            baseline_kind=self.baseline_kind,
            baseline_lookback=self.baseline_lookback,
            scale_kind=self.scale_kind,
            scale_lookback=self.scale_lookback,
            z_threshold=self.z_threshold,
            gradient_threshold=self.gradient_threshold,
            gradient_horizon=self.gradient_horizon,
            acceleration_threshold=self.acceleration_threshold,
            acceleration_horizon=self.acceleration_horizon,
            persistence_bars=self.persistence_bars,
            require_baseline_slope=self.require_baseline_slope,
            baseline_slope_horizon=self.baseline_slope_horizon,
            allow_short_signals=self.allow_short_signals,
        )
        return signal_plan[-1] if signal_plan else ActionRequest()


@lru_cache(maxsize=48)
def _cached_signal_plan(
    series: BarHistorySeries,
    baseline_kind: str,
    baseline_lookback: int,
    scale_kind: str,
    scale_lookback: int,
    z_threshold: float,
    gradient_threshold: float | None,
    gradient_horizon: int,
    acceleration_threshold: float | None,
    acceleration_horizon: int,
    persistence_bars: int,
    require_baseline_slope: bool,
    baseline_slope_horizon: int,
    allow_short_signals: bool,
) -> tuple[ActionRequest, ...]:
    return _compute_flat_signal_plan(
        series.bars,
        baseline_kind=baseline_kind,
        baseline_lookback=baseline_lookback,
        scale_kind=scale_kind,
        scale_lookback=scale_lookback,
        z_threshold=z_threshold,
        gradient_threshold=gradient_threshold,
        gradient_horizon=gradient_horizon,
        acceleration_threshold=acceleration_threshold,
        acceleration_horizon=acceleration_horizon,
        persistence_bars=persistence_bars,
        require_baseline_slope=require_baseline_slope,
        baseline_slope_horizon=baseline_slope_horizon,
        allow_short_signals=allow_short_signals,
    )


def _compute_flat_signal_plan(
    bars: Sequence[Bar],
    *,
    baseline_kind: str,
    baseline_lookback: int,
    scale_kind: str,
    scale_lookback: int,
    z_threshold: float,
    gradient_threshold: float | None,
    gradient_horizon: int,
    acceleration_threshold: float | None,
    acceleration_horizon: int,
    persistence_bars: int,
    require_baseline_slope: bool,
    baseline_slope_horizon: int,
    allow_short_signals: bool,
) -> tuple[ActionRequest, ...]:
    if not bars:
        return ()

    required_history = _required_history(
        baseline_kind=baseline_kind,
        baseline_lookback=baseline_lookback,
        scale_kind=scale_kind,
        scale_lookback=scale_lookback,
        gradient_horizon=gradient_horizon,
        acceleration_horizon=acceleration_horizon,
        persistence_bars=persistence_bars,
        require_baseline_slope=require_baseline_slope,
        baseline_slope_horizon=baseline_slope_horizon,
    )

    closes = tuple(float(bar.close) for bar in bars)
    highs = tuple(float(bar.high) for bar in bars)
    lows = tuple(float(bar.low) for bar in bars)

    baseline = _baseline_series(closes, baseline_kind, baseline_lookback)
    scale = _scale_series(
        closes=closes,
        highs=highs,
        lows=lows,
        baseline=baseline,
        scale_kind=scale_kind,
        scale_lookback=scale_lookback,
    )
    z_series = _zscore_series(closes, baseline, scale)
    gradient_series = _diff_series(z_series, gradient_horizon, divide_by_horizon=True)
    acceleration_series = _diff_series(
        gradient_series,
        acceleration_horizon,
        divide_by_horizon=False,
    )

    actions: list[ActionRequest] = []
    for index, z_value in enumerate(z_series):
        if index + 1 < required_history or z_value is None:
            actions.append(ActionRequest())
            continue

        gradient_value = gradient_series[index]
        if gradient_threshold is not None and gradient_value is None:
            actions.append(ActionRequest())
            continue

        acceleration_value = acceleration_series[index]
        if acceleration_threshold is not None and acceleration_value is None:
            actions.append(ActionRequest())
            continue

        baseline_slope = None
        if require_baseline_slope:
            baseline_slope = _value_diff_at(
                baseline,
                index=index,
                horizon=baseline_slope_horizon,
            )
            if baseline_slope is None:
                actions.append(ActionRequest())
                continue

        if _passes_long_conditions(
            z_series=z_series,
            index=index,
            z_threshold=z_threshold,
            persistence_bars=persistence_bars,
            gradient_threshold=gradient_threshold,
            gradient_value=gradient_value,
            acceleration_threshold=acceleration_threshold,
            acceleration_value=acceleration_value,
            require_baseline_slope=require_baseline_slope,
            baseline_slope=baseline_slope,
        ):
            actions.append(
                ActionRequest(
                    action_type="enter_long",
                    reason=_build_reason(
                        direction="long",
                        z_value=z_value,
                        gradient_value=gradient_value,
                        acceleration_value=acceleration_value,
                        baseline_kind=baseline_kind,
                        scale_kind=scale_kind,
                    ),
                )
            )
            continue

        if allow_short_signals and _passes_short_conditions(
            z_series=z_series,
            index=index,
            z_threshold=z_threshold,
            persistence_bars=persistence_bars,
            gradient_threshold=gradient_threshold,
            gradient_value=gradient_value,
            acceleration_threshold=acceleration_threshold,
            acceleration_value=acceleration_value,
            require_baseline_slope=require_baseline_slope,
            baseline_slope=baseline_slope,
        ):
            actions.append(
                ActionRequest(
                    action_type="enter_short",
                    reason=_build_reason(
                        direction="short",
                        z_value=z_value,
                        gradient_value=gradient_value,
                        acceleration_value=acceleration_value,
                        baseline_kind=baseline_kind,
                        scale_kind=scale_kind,
                    ),
                )
            )
            continue

        actions.append(ActionRequest())

    return tuple(actions)


def _passes_long_conditions(
    *,
    z_series: Sequence[float | None],
    index: int,
    z_threshold: float,
    persistence_bars: int,
    gradient_threshold: float | None,
    gradient_value: float | None,
    acceleration_threshold: float | None,
    acceleration_value: float | None,
    require_baseline_slope: bool,
    baseline_slope: float | None,
) -> bool:
    if not _recent_all_at_least(z_series, index=index, count=persistence_bars, threshold=z_threshold):
        return False
    if gradient_threshold is not None:
        if gradient_value is None or gradient_value < gradient_threshold:
            return False
    if acceleration_threshold is not None:
        if acceleration_value is None or acceleration_value < acceleration_threshold:
            return False
    if require_baseline_slope:
        if baseline_slope is None or baseline_slope <= 0.0:
            return False
    return True


def _passes_short_conditions(
    *,
    z_series: Sequence[float | None],
    index: int,
    z_threshold: float,
    persistence_bars: int,
    gradient_threshold: float | None,
    gradient_value: float | None,
    acceleration_threshold: float | None,
    acceleration_value: float | None,
    require_baseline_slope: bool,
    baseline_slope: float | None,
) -> bool:
    if not _recent_all_at_most(z_series, index=index, count=persistence_bars, threshold=-z_threshold):
        return False
    if gradient_threshold is not None:
        if gradient_value is None or gradient_value > -gradient_threshold:
            return False
    if acceleration_threshold is not None:
        if acceleration_value is None or acceleration_value > -acceleration_threshold:
            return False
    if require_baseline_slope:
        if baseline_slope is None or baseline_slope >= 0.0:
            return False
    return True


def _required_history(
    *,
    baseline_kind: str,
    baseline_lookback: int,
    scale_kind: str,
    scale_lookback: int,
    gradient_horizon: int,
    acceleration_horizon: int,
    persistence_bars: int,
    require_baseline_slope: bool,
    baseline_slope_horizon: int,
) -> int:
    baseline_history = baseline_lookback
    if baseline_kind == "ema":
        baseline_history = baseline_lookback

    if scale_kind == "atr":
        scale_history = scale_lookback + 1
    else:
        scale_history = baseline_history + scale_lookback - 1

    core_history = max(baseline_history, scale_history)
    derivative_history = gradient_horizon + acceleration_horizon
    persistence_history = persistence_bars - 1
    slope_history = baseline_slope_horizon if require_baseline_slope else 0
    return core_history + max(derivative_history, persistence_history, slope_history)


def _baseline_series(values: Sequence[float], kind: str, lookback: int) -> list[float | None]:
    if kind == "ema":
        return _ema_series(values, lookback)
    if kind == "sma":
        return _rolling_mean(values, lookback)
    if kind == "median":
        return _rolling_median(values, lookback)
    raise ValueError(f"Unsupported baseline kind: {kind}.")


def _scale_series(
    *,
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    baseline: Sequence[float | None],
    scale_kind: str,
    scale_lookback: int,
) -> list[float | None]:
    if scale_kind == "atr":
        true_ranges = _true_range_series(highs, lows, closes)
        return _rolling_mean(true_ranges, scale_lookback)

    residuals = [
        None if base is None else close - base
        for close, base in zip(closes, baseline, strict=False)
    ]
    if scale_kind == "residual_std":
        return _rolling_std_nullable(residuals, scale_lookback)
    if scale_kind == "residual_mad":
        return _rolling_mad_nullable(residuals, scale_lookback)
    raise ValueError(f"Unsupported scale kind: {scale_kind}.")


def _zscore_series(
    closes: Sequence[float],
    baseline: Sequence[float | None],
    scale: Sequence[float | None],
) -> list[float | None]:
    output: list[float | None] = []
    for close, base, scaler in zip(closes, baseline, scale, strict=False):
        if base is None or scaler is None or scaler <= 0.0:
            output.append(None)
            continue
        output.append((close - base) / scaler)
    return output


def _true_range_series(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> list[float]:
    output: list[float] = []
    previous_close: float | None = None
    for high, low, close in zip(highs, lows, closes, strict=False):
        if previous_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - previous_close), abs(low - previous_close))
        output.append(tr)
        previous_close = close
    return output


def _rolling_mean(values: Sequence[float], lookback: int) -> list[float | None]:
    output: list[float | None] = []
    window: deque[float] = deque()
    window_sum = 0.0
    for value in values:
        window.append(value)
        window_sum += value
        if len(window) > lookback:
            window_sum -= window.popleft()
        if len(window) == lookback:
            output.append(window_sum / lookback)
        else:
            output.append(None)
    return output


def _rolling_median(values: Sequence[float], lookback: int) -> list[float | None]:
    output: list[float | None] = []
    window: deque[float] = deque()
    ordered_window: list[float] = []
    for value in values:
        window.append(value)
        insort(ordered_window, value)
        if len(window) > lookback:
            removed = window.popleft()
            removal_index = bisect_left(ordered_window, removed)
            del ordered_window[removal_index]
        if len(window) == lookback:
            output.append(_median_from_sorted_values(ordered_window))
        else:
            output.append(None)
    return output


def _ema_series(values: Sequence[float], lookback: int) -> list[float | None]:
    alpha = 2.0 / (lookback + 1.0)
    ema: float | None = None
    output: list[float | None] = []
    for index, value in enumerate(values):
        ema = value if ema is None else alpha * value + (1.0 - alpha) * ema
        output.append(ema if index + 1 >= lookback else None)
    return output


def _rolling_std_nullable(values: Sequence[float | None], lookback: int) -> list[float | None]:
    output: list[float | None] = []
    window: deque[float] = deque()
    window_sum = 0.0
    window_sum_sq = 0.0
    for value in values:
        if value is None:
            output.append(None)
            continue
        numeric_value = float(value)
        window.append(numeric_value)
        window_sum += numeric_value
        window_sum_sq += numeric_value * numeric_value
        if len(window) > lookback:
            removed = window.popleft()
            window_sum -= removed
            window_sum_sq -= removed * removed
        if len(window) == lookback:
            mean = window_sum / lookback
            variance = max((window_sum_sq / lookback) - (mean * mean), 0.0)
            output.append(sqrt(variance))
        else:
            output.append(None)
    return output


def _rolling_mad_nullable(values: Sequence[float | None], lookback: int) -> list[float | None]:
    output: list[float | None] = []
    window: deque[float] = deque()
    for value in values:
        if value is None:
            output.append(None)
            continue
        numeric_value = float(value)
        window.append(numeric_value)
        if len(window) > lookback:
            window.popleft()
        if len(window) == lookback:
            ordered_window = sorted(window)
            median = _median_from_sorted_values(ordered_window)
            deviations = sorted(abs(item - median) for item in window)
            output.append(_median_from_sorted_values(deviations))
        else:
            output.append(None)
    return output


def _diff_series(
    values: Sequence[float | None],
    horizon: int,
    *,
    divide_by_horizon: bool,
) -> list[float | None]:
    output: list[float | None] = []
    for index, value in enumerate(values):
        prior_index = index - horizon
        if prior_index < 0 or value is None or values[prior_index] is None:
            output.append(None)
            continue
        difference = value - values[prior_index]
        output.append(difference / horizon if divide_by_horizon else difference)
    return output


def _value_diff_at(
    values: Sequence[float | None],
    *,
    index: int,
    horizon: int,
) -> float | None:
    prior_index = index - horizon
    if prior_index < 0:
        return None
    current = values[index]
    prior = values[prior_index]
    if current is None or prior is None:
        return None
    return current - prior


def _recent_all_at_least(
    values: Sequence[float | None],
    *,
    index: int,
    count: int,
    threshold: float,
) -> bool:
    start = index - count + 1
    if start < 0:
        return False
    for offset in range(start, index + 1):
        value = values[offset]
        if value is None or value < threshold:
            return False
    return True


def _recent_all_at_most(
    values: Sequence[float | None],
    *,
    index: int,
    count: int,
    threshold: float,
) -> bool:
    start = index - count + 1
    if start < 0:
        return False
    for offset in range(start, index + 1):
        value = values[offset]
        if value is None or value > threshold:
            return False
    return True


def _median_from_sorted_values(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value.")
    midpoint = len(values) // 2
    if len(values) % 2 == 1:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2.0


def _build_reason(
    *,
    direction: str,
    z_value: float,
    gradient_value: float | None,
    acceleration_value: float | None,
    baseline_kind: str,
    scale_kind: str,
) -> str:
    parts = [
        f"zscore_state_{direction}",
        f"baseline={baseline_kind}",
        f"scale={scale_kind}",
        f"z={z_value:.4f}",
    ]
    if gradient_value is not None:
        parts.append(f"dz={gradient_value:.4f}")
    if acceleration_value is not None:
        parts.append(f"ddz={acceleration_value:.4f}")
    return " ".join(parts)


DEFAULT_ENTRY_ZSCORE_STATE = ZScoreStateEntryNode()
DEFAULT_ENTRY_ZSCORE_STATE_CONTRACT = build_entry_zscore_state_contract()

register_entry(
    DEFAULT_ENTRY_ZSCORE_STATE_NAME,
    DEFAULT_ENTRY_ZSCORE_STATE,
    DEFAULT_ENTRY_ZSCORE_STATE_CONTRACT,
)


__all__ = [
    "DEFAULT_ENTRY_ZSCORE_STATE",
    "DEFAULT_ENTRY_ZSCORE_STATE_CONTRACT",
    "DEFAULT_ENTRY_ZSCORE_STATE_NAME",
    "ZScoreStateEntryNode",
    "build_entry_zscore_state_contract",
]
