from __future__ import annotations

from collections import deque
from math import sqrt
from typing import Iterable, Sequence

from trading_lab.contracts import ActionRequest, DecisionContext


def closes_from_ctx(ctx: DecisionContext) -> list[float]:
    return [float(bar.close) for bar in ctx.history]


def highs_from_ctx(ctx: DecisionContext) -> list[float]:
    return [float(bar.high) for bar in ctx.history]


def lows_from_ctx(ctx: DecisionContext) -> list[float]:
    return [float(bar.low) for bar in ctx.history]


def position_direction(ctx: DecisionContext) -> str | None:
    position = ctx.position
    if bool(getattr(position, "is_flat", False)):
        return None
    if bool(getattr(position, "is_long", False)):
        return "long"
    if bool(getattr(position, "is_short", False)):
        return "short"
    quantity = getattr(position, "quantity", None)
    if quantity is None:
        quantity = getattr(position, "size", None)
    if quantity is None:
        return None
    quantity_value = float(quantity)
    if quantity_value > 0.0:
        return "long"
    if quantity_value < 0.0:
        return "short"
    return None



def exit_action_type(direction: str) -> str:
    if direction in {"long", "short"}:
        return "close"
    raise ValueError(f"Unsupported direction: {direction}.")



def entry_price_from_ctx(ctx: DecisionContext) -> float | None:
    position = ctx.position
    for attribute in (
        "entry_price",
        "avg_entry_price",
        "average_entry_price",
        "average_price",
        "cost_basis_price",
    ):
        value = getattr(position, attribute, None)
        if value is not None:
            return float(value)
    return None



def bars_held_from_ctx(ctx: DecisionContext) -> int | None:
    position = ctx.position
    for attribute in (
        "bars_held",
        "bars_in_position",
        "hold_bars",
        "age_bars",
        "holding_period_bars",
    ):
        value = getattr(position, attribute, None)
        if value is not None:
            return int(value)
    return None



def current_close(ctx: DecisionContext) -> float:
    return float(ctx.history[-1].close)



def open_profit_points(ctx: DecisionContext) -> float | None:
    direction = position_direction(ctx)
    entry_price = entry_price_from_ctx(ctx)
    if direction is None or entry_price is None:
        return None
    close = current_close(ctx)
    if direction == "long":
        return close - entry_price
    return entry_price - close



def action_request_exit(direction: str, reason: str) -> ActionRequest:
    return ActionRequest(action_type=exit_action_type(direction), reason=reason)



def slice_since_entry(values: Sequence[float], bars_held: int) -> list[float]:
    if bars_held < 0:
        raise ValueError("bars_held must be non-negative.")
    count = min(len(values), bars_held + 1)
    return list(values[-count:])



def true_range_series(
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



def rolling_mean(values: Sequence[float], lookback: int) -> list[float | None]:
    output: list[float | None] = []
    window: deque[float] = deque()
    window_sum = 0.0
    for value in values:
        window.append(value)
        window_sum += value
        if len(window) > lookback:
            window_sum -= window.popleft()
        output.append(window_sum / lookback if len(window) == lookback else None)
    return output



def rolling_median(values: Sequence[float], lookback: int) -> list[float | None]:
    output: list[float | None] = []
    window: deque[float] = deque()
    for value in values:
        window.append(value)
        if len(window) > lookback:
            window.popleft()
        if len(window) == lookback:
            ordered = sorted(window)
            midpoint = lookback // 2
            if lookback % 2 == 1:
                output.append(ordered[midpoint])
            else:
                output.append((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)
        else:
            output.append(None)
    return output



def ema_series(values: Sequence[float], lookback: int) -> list[float | None]:
    alpha = 2.0 / (lookback + 1.0)
    ema: float | None = None
    output: list[float | None] = []
    for index, value in enumerate(values):
        ema = value if ema is None else alpha * value + (1.0 - alpha) * ema
        output.append(ema if index + 1 >= lookback else None)
    return output



def rolling_std(values: Sequence[float], lookback: int) -> list[float | None]:
    output: list[float | None] = []
    window: deque[float] = deque()
    for value in values:
        window.append(value)
        if len(window) > lookback:
            window.popleft()
        output.append(std(window) if len(window) == lookback else None)
    return output



def rolling_std_nullable(values: Sequence[float | None], lookback: int) -> list[float | None]:
    output: list[float | None] = []
    window: deque[float] = deque()
    for value in values:
        if value is None:
            output.append(None)
            continue
        window.append(value)
        if len(window) > lookback:
            window.popleft()
        output.append(std(window) if len(window) == lookback else None)
    return output



def rolling_mad_nullable(values: Sequence[float | None], lookback: int) -> list[float | None]:
    output: list[float | None] = []
    window: deque[float] = deque()
    for value in values:
        if value is None:
            output.append(None)
            continue
        window.append(value)
        if len(window) > lookback:
            window.popleft()
        if len(window) == lookback:
            center = median(window)
            deviations = [abs(item - center) for item in window]
            output.append(median(deviations))
        else:
            output.append(None)
    return output



def mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    if count == 0:
        raise ValueError("mean requires at least one value.")
    return total / count



def std(values: Iterable[float]) -> float:
    as_list = list(values)
    if len(as_list) < 2:
        return 0.0
    avg = mean(as_list)
    variance = sum((value - avg) ** 2 for value in as_list) / len(as_list)
    return sqrt(variance)



def median(values: Iterable[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median requires at least one value.")
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0



def macd_histogram_series(
    values: Sequence[float],
    *,
    fast_lookback: int,
    slow_lookback: int,
    signal_lookback: int,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    fast = ema_series(values, fast_lookback)
    slow = ema_series(values, slow_lookback)
    macd_line: list[float | None] = []
    macd_numeric: list[float] = []
    numeric_to_full_index: list[int] = []
    for index, (fast_value, slow_value) in enumerate(zip(fast, slow, strict=False)):
        if fast_value is None or slow_value is None:
            macd_line.append(None)
            continue
        value = fast_value - slow_value
        macd_line.append(value)
        macd_numeric.append(value)
        numeric_to_full_index.append(index)

    signal_numeric = ema_series(macd_numeric, signal_lookback)
    signal_line: list[float | None] = [None] * len(values)
    for numeric_index, full_index in enumerate(numeric_to_full_index):
        signal_line[full_index] = signal_numeric[numeric_index]

    histogram: list[float | None] = []
    for macd_value, signal_value in zip(macd_line, signal_line, strict=False):
        if macd_value is None or signal_value is None:
            histogram.append(None)
            continue
        histogram.append(macd_value - signal_value)
    return macd_line, signal_line, histogram



def atr_series(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    lookback: int,
) -> list[float | None]:
    return rolling_mean(true_range_series(highs, lows, closes), lookback)



def baseline_series(values: Sequence[float], kind: str, lookback: int) -> list[float | None]:
    if kind == "ema":
        return ema_series(values, lookback)
    if kind == "sma":
        return rolling_mean(values, lookback)
    if kind == "median":
        return rolling_median(values, lookback)
    raise ValueError(f"Unsupported baseline kind: {kind}.")



def scale_series(
    *,
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    baseline: Sequence[float | None],
    scale_kind: str,
    scale_lookback: int,
) -> list[float | None]:
    if scale_kind == "atr":
        return atr_series(highs, lows, closes, scale_lookback)
    residuals = [
        None if base is None else close - base
        for close, base in zip(closes, baseline, strict=False)
    ]
    if scale_kind == "residual_std":
        return rolling_std_nullable(residuals, scale_lookback)
    if scale_kind == "residual_mad":
        return rolling_mad_nullable(residuals, scale_lookback)
    raise ValueError(f"Unsupported scale kind: {scale_kind}.")



def zscore_series(
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



def diff_series(
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
