from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_lab.contracts import (  # noqa: E402
    Bar,
    BarHistorySeries,
    CostAssumptions,
    DecisionContext,
    InstrumentMeta,
    PortfolioState,
    PositionState,
    SessionInfo,
)
from trading_lab.nodes.exit_composite_release import CompositeReleaseExitNode  # noqa: E402
from trading_lab.nodes.exit_macd_release import MacdReleaseExitNode  # noqa: E402
from trading_lab.nodes.exit_mfe_giveback import MfeGivebackExitNode  # noqa: E402
from trading_lab.nodes.exit_no_progress import NoProgressExitNode  # noqa: E402
from trading_lab.nodes.exit_profit_target import ProfitTargetExitNode  # noqa: E402
from trading_lab.nodes.exit_shared import bars_held_from_ctx, slice_since_entry  # noqa: E402
from trading_lab.nodes.exit_trailing_atr import TrailingAtrExitNode  # noqa: E402
from trading_lab.nodes.exit_zscore_release import ZScoreReleaseExitNode  # noqa: E402


def _bars_from_closes(closes: list[float]) -> tuple[Bar, ...]:
    output = []
    for index, close in enumerate(closes):
        output.append(
            Bar(
                timestamp=datetime(2024, 1, 1, 9, 30) + timedelta(minutes=index),
                symbol="TEST",
                open=close,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1_000.0,
            )
        )
    return tuple(output)


def _bars_from_ohlc(rows: list[tuple[float, float, float, float]]) -> tuple[Bar, ...]:
    output = []
    for index, (open_, high, low, close) in enumerate(rows):
        output.append(
            Bar(
                timestamp=datetime(2024, 1, 1, 9, 30) + timedelta(minutes=index),
                symbol="TEST",
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=1_000.0,
            )
        )
    return tuple(output)


def _context(
    bars: tuple[Bar, ...],
    *,
    bar_index: int,
    entry_index: int,
    use_history_window: bool,
) -> DecisionContext:
    instrument = InstrumentMeta(
        symbol="TEST",
        price_increment=0.01,
        quantity_increment=1.0,
    )
    position = PositionState(
        symbol="TEST",
        side="long",
        quantity=1.0,
        entry_price=bars[entry_index].close,
        entry_time=bars[entry_index].timestamp,
    )
    history = (
        BarHistorySeries(bars).window(bar_index + 1)
        if use_history_window
        else bars[: bar_index + 1]
    )
    return DecisionContext(
        bar=bars[bar_index],
        history=history,
        instrument=instrument,
        costs=CostAssumptions(),
        position=position,
        portfolio=PortfolioState(
            cash=100_000.0,
            equity=100_000.0,
            positions=(position,),
        ),
        session=SessionInfo(bar_index=bar_index, bars_total=len(bars)),
    )


@pytest.mark.parametrize("use_history_window", [False, True])
def test_bars_held_from_ctx_derives_from_position_entry_time(use_history_window: bool) -> None:
    bars = _bars_from_closes([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    ctx = _context(
        bars,
        bar_index=4,
        entry_index=2,
        use_history_window=use_history_window,
    )

    assert bars_held_from_ctx(ctx) == 3


def test_slice_since_entry_uses_bars_in_trade_count() -> None:
    assert slice_since_entry([10.0, 20.0, 30.0, 40.0, 50.0], 3) == [30.0, 40.0, 50.0]


@pytest.mark.parametrize("use_history_window", [False, True])
def test_no_progress_exit_uses_derived_holding_age(use_history_window: bool) -> None:
    bars = _bars_from_closes([100.0, 101.0, 100.0, 99.0, 99.0, 98.0])
    ctx = _context(
        bars,
        bar_index=4,
        entry_index=2,
        use_history_window=use_history_window,
    )

    result = NoProgressExitNode(evaluation_bars=3, min_open_profit_points=0.0)(ctx)

    assert result.action_type == "close"
    assert "no_progress" in result.reason


@pytest.mark.parametrize("use_history_window", [False, True])
def test_composite_release_max_hold_uses_derived_holding_age(use_history_window: bool) -> None:
    bars = _bars_from_closes([100.0, 101.0, 102.0, 101.0, 100.0, 99.0])
    ctx = _context(
        bars,
        bar_index=4,
        entry_index=2,
        use_history_window=use_history_window,
    )

    result = CompositeReleaseExitNode(max_hold_bars=3)(ctx)

    assert result.action_type == "close"
    assert "max_hold_bars=3" in result.reason


@pytest.mark.parametrize("use_history_window", [False, True])
def test_mfe_giveback_ignores_pre_entry_extremes(use_history_window: bool) -> None:
    bars = _bars_from_ohlc(
        [
            (100.0, 100.5, 99.5, 100.0),
            (100.0, 125.0, 99.5, 100.0),
            (100.0, 100.5, 99.5, 100.0),
            (100.8, 101.4, 100.2, 101.2),
            (101.1, 101.5, 100.8, 101.0),
            (101.0, 101.4, 100.5, 100.9),
        ]
    )
    ctx = _context(
        bars,
        bar_index=5,
        entry_index=2,
        use_history_window=use_history_window,
    )

    result = MfeGivebackExitNode(
        activation_profit_points=1.0,
        giveback_kind="fraction",
        giveback_fraction=0.5,
    )(ctx)

    assert result.action_type == "hold"


def test_zscore_release_history_window_matches_tuple_history() -> None:
    bars = _bars_from_closes([100.0, 101.0, 103.0, 102.0, 100.0, 99.0, 98.0])
    node = ZScoreReleaseExitNode(
        baseline_kind="sma",
        baseline_lookback=2,
        scale_kind="atr",
        scale_lookback=2,
        release_kind="threshold_cross",
        z_exit_threshold=0.0,
        confirm_bars=1,
    )

    tuple_result = node(_context(bars, bar_index=5, entry_index=2, use_history_window=False))
    window_result = node(_context(bars, bar_index=5, entry_index=2, use_history_window=True))

    assert window_result == tuple_result


def test_macd_release_history_window_matches_tuple_history() -> None:
    bars = _bars_from_closes([100.0, 101.0, 102.0, 103.0, 101.0, 99.0, 98.0, 97.0])
    node = MacdReleaseExitNode(
        fast_lookback=2,
        slow_lookback=4,
        signal_lookback=2,
        release_kind="histogram_cross",
        confirm_bars=1,
    )

    tuple_result = node(_context(bars, bar_index=6, entry_index=2, use_history_window=False))
    window_result = node(_context(bars, bar_index=6, entry_index=2, use_history_window=True))

    assert window_result == tuple_result


def test_profit_target_atr_history_window_matches_tuple_history() -> None:
    bars = _bars_from_ohlc(
        [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.5, 99.5, 101.0),
            (101.0, 102.0, 100.0, 101.0),
            (101.0, 103.5, 100.5, 103.0),
            (103.0, 104.0, 102.5, 103.5),
        ]
    )
    node = ProfitTargetExitNode(
        target_kind="atr_from_entry",
        atr_lookback=2,
        atr_multiple=1.0,
    )

    tuple_result = node(_context(bars, bar_index=4, entry_index=2, use_history_window=False))
    window_result = node(_context(bars, bar_index=4, entry_index=2, use_history_window=True))

    assert window_result == tuple_result


def test_trailing_atr_history_window_matches_tuple_history() -> None:
    bars = _bars_from_ohlc(
        [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 102.0, 99.5, 101.5),
            (101.5, 103.0, 101.0, 102.0),
            (102.0, 103.5, 101.5, 103.0),
            (103.0, 103.2, 100.0, 100.5),
            (100.5, 101.0, 99.5, 100.0),
        ]
    )
    node = TrailingAtrExitNode(
        atr_lookback=2,
        atr_multiple=0.5,
        reference_kind="highest_high",
        activation_bars=1,
    )

    tuple_result = node(_context(bars, bar_index=4, entry_index=2, use_history_window=False))
    window_result = node(_context(bars, bar_index=4, entry_index=2, use_history_window=True))

    assert window_result == tuple_result


def test_composite_release_history_window_matches_tuple_history() -> None:
    bars = _bars_from_ohlc(
        [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 102.0, 99.5, 101.5),
            (101.5, 103.0, 101.0, 102.5),
            (102.5, 104.0, 102.0, 103.5),
            (103.5, 103.6, 100.5, 101.0),
            (101.0, 101.2, 99.5, 100.0),
        ]
    )
    node = CompositeReleaseExitNode(
        trailing_atr_lookback=2,
        trailing_atr_multiple=0.5,
        trailing_activation_bars=1,
        z_exit_threshold=0.0,
        z_baseline_kind="sma",
        z_baseline_lookback=2,
        z_scale_kind="atr",
        z_scale_lookback=2,
        z_confirm_bars=1,
    )

    tuple_result = node(_context(bars, bar_index=4, entry_index=2, use_history_window=False))
    window_result = node(_context(bars, bar_index=4, entry_index=2, use_history_window=True))

    assert window_result == tuple_result
