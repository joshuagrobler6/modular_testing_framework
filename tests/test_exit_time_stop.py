from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

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
from trading_lab.nodes.exit_time_stop import TimeStopExitNode  # noqa: E402


def _bars(count: int) -> tuple[Bar, ...]:
    output = []
    for index in range(count):
        price = 100.0 + index
        output.append(
            Bar(
                timestamp=datetime(2024, 1, 1, 9, 30) + timedelta(minutes=index),
                symbol="TEST",
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price + 0.5,
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


def test_time_stop_exit_history_window_matches_tuple_history() -> None:
    bars = _bars(8)
    node = TimeStopExitNode(hold_bars=3)

    tuple_result = node(_context(bars, bar_index=4, entry_index=2, use_history_window=False))
    window_result = node(_context(bars, bar_index=4, entry_index=2, use_history_window=True))

    assert tuple_result == window_result
    assert tuple_result.action_type == "close"


def test_time_stop_exit_history_window_can_hold_before_threshold() -> None:
    bars = _bars(8)
    node = TimeStopExitNode(hold_bars=4)

    result = node(_context(bars, bar_index=4, entry_index=2, use_history_window=True))

    assert result.action_type == "hold"
