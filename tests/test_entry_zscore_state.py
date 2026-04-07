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
from trading_lab.nodes.entry_zscore_state import (  # noqa: E402
    ZScoreStateEntryNode,
    build_entry_zscore_state_contract,
)
from trading_lab.registry import registry  # noqa: E402


def _context_from_closes(closes: list[float]) -> DecisionContext:
    bars = _bars_from_closes(closes)
    return _context_from_bars(bars, bar_index=len(bars) - 1)


def _bars_from_closes(closes: list[float]) -> tuple[Bar, ...]:
    bars = []
    for index, close in enumerate(closes):
        open_price = closes[index - 1] if index > 0 else close
        bars.append(
            Bar(
                timestamp=datetime(2024, 1, 1) + timedelta(days=index),
                symbol="TEST",
                open=open_price,
                high=max(open_price, close) + 0.5,
                low=min(open_price, close) - 0.5,
                close=close,
                volume=1_000.0,
            )
        )
    return tuple(bars)


def _context_from_bars(
    bars: tuple[Bar, ...],
    *,
    bar_index: int,
    use_history_window: bool = False,
) -> DecisionContext:
    instrument = InstrumentMeta(
        symbol="TEST",
        price_increment=0.01,
        quantity_increment=1.0,
    )
    position = PositionState(symbol="TEST")
    if use_history_window:
        history = BarHistorySeries(bars).window(bar_index + 1)
    else:
        history = bars[: bar_index + 1]
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


def test_default_registry_autoloads_zscore_entry_node() -> None:
    assert "entry_zscore_state" in registry.available("entry")


def test_zscore_entry_emits_long_signal_on_positive_residual_breakout() -> None:
    node = ZScoreStateEntryNode(
        baseline_kind="ema",
        baseline_lookback=5,
        scale_kind="atr",
        scale_lookback=3,
        z_threshold=0.5,
    )

    result = node(_context_from_closes([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 110]))

    assert result.action_type == "enter_long"
    assert "zscore_state_long" in result.reason


def test_zscore_entry_emits_short_signal_on_negative_residual_breakout() -> None:
    node = ZScoreStateEntryNode(
        baseline_kind="ema",
        baseline_lookback=5,
        scale_kind="atr",
        scale_lookback=3,
        z_threshold=0.25,
    )

    result = node(_context_from_closes([110, 100, 100, 100, 100, 100, 100, 100, 100, 100, 90]))

    assert result.action_type == "enter_short"
    assert "zscore_state_short" in result.reason


def test_zscore_contract_preserves_parameters_in_manifest() -> None:
    contract = build_entry_zscore_state_contract(
        name="entry_zs_custom",
        baseline_kind="median",
        baseline_lookback=30,
        scale_kind="residual_mad",
        scale_lookback=12,
        z_threshold=1.25,
        persistence_bars=2,
    )

    assert contract.name == "entry_zs_custom"
    assert contract.spec.required_history >= 30
    assert contract.manifest["parameters"]["baseline_kind"] == "median"
    assert contract.manifest["parameters"]["scale_kind"] == "residual_mad"


def test_zscore_history_window_path_matches_tuple_history_path() -> None:
    closes = [
        100.0,
        100.2,
        100.4,
        99.8,
        100.7,
        101.1,
        100.5,
        101.4,
        102.0,
        101.2,
        102.3,
        103.1,
        102.4,
        103.6,
        104.2,
    ]
    bars = _bars_from_closes(closes)
    node = ZScoreStateEntryNode(
        baseline_kind="ema",
        baseline_lookback=5,
        scale_kind="residual_std",
        scale_lookback=4,
        z_threshold=0.5,
        gradient_threshold=0.05,
        gradient_horizon=2,
        persistence_bars=2,
        require_baseline_slope=True,
        baseline_slope_horizon=3,
    )

    tuple_actions = []
    window_actions = []
    for bar_index in range(len(bars)):
        tuple_actions.append(node(_context_from_bars(bars, bar_index=bar_index)))
        window_actions.append(
            node(
                _context_from_bars(
                    bars,
                    bar_index=bar_index,
                    use_history_window=True,
                )
            )
        )

    assert tuple_actions == window_actions
