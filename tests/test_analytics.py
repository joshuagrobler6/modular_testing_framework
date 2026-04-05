from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_lab.analytics import (  # noqa: E402
    BEHAVIORAL_METRIC_NAMES,
    DEFAULT_METRIC_REGISTRY,
    EQUITY_METRIC_NAMES,
    TRADE_METRIC_NAMES,
    MetricDependencyError,
    MetricRegistry,
    compute_behavioral_metrics,
    compute_equity_metrics,
    compute_metrics,
    compute_trade_metrics,
)
from trading_lab.contracts import BacktestResult, BacktestSpec, InstrumentMeta  # noqa: E402
from trading_lab.schemas import RESERVED_LEDGER_COLUMNS  # noqa: E402


def _with_reserved_columns(
    df: pd.DataFrame,
    *,
    contract_version: str = "1.0",
) -> pd.DataFrame:
    result = df.copy()
    for column in RESERVED_LEDGER_COLUMNS:
        if column == "contract_version":
            result[column] = contract_version
        else:
            result[column] = None
    return result


def _make_trade_ledger() -> pd.DataFrame:
    return _with_reserved_columns(
        pd.DataFrame(
            [
                {
                    "trade_id": "trade-1",
                    "symbol": "AAPL",
                    "side": "long",
                    "entry_ts": pd.Timestamp("2024-02-29"),
                    "exit_ts": pd.Timestamp("2024-04-30"),
                    "entry_price": 100.0,
                    "exit_price": 112.0,
                    "qty": 10.0,
                    "gross_pnl": 110.0,
                    "net_pnl": 100.0,
                    "mfe": 140.0,
                    "mae": -20.0,
                    "exit_efficiency": 0.8,
                    "bars_held": 2,
                    "exit_reason": "tp-hit",
                    "fees": 10.0,
                },
                {
                    "trade_id": "trade-2",
                    "symbol": "MSFT",
                    "side": "short",
                    "entry_ts": pd.Timestamp("2024-04-30"),
                    "exit_ts": pd.Timestamp("2024-07-31"),
                    "entry_price": 90.0,
                    "exit_price": 96.0,
                    "qty": 8.0,
                    "gross_pnl": -40.0,
                    "net_pnl": -50.0,
                    "mfe": 24.0,
                    "mae": -40.0,
                    "exit_efficiency": -0.5,
                    "bars_held": 3,
                    "exit_reason": "stop-hit",
                    "fees": 10.0,
                },
                {
                    "trade_id": "trade-3",
                    "symbol": "AAPL",
                    "side": "long",
                    "entry_ts": pd.Timestamp("2024-05-31"),
                    "exit_ts": pd.Timestamp("2024-06-30"),
                    "entry_price": 80.0,
                    "exit_price": 114.0,
                    "qty": 5.0,
                    "gross_pnl": 160.0,
                    "net_pnl": 150.0,
                    "mfe": 180.0,
                    "mae": -10.0,
                    "exit_efficiency": 0.9,
                    "bars_held": 1,
                    "exit_reason": "time-stop",
                    "fees": 10.0,
                },
            ]
        )
    )


def _make_fill_log() -> pd.DataFrame:
    return _with_reserved_columns(
        pd.DataFrame(
            [
                {
                    "fill_id": "fill-1",
                    "order_id": "ord-1",
                    "ts_fill": pd.Timestamp("2024-02-29"),
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 10.0,
                    "fill_price": 100.0,
                    "fees": 5.0,
                    "slippage": 1.0,
                    "gross_notional": 500.0,
                },
                {
                    "fill_id": "fill-2",
                    "order_id": "ord-2",
                    "ts_fill": pd.Timestamp("2024-04-30"),
                    "symbol": "AAPL",
                    "side": "sell",
                    "qty": 10.0,
                    "fill_price": 112.0,
                    "fees": 5.0,
                    "slippage": 1.0,
                    "gross_notional": 610.0,
                },
                {
                    "fill_id": "fill-3",
                    "order_id": "ord-3",
                    "ts_fill": pd.Timestamp("2024-04-30"),
                    "symbol": "MSFT",
                    "side": "sell",
                    "qty": 8.0,
                    "fill_price": 90.0,
                    "fees": 5.0,
                    "slippage": 1.0,
                    "gross_notional": 700.0,
                },
                {
                    "fill_id": "fill-4",
                    "order_id": "ord-4",
                    "ts_fill": pd.Timestamp("2024-07-31"),
                    "symbol": "MSFT",
                    "side": "buy",
                    "qty": 8.0,
                    "fill_price": 96.0,
                    "fees": 5.0,
                    "slippage": 1.0,
                    "gross_notional": 650.0,
                },
                {
                    "fill_id": "fill-5",
                    "order_id": "ord-5",
                    "ts_fill": pd.Timestamp("2024-05-31"),
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 5.0,
                    "fill_price": 80.0,
                    "fees": 5.0,
                    "slippage": 1.0,
                    "gross_notional": 400.0,
                },
                {
                    "fill_id": "fill-6",
                    "order_id": "ord-6",
                    "ts_fill": pd.Timestamp("2024-06-30"),
                    "symbol": "AAPL",
                    "side": "sell",
                    "qty": 5.0,
                    "fill_price": 114.0,
                    "fees": 5.0,
                    "slippage": 1.0,
                    "gross_notional": 470.0,
                },
            ]
        )
    )


def _make_equity_curve() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
            "2024-03-31",
            "2024-04-30",
            "2024-05-31",
            "2024-06-30",
            "2024-07-31",
        ]
    )
    equity = pd.Series([1000.0, 1050.0, 1100.0, 1045.0, 1030.0, 1200.0, 1200.0])
    net_exposure = pd.Series([0.0, 500.0, 500.0, -700.0, -300.0, 300.0, 0.0])
    cash = equity - net_exposure
    gross_exposure = pd.Series([0.0, 500.0, 500.0, 700.0, 1100.0, 300.0, 0.0])
    drawdown = equity / equity.cummax() - 1.0

    return _with_reserved_columns(
        pd.DataFrame(
            {
                "ts": timestamps,
                "cash": cash,
                "equity": equity,
                "realized_pnl": [0.0, 0.0, 0.0, 100.0, 100.0, 250.0, 200.0],
                "unrealized_pnl": [0.0, 50.0, 100.0, -5.0, -20.0, 0.0, 0.0],
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "drawdown": drawdown,
            }
        )
    )


def _make_decision_log() -> pd.DataFrame:
    return _with_reserved_columns(
        pd.DataFrame(
            [
                {
                    "ts": pd.Timestamp("2024-01-31"),
                    "symbol": "AAPL",
                    "entry_action": "none",
                    "exit_action": "none",
                    "risk_approved": False,
                    "target_units": 0.0,
                    "resolved_action": "hold",
                    "reason": "",
                    "metadata": {},
                }
            ]
        )
    )


def _make_order_log() -> pd.DataFrame:
    return _with_reserved_columns(
        pd.DataFrame(
            [
                {
                    "order_id": "ord-1",
                    "ts_submitted": pd.Timestamp("2024-02-28"),
                    "symbol": "AAPL",
                    "side": "buy",
                    "qty": 10.0,
                    "order_type": "market",
                    "price_reference": 99.5,
                    "status": "filled",
                }
            ]
        )
    )


def _make_result() -> BacktestResult:
    return BacktestResult(
        spec=BacktestSpec(
            name="analytics-test",
            instrument=InstrumentMeta(
                symbol="AAPL",
                price_increment=0.01,
                quantity_increment=1.0,
            ),
            entry_node="entry_sma_cross",
            exit_node="exit_time_stop",
            risk_node="risk_fixed_fraction",
            initial_cash=1000.0,
        ),
        decision_log=_make_decision_log(),
        order_log=_make_order_log(),
        fill_log=_make_fill_log(),
        trade_ledger=_make_trade_ledger(),
        equity_curve=_make_equity_curve(),
    )


def test_metric_registry_exposes_core_metric_groups_and_dependencies() -> None:
    assert set(DEFAULT_METRIC_REGISTRY.available("trade")) == set(TRADE_METRIC_NAMES)
    assert set(DEFAULT_METRIC_REGISTRY.available("equity")) == set(EQUITY_METRIC_NAMES)
    assert set(DEFAULT_METRIC_REGISTRY.available("behavioral")) == set(
        BEHAVIORAL_METRIC_NAMES
    )

    turnover = DEFAULT_METRIC_REGISTRY.definition("turnover")
    assert turnover.group == "equity"
    assert turnover.required_ledgers == {
        "equity_curve": ("equity",),
        "fill_log": ("gross_notional",),
    }

    pnl_by_symbol = DEFAULT_METRIC_REGISTRY.definition("pnl_by_symbol")
    assert pnl_by_symbol.required_ledgers == {
        "trade_ledger": ("symbol", "net_pnl")
    }


def test_compute_trade_metrics_from_ledgers() -> None:
    metrics = compute_trade_metrics(_make_trade_ledger())

    assert metrics["trade_count"] == 3
    assert metrics["win_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["avg_win"] == pytest.approx(125.0)
    assert metrics["avg_loss"] == pytest.approx(-50.0)
    assert metrics["payoff_ratio"] == pytest.approx(2.5)
    assert metrics["expectancy_per_trade"] == pytest.approx(200.0 / 3.0)
    assert metrics["avg_bars_held"] == pytest.approx(2.0)
    assert metrics["gross_pnl"] == pytest.approx(230.0)
    assert metrics["net_pnl"] == pytest.approx(200.0)
    assert metrics["avg_mfe"] == pytest.approx((140.0 + 24.0 + 180.0) / 3.0)
    assert metrics["avg_mae"] == pytest.approx((-20.0 - 40.0 - 10.0) / 3.0)
    assert metrics["avg_exit_efficiency"] == pytest.approx((0.8 - 0.5 + 0.9) / 3.0)


def test_compute_equity_and_behavioral_metrics_from_ledgers() -> None:
    trade_ledger = _make_trade_ledger()
    fill_log = _make_fill_log()
    equity_curve = _make_equity_curve()

    equity_metrics = compute_equity_metrics(equity_curve, fill_log)
    behavioral_metrics = compute_behavioral_metrics(trade_ledger, fill_log)

    returns = equity_curve["equity"].pct_change().dropna()
    elapsed_seconds = (
        equity_curve["ts"].iloc[-1] - equity_curve["ts"].iloc[0]
    ).total_seconds()
    periods_per_year = (
        365.25 * 24.0 * 60.0 * 60.0
        / equity_curve["ts"].diff().dropna().dt.total_seconds().median()
    )
    downside_returns = returns.loc[returns < 0.0]
    expected_total_return = 1200.0 / 1000.0 - 1.0
    expected_annualized_return = (1200.0 / 1000.0) ** (
        (365.25 * 24.0 * 60.0 * 60.0) / elapsed_seconds
    ) - 1.0
    expected_annualized_vol = float(returns.std(ddof=0)) * math.sqrt(periods_per_year)
    expected_sharpe = (
        float(returns.mean()) / float(returns.std(ddof=0)) * math.sqrt(periods_per_year)
    )
    expected_sortino = (
        float(returns.mean())
        / float(downside_returns.std(ddof=0))
        * math.sqrt(periods_per_year)
    )
    expected_turnover = fill_log["gross_notional"].sum() / equity_curve["equity"].mean()

    assert equity_metrics["total_return"] == pytest.approx(expected_total_return)
    assert equity_metrics["annualized_return"] == pytest.approx(expected_annualized_return)
    assert equity_metrics["annualized_vol"] == pytest.approx(expected_annualized_vol)
    assert equity_metrics["sharpe"] == pytest.approx(expected_sharpe)
    assert equity_metrics["sortino"] == pytest.approx(expected_sortino)
    assert equity_metrics["max_drawdown"] == pytest.approx((1030.0 / 1100.0) - 1.0)
    assert equity_metrics["drawdown_duration"] == 2
    assert equity_metrics["turnover"] == pytest.approx(expected_turnover)
    assert equity_metrics["time_in_market"] == pytest.approx(5.0 / 7.0)

    assert behavioral_metrics["pnl_by_symbol"] == {
        "AAPL": pytest.approx(250.0),
        "MSFT": pytest.approx(-50.0),
    }
    assert behavioral_metrics["pnl_by_side"] == {
        "long": pytest.approx(250.0),
        "short": pytest.approx(-50.0),
    }
    assert behavioral_metrics["holding_period_distribution"] == {1: 1, 2: 1, 3: 1}
    assert behavioral_metrics["costs_breakdown"] == {
        "fees": pytest.approx(30.0),
        "slippage": pytest.approx(6.0),
        "total_costs": pytest.approx(36.0),
    }


def test_compute_metrics_merges_all_metric_groups_from_backtest_result() -> None:
    metrics = compute_metrics(_make_result())

    assert metrics["trade_count"] == 3
    assert metrics["gross_pnl"] == pytest.approx(230.0)
    assert metrics["avg_mfe"] == pytest.approx((140.0 + 24.0 + 180.0) / 3.0)
    assert metrics["total_return"] == pytest.approx(0.2)
    assert metrics["max_drawdown"] < 0.0
    assert metrics["pnl_by_symbol"]["AAPL"] == pytest.approx(250.0)
    assert metrics["holding_period_distribution"] == {1: 1, 2: 1, 3: 1}
    assert metrics["costs_breakdown"]["total_costs"] == pytest.approx(36.0)


def test_metric_dependency_failures_are_clear_for_future_column_requirements() -> None:
    registry = MetricRegistry()

    @registry.metric(
        name="future_trade_metric",
        group="trade",
        required_ledgers={"trade_ledger": ("net_pnl", "future_column")},
    )
    def future_trade_metric(ledgers):
        return 0.0

    with pytest.raises(
        MetricDependencyError,
        match="requires columns missing from 'trade_ledger': \\['future_column'\\]",
    ):
        registry.compute(("future_trade_metric",), {"trade_ledger": _make_trade_ledger()})


def test_analytics_validate_inputs_before_metric_computation() -> None:
    invalid_trade_ledger = _make_trade_ledger().copy()
    invalid_trade_ledger.loc[0, "side"] = "flat"

    with pytest.raises((SchemaError, SchemaErrors)):
        compute_trade_metrics(invalid_trade_ledger)
