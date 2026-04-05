from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import pandas as pd

from trading_lab.contracts import BacktestResult
from trading_lab.schemas import (
    DecisionLogSchema,
    EquityCurveSchema,
    FillLogSchema,
    OrderLogSchema,
    TradeLedgerSchema,
)

LedgerName = Literal[
    "decision_log",
    "order_log",
    "fill_log",
    "trade_ledger",
    "equity_curve",
]
MetricGroup = Literal["trade", "equity", "behavioral"]
LedgerFrames = Mapping[LedgerName, pd.DataFrame]
MetricFn = Callable[[LedgerFrames], Any]

_SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0
_LEDGER_SCHEMAS = {
    "decision_log": DecisionLogSchema,
    "order_log": OrderLogSchema,
    "fill_log": FillLogSchema,
    "trade_ledger": TradeLedgerSchema,
    "equity_curve": EquityCurveSchema,
}


class MetricDependencyError(ValueError):
    """Raised when a metric requires ledgers or columns that are not available."""


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    name: str
    group: MetricGroup
    required_ledgers: dict[LedgerName, tuple[str, ...]]
    fn: MetricFn
    description: str = ""


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, MetricDefinition] = {}

    def register(self, definition: MetricDefinition) -> MetricDefinition:
        if definition.name in self._metrics:
            raise ValueError(f"metric {definition.name!r} is already registered.")
        self._metrics[definition.name] = definition
        return definition

    def metric(
        self,
        *,
        name: str,
        group: MetricGroup,
        required_ledgers: Mapping[LedgerName, tuple[str, ...] | list[str]],
        description: str = "",
    ) -> Callable[[MetricFn], MetricFn]:
        normalized_requirements = {
            ledger_name: tuple(columns)
            for ledger_name, columns in required_ledgers.items()
        }

        def decorator(fn: MetricFn) -> MetricFn:
            self.register(
                MetricDefinition(
                    name=name,
                    group=group,
                    required_ledgers=normalized_requirements,
                    fn=fn,
                    description=description,
                )
            )
            return fn

        return decorator

    def definition(self, name: str) -> MetricDefinition:
        try:
            return self._metrics[name]
        except KeyError as exc:
            raise KeyError(f"metric {name!r} is not registered.") from exc

    def available(self, group: MetricGroup | None = None) -> tuple[str, ...]:
        if group is None:
            return tuple(sorted(self._metrics))
        return tuple(
            sorted(name for name, definition in self._metrics.items() if definition.group == group)
        )

    def compute(
        self,
        metric_names: tuple[str, ...] | list[str],
        ledgers: LedgerFrames,
    ) -> dict[str, Any]:
        requested = tuple(metric_names)
        self._validate_dependencies(requested, ledgers)
        return {name: self.definition(name).fn(ledgers) for name in requested}

    def _validate_dependencies(
        self,
        metric_names: tuple[str, ...],
        ledgers: LedgerFrames,
    ) -> None:
        for name in metric_names:
            definition = self.definition(name)
            for ledger_name, required_columns in definition.required_ledgers.items():
                if ledger_name not in ledgers:
                    raise MetricDependencyError(
                        f"metric {name!r} requires ledger {ledger_name!r}, which was not supplied."
                    )
                available_columns = set(ledgers[ledger_name].columns)
                missing_columns = [
                    column for column in required_columns if column not in available_columns
                ]
                if missing_columns:
                    raise MetricDependencyError(
                        f"metric {name!r} requires columns missing from {ledger_name!r}: "
                        f"{missing_columns}."
                    )


DEFAULT_METRIC_REGISTRY = MetricRegistry()

TRADE_METRIC_NAMES = (
    "trade_count",
    "win_rate",
    "avg_win",
    "avg_loss",
    "payoff_ratio",
    "expectancy_per_trade",
    "avg_bars_held",
    "gross_pnl",
    "net_pnl",
    "avg_mfe",
    "avg_mae",
    "avg_exit_efficiency",
)
EQUITY_METRIC_NAMES = (
    "total_return",
    "annualized_return",
    "annualized_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "drawdown_duration",
    "turnover",
    "time_in_market",
)
BEHAVIORAL_METRIC_NAMES = (
    "pnl_by_symbol",
    "pnl_by_side",
    "costs_breakdown",
    "holding_period_distribution",
)


def _validate_backtest_result(result: BacktestResult) -> dict[LedgerName, pd.DataFrame]:
    if not isinstance(result, BacktestResult):
        raise TypeError("result must be a BacktestResult instance.")
    return {
        "decision_log": DecisionLogSchema.validate(result.decision_log.copy(), lazy=True),
        "order_log": OrderLogSchema.validate(result.order_log.copy(), lazy=True),
        "fill_log": FillLogSchema.validate(result.fill_log.copy(), lazy=True),
        "trade_ledger": TradeLedgerSchema.validate(result.trade_ledger.copy(), lazy=True),
        "equity_curve": EquityCurveSchema.validate(result.equity_curve.copy(), lazy=True),
    }


def _validate_ledger(name: LedgerName, df: pd.DataFrame) -> pd.DataFrame:
    schema = _LEDGER_SCHEMAS[name]
    return schema.validate(df.copy(), lazy=True)


def _infer_periods_per_year(equity_curve: pd.DataFrame) -> float:
    if len(equity_curve) < 2:
        return 0.0
    deltas = equity_curve["ts"].diff().dropna().dt.total_seconds()
    positive_deltas = deltas[deltas > 0.0]
    if positive_deltas.empty:
        return 0.0
    median_seconds = float(positive_deltas.median())
    if median_seconds <= 0.0:
        return 0.0
    return _SECONDS_PER_YEAR / median_seconds


def _drawdown_duration(drawdowns: pd.Series) -> int:
    max_duration = 0
    current_duration = 0
    for value in drawdowns:
        if float(value) < 0.0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    return max_duration


def _trade_ledger(ledgers: LedgerFrames) -> pd.DataFrame:
    return ledgers["trade_ledger"]


def _fill_log(ledgers: LedgerFrames) -> pd.DataFrame:
    return ledgers["fill_log"]


def _equity_curve(ledgers: LedgerFrames) -> pd.DataFrame:
    return ledgers["equity_curve"]


@DEFAULT_METRIC_REGISTRY.metric(
    name="trade_count",
    group="trade",
    required_ledgers={"trade_ledger": ("trade_id",)},
    description="Number of closed trades in the canonical trade ledger.",
)
def _metric_trade_count(ledgers: LedgerFrames) -> int:
    return int(len(_trade_ledger(ledgers)))


@DEFAULT_METRIC_REGISTRY.metric(
    name="win_rate",
    group="trade",
    required_ledgers={"trade_ledger": ("net_pnl",)},
)
def _metric_win_rate(ledgers: LedgerFrames) -> float:
    trade_ledger = _trade_ledger(ledgers)
    trade_count = len(trade_ledger)
    if trade_count == 0:
        return 0.0
    return float((trade_ledger["net_pnl"] > 0.0).mean())


@DEFAULT_METRIC_REGISTRY.metric(
    name="avg_win",
    group="trade",
    required_ledgers={"trade_ledger": ("net_pnl",)},
)
def _metric_avg_win(ledgers: LedgerFrames) -> float:
    wins = _trade_ledger(ledgers).loc[_trade_ledger(ledgers)["net_pnl"] > 0.0, "net_pnl"]
    return float(wins.mean()) if not wins.empty else 0.0


@DEFAULT_METRIC_REGISTRY.metric(
    name="avg_loss",
    group="trade",
    required_ledgers={"trade_ledger": ("net_pnl",)},
)
def _metric_avg_loss(ledgers: LedgerFrames) -> float:
    losses = _trade_ledger(ledgers).loc[
        _trade_ledger(ledgers)["net_pnl"] < 0.0, "net_pnl"
    ]
    return float(losses.mean()) if not losses.empty else 0.0


@DEFAULT_METRIC_REGISTRY.metric(
    name="payoff_ratio",
    group="trade",
    required_ledgers={"trade_ledger": ("net_pnl",)},
)
def _metric_payoff_ratio(ledgers: LedgerFrames) -> float:
    avg_win = _metric_avg_win(ledgers)
    avg_loss = _metric_avg_loss(ledgers)
    if avg_loss < 0.0:
        return float(avg_win / abs(avg_loss))
    if avg_win > 0.0:
        return float(math.inf)
    return 0.0


@DEFAULT_METRIC_REGISTRY.metric(
    name="expectancy_per_trade",
    group="trade",
    required_ledgers={"trade_ledger": ("net_pnl",)},
)
def _metric_expectancy_per_trade(ledgers: LedgerFrames) -> float:
    trade_ledger = _trade_ledger(ledgers)
    return float(trade_ledger["net_pnl"].mean()) if not trade_ledger.empty else 0.0


@DEFAULT_METRIC_REGISTRY.metric(
    name="avg_bars_held",
    group="trade",
    required_ledgers={"trade_ledger": ("bars_held",)},
)
def _metric_avg_bars_held(ledgers: LedgerFrames) -> float:
    trade_ledger = _trade_ledger(ledgers)
    return float(trade_ledger["bars_held"].mean()) if not trade_ledger.empty else 0.0


@DEFAULT_METRIC_REGISTRY.metric(
    name="gross_pnl",
    group="trade",
    required_ledgers={"trade_ledger": ("gross_pnl",)},
)
def _metric_gross_pnl(ledgers: LedgerFrames) -> float:
    return float(_trade_ledger(ledgers)["gross_pnl"].sum())


@DEFAULT_METRIC_REGISTRY.metric(
    name="net_pnl",
    group="trade",
    required_ledgers={"trade_ledger": ("net_pnl",)},
)
def _metric_net_pnl(ledgers: LedgerFrames) -> float:
    return float(_trade_ledger(ledgers)["net_pnl"].sum())


@DEFAULT_METRIC_REGISTRY.metric(
    name="avg_mfe",
    group="trade",
    required_ledgers={"trade_ledger": ("mfe",)},
)
def _metric_avg_mfe(ledgers: LedgerFrames) -> float:
    trade_ledger = _trade_ledger(ledgers)
    return float(trade_ledger["mfe"].mean()) if not trade_ledger.empty else 0.0


@DEFAULT_METRIC_REGISTRY.metric(
    name="avg_mae",
    group="trade",
    required_ledgers={"trade_ledger": ("mae",)},
)
def _metric_avg_mae(ledgers: LedgerFrames) -> float:
    trade_ledger = _trade_ledger(ledgers)
    return float(trade_ledger["mae"].mean()) if not trade_ledger.empty else 0.0


@DEFAULT_METRIC_REGISTRY.metric(
    name="avg_exit_efficiency",
    group="trade",
    required_ledgers={"trade_ledger": ("exit_efficiency",)},
)
def _metric_avg_exit_efficiency(ledgers: LedgerFrames) -> float:
    trade_ledger = _trade_ledger(ledgers)
    return (
        float(trade_ledger["exit_efficiency"].mean())
        if not trade_ledger.empty
        else 0.0
    )


@DEFAULT_METRIC_REGISTRY.metric(
    name="total_return",
    group="equity",
    required_ledgers={"equity_curve": ("equity",)},
)
def _metric_total_return(ledgers: LedgerFrames) -> float:
    equity_curve = _equity_curve(ledgers)
    if equity_curve.empty:
        return 0.0
    initial_equity = float(equity_curve["equity"].iloc[0])
    final_equity = float(equity_curve["equity"].iloc[-1])
    return (final_equity / initial_equity) - 1.0 if initial_equity != 0.0 else 0.0


@DEFAULT_METRIC_REGISTRY.metric(
    name="annualized_return",
    group="equity",
    required_ledgers={"equity_curve": ("ts", "equity")},
)
def _metric_annualized_return(ledgers: LedgerFrames) -> float:
    equity_curve = _equity_curve(ledgers)
    if equity_curve.empty:
        return 0.0
    initial_equity = float(equity_curve["equity"].iloc[0])
    final_equity = float(equity_curve["equity"].iloc[-1])
    elapsed_seconds = float(
        (equity_curve["ts"].iloc[-1] - equity_curve["ts"].iloc[0]).total_seconds()
    )
    if initial_equity > 0.0 and final_equity > 0.0 and elapsed_seconds > 0.0:
        return float(
            (final_equity / initial_equity) ** (_SECONDS_PER_YEAR / elapsed_seconds) - 1.0
        )
    return 0.0


def _equity_returns(equity_curve: pd.DataFrame) -> pd.Series:
    return equity_curve["equity"].pct_change().dropna()


@DEFAULT_METRIC_REGISTRY.metric(
    name="annualized_vol",
    group="equity",
    required_ledgers={"equity_curve": ("ts", "equity")},
)
def _metric_annualized_vol(ledgers: LedgerFrames) -> float:
    equity_curve = _equity_curve(ledgers)
    returns = _equity_returns(equity_curve)
    periods_per_year = _infer_periods_per_year(equity_curve)
    if returns.empty or periods_per_year <= 0.0:
        return 0.0
    return float(returns.std(ddof=0)) * math.sqrt(periods_per_year)


@DEFAULT_METRIC_REGISTRY.metric(
    name="sharpe",
    group="equity",
    required_ledgers={"equity_curve": ("ts", "equity")},
)
def _metric_sharpe(ledgers: LedgerFrames) -> float:
    equity_curve = _equity_curve(ledgers)
    returns = _equity_returns(equity_curve)
    periods_per_year = _infer_periods_per_year(equity_curve)
    if returns.empty or periods_per_year <= 0.0:
        return 0.0
    return_vol = float(returns.std(ddof=0))
    if return_vol <= 0.0:
        return 0.0
    return float(returns.mean()) / return_vol * math.sqrt(periods_per_year)


@DEFAULT_METRIC_REGISTRY.metric(
    name="sortino",
    group="equity",
    required_ledgers={"equity_curve": ("ts", "equity")},
)
def _metric_sortino(ledgers: LedgerFrames) -> float:
    equity_curve = _equity_curve(ledgers)
    returns = _equity_returns(equity_curve)
    periods_per_year = _infer_periods_per_year(equity_curve)
    if returns.empty or periods_per_year <= 0.0:
        return 0.0
    downside_returns = returns.loc[returns < 0.0]
    downside_vol = float(downside_returns.std(ddof=0)) if not downside_returns.empty else 0.0
    if downside_vol <= 0.0:
        return 0.0
    return float(returns.mean()) / downside_vol * math.sqrt(periods_per_year)


@DEFAULT_METRIC_REGISTRY.metric(
    name="max_drawdown",
    group="equity",
    required_ledgers={"equity_curve": ("drawdown",)},
)
def _metric_max_drawdown(ledgers: LedgerFrames) -> float:
    equity_curve = _equity_curve(ledgers)
    return float(equity_curve["drawdown"].min()) if not equity_curve.empty else 0.0


@DEFAULT_METRIC_REGISTRY.metric(
    name="drawdown_duration",
    group="equity",
    required_ledgers={"equity_curve": ("drawdown",)},
)
def _metric_drawdown_duration(ledgers: LedgerFrames) -> int:
    equity_curve = _equity_curve(ledgers)
    return _drawdown_duration(equity_curve["drawdown"]) if not equity_curve.empty else 0


@DEFAULT_METRIC_REGISTRY.metric(
    name="turnover",
    group="equity",
    required_ledgers={"equity_curve": ("equity",), "fill_log": ("gross_notional",)},
)
def _metric_turnover(ledgers: LedgerFrames) -> float:
    equity_curve = _equity_curve(ledgers)
    average_equity = float(equity_curve["equity"].mean()) if not equity_curve.empty else 0.0
    if average_equity == 0.0:
        return 0.0
    return float(_fill_log(ledgers)["gross_notional"].sum()) / average_equity


@DEFAULT_METRIC_REGISTRY.metric(
    name="time_in_market",
    group="equity",
    required_ledgers={"equity_curve": ("gross_exposure",)},
)
def _metric_time_in_market(ledgers: LedgerFrames) -> float:
    equity_curve = _equity_curve(ledgers)
    return float((equity_curve["gross_exposure"] > 0.0).mean()) if not equity_curve.empty else 0.0


@DEFAULT_METRIC_REGISTRY.metric(
    name="pnl_by_symbol",
    group="behavioral",
    required_ledgers={"trade_ledger": ("symbol", "net_pnl")},
)
def _metric_pnl_by_symbol(ledgers: LedgerFrames) -> dict[str, float]:
    return {
        str(symbol): float(value)
        for symbol, value in _trade_ledger(ledgers).groupby("symbol")["net_pnl"].sum().items()
    }


@DEFAULT_METRIC_REGISTRY.metric(
    name="pnl_by_side",
    group="behavioral",
    required_ledgers={"trade_ledger": ("side", "net_pnl")},
)
def _metric_pnl_by_side(ledgers: LedgerFrames) -> dict[str, float]:
    return {
        str(side): float(value)
        for side, value in _trade_ledger(ledgers).groupby("side")["net_pnl"].sum().items()
    }


@DEFAULT_METRIC_REGISTRY.metric(
    name="costs_breakdown",
    group="behavioral",
    required_ledgers={"fill_log": ("fees", "slippage")},
)
def _metric_costs_breakdown(ledgers: LedgerFrames) -> dict[str, float]:
    fees = float(_fill_log(ledgers)["fees"].sum())
    slippage = float(_fill_log(ledgers)["slippage"].sum())
    return {
        "fees": fees,
        "slippage": slippage,
        "total_costs": fees + slippage,
    }


@DEFAULT_METRIC_REGISTRY.metric(
    name="holding_period_distribution",
    group="behavioral",
    required_ledgers={"trade_ledger": ("bars_held",)},
)
def _metric_holding_period_distribution(ledgers: LedgerFrames) -> dict[int, int]:
    return {
        int(period): int(count)
        for period, count in _trade_ledger(ledgers)["bars_held"]
        .value_counts()
        .sort_index()
        .items()
    }


def _compute_from_registry(
    metric_names: tuple[str, ...],
    ledgers: LedgerFrames,
    *,
    registry: MetricRegistry = DEFAULT_METRIC_REGISTRY,
) -> dict[str, Any]:
    return registry.compute(metric_names, ledgers)


def compute_trade_metrics(trade_ledger: pd.DataFrame) -> dict[str, Any]:
    validated_ledgers: dict[LedgerName, pd.DataFrame] = {
        "trade_ledger": _validate_ledger("trade_ledger", trade_ledger)
    }
    return _compute_from_registry(TRADE_METRIC_NAMES, validated_ledgers)


def compute_equity_metrics(
    equity_curve: pd.DataFrame,
    fill_log: pd.DataFrame,
) -> dict[str, Any]:
    validated_ledgers: dict[LedgerName, pd.DataFrame] = {
        "equity_curve": _validate_ledger("equity_curve", equity_curve),
        "fill_log": _validate_ledger("fill_log", fill_log),
    }
    return _compute_from_registry(EQUITY_METRIC_NAMES, validated_ledgers)


def compute_behavioral_metrics(
    trade_ledger: pd.DataFrame,
    fill_log: pd.DataFrame,
) -> dict[str, Any]:
    validated_ledgers: dict[LedgerName, pd.DataFrame] = {
        "trade_ledger": _validate_ledger("trade_ledger", trade_ledger),
        "fill_log": _validate_ledger("fill_log", fill_log),
    }
    return _compute_from_registry(BEHAVIORAL_METRIC_NAMES, validated_ledgers)


def compute_metrics(result: BacktestResult) -> dict[str, Any]:
    ledgers = _validate_backtest_result(result)
    metric_names = TRADE_METRIC_NAMES + EQUITY_METRIC_NAMES + BEHAVIORAL_METRIC_NAMES
    return _compute_from_registry(metric_names, ledgers)


__all__ = [
    "BEHAVIORAL_METRIC_NAMES",
    "DEFAULT_METRIC_REGISTRY",
    "EQUITY_METRIC_NAMES",
    "MetricDefinition",
    "MetricDependencyError",
    "MetricRegistry",
    "TRADE_METRIC_NAMES",
    "compute_behavioral_metrics",
    "compute_equity_metrics",
    "compute_metrics",
    "compute_trade_metrics",
]
