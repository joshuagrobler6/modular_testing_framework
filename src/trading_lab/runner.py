from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Real
from typing import Any, Callable, Literal

import pandas as pd

from trading_lab.analytics import compute_metrics
from trading_lab.contracts import DEFAULT_CONTRACT_VERSION, BacktestResult
from trading_lab.engine import run_backtest
from trading_lab.experiments import (
    ExperimentSpec,
    VariantSpec,
    label_fold,
    serialize_manifest,
)
from trading_lab.registry import NodeRegistry

RunPhase = Literal["cv", "holdout"]
VariantStatus = Literal["completed", "pruned", "failed"]
SkipReason = Literal["runtime_budget_reached", "max_variants_reached"]

BacktestRunner = Callable[..., BacktestResult]
MetricsFn = Callable[[BacktestResult], dict[str, Any]]
TimeFn = Callable[[], float]

_DEFAULT_RANK_METRIC = "net_pnl"
_SUM_METRICS = frozenset({"trade_count", "gross_pnl", "net_pnl"})
_WEIGHTED_BY_TRADES_METRICS = frozenset(
    {"avg_bars_held", "avg_mfe", "avg_mae", "avg_exit_efficiency"}
)
_MEAN_METRICS = frozenset(
    {
        "total_return",
        "annualized_return",
        "annualized_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "drawdown_duration",
        "turnover",
        "time_in_market",
    }
)
_MAPPING_SUM_METRICS = frozenset(
    {"pnl_by_symbol", "pnl_by_side", "costs_breakdown", "holding_period_distribution"}
)


def _require_datetime(name: str, value: object) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(value).__name__}.")


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}.")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}.")


def _require_number(
    name: str,
    value: object,
    *,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, got {type(value).__name__}.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value}.")
    if non_negative and numeric < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return numeric


def _require_contract_version(name: str, value: str) -> None:
    _require_non_empty(name, value)


def _normalize_string_tuple(
    name: str,
    value: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if value is None:
        return ()
    normalized = tuple(value)
    for item in normalized:
        _require_non_empty(name, item)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")
    return normalized


def _copy_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return dict(value)


def _fold_key(variant_id: str, phase: RunPhase, label: str) -> str:
    return f"{variant_id}:{phase}:{label}"


def _experiment_run_id(experiment: ExperimentSpec) -> str:
    digest = hashlib.sha256(serialize_manifest(experiment).encode("utf-8")).hexdigest()
    return f"experiment-{digest[:16]}"


def _slice_window(data: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if "ts" not in data.columns:
        raise ValueError("experiment data must contain a 'ts' column.")
    mask = (data["ts"] >= start) & (data["ts"] < end)
    return data.loc[mask].reset_index(drop=True)


def _run_id_from_result(result: BacktestResult) -> str | None:
    manifest = result.artifacts.get("run_manifest")
    if not isinstance(manifest, dict):
        return None
    run_id = manifest.get("run_id")
    return str(run_id) if isinstance(run_id, str) and run_id.strip() else None


def _final_equity(result: BacktestResult) -> float | None:
    if "equity" not in result.equity_curve.columns or result.equity_curve.empty:
        return None
    return float(result.equity_curve["equity"].iloc[-1])


def _metric_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _contains_invalid_numeric(value: object, *, prefix: str) -> str | None:
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            reason = _contains_invalid_numeric(value[key], prefix=child_prefix)
            if reason is not None:
                return reason
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            reason = _contains_invalid_numeric(item, prefix=child_prefix)
            if reason is not None:
                return reason
        return None
    if isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return prefix
    return None


def _invalid_numeric_state_reason(
    *,
    result: BacktestResult,
    metrics: dict[str, Any],
) -> str | None:
    metric_reason = _contains_invalid_numeric(metrics, prefix="metrics")
    if metric_reason is not None:
        return metric_reason

    for frame_name in (
        "decision_log",
        "order_log",
        "fill_log",
        "trade_ledger",
        "equity_curve",
    ):
        frame = getattr(result, frame_name)
        numeric_columns = frame.select_dtypes(include="number")
        for column_name in numeric_columns.columns:
            for row_index, value in enumerate(numeric_columns[column_name].tolist()):
                if isinstance(value, Real) and not isinstance(value, bool):
                    if not math.isfinite(float(value)):
                        return f"{frame_name}.{column_name}[{row_index}]"
    return None


def _should_apply_early_thresholds(
    *,
    pruning_config: Any,
    metrics: dict[str, Any],
    bars_observed: int,
) -> bool:
    if not pruning_config.early_metric_thresholds:
        return False
    if pruning_config.early_min_trades is None and pruning_config.early_min_bars is None:
        return True

    trade_count_value = _metric_number(metrics.get("trade_count"))
    meets_trade_gate = (
        pruning_config.early_min_trades is not None
        and trade_count_value is not None
        and int(trade_count_value) >= pruning_config.early_min_trades
    )
    meets_bar_gate = (
        pruning_config.early_min_bars is not None
        and bars_observed >= pruning_config.early_min_bars
    )
    return meets_trade_gate or meets_bar_gate


def _numeric_mapping_sum(values: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for value in values for key in value})
    aggregated: dict[str, float] = {}
    for key in keys:
        total = 0.0
        found = False
        for value in values:
            if key not in value:
                continue
            number = value[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                continue
            total += float(number)
            found = True
        if found:
            aggregated[key] = total
    return aggregated


def aggregate_cv_metrics(
    fold_summaries: tuple["FoldRunSummary", ...] | list["FoldRunSummary"],
) -> dict[str, Any]:
    summaries = tuple(fold_summaries)
    if not summaries:
        return {}

    metrics_by_fold = [summary.metrics for summary in summaries]
    trade_counts = [
        int(metrics.get("trade_count", 0))
        for metrics in metrics_by_fold
        if isinstance(metrics.get("trade_count", 0), (int, float))
    ]
    total_trades = int(sum(trade_counts))

    aggregated: dict[str, Any] = {}
    if total_trades > 0:
        aggregated["trade_count"] = total_trades
        net_pnl = sum(float(metrics.get("net_pnl", 0.0)) for metrics in metrics_by_fold)
        gross_pnl = sum(
            float(metrics.get("gross_pnl", 0.0)) for metrics in metrics_by_fold
        )
        wins = sum(
            float(metrics.get("win_rate", 0.0)) * float(metrics.get("trade_count", 0))
            for metrics in metrics_by_fold
        )
        losses = sum(
            float(metrics.get("trade_count", 0))
            - float(metrics.get("win_rate", 0.0)) * float(metrics.get("trade_count", 0))
            for metrics in metrics_by_fold
        )
        win_notional = sum(
            float(metrics.get("avg_win", 0.0))
            * float(metrics.get("win_rate", 0.0))
            * float(metrics.get("trade_count", 0))
            for metrics in metrics_by_fold
        )
        loss_notional = sum(
            float(metrics.get("avg_loss", 0.0))
            * (
                float(metrics.get("trade_count", 0))
                - float(metrics.get("win_rate", 0.0)) * float(metrics.get("trade_count", 0))
            )
            for metrics in metrics_by_fold
        )
        aggregated["gross_pnl"] = gross_pnl
        aggregated["net_pnl"] = net_pnl
        aggregated["win_rate"] = wins / total_trades
        aggregated["avg_win"] = win_notional / wins if wins > 0.0 else 0.0
        aggregated["avg_loss"] = loss_notional / losses if losses > 0.0 else 0.0
        if aggregated["avg_loss"] < 0.0:
            aggregated["payoff_ratio"] = aggregated["avg_win"] / abs(
                aggregated["avg_loss"]
            )
        elif aggregated["avg_win"] > 0.0:
            aggregated["payoff_ratio"] = float("inf")
        else:
            aggregated["payoff_ratio"] = 0.0
        aggregated["expectancy_per_trade"] = net_pnl / total_trades
        for metric_name in _WEIGHTED_BY_TRADES_METRICS:
            weighted_total = sum(
                float(metrics.get(metric_name, 0.0))
                * float(metrics.get("trade_count", 0))
                for metrics in metrics_by_fold
            )
            aggregated[metric_name] = weighted_total / total_trades
    else:
        for metric_name in (
            "trade_count",
            "win_rate",
            "avg_win",
            "avg_loss",
            "payoff_ratio",
            "expectancy_per_trade",
            *sorted(_WEIGHTED_BY_TRADES_METRICS),
            "gross_pnl",
            "net_pnl",
        ):
            aggregated[metric_name] = 0.0 if metric_name != "trade_count" else 0

    for metric_name in _MEAN_METRICS:
        values = [
            float(metrics[metric_name])
            for metrics in metrics_by_fold
            if isinstance(metrics.get(metric_name), (int, float))
        ]
        aggregated[metric_name] = sum(values) / len(values) if values else 0.0

    for metric_name in _MAPPING_SUM_METRICS:
        mapping_values = [
            value
            for metrics in metrics_by_fold
            if isinstance((value := metrics.get(metric_name)), dict)
        ]
        aggregated[metric_name] = _numeric_mapping_sum(mapping_values)

    handled_metric_names = (
        _SUM_METRICS
        | _WEIGHTED_BY_TRADES_METRICS
        | _MEAN_METRICS
        | _MAPPING_SUM_METRICS
        | {
            "win_rate",
            "avg_win",
            "avg_loss",
            "payoff_ratio",
            "expectancy_per_trade",
        }
    )
    extra_metric_names = sorted(
        {
            key
            for metrics in metrics_by_fold
            for key in metrics
            if key not in handled_metric_names
        }
    )
    for metric_name in extra_metric_names:
        values = [metrics[metric_name] for metrics in metrics_by_fold if metric_name in metrics]
        if all(isinstance(value, dict) for value in values):
            aggregated[metric_name] = _numeric_mapping_sum(values)  # type: ignore[arg-type]
            continue
        numeric_values = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if numeric_values:
            aggregated[metric_name] = sum(numeric_values) / len(numeric_values)

    aggregated["fold_count"] = len(summaries)
    return aggregated


def _prune_reason_after_fold(
    *,
    pruning_config: Any,
    result: BacktestResult,
    source_metrics: dict[str, Any],
    metrics: dict[str, Any],
    final_equity: float | None,
    bars_observed: int,
) -> str | None:
    if pruning_config.stop_on_invalid_numeric_state:
        invalid_reason = _invalid_numeric_state_reason(
            result=result,
            metrics=source_metrics,
        )
        if invalid_reason is None:
            invalid_reason = _invalid_numeric_state_reason(result=result, metrics=metrics)
        if invalid_reason is not None:
            return f"stop_on_invalid_numeric_state:{invalid_reason}"
    if (
        pruning_config.stop_on_zero_equity
        and final_equity is not None
        and final_equity <= 0.0
    ):
        return f"stop_on_zero_equity:final_equity={float(final_equity)}"
    if pruning_config.max_drawdown_threshold is not None:
        observed_drawdown = _metric_number(metrics.get("max_drawdown"))
        if (
            observed_drawdown is not None
            and observed_drawdown <= float(pruning_config.max_drawdown_threshold)
        ):
            return (
                "max_drawdown_threshold:"
                f"max_drawdown={observed_drawdown}<="
                f"{float(pruning_config.max_drawdown_threshold)}"
            )
    if _should_apply_early_thresholds(
        pruning_config=pruning_config,
        metrics=metrics,
        bars_observed=bars_observed,
    ):
        trade_count_value = _metric_number(metrics.get("trade_count"))
        trade_count_label = int(trade_count_value) if trade_count_value is not None else None
        for metric_name in sorted(pruning_config.early_metric_thresholds):
            threshold = float(pruning_config.early_metric_thresholds[metric_name])
            metric_value = _metric_number(metrics.get(metric_name))
            if metric_value is not None and metric_value < threshold:
                return (
                    "early_metric_threshold:"
                    f"{metric_name}={metric_value}<{threshold}"
                    f" after trades={trade_count_label} bars={bars_observed}"
                )
    return None


def _prune_reason_after_cv(
    *,
    pruning_config: Any,
    metrics: dict[str, Any],
) -> str | None:
    if pruning_config.min_trades is not None:
        trade_count = _metric_number(metrics.get("trade_count"))
        if trade_count is not None and int(trade_count) < pruning_config.min_trades:
            return f"min_trades:trade_count={int(trade_count)}<{pruning_config.min_trades}"
    return None


@dataclass(frozen=True, slots=True)
class FoldRunSummary:
    variant_id: str
    phase: RunPhase
    label: str
    window_start: datetime
    window_end: datetime
    runtime_seconds: float
    metrics: dict[str, Any]
    bars_observed: int
    run_id: str | None = None
    final_equity: float | None = None
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("variant_id", self.variant_id)
        if self.phase not in {"cv", "holdout"}:
            raise ValueError(f"unsupported phase: {self.phase!r}.")
        _require_non_empty("label", self.label)
        _require_datetime("window_start", self.window_start)
        _require_datetime("window_end", self.window_end)
        _require_number("runtime_seconds", self.runtime_seconds, non_negative=True)
        if not isinstance(self.bars_observed, int) or self.bars_observed < 0:
            raise ValueError("bars_observed must be a non-negative int.")
        if self.window_start >= self.window_end:
            raise ValueError("window_start must be earlier than window_end.")
        if self.run_id is not None:
            _require_non_empty("run_id", self.run_id)
        if self.final_equity is not None:
            _require_number("final_equity", self.final_equity)
        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dict.")
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(self, "metrics", _copy_mapping(self.metrics))


@dataclass(frozen=True, slots=True)
class VariantCvSummary:
    variant_id: str
    status: VariantStatus
    fold_labels: tuple[str, ...]
    cv_metrics: dict[str, Any]
    ranking_metrics: dict[str, Any]
    runtime_seconds: float
    estimated_remaining_seconds: float
    holdout_executed: bool = False
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("variant_id", self.variant_id)
        if self.status not in {"completed", "pruned", "failed"}:
            raise ValueError(f"unsupported status: {self.status!r}.")
        object.__setattr__(
            self,
            "fold_labels",
            _normalize_string_tuple("fold_labels", self.fold_labels),
        )
        if not isinstance(self.cv_metrics, dict):
            raise TypeError("cv_metrics must be a dict.")
        if not isinstance(self.ranking_metrics, dict):
            raise TypeError("ranking_metrics must be a dict.")
        _require_number("runtime_seconds", self.runtime_seconds, non_negative=True)
        _require_number(
            "estimated_remaining_seconds",
            self.estimated_remaining_seconds,
            non_negative=True,
        )
        _require_bool("holdout_executed", self.holdout_executed)
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(self, "cv_metrics", _copy_mapping(self.cv_metrics))
        object.__setattr__(self, "ranking_metrics", _copy_mapping(self.ranking_metrics))


@dataclass(frozen=True, slots=True)
class HoldoutRunSummary:
    variant_id: str
    label: str
    window_start: datetime
    window_end: datetime
    runtime_seconds: float
    metrics: dict[str, Any]
    bars_observed: int
    run_id: str | None = None
    final_equity: float | None = None
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("variant_id", self.variant_id)
        _require_non_empty("label", self.label)
        _require_datetime("window_start", self.window_start)
        _require_datetime("window_end", self.window_end)
        _require_number("runtime_seconds", self.runtime_seconds, non_negative=True)
        if not isinstance(self.bars_observed, int) or self.bars_observed < 0:
            raise ValueError("bars_observed must be a non-negative int.")
        if self.window_start >= self.window_end:
            raise ValueError("window_start must be earlier than window_end.")
        if self.run_id is not None:
            _require_non_empty("run_id", self.run_id)
        if self.final_equity is not None:
            _require_number("final_equity", self.final_equity)
        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dict.")
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(self, "metrics", _copy_mapping(self.metrics))


@dataclass(frozen=True, slots=True)
class RuntimeCheckpoint:
    variant_id: str
    phase: RunPhase
    label: str
    elapsed_seconds: float
    estimated_remaining_seconds: float
    completed_units: int
    remaining_units: int
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("variant_id", self.variant_id)
        if self.phase not in {"cv", "holdout"}:
            raise ValueError(f"unsupported phase: {self.phase!r}.")
        _require_non_empty("label", self.label)
        _require_number("elapsed_seconds", self.elapsed_seconds, non_negative=True)
        _require_number(
            "estimated_remaining_seconds",
            self.estimated_remaining_seconds,
            non_negative=True,
        )
        if not isinstance(self.completed_units, int) or self.completed_units < 0:
            raise ValueError("completed_units must be a non-negative int.")
        if not isinstance(self.remaining_units, int) or self.remaining_units < 0:
            raise ValueError("remaining_units must be a non-negative int.")
        _require_contract_version("contract_version", self.contract_version)


@dataclass(frozen=True, slots=True)
class RuntimeSummary:
    total_runtime_seconds: float
    variant_runtimes: dict[str, float]
    fold_runtimes: dict[str, float]
    estimated_remaining_seconds: float
    progress: tuple[RuntimeCheckpoint, ...]
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_number("total_runtime_seconds", self.total_runtime_seconds, non_negative=True)
        _require_number(
            "estimated_remaining_seconds",
            self.estimated_remaining_seconds,
            non_negative=True,
        )
        if not isinstance(self.variant_runtimes, dict):
            raise TypeError("variant_runtimes must be a dict.")
        if not isinstance(self.fold_runtimes, dict):
            raise TypeError("fold_runtimes must be a dict.")
        normalized_variant_runtimes: dict[str, float] = {}
        normalized_fold_runtimes: dict[str, float] = {}
        for key, value in self.variant_runtimes.items():
            _require_non_empty("variant_runtimes key", key)
            normalized_variant_runtimes[key] = _require_number(
                f"variant_runtimes[{key!r}]",
                value,
                non_negative=True,
            )
        for key, value in self.fold_runtimes.items():
            _require_non_empty("fold_runtimes key", key)
            normalized_fold_runtimes[key] = _require_number(
                f"fold_runtimes[{key!r}]",
                value,
                non_negative=True,
            )
        if any(not isinstance(checkpoint, RuntimeCheckpoint) for checkpoint in self.progress):
            raise TypeError("progress must contain RuntimeCheckpoint objects only.")
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(self, "variant_runtimes", normalized_variant_runtimes)
        object.__setattr__(self, "fold_runtimes", normalized_fold_runtimes)
        object.__setattr__(self, "progress", tuple(self.progress))


@dataclass(frozen=True, slots=True)
class PruneRecord:
    variant_id: str
    stage: Literal["after_fold", "after_cv"]
    reason: str
    fold_label: str | None = None
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("variant_id", self.variant_id)
        if self.stage not in {"after_fold", "after_cv"}:
            raise ValueError(f"unsupported stage: {self.stage!r}.")
        _require_non_empty("reason", self.reason)
        if self.fold_label is not None:
            _require_non_empty("fold_label", self.fold_label)
        if not isinstance(self.metrics_snapshot, dict):
            raise TypeError("metrics_snapshot must be a dict.")
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(self, "metrics_snapshot", _copy_mapping(self.metrics_snapshot))


@dataclass(frozen=True, slots=True)
class FailureRecord:
    variant_id: str
    phase: RunPhase
    label: str
    error_type: str
    message: str
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("variant_id", self.variant_id)
        if self.phase not in {"cv", "holdout"}:
            raise ValueError(f"unsupported phase: {self.phase!r}.")
        _require_non_empty("label", self.label)
        _require_non_empty("error_type", self.error_type)
        _require_non_empty("message", self.message)
        _require_contract_version("contract_version", self.contract_version)


@dataclass(frozen=True, slots=True)
class SkipRecord:
    variant_id: str
    reason: SkipReason
    elapsed_seconds: float
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("variant_id", self.variant_id)
        if self.reason not in {"runtime_budget_reached", "max_variants_reached"}:
            raise ValueError(f"unsupported skip reason: {self.reason!r}.")
        _require_number("elapsed_seconds", self.elapsed_seconds, non_negative=True)
        _require_contract_version("contract_version", self.contract_version)


@dataclass(frozen=True, slots=True)
class VariantRanking:
    variant_id: str
    rank: int
    metric_name: str
    metric_value: float
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("variant_id", self.variant_id)
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be a positive int.")
        _require_non_empty("metric_name", self.metric_name)
        _require_number("metric_value", self.metric_value)
        _require_contract_version("contract_version", self.contract_version)


@dataclass(frozen=True, slots=True)
class ExperimentExecutionResult:
    experiment: ExperimentSpec
    run_id: str
    fold_summaries: tuple[FoldRunSummary, ...]
    cv_summaries: tuple[VariantCvSummary, ...]
    holdout_summaries: tuple[HoldoutRunSummary, ...]
    cv_rankings: tuple[VariantRanking, ...]
    runtime_summary: RuntimeSummary
    prune_records: tuple[PruneRecord, ...]
    failure_records: tuple[FailureRecord, ...]
    skip_records: tuple[SkipRecord, ...]
    run_results: dict[str, BacktestResult]
    completed_variant_ids: tuple[str, ...]
    pruned_variant_ids: tuple[str, ...]
    failed_variant_ids: tuple[str, ...]
    skipped_variant_ids: tuple[str, ...]
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, ExperimentSpec):
            raise TypeError("experiment must be an ExperimentSpec instance.")
        _require_non_empty("run_id", self.run_id)
        for name, value, expected_type in (
            ("fold_summaries", self.fold_summaries, FoldRunSummary),
            ("cv_summaries", self.cv_summaries, VariantCvSummary),
            ("holdout_summaries", self.holdout_summaries, HoldoutRunSummary),
            ("cv_rankings", self.cv_rankings, VariantRanking),
            ("prune_records", self.prune_records, PruneRecord),
            ("failure_records", self.failure_records, FailureRecord),
            ("skip_records", self.skip_records, SkipRecord),
        ):
            if any(not isinstance(item, expected_type) for item in value):
                raise TypeError(f"{name} must contain {expected_type.__name__} objects only.")
        if not isinstance(self.runtime_summary, RuntimeSummary):
            raise TypeError("runtime_summary must be a RuntimeSummary instance.")
        if not isinstance(self.run_results, dict):
            raise TypeError("run_results must be a dict.")
        normalized_run_results: dict[str, BacktestResult] = {}
        for key, value in self.run_results.items():
            _require_non_empty("run_results key", key)
            if not isinstance(value, BacktestResult):
                raise TypeError("run_results values must be BacktestResult instances.")
            normalized_run_results[key] = value
        object.__setattr__(self, "run_results", normalized_run_results)
        object.__setattr__(
            self,
            "completed_variant_ids",
            _normalize_string_tuple("completed_variant_ids", self.completed_variant_ids),
        )
        object.__setattr__(
            self,
            "pruned_variant_ids",
            _normalize_string_tuple("pruned_variant_ids", self.pruned_variant_ids),
        )
        object.__setattr__(
            self,
            "failed_variant_ids",
            _normalize_string_tuple("failed_variant_ids", self.failed_variant_ids),
        )
        object.__setattr__(
            self,
            "skipped_variant_ids",
            _normalize_string_tuple("skipped_variant_ids", self.skipped_variant_ids),
        )
        _require_contract_version("contract_version", self.contract_version)


def _estimate_remaining_seconds(
    *,
    total_runtime_seconds: float,
    completed_units: int,
    cancelled_units: int,
    total_planned_units: int,
) -> float:
    if completed_units <= 0:
        return 0.0
    remaining_units = max(total_planned_units - completed_units - cancelled_units, 0)
    if remaining_units == 0:
        return 0.0
    average_unit_runtime = total_runtime_seconds / completed_units
    return average_unit_runtime * remaining_units


def _append_runtime_checkpoint(
    *,
    checkpoints: list[RuntimeCheckpoint],
    variant_id: str,
    phase: RunPhase,
    label: str,
    total_runtime_seconds: float,
    completed_units: int,
    cancelled_units: int,
    total_planned_units: int,
    contract_version: str,
) -> float:
    estimated_remaining_seconds = _estimate_remaining_seconds(
        total_runtime_seconds=total_runtime_seconds,
        completed_units=completed_units,
        cancelled_units=cancelled_units,
        total_planned_units=total_planned_units,
    )
    checkpoints.append(
        RuntimeCheckpoint(
            variant_id=variant_id,
            phase=phase,
            label=label,
            elapsed_seconds=total_runtime_seconds,
            estimated_remaining_seconds=estimated_remaining_seconds,
            completed_units=completed_units,
            remaining_units=max(total_planned_units - completed_units - cancelled_units, 0),
            contract_version=contract_version,
        )
    )
    return estimated_remaining_seconds


def _run_single_backtest(
    *,
    backtest_fn: BacktestRunner,
    spec: VariantSpec,
    data_slice: pd.DataFrame,
    node_registry: NodeRegistry | None,
) -> BacktestResult:
    return backtest_fn(spec.backtest_spec, data_slice, node_registry=node_registry)


def _fold_summary(
    *,
    variant_id: str,
    phase: RunPhase,
    label: str,
    window_start: datetime,
    window_end: datetime,
    runtime_seconds: float,
    result: BacktestResult,
    metrics: dict[str, Any],
    contract_version: str,
) -> FoldRunSummary:
    return FoldRunSummary(
        variant_id=variant_id,
        phase=phase,
        label=label,
        window_start=window_start,
        window_end=window_end,
        runtime_seconds=runtime_seconds,
        metrics=metrics,
        bars_observed=int(len(result.equity_curve)),
        run_id=_run_id_from_result(result),
        final_equity=_final_equity(result),
        contract_version=contract_version,
    )


def _build_rankings(
    cv_summaries: tuple[VariantCvSummary, ...],
    *,
    metric_name: str = _DEFAULT_RANK_METRIC,
    contract_version: str,
) -> tuple[VariantRanking, ...]:
    rankable = [
        summary
        for summary in cv_summaries
        if _metric_number(summary.ranking_metrics.get(metric_name)) is not None
    ]
    ordered = sorted(
        rankable,
        key=lambda summary: (
            -float(_metric_number(summary.ranking_metrics[metric_name])),
            summary.variant_id,
        ),
    )
    return tuple(
        VariantRanking(
            variant_id=summary.variant_id,
            rank=index + 1,
            metric_name=metric_name,
            metric_value=float(_metric_number(summary.ranking_metrics[metric_name])),
            contract_version=contract_version,
        )
        for index, summary in enumerate(ordered)
    )


def run_experiment(
    experiment: ExperimentSpec,
    data: pd.DataFrame,
    *,
    node_registry: NodeRegistry | None = None,
    backtest_fn: BacktestRunner = run_backtest,
    metrics_fn: MetricsFn = compute_metrics,
    time_fn: TimeFn = time.perf_counter,
) -> ExperimentExecutionResult:
    if not isinstance(experiment, ExperimentSpec):
        raise TypeError("experiment must be an ExperimentSpec instance.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not callable(backtest_fn):
        raise TypeError("backtest_fn must be callable.")
    if not callable(metrics_fn):
        raise TypeError("metrics_fn must be callable.")
    if not callable(time_fn):
        raise TypeError("time_fn must be callable.")

    run_id = _experiment_run_id(experiment)
    variants = experiment.variants
    max_launches = (
        min(len(variants), experiment.search.max_variants)
        if experiment.search.max_variants is not None
        else len(variants)
    )
    launchable_variants = variants[:max_launches]
    max_variants_skipped = variants[max_launches:]
    units_per_variant = len(experiment.folds) + (1 if experiment.holdout is not None else 0)
    total_planned_units = len(launchable_variants) * units_per_variant

    fold_summaries: list[FoldRunSummary] = []
    cv_summaries: list[VariantCvSummary] = []
    holdout_summaries: list[HoldoutRunSummary] = []
    prune_records: list[PruneRecord] = []
    failure_records: list[FailureRecord] = []
    skip_records: list[SkipRecord] = []
    progress: list[RuntimeCheckpoint] = []
    variant_runtimes: dict[str, float] = {}
    fold_runtimes: dict[str, float] = {}
    run_results: dict[str, BacktestResult] = {}

    completed_variant_ids: list[str] = []
    pruned_variant_ids: list[str] = []
    failed_variant_ids: list[str] = []
    skipped_variant_ids: list[str] = []

    total_runtime_seconds = 0.0
    completed_units = 0
    cancelled_units = 0
    latest_estimated_remaining_seconds = 0.0

    for variant_index, variant in enumerate(launchable_variants):
        runtime_budget = experiment.search.max_runtime_seconds
        if runtime_budget is not None and total_runtime_seconds >= float(runtime_budget):
            remaining_variants = launchable_variants[variant_index:]
            cancelled_units += len(remaining_variants) * units_per_variant
            for skipped_variant in remaining_variants:
                skip_records.append(
                    SkipRecord(
                        variant_id=skipped_variant.variant_id,
                        reason="runtime_budget_reached",
                        elapsed_seconds=total_runtime_seconds,
                        contract_version=experiment.contract_version,
                    )
                )
                skipped_variant_ids.append(skipped_variant.variant_id)
            break

        variant_fold_summaries: list[FoldRunSummary] = []
        variant_runtime_seconds = 0.0
        variant_status: VariantStatus = "completed"
        holdout_executed = False

        for fold in experiment.folds:
            fold_label = label_fold(fold)
            fold_data = _slice_window(data, fold.validation_start, fold.validation_end)
            if fold_data.empty:
                failure_records.append(
                    FailureRecord(
                        variant_id=variant.variant_id,
                        phase="cv",
                        label=fold_label,
                        error_type="ValueError",
                        message="validation fold has no rows in the provided data window.",
                        contract_version=experiment.contract_version,
                    )
                )
                variant_status = "failed"
                failed_variant_ids.append(variant.variant_id)
                cancelled_units += units_per_variant - len(variant_fold_summaries)
                break

            started = float(time_fn())
            try:
                backtest_result = _run_single_backtest(
                    backtest_fn=backtest_fn,
                    spec=variant,
                    data_slice=fold_data,
                    node_registry=node_registry,
                )
                metrics = metrics_fn(backtest_result)
            except Exception as exc:
                ended = float(time_fn())
                runtime_seconds = max(ended - started, 0.0)
                total_runtime_seconds += runtime_seconds
                variant_runtime_seconds += runtime_seconds
                fold_runtimes[_fold_key(variant.variant_id, "cv", fold_label)] = runtime_seconds
                variant_runtimes[variant.variant_id] = variant_runtime_seconds
                failure_records.append(
                    FailureRecord(
                        variant_id=variant.variant_id,
                        phase="cv",
                        label=fold_label,
                        error_type=type(exc).__name__,
                        message=str(exc),
                        contract_version=experiment.contract_version,
                    )
                )
                variant_status = "failed"
                failed_variant_ids.append(variant.variant_id)
                cancelled_units += units_per_variant - len(variant_fold_summaries)
                break

            ended = float(time_fn())
            runtime_seconds = max(ended - started, 0.0)
            total_runtime_seconds += runtime_seconds
            variant_runtime_seconds += runtime_seconds
            completed_units += 1

            summary = _fold_summary(
                variant_id=variant.variant_id,
                phase="cv",
                label=fold_label,
                window_start=fold.validation_start,
                window_end=fold.validation_end,
                runtime_seconds=runtime_seconds,
                result=backtest_result,
                metrics=metrics,
                contract_version=experiment.contract_version,
            )
            fold_summaries.append(summary)
            variant_fold_summaries.append(summary)
            run_results[_fold_key(variant.variant_id, "cv", fold_label)] = backtest_result
            fold_runtimes[_fold_key(variant.variant_id, "cv", fold_label)] = runtime_seconds
            latest_estimated_remaining_seconds = _append_runtime_checkpoint(
                checkpoints=progress,
                variant_id=variant.variant_id,
                phase="cv",
                label=fold_label,
                total_runtime_seconds=total_runtime_seconds,
                completed_units=completed_units,
                cancelled_units=cancelled_units,
                total_planned_units=total_planned_units,
                contract_version=experiment.contract_version,
            )

            partial_cv_metrics = aggregate_cv_metrics(variant_fold_summaries)
            prune_reason = _prune_reason_after_fold(
                pruning_config=experiment.pruning,
                result=backtest_result,
                source_metrics=metrics,
                metrics=partial_cv_metrics,
                final_equity=summary.final_equity,
                bars_observed=sum(
                    fold_summary.bars_observed for fold_summary in variant_fold_summaries
                ),
            )
            if prune_reason is not None:
                variant_status = "pruned"
                pruned_variant_ids.append(variant.variant_id)
                prune_records.append(
                    PruneRecord(
                        variant_id=variant.variant_id,
                        stage="after_fold",
                        reason=prune_reason,
                        fold_label=fold_label,
                        metrics_snapshot=partial_cv_metrics,
                        contract_version=experiment.contract_version,
                    )
                )
                remaining_units = units_per_variant - len(variant_fold_summaries)
                cancelled_units += max(remaining_units, 0)
                break

        cv_metrics = aggregate_cv_metrics(variant_fold_summaries)

        if variant_status == "completed":
            prune_reason = _prune_reason_after_cv(
                pruning_config=experiment.pruning,
                metrics=cv_metrics,
            )
            if prune_reason is not None:
                variant_status = "pruned"
                pruned_variant_ids.append(variant.variant_id)
                prune_records.append(
                    PruneRecord(
                        variant_id=variant.variant_id,
                        stage="after_cv",
                        reason=prune_reason,
                        metrics_snapshot=cv_metrics,
                        contract_version=experiment.contract_version,
                    )
                )
                if experiment.holdout is not None:
                    cancelled_units += 1

        if variant_status == "completed" and experiment.holdout is not None:
            holdout = experiment.holdout
            holdout_data = _slice_window(data, holdout.start, holdout.end)
            if holdout_data.empty:
                failure_records.append(
                    FailureRecord(
                        variant_id=variant.variant_id,
                        phase="holdout",
                        label=holdout.label,
                        error_type="ValueError",
                        message="holdout window has no rows in the provided data window.",
                        contract_version=experiment.contract_version,
                    )
                )
                failed_variant_ids.append(variant.variant_id)
                variant_status = "failed"
            else:
                started = float(time_fn())
                try:
                    holdout_result = _run_single_backtest(
                        backtest_fn=backtest_fn,
                        spec=variant,
                        data_slice=holdout_data,
                        node_registry=node_registry,
                    )
                    holdout_metrics = metrics_fn(holdout_result)
                except Exception as exc:
                    ended = float(time_fn())
                    runtime_seconds = max(ended - started, 0.0)
                    total_runtime_seconds += runtime_seconds
                    variant_runtime_seconds += runtime_seconds
                    fold_runtimes[_fold_key(variant.variant_id, "holdout", holdout.label)] = runtime_seconds
                    variant_runtimes[variant.variant_id] = variant_runtime_seconds
                    failure_records.append(
                        FailureRecord(
                            variant_id=variant.variant_id,
                            phase="holdout",
                            label=holdout.label,
                            error_type=type(exc).__name__,
                            message=str(exc),
                            contract_version=experiment.contract_version,
                        )
                    )
                    failed_variant_ids.append(variant.variant_id)
                    variant_status = "failed"
                else:
                    ended = float(time_fn())
                    runtime_seconds = max(ended - started, 0.0)
                    total_runtime_seconds += runtime_seconds
                    variant_runtime_seconds += runtime_seconds
                    completed_units += 1
                    holdout_executed = True
                    holdout_summary = HoldoutRunSummary(
                        variant_id=variant.variant_id,
                        label=holdout.label,
                        window_start=holdout.start,
                        window_end=holdout.end,
                        runtime_seconds=runtime_seconds,
                        metrics=holdout_metrics,
                        bars_observed=int(len(holdout_result.equity_curve)),
                        run_id=_run_id_from_result(holdout_result),
                        final_equity=_final_equity(holdout_result),
                        contract_version=experiment.contract_version,
                    )
                    holdout_summaries.append(holdout_summary)
                    run_results[_fold_key(variant.variant_id, "holdout", holdout.label)] = holdout_result
                    fold_runtimes[_fold_key(variant.variant_id, "holdout", holdout.label)] = runtime_seconds
                    latest_estimated_remaining_seconds = _append_runtime_checkpoint(
                        checkpoints=progress,
                        variant_id=variant.variant_id,
                        phase="holdout",
                        label=holdout.label,
                        total_runtime_seconds=total_runtime_seconds,
                        completed_units=completed_units,
                        cancelled_units=cancelled_units,
                        total_planned_units=total_planned_units,
                        contract_version=experiment.contract_version,
                    )

        variant_runtimes[variant.variant_id] = variant_runtime_seconds

        if variant_status == "completed":
            completed_variant_ids.append(variant.variant_id)

        cv_summaries.append(
            VariantCvSummary(
                variant_id=variant.variant_id,
                status=variant_status,
                fold_labels=tuple(summary.label for summary in variant_fold_summaries),
                cv_metrics=cv_metrics,
                ranking_metrics=cv_metrics,
                runtime_seconds=variant_runtime_seconds,
                estimated_remaining_seconds=latest_estimated_remaining_seconds,
                holdout_executed=holdout_executed,
                contract_version=experiment.contract_version,
            )
        )

    for skipped_variant in max_variants_skipped:
        skip_records.append(
            SkipRecord(
                variant_id=skipped_variant.variant_id,
                reason="max_variants_reached",
                elapsed_seconds=total_runtime_seconds,
                contract_version=experiment.contract_version,
            )
        )
        skipped_variant_ids.append(skipped_variant.variant_id)

    runtime_summary = RuntimeSummary(
        total_runtime_seconds=total_runtime_seconds,
        variant_runtimes=variant_runtimes,
        fold_runtimes=fold_runtimes,
        estimated_remaining_seconds=_estimate_remaining_seconds(
            total_runtime_seconds=total_runtime_seconds,
            completed_units=completed_units,
            cancelled_units=cancelled_units,
            total_planned_units=total_planned_units,
        ),
        progress=tuple(progress),
        contract_version=experiment.contract_version,
    )

    cv_rankings = _build_rankings(
        tuple(cv_summaries),
        contract_version=experiment.contract_version,
    )

    return ExperimentExecutionResult(
        experiment=experiment,
        run_id=run_id,
        fold_summaries=tuple(fold_summaries),
        cv_summaries=tuple(cv_summaries),
        holdout_summaries=tuple(holdout_summaries),
        cv_rankings=cv_rankings,
        runtime_summary=runtime_summary,
        prune_records=tuple(prune_records),
        failure_records=tuple(failure_records),
        skip_records=tuple(skip_records),
        run_results=run_results,
        completed_variant_ids=tuple(completed_variant_ids),
        pruned_variant_ids=tuple(pruned_variant_ids),
        failed_variant_ids=tuple(failed_variant_ids),
        skipped_variant_ids=tuple(skipped_variant_ids),
        contract_version=experiment.contract_version,
    )


__all__ = [
    "ExperimentExecutionResult",
    "FailureRecord",
    "FoldRunSummary",
    "HoldoutRunSummary",
    "PruneRecord",
    "RuntimeCheckpoint",
    "RuntimeSummary",
    "SkipRecord",
    "VariantCvSummary",
    "VariantRanking",
    "aggregate_cv_metrics",
    "run_experiment",
]
