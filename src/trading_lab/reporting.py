from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Literal

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trading_lab.contracts import DEFAULT_CONTRACT_VERSION, BacktestResult
from trading_lab.experiments import serialize_manifest
from trading_lab.runner import ExperimentExecutionResult

_SHEET_ORDER = (
    "run_summary",
    "variant_summary",
    "fold_metrics",
    "holdout_metrics",
    "failures_prunes",
    "config",
)

TargetPhase = Literal["cv", "holdout"]


@dataclass(frozen=True, slots=True)
class DeepDiveTarget:
    variant_id: str
    phase: TargetPhase
    label: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    contract_version: str = DEFAULT_CONTRACT_VERSION


@dataclass(frozen=True, slots=True)
class DeepDiveArtifactSet:
    target: DeepDiveTarget
    target_dir: Path
    equity_plot_path: Path | None
    price_plot_path: Path | None
    trade_log_path: Path | None
    contract_version: str = DEFAULT_CONTRACT_VERSION


def _require_result(result: object) -> ExperimentExecutionResult:
    if not isinstance(result, ExperimentExecutionResult):
        raise TypeError("result must be an ExperimentExecutionResult instance.")
    return result


def _normalize_string_tuple(
    name: str,
    values: Iterable[str] | None,
) -> tuple[str, ...]:
    if values is None:
        return ()
    normalized = tuple(values)
    for value in normalized:
        if not isinstance(value, str):
            raise TypeError(f"{name} must contain strings only.")
        if not value.strip():
            raise ValueError(f"{name} must not contain empty strings.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")
    return normalized


def _result_key(variant_id: str, phase: TargetPhase, label: str) -> str:
    return f"{variant_id}:{phase}:{label}"


def _slug_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value)
    return safe.strip("-") or "item"


def _normalize_metric_value(value: object) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key in sorted(value, key=lambda item: str(item)):
            normalized[str(key)] = _normalize_metric_value(value[key])
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_metric_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_normalize_metric_value(item) for item in value]
        return sorted(normalized_items, key=lambda item: str(item))
    return value


def _metric_cell(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, pd.Timestamp):
        return value
    return serialize_manifest(_normalize_metric_value(value))


def _variant_map(result: ExperimentExecutionResult) -> dict[str, Any]:
    return {variant.variant_id: variant for variant in result.experiment.variants}


def _fold_summary_map(
    result: ExperimentExecutionResult,
) -> dict[tuple[str, str], Any]:
    return {(summary.variant_id, summary.label): summary for summary in result.fold_summaries}


def _holdout_summary_map(result: ExperimentExecutionResult) -> dict[str, Any]:
    return {summary.variant_id: summary for summary in result.holdout_summaries}


def _run_result_for_target(
    result: ExperimentExecutionResult,
    target: DeepDiveTarget,
) -> BacktestResult:
    key = _result_key(target.variant_id, target.phase, target.label)
    if key not in result.run_results:
        raise ValueError(
            f"deep dive target {target.variant_id!r}/{target.phase}/{target.label!r} "
            "has no recorded BacktestResult. Rerun with retained run_results for deep-dive artifacts."
        )
    return result.run_results[key]


def _node_parameters_summary(variant: Any, contract_name: str) -> str:
    contract = getattr(variant, contract_name)
    parameters = contract.manifest.get("parameters", {})
    return serialize_manifest(parameters if isinstance(parameters, dict) else {})


def _variant_base_row(variant: Any, variant_order: int) -> dict[str, object]:
    return {
        "variant_order": variant_order,
        "variant_id": variant.variant_id,
        "strategy_name": variant.backtest_spec.name,
        "symbol": variant.backtest_spec.instrument.symbol,
        "entry_node": variant.entry_contract.name,
        "exit_node": variant.exit_contract.name,
        "risk_node": variant.risk_contract.name,
        "entry_version": variant.entry_contract.spec.version,
        "exit_version": variant.exit_contract.spec.version,
        "risk_version": variant.risk_contract.spec.version,
        "entry_params": _node_parameters_summary(variant, "entry_contract"),
        "exit_params": _node_parameters_summary(variant, "exit_contract"),
        "risk_params": _node_parameters_summary(variant, "risk_contract"),
    }


def _variant_order_map(result: ExperimentExecutionResult) -> dict[str, int]:
    return {
        variant.variant_id: index
        for index, variant in enumerate(result.experiment.variants, start=1)
    }


def _variant_status_map(result: ExperimentExecutionResult) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for summary in result.cv_summaries:
        statuses[summary.variant_id] = summary.status
    for variant_id in result.skipped_variant_ids:
        statuses[variant_id] = "skipped"
    return statuses


def _metric_columns(metric_dicts: list[dict[str, Any]], *, prefix: str) -> tuple[str, ...]:
    keys = sorted({key for metric_dict in metric_dicts for key in metric_dict})
    return tuple(f"{prefix}{key}" for key in keys)


def _run_summary_frame(result: ExperimentExecutionResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_name": result.experiment.name,
                "run_id": result.run_id,
                "contract_version": result.contract_version,
                "variant_count": len(result.experiment.variants),
                "fold_count": len(result.experiment.folds),
                "holdout_enabled": result.experiment.holdout is not None,
                "completed_variants": len(result.completed_variant_ids),
                "pruned_variants": len(result.pruned_variant_ids),
                "failed_variants": len(result.failed_variant_ids),
                "skipped_variants": len(result.skipped_variant_ids),
                "completed_fold_runs": len(result.fold_summaries),
                "completed_holdout_runs": len(result.holdout_summaries),
                "total_runtime_seconds": result.runtime_summary.total_runtime_seconds,
                "estimated_remaining_seconds": result.runtime_summary.estimated_remaining_seconds,
            }
        ]
    )


def _variant_summary_frame(result: ExperimentExecutionResult) -> pd.DataFrame:
    variant_order_map = _variant_order_map(result)
    status_map = _variant_status_map(result)
    cv_summary_map = {summary.variant_id: summary for summary in result.cv_summaries}
    cv_rank_map = {ranking.variant_id: ranking.rank for ranking in result.cv_rankings}
    metric_columns = _metric_columns(
        [summary.cv_metrics for summary in result.cv_summaries],
        prefix="cv_",
    )
    rows: list[dict[str, object]] = []
    for variant in result.experiment.variants:
        summary = cv_summary_map.get(variant.variant_id)
        row = {
            **_variant_base_row(variant, variant_order_map[variant.variant_id]),
            "status": status_map.get(variant.variant_id, "not_run"),
            "cv_rank": cv_rank_map.get(variant.variant_id),
            "cv_runtime_seconds": summary.runtime_seconds if summary is not None else 0.0,
            "cv_estimated_remaining_seconds": (
                summary.estimated_remaining_seconds if summary is not None else None
            ),
            "cv_fold_count": len(summary.fold_labels) if summary is not None else 0,
            "holdout_executed": summary.holdout_executed if summary is not None else False,
        }
        metric_values = summary.cv_metrics if summary is not None else {}
        for column_name in metric_columns:
            metric_name = column_name.removeprefix("cv_")
            row[column_name] = _metric_cell(metric_values.get(metric_name))
        rows.append(row)
    return pd.DataFrame(rows)


def _fold_metrics_frame(result: ExperimentExecutionResult) -> pd.DataFrame:
    variant_order_map = _variant_order_map(result)
    status_map = _variant_status_map(result)
    fold_order_map = {
        label: index
        for index, label in enumerate(
            [fold.fold_label for fold in result.experiment.folds],
            start=1,
        )
    }
    fold_summary_map = {
        (summary.variant_id, summary.label): summary for summary in result.fold_summaries
    }
    metric_columns = _metric_columns(
        [summary.metrics for summary in result.fold_summaries],
        prefix="metric_",
    )
    rows: list[dict[str, object]] = []
    for variant in result.experiment.variants:
        variant_status = status_map.get(variant.variant_id, "not_run")
        for fold in result.experiment.folds:
            summary = fold_summary_map.get((variant.variant_id, fold.fold_label))
            row = {
                **_variant_base_row(variant, variant_order_map[variant.variant_id]),
                "fold_order": fold_order_map[fold.fold_label],
                "fold_label": fold.fold_label,
                "variant_status": variant_status,
                "fold_status": "completed" if summary is not None else "not_run",
                "runtime_seconds": summary.runtime_seconds if summary is not None else 0.0,
                "bars_observed": summary.bars_observed if summary is not None else 0,
                "run_id": summary.run_id if summary is not None else None,
            }
            metric_values = summary.metrics if summary is not None else {}
            for column_name in metric_columns:
                metric_name = column_name.removeprefix("metric_")
                row[column_name] = _metric_cell(metric_values.get(metric_name))
            rows.append(row)
    return pd.DataFrame(rows)


def _holdout_metrics_frame(result: ExperimentExecutionResult) -> pd.DataFrame:
    base_columns = [
        "variant_order",
        "variant_id",
        "strategy_name",
        "symbol",
        "entry_node",
        "exit_node",
        "risk_node",
        "entry_version",
        "exit_version",
        "risk_version",
        "entry_params",
        "exit_params",
        "risk_params",
        "holdout_label",
        "variant_status",
        "holdout_status",
        "runtime_seconds",
        "run_id",
    ]
    holdout_metric_columns = _metric_columns(
        [summary.metrics for summary in result.holdout_summaries],
        prefix="metric_",
    )
    if result.experiment.holdout is None:
        return pd.DataFrame(columns=base_columns + list(holdout_metric_columns))

    variant_order_map = _variant_order_map(result)
    status_map = _variant_status_map(result)
    holdout_summary_map = {
        summary.variant_id: summary for summary in result.holdout_summaries
    }
    rows: list[dict[str, object]] = []
    for variant in result.experiment.variants:
        summary = holdout_summary_map.get(variant.variant_id)
        row = {
            **_variant_base_row(variant, variant_order_map[variant.variant_id]),
            "holdout_label": result.experiment.holdout.label,
            "variant_status": status_map.get(variant.variant_id, "not_run"),
            "holdout_status": "completed" if summary is not None else "not_run",
            "runtime_seconds": summary.runtime_seconds if summary is not None else 0.0,
            "run_id": summary.run_id if summary is not None else None,
        }
        metric_values = summary.metrics if summary is not None else {}
        for column_name in holdout_metric_columns:
            metric_name = column_name.removeprefix("metric_")
            row[column_name] = _metric_cell(metric_values.get(metric_name))
        rows.append(row)
    return pd.DataFrame(rows)


def _failures_prunes_frame(result: ExperimentExecutionResult) -> pd.DataFrame:
    variant_order_map = _variant_order_map(result)
    status_map = _variant_status_map(result)
    type_order = {"prune": 0, "failure": 1, "skip": 2}
    rows: list[dict[str, object]] = []

    for record in result.prune_records:
        rows.append(
            {
                "variant_order": variant_order_map.get(record.variant_id, 0),
                "variant_id": record.variant_id,
                "variant_status": status_map.get(record.variant_id, "not_run"),
                "record_type": "prune",
                "phase_or_stage": record.stage,
                "label": record.fold_label,
                "reason": record.reason,
                "error_type": None,
                "message": None,
                "elapsed_seconds": None,
                "details": _metric_cell(record.metrics_snapshot),
            }
        )
    for record in result.failure_records:
        rows.append(
            {
                "variant_order": variant_order_map.get(record.variant_id, 0),
                "variant_id": record.variant_id,
                "variant_status": status_map.get(record.variant_id, "failed"),
                "record_type": "failure",
                "phase_or_stage": record.phase,
                "label": record.label,
                "reason": None,
                "error_type": record.error_type,
                "message": record.message,
                "elapsed_seconds": None,
                "details": None,
            }
        )
    for record in result.skip_records:
        rows.append(
            {
                "variant_order": variant_order_map.get(record.variant_id, 0),
                "variant_id": record.variant_id,
                "variant_status": status_map.get(record.variant_id, "skipped"),
                "record_type": "skip",
                "phase_or_stage": None,
                "label": None,
                "reason": record.reason,
                "error_type": None,
                "message": None,
                "elapsed_seconds": record.elapsed_seconds,
                "details": None,
            }
        )

    rows.sort(
        key=lambda row: (
            int(row["variant_order"]),
            type_order[str(row["record_type"])],
            "" if row["label"] is None else str(row["label"]),
            "" if row["phase_or_stage"] is None else str(row["phase_or_stage"]),
        )
    )
    return pd.DataFrame(rows)


def _config_frame(result: ExperimentExecutionResult) -> pd.DataFrame:
    experiment = result.experiment
    rows: list[dict[str, object]] = []

    def add(section: str, key: str, value: object) -> None:
        rows.append({"section": section, "key": key, "value": _metric_cell(value)})

    add("experiment", "name", experiment.name)
    add("experiment", "run_id", result.run_id)
    add("experiment", "contract_version", experiment.contract_version)
    add("experiment", "variant_ids", [variant.variant_id for variant in experiment.variants])
    add("experiment", "fold_labels", [fold.fold_label for fold in experiment.folds])
    add(
        "experiment",
        "holdout_label",
        experiment.holdout.label if experiment.holdout is not None else None,
    )

    add("search", "mode", experiment.search.mode)
    add("search", "max_variants", experiment.search.max_variants)
    add("search", "max_runtime_seconds", experiment.search.max_runtime_seconds)
    add("search", "max_parallel_variants", experiment.search.max_parallel_variants)
    add("search", "random_seed", experiment.search.random_seed)

    add("pruning", "stop_on_zero_equity", experiment.pruning.stop_on_zero_equity)
    add(
        "pruning",
        "stop_on_invalid_numeric_state",
        experiment.pruning.stop_on_invalid_numeric_state,
    )
    add("pruning", "min_trades", experiment.pruning.min_trades)
    add("pruning", "max_drawdown_threshold", experiment.pruning.max_drawdown_threshold)
    add("pruning", "early_metric_thresholds", experiment.pruning.early_metric_thresholds)
    add("pruning", "early_min_trades", experiment.pruning.early_min_trades)
    add("pruning", "early_min_bars", experiment.pruning.early_min_bars)

    add("output", "output_dir", experiment.outputs.output_dir)
    add("output", "export_summary_excel", experiment.outputs.export_summary_excel)
    add("output", "summary_excel_name", experiment.outputs.summary_excel_name)
    add("output", "write_run_manifests", experiment.outputs.write_run_manifests)

    if experiment.deep_dive is not None:
        add("deep_dive", "selected_variant_ids", experiment.deep_dive.selected_variant_ids)
        add("deep_dive", "selected_folds", experiment.deep_dive.selected_folds)
        add("deep_dive", "include_holdout", experiment.deep_dive.include_holdout)
        add("deep_dive", "generate_trade_log", experiment.deep_dive.generate_trade_log)
        add("deep_dive", "generate_equity_plot", experiment.deep_dive.generate_equity_plot)
        add("deep_dive", "generate_price_plot", experiment.deep_dive.generate_price_plot)

    return pd.DataFrame(rows)


def select_deep_dive_targets(
    result: ExperimentExecutionResult,
    *,
    selected_variant_ids: Iterable[str] | None = None,
    selected_folds: Iterable[str] | None = None,
    include_holdout: bool | None = None,
) -> tuple[DeepDiveTarget, ...]:
    resolved = _require_result(result)
    config = resolved.experiment.deep_dive

    variant_ids = _normalize_string_tuple(
        "selected_variant_ids",
        config.selected_variant_ids if selected_variant_ids is None and config is not None else selected_variant_ids,
    )
    if not variant_ids:
        raise ValueError("selected_variant_ids must contain at least one variant_id.")

    fold_labels = _normalize_string_tuple(
        "selected_folds",
        config.selected_folds if selected_folds is None and config is not None else selected_folds,
    )
    holdout_requested = (
        config.include_holdout if include_holdout is None and config is not None else bool(include_holdout)
    )

    variant_map = _variant_map(resolved)
    unknown_variants = [variant_id for variant_id in variant_ids if variant_id not in variant_map]
    if unknown_variants:
        raise ValueError(f"unknown variant_ids requested for deep dive: {unknown_variants!r}.")

    experiment_fold_labels = tuple(fold.fold_label for fold in resolved.experiment.folds)
    unknown_folds = [fold_label for fold_label in fold_labels if fold_label not in experiment_fold_labels]
    if unknown_folds:
        raise ValueError(f"unknown fold labels requested for deep dive: {unknown_folds!r}.")

    if holdout_requested and resolved.experiment.holdout is None:
        raise ValueError("holdout deep dive requested, but the experiment has no holdout.")

    fold_summary_map = _fold_summary_map(resolved)
    holdout_summary_map = _holdout_summary_map(resolved)
    targets: list[DeepDiveTarget] = []

    for variant in resolved.experiment.variants:
        if variant.variant_id not in variant_ids:
            continue

        if fold_labels:
            for fold in resolved.experiment.folds:
                if fold.fold_label not in fold_labels:
                    continue
                summary = fold_summary_map.get((variant.variant_id, fold.fold_label))
                if summary is None:
                    raise ValueError(
                        f"deep dive requested for {variant.variant_id!r}/{fold.fold_label!r}, "
                        "but that fold did not complete."
                    )
                targets.append(
                    DeepDiveTarget(
                        variant_id=summary.variant_id,
                        phase="cv",
                        label=summary.label,
                        window_start=pd.Timestamp(summary.window_start),
                        window_end=pd.Timestamp(summary.window_end),
                        contract_version=summary.contract_version,
                    )
                )
        elif not holdout_requested:
            default_summary = next(
                (
                    fold_summary_map[(variant.variant_id, fold.fold_label)]
                    for fold in resolved.experiment.folds
                    if (variant.variant_id, fold.fold_label) in fold_summary_map
                ),
                None,
            )
            if default_summary is not None:
                targets.append(
                    DeepDiveTarget(
                        variant_id=default_summary.variant_id,
                        phase="cv",
                        label=default_summary.label,
                        window_start=pd.Timestamp(default_summary.window_start),
                        window_end=pd.Timestamp(default_summary.window_end),
                        contract_version=default_summary.contract_version,
                    )
                )
            else:
                holdout_summary = holdout_summary_map.get(variant.variant_id)
                if holdout_summary is None:
                    raise ValueError(
                        f"variant {variant.variant_id!r} has no completed fold or holdout run "
                        "available for default deep dive selection."
                    )
                targets.append(
                    DeepDiveTarget(
                        variant_id=holdout_summary.variant_id,
                        phase="holdout",
                        label=holdout_summary.label,
                        window_start=pd.Timestamp(holdout_summary.window_start),
                        window_end=pd.Timestamp(holdout_summary.window_end),
                        contract_version=holdout_summary.contract_version,
                    )
                )

        if holdout_requested:
            holdout_summary = holdout_summary_map.get(variant.variant_id)
            if holdout_summary is None:
                raise ValueError(
                    f"deep dive requested for holdout on variant {variant.variant_id!r}, "
                    "but that holdout run did not complete."
                )
            targets.append(
                DeepDiveTarget(
                    variant_id=holdout_summary.variant_id,
                    phase="holdout",
                    label=holdout_summary.label,
                    window_start=pd.Timestamp(holdout_summary.window_start),
                    window_end=pd.Timestamp(holdout_summary.window_end),
                    contract_version=holdout_summary.contract_version,
                )
            )

    return tuple(targets)


def build_trade_log_frame(
    result: ExperimentExecutionResult,
    target: DeepDiveTarget,
) -> pd.DataFrame:
    resolved = _require_result(result)
    if not isinstance(target, DeepDiveTarget):
        raise TypeError("target must be a DeepDiveTarget instance.")

    trade_columns = [
        "variant_id",
        "target_phase",
        "target_label",
        "trade_id",
        "symbol",
        "side",
        "entry_ts",
        "exit_ts",
        "qty",
        "entry_price",
        "exit_price",
        "gross_pnl",
        "net_pnl",
        "bars_held",
        "exit_reason",
        "fees",
        "equity_after_trade",
    ]
    backtest_result = _run_result_for_target(resolved, target)
    trade_ledger = backtest_result.trade_ledger.copy()
    if trade_ledger.empty:
        return pd.DataFrame(columns=trade_columns)

    required_columns = (
        "trade_id",
        "symbol",
        "side",
        "entry_ts",
        "exit_ts",
        "qty",
        "entry_price",
        "exit_price",
        "gross_pnl",
        "net_pnl",
        "bars_held",
        "exit_reason",
    )
    missing_columns = [column for column in required_columns if column not in trade_ledger.columns]
    if missing_columns:
        raise ValueError(
            f"trade_ledger is missing required deep-dive columns: {missing_columns!r}."
        )

    ordered = trade_ledger.sort_values(
        by=["exit_ts", "entry_ts", "trade_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if "fees" not in ordered.columns:
        ordered["fees"] = 0.0

    equity_after_trade = pd.Series([pd.NA] * len(ordered), dtype="object")
    equity_curve = backtest_result.equity_curve.copy()
    if {"ts", "equity"}.issubset(equity_curve.columns) and not equity_curve.empty:
        equity_reference = equity_curve[["ts", "equity"]].sort_values("ts").reset_index(drop=True)
        merged = pd.merge_asof(
            ordered[["exit_ts"]].sort_values("exit_ts").reset_index(),
            equity_reference,
            left_on="exit_ts",
            right_on="ts",
            direction="backward",
        )
        equity_after_trade = (
            merged.sort_values("index")["equity"].reset_index(drop=True).astype("object")
        )

    return pd.DataFrame(
        {
            "variant_id": target.variant_id,
            "target_phase": target.phase,
            "target_label": target.label,
            "trade_id": ordered["trade_id"],
            "symbol": ordered["symbol"],
            "side": ordered["side"],
            "entry_ts": ordered["entry_ts"],
            "exit_ts": ordered["exit_ts"],
            "qty": ordered["qty"],
            "entry_price": ordered["entry_price"],
            "exit_price": ordered["exit_price"],
            "gross_pnl": ordered["gross_pnl"],
            "net_pnl": ordered["net_pnl"],
            "bars_held": ordered["bars_held"],
            "exit_reason": ordered["exit_reason"],
            "fees": ordered["fees"],
            "equity_after_trade": equity_after_trade,
        }
    )


def build_price_plot_frame(
    result: ExperimentExecutionResult,
    data: pd.DataFrame,
    target: DeepDiveTarget,
) -> pd.DataFrame:
    resolved = _require_result(result)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(target, DeepDiveTarget):
        raise TypeError("target must be a DeepDiveTarget instance.")
    if "ts" not in data.columns:
        raise ValueError("data must contain a 'ts' column for price plotting.")

    variant = _variant_map(resolved)[target.variant_id]
    symbol = variant.backtest_spec.instrument.symbol
    mask = (data["ts"] >= target.window_start) & (data["ts"] < target.window_end)
    if "symbol" in data.columns:
        mask &= data["symbol"] == symbol
    scoped = data.loc[mask].sort_values("ts", kind="mergesort").reset_index(drop=True)
    return scoped


def _build_equity_plot_frame(
    result: ExperimentExecutionResult,
    target: DeepDiveTarget,
) -> pd.DataFrame:
    equity_curve = _run_result_for_target(result, target).equity_curve.copy()
    if equity_curve.empty:
        return equity_curve
    if "ts" not in equity_curve.columns:
        raise ValueError("equity_curve must contain a 'ts' column for deep-dive plotting.")
    return equity_curve.sort_values("ts", kind="mergesort").reset_index(drop=True)


def _build_trade_marker_frame(
    result: ExperimentExecutionResult,
    target: DeepDiveTarget,
) -> pd.DataFrame:
    trade_log = build_trade_log_frame(result, target)
    marker_columns = ["ts", "price", "side", "event_type", "trade_id"]
    if trade_log.empty:
        return pd.DataFrame(columns=marker_columns)

    entries = pd.DataFrame(
        {
            "ts": trade_log["entry_ts"],
            "price": trade_log["entry_price"],
            "side": trade_log["side"],
            "event_type": "entry",
            "trade_id": trade_log["trade_id"],
        }
    )
    exits = pd.DataFrame(
        {
            "ts": trade_log["exit_ts"],
            "price": trade_log["exit_price"],
            "side": trade_log["side"],
            "event_type": "exit",
            "trade_id": trade_log["trade_id"],
        }
    )
    return (
        pd.concat([entries, exits], ignore_index=True)
        .sort_values(["ts", "event_type", "trade_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _write_equity_plot(
    result: ExperimentExecutionResult,
    target: DeepDiveTarget,
    target_dir: Path,
) -> Path:
    equity_frame = _build_equity_plot_frame(result, target)
    plot_path = target_dir / "equity_curve.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    if equity_frame.empty or "equity" not in equity_frame.columns:
        ax.text(0.5, 0.5, "No equity data", ha="center", va="center", transform=ax.transAxes)
    else:
        color = "#d95f02" if target.phase == "holdout" else "#1f77b4"
        ax.plot(equity_frame["ts"], equity_frame["equity"], color=color, linewidth=2.0)
        ax.axvline(target.window_start, color="#666666", linestyle="--", linewidth=1.0)
        ax.axvline(target.window_end, color="#666666", linestyle="--", linewidth=1.0)
        if target.phase == "holdout":
            ax.axvspan(target.window_start, target.window_end, color="#fdd0a2", alpha=0.25)
        ax.set_ylabel("Equity")
    ax.set_title(f"{target.variant_id} {target.phase}:{target.label} equity")
    ax.set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def _write_price_plot(
    result: ExperimentExecutionResult,
    data: pd.DataFrame,
    target: DeepDiveTarget,
    target_dir: Path,
) -> Path:
    price_frame = build_price_plot_frame(result, data, target)
    markers = _build_trade_marker_frame(result, target)
    plot_path = target_dir / "price_entries_exits.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    if price_frame.empty:
        ax.text(0.5, 0.5, "No price data", ha="center", va="center", transform=ax.transAxes)
    else:
        price_column = "close" if "close" in price_frame.columns else "open"
        if price_column not in price_frame.columns:
            raise ValueError(
                "price plotting requires either a 'close' or 'open' column in the data."
            )
        ax.plot(price_frame["ts"], price_frame[price_column], color="#444444", linewidth=1.5)
        ax.axvline(target.window_start, color="#666666", linestyle="--", linewidth=1.0)
        ax.axvline(target.window_end, color="#666666", linestyle="--", linewidth=1.0)
        if target.phase == "holdout":
            ax.axvspan(target.window_start, target.window_end, color="#fdd0a2", alpha=0.2)
        style_map = {
            ("entry", "long"): {"marker": "^", "color": "#2ca02c", "label": "Long entry"},
            ("entry", "short"): {"marker": "v", "color": "#d62728", "label": "Short entry"},
            ("exit", "long"): {"marker": "x", "color": "#111111", "label": "Long exit"},
            ("exit", "short"): {"marker": "x", "color": "#ff7f0e", "label": "Short exit"},
        }
        for (event_type, side), frame in markers.groupby(["event_type", "side"], sort=False):
            style = style_map[(str(event_type), str(side))]
            ax.scatter(
                frame["ts"],
                frame["price"],
                marker=style["marker"],
                color=style["color"],
                s=45,
                label=style["label"],
                zorder=3,
            )
        if not markers.empty:
            handles, labels = ax.get_legend_handles_labels()
            deduped: dict[str, Any] = {}
            for handle, label in zip(handles, labels):
                deduped.setdefault(label, handle)
            ax.legend(deduped.values(), deduped.keys(), loc="best")
        ax.set_ylabel(price_column.capitalize())
    ax.set_title(f"{target.variant_id} {target.phase}:{target.label} price")
    ax.set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def export_deep_dive_artifacts(
    result: ExperimentExecutionResult,
    data: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    selected_variant_ids: Iterable[str] | None = None,
    selected_folds: Iterable[str] | None = None,
    include_holdout: bool | None = None,
    generate_trade_log: bool | None = None,
    generate_equity_plot: bool | None = None,
    generate_price_plot: bool | None = None,
) -> tuple[DeepDiveArtifactSet, ...]:
    resolved = _require_result(result)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    config = resolved.experiment.deep_dive
    trade_log_enabled = (
        config.generate_trade_log
        if generate_trade_log is None and config is not None
        else True if generate_trade_log is None else bool(generate_trade_log)
    )
    equity_plot_enabled = (
        config.generate_equity_plot
        if generate_equity_plot is None and config is not None
        else True if generate_equity_plot is None else bool(generate_equity_plot)
    )
    price_plot_enabled = (
        config.generate_price_plot
        if generate_price_plot is None and config is not None
        else True if generate_price_plot is None else bool(generate_price_plot)
    )

    targets = select_deep_dive_targets(
        resolved,
        selected_variant_ids=selected_variant_ids,
        selected_folds=selected_folds,
        include_holdout=include_holdout,
    )
    root_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(resolved.experiment.outputs.output_dir) / "deep_dive"
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[DeepDiveArtifactSet] = []
    for target in targets:
        target_dir = (
            root_dir
            / _slug_component(target.variant_id)
            / f"{target.phase}_{_slug_component(target.label)}"
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        equity_plot_path = (
            _write_equity_plot(resolved, target, target_dir) if equity_plot_enabled else None
        )
        price_plot_path = (
            _write_price_plot(resolved, data, target, target_dir) if price_plot_enabled else None
        )
        trade_log_path: Path | None = None
        if trade_log_enabled:
            trade_log_path = target_dir / "trade_log.csv"
            build_trade_log_frame(resolved, target).to_csv(trade_log_path, index=False)

        artifacts.append(
            DeepDiveArtifactSet(
                target=target,
                target_dir=target_dir,
                equity_plot_path=equity_plot_path,
                price_plot_path=price_plot_path,
                trade_log_path=trade_log_path,
                contract_version=target.contract_version,
            )
        )

    return tuple(artifacts)


def build_summary_frames(result: ExperimentExecutionResult) -> dict[str, pd.DataFrame]:
    resolved = _require_result(result)
    frames = {
        "run_summary": _run_summary_frame(resolved),
        "variant_summary": _variant_summary_frame(resolved),
        "fold_metrics": _fold_metrics_frame(resolved),
        "holdout_metrics": _holdout_metrics_frame(resolved),
        "failures_prunes": _failures_prunes_frame(resolved),
        "config": _config_frame(resolved),
    }
    return {sheet: frames[sheet] for sheet in _SHEET_ORDER}


def export_summary_workbook(
    result: ExperimentExecutionResult,
    path: str | Path,
) -> Path:
    frames = build_summary_frames(result)
    target_path = Path(path)
    if target_path.parent != Path():
        target_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
        for sheet_name in _SHEET_ORDER:
            frames[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
    return target_path


__all__ = [
    "DeepDiveArtifactSet",
    "DeepDiveTarget",
    "build_price_plot_frame",
    "build_summary_frames",
    "build_trade_log_frame",
    "export_summary_workbook",
    "export_deep_dive_artifacts",
    "select_deep_dive_targets",
]
