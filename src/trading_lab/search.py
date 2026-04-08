from __future__ import annotations

import importlib
import json
import math
import time
import types
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Union, get_args, get_origin, get_type_hints

import pandas as pd

from trading_lab.contracts import BacktestSpec, DEFAULT_CONTRACT_VERSION
from trading_lab.experiments import (
    DeepDiveConfig,
    ExperimentSpec,
    FoldSpec,
    HoldoutSpec,
    OutputConfig,
    PruningConfig,
    SearchConfig,
    VariantSpec,
    generate_variants,
    serialize_manifest,
)
from trading_lab.reporting import DeepDiveArtifactSet, export_deep_dive_artifacts, export_summary_workbook
from trading_lab.runner import ExperimentExecutionResult, run_experiment

SearchMode = Literal["grid", "random", "optuna"]
ObjectiveMode = Literal["single_metric", "composite"]
SearchDirection = Literal["maximize", "minimize"]
VariantStatus = Literal["completed", "pruned", "failed", "skipped"]

RunnerFn = Callable[..., ExperimentExecutionResult]
TimeFn = Callable[[], float]
VariantFactory = Callable[[Any], VariantSpec]


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}.")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_number(
    name: str,
    value: object,
    *,
    allow_none: bool = False,
) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be a finite number, got None.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, got {type(value).__name__}.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value}.")
    return numeric


def _require_contract_version(name: str, value: str) -> None:
    _require_non_empty(name, value)


def _copy_mapping(value: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if value is None else dict(value)


def _normalize_constraints(
    constraints: Iterable["MetricConstraint"] | None,
) -> tuple["MetricConstraint", ...]:
    if constraints is None:
        return ()
    normalized = tuple(constraints)
    if any(not isinstance(item, MetricConstraint) for item in normalized):
        raise TypeError("constraints must contain MetricConstraint objects only.")
    return normalized


def _normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    if weights is None:
        return {}
    if not isinstance(weights, dict):
        raise TypeError("composite_weights must be a dict.")
    normalized: dict[str, float] = {}
    for metric_name, weight in sorted(weights.items()):
        _require_non_empty("composite weight metric", metric_name)
        numeric = _require_number(f"composite_weights[{metric_name!r}]", weight)
        assert numeric is not None
        normalized[metric_name] = numeric
    return normalized


def _normalize_runner_kwargs(
    runner_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    if runner_kwargs is None:
        return {}
    if not isinstance(runner_kwargs, dict):
        raise TypeError("runner_kwargs must be a dict.")
    return dict(runner_kwargs)


def _metric_number(metrics: dict[str, Any], metric_name: str) -> float | None:
    if metric_name not in metrics:
        return None
    value = metrics[metric_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _objective_score(value: float, direction: SearchDirection) -> float:
    return value if direction == "maximize" else -value


def _worst_objective_value(direction: SearchDirection) -> float:
    return -1.0e308 if direction == "maximize" else 1.0e308


@dataclass(frozen=True, slots=True)
class MetricConstraint:
    metric_name: str
    minimum: float | None = None
    maximum: float | None = None
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("metric_name", self.metric_name)
        minimum = _require_number("minimum", self.minimum, allow_none=True)
        maximum = _require_number("maximum", self.maximum, allow_none=True)
        _require_contract_version("contract_version", self.contract_version)
        if minimum is None and maximum is None:
            raise ValueError("MetricConstraint requires at least one bound.")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("MetricConstraint minimum cannot exceed maximum.")


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    mode: ObjectiveMode
    direction: SearchDirection = "maximize"
    metric_name: str | None = None
    composite_weights: dict[str, float] = field(default_factory=dict)
    constraints: tuple[MetricConstraint, ...] = ()
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.mode not in {"single_metric", "composite"}:
            raise ValueError(f"unsupported objective mode: {self.mode!r}.")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError(f"unsupported objective direction: {self.direction!r}.")
        if self.metric_name is not None:
            _require_non_empty("metric_name", self.metric_name)
        normalized_weights = _normalize_weights(self.composite_weights)
        normalized_constraints = _normalize_constraints(self.constraints)
        _require_contract_version("contract_version", self.contract_version)

        if self.mode == "single_metric":
            if self.metric_name is None:
                raise ValueError("single_metric objectives require metric_name.")
            if normalized_weights:
                raise ValueError(
                    "single_metric objectives must not define composite_weights."
                )
        else:
            if not normalized_weights:
                raise ValueError("composite objectives require composite_weights.")
            if self.metric_name is not None:
                raise ValueError("composite objectives must not define metric_name.")

        object.__setattr__(self, "composite_weights", normalized_weights)
        object.__setattr__(self, "constraints", normalized_constraints)

    @classmethod
    def single_metric(
        cls,
        metric_name: str,
        *,
        direction: SearchDirection = "maximize",
        constraints: Iterable[MetricConstraint] | None = None,
        contract_version: str = DEFAULT_CONTRACT_VERSION,
    ) -> "ObjectiveConfig":
        return cls(
            mode="single_metric",
            direction=direction,
            metric_name=metric_name,
            constraints=tuple(constraints or ()),
            contract_version=contract_version,
        )

    @classmethod
    def composite(
        cls,
        composite_weights: dict[str, float],
        *,
        direction: SearchDirection = "maximize",
        constraints: Iterable[MetricConstraint] | None = None,
        contract_version: str = DEFAULT_CONTRACT_VERSION,
    ) -> "ObjectiveConfig":
        return cls(
            mode="composite",
            direction=direction,
            composite_weights=composite_weights,
            constraints=tuple(constraints or ()),
            contract_version=contract_version,
        )


@dataclass(frozen=True, slots=True)
class VariantEvaluation:
    variant_id: str
    status: VariantStatus
    cv_metrics: dict[str, Any]
    holdout_metrics: dict[str, Any] | None = None
    objective_value: float | None = None
    objective_score: float | None = None
    feasible: bool = False
    constraint_violations: tuple[str, ...] = ()
    runtime_seconds: float = 0.0
    rank: int | None = None
    source_label: str | None = None
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("variant_id", self.variant_id)
        if self.status not in {"completed", "pruned", "failed", "skipped"}:
            raise ValueError(f"unsupported variant status: {self.status!r}.")
        if not isinstance(self.cv_metrics, dict):
            raise TypeError("cv_metrics must be a dict.")
        if self.holdout_metrics is not None and not isinstance(self.holdout_metrics, dict):
            raise TypeError("holdout_metrics must be a dict when provided.")
        if self.objective_value is not None:
            _require_number("objective_value", self.objective_value)
        if self.objective_score is not None:
            _require_number("objective_score", self.objective_score)
        if any(not isinstance(item, str) or not item.strip() for item in self.constraint_violations):
            raise ValueError("constraint_violations must contain non-empty strings only.")
        runtime_seconds = _require_number("runtime_seconds", self.runtime_seconds)
        assert runtime_seconds is not None
        if runtime_seconds < 0.0:
            raise ValueError("runtime_seconds must be non-negative.")
        if self.rank is not None and (not isinstance(self.rank, int) or self.rank < 1):
            raise ValueError("rank must be a positive int when provided.")
        if self.source_label is not None:
            _require_non_empty("source_label", self.source_label)
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(self, "cv_metrics", dict(self.cv_metrics))
        object.__setattr__(self, "holdout_metrics", _copy_mapping(self.holdout_metrics))
        object.__setattr__(self, "constraint_violations", tuple(self.constraint_violations))


@dataclass(frozen=True, slots=True)
class OptunaTrialRecord:
    trial_number: int
    trial_name: str
    variant_id: str
    experiment_name: str
    objective_value: float | None
    objective_score: float | None
    feasible: bool
    status: VariantStatus
    runtime_seconds: float
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.trial_number, int) or self.trial_number < 0:
            raise ValueError("trial_number must be a non-negative int.")
        _require_non_empty("trial_name", self.trial_name)
        _require_non_empty("variant_id", self.variant_id)
        _require_non_empty("experiment_name", self.experiment_name)
        if self.objective_value is not None:
            _require_number("objective_value", self.objective_value)
        if self.objective_score is not None:
            _require_number("objective_score", self.objective_score)
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be a bool.")
        if self.status not in {"completed", "pruned", "failed", "skipped"}:
            raise ValueError(f"unsupported variant status: {self.status!r}.")
        runtime_seconds = _require_number("runtime_seconds", self.runtime_seconds)
        assert runtime_seconds is not None
        if runtime_seconds < 0.0:
            raise ValueError("runtime_seconds must be non-negative.")
        _require_contract_version("contract_version", self.contract_version)


@dataclass(frozen=True, slots=True)
class SearchExecutionResult:
    mode: SearchMode
    objective: ObjectiveConfig
    evaluations: tuple[VariantEvaluation, ...]
    best_variant_id: str | None = None
    best_objective_value: float | None = None
    experiment_results: tuple[ExperimentExecutionResult, ...] = ()
    trial_records: tuple[OptunaTrialRecord, ...] = ()
    study: object | None = None
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.mode not in {"grid", "random", "optuna"}:
            raise ValueError(f"unsupported search execution mode: {self.mode!r}.")
        if not isinstance(self.objective, ObjectiveConfig):
            raise TypeError("objective must be an ObjectiveConfig instance.")
        if any(not isinstance(item, VariantEvaluation) for item in self.evaluations):
            raise TypeError("evaluations must contain VariantEvaluation objects only.")
        if self.best_variant_id is not None:
            _require_non_empty("best_variant_id", self.best_variant_id)
        if self.best_objective_value is not None:
            _require_number("best_objective_value", self.best_objective_value)
        if any(
            not isinstance(result, ExperimentExecutionResult)
            for result in self.experiment_results
        ):
            raise TypeError(
                "experiment_results must contain ExperimentExecutionResult objects only."
            )
        if any(not isinstance(item, OptunaTrialRecord) for item in self.trial_records):
            raise TypeError("trial_records must contain OptunaTrialRecord objects only.")
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(self, "evaluations", tuple(self.evaluations))
        object.__setattr__(self, "experiment_results", tuple(self.experiment_results))
        object.__setattr__(self, "trial_records", tuple(self.trial_records))


@dataclass(frozen=True, slots=True)
class SearchRunConfig:
    experiment: ExperimentSpec
    objective: ObjectiveConfig
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment", _coerce_experiment_spec(self.experiment))
        if not isinstance(self.experiment, _current_experiment_spec_class()):
            raise TypeError("experiment must be an ExperimentSpec instance.")
        if not isinstance(self.objective, ObjectiveConfig):
            raise TypeError("objective must be an ObjectiveConfig instance.")
        _require_contract_version("contract_version", self.contract_version)
        if self.experiment.contract_version != self.contract_version:
            raise ValueError(
                "experiment.contract_version must match SearchRunConfig.contract_version."
            )
        if self.objective.contract_version != self.contract_version:
            raise ValueError(
                "objective.contract_version must match SearchRunConfig.contract_version."
            )


@dataclass(frozen=True, slots=True)
class SearchEntrypointResult:
    run_config: SearchRunConfig
    search_result: SearchExecutionResult
    summary_workbook_path: Path
    deep_dive_artifacts: tuple[DeepDiveArtifactSet, ...] = ()
    runtime_summary: str = ""
    stopping_reason: str = ""
    reproducibility_manifest: dict[str, Any] = field(default_factory=dict)
    manifest_path: Path | None = None
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.run_config, SearchRunConfig):
            raise TypeError("run_config must be a SearchRunConfig instance.")
        if not isinstance(self.search_result, SearchExecutionResult):
            raise TypeError("search_result must be a SearchExecutionResult instance.")
        if not isinstance(self.summary_workbook_path, Path):
            raise TypeError("summary_workbook_path must be a pathlib.Path.")
        if any(not isinstance(item, DeepDiveArtifactSet) for item in self.deep_dive_artifacts):
            raise TypeError(
                "deep_dive_artifacts must contain DeepDiveArtifactSet objects only."
            )
        if not isinstance(self.runtime_summary, str):
            raise TypeError("runtime_summary must be a string.")
        if not isinstance(self.stopping_reason, str):
            raise TypeError("stopping_reason must be a string.")
        if not isinstance(self.reproducibility_manifest, dict):
            raise TypeError("reproducibility_manifest must be a dict.")
        if self.manifest_path is not None and not isinstance(self.manifest_path, Path):
            raise TypeError("manifest_path must be a pathlib.Path when provided.")
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(self, "deep_dive_artifacts", tuple(self.deep_dive_artifacts))
        object.__setattr__(
            self,
            "reproducibility_manifest",
            dict(self.reproducibility_manifest),
        )


def serialize_search_run_config(config: SearchRunConfig) -> str:
    if not isinstance(config, SearchRunConfig):
        raise TypeError("config must be a SearchRunConfig instance.")
    return serialize_manifest(config)


def _load_serialized_payload(
    config: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if isinstance(config, Path):
        payload_text = config.read_text(encoding="utf-8")
    elif isinstance(config, str):
        stripped = config.strip()
        if stripped.startswith("{"):
            payload_text = stripped
        else:
            path = Path(config)
            if not path.exists():
                raise FileNotFoundError(f"serialized config path not found: {config!r}.")
            payload_text = path.read_text(encoding="utf-8")
    else:
        raise TypeError(
            "serialized config must be a dict, JSON string, or filesystem path."
        )

    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise TypeError("serialized config payload must decode to a JSON object.")
    return payload


def _current_experiment_spec_class() -> type[Any]:
    return importlib.import_module("trading_lab.experiments").ExperimentSpec


def _coerce_experiment_spec(value: object) -> object:
    current_experiment_spec = _current_experiment_spec_class()
    if isinstance(value, current_experiment_spec):
        return value
    if isinstance(value, dict):
        return _deserialize_dataclass(value, current_experiment_spec)
    if hasattr(value, "__dataclass_fields__") and type(value).__name__ == "ExperimentSpec":
        payload = json.loads(serialize_manifest(value))
        return _deserialize_dataclass(payload, current_experiment_spec)
    return value


def _deserialize_value(value: Any, annotation: Any) -> Any:
    if annotation is Any:
        return value
    if value is None:
        return None

    origin = get_origin(annotation)
    if origin is Literal:
        return value
    if origin in {Union, types.UnionType}:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _deserialize_value(value, args[0])
        for arg in args:
            try:
                return _deserialize_value(value, arg)
            except Exception:
                continue
        return value
    if origin is tuple:
        args = get_args(annotation)
        item_type = Any if not args else args[0]
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_deserialize_value(item, item_type) for item in value)
        return tuple(
            _deserialize_value(item, args[index] if index < len(args) else Any)
            for index, item in enumerate(value)
        )
    if origin is list:
        item_type = get_args(annotation)[0] if get_args(annotation) else Any
        return [_deserialize_value(item, item_type) for item in value]
    if origin is dict:
        args = get_args(annotation)
        key_type = args[0] if len(args) > 0 else Any
        value_type = args[1] if len(args) > 1 else Any
        return {
            _deserialize_value(key, key_type): _deserialize_value(item, value_type)
            for key, item in value.items()
        }
    if annotation is datetime:
        return datetime.fromisoformat(value) if isinstance(value, str) else value
    if annotation is Path:
        return value if isinstance(value, Path) else Path(str(value))
    if isinstance(annotation, type) and hasattr(annotation, "__dataclass_fields__"):
        return _deserialize_dataclass(value, annotation)
    return value


def _deserialize_dataclass(mapping: dict[str, Any], cls: type[Any]) -> Any:
    if not isinstance(mapping, dict):
        raise TypeError(f"expected dict payload for {cls.__name__}.")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field_info in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        if field_info.name not in mapping:
            continue
        annotation = hints.get(field_info.name, Any)
        kwargs[field_info.name] = _deserialize_value(mapping[field_info.name], annotation)
    return cls(**kwargs)


def load_search_run_config(
    config: SearchRunConfig | ExperimentSpec | dict[str, Any] | str | Path,
    *,
    objective: ObjectiveConfig | None = None,
) -> SearchRunConfig:
    current_experiment_spec = _current_experiment_spec_class()

    if isinstance(config, SearchRunConfig):
        if objective is not None and objective != config.objective:
            raise ValueError(
                "objective must not be provided separately when config is a SearchRunConfig."
            )
        return config
    config = _coerce_experiment_spec(config)
    if isinstance(config, current_experiment_spec):
        if not isinstance(objective, ObjectiveConfig):
            raise TypeError(
                "objective must be provided when config is an ExperimentSpec."
            )
        return SearchRunConfig(
            experiment=config,
            objective=objective,
            contract_version=config.contract_version,
        )

    payload = _load_serialized_payload(config)
    if "experiment" in payload and "objective" in payload:
        experiment = _deserialize_dataclass(payload["experiment"], current_experiment_spec)
        parsed_objective = _deserialize_dataclass(payload["objective"], ObjectiveConfig)
        return SearchRunConfig(
            experiment=experiment,
            objective=parsed_objective,
            contract_version=payload.get("contract_version", experiment.contract_version),
        )

    experiment = _deserialize_dataclass(payload, current_experiment_spec)
    if not isinstance(objective, ObjectiveConfig):
        raise TypeError(
            "objective must be provided when serialized config contains only an ExperimentSpec."
        )
    return SearchRunConfig(
        experiment=experiment,
        objective=objective,
        contract_version=experiment.contract_version,
    )


def evaluate_objective(
    metrics: dict[str, Any],
    objective: ObjectiveConfig,
) -> tuple[float | None, float | None, tuple[str, ...]]:
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a dict.")
    if not isinstance(objective, ObjectiveConfig):
        raise TypeError("objective must be an ObjectiveConfig instance.")

    violations: list[str] = []
    for constraint in objective.constraints:
        metric_value = _metric_number(metrics, constraint.metric_name)
        if metric_value is None:
            violations.append(f"missing_constraint_metric:{constraint.metric_name}")
            continue
        if constraint.minimum is not None and metric_value < float(constraint.minimum):
            violations.append(
                f"{constraint.metric_name}<{float(constraint.minimum)}"
            )
        if constraint.maximum is not None and metric_value > float(constraint.maximum):
            violations.append(
                f"{constraint.metric_name}>{float(constraint.maximum)}"
            )

    objective_value: float | None
    if objective.mode == "single_metric":
        assert objective.metric_name is not None
        objective_value = _metric_number(metrics, objective.metric_name)
        if objective_value is None:
            violations.append(f"missing_objective_metric:{objective.metric_name}")
    else:
        total = 0.0
        missing_metrics: list[str] = []
        for metric_name, weight in objective.composite_weights.items():
            metric_value = _metric_number(metrics, metric_name)
            if metric_value is None:
                missing_metrics.append(metric_name)
                continue
            total += weight * metric_value
        if missing_metrics:
            for metric_name in missing_metrics:
                violations.append(f"missing_objective_metric:{metric_name}")
            objective_value = None
        else:
            objective_value = total

    if objective_value is None or violations:
        return objective_value, None, tuple(violations)
    return (
        objective_value,
        _objective_score(objective_value, objective.direction),
        (),
    )


def _rank_evaluations(
    evaluations: tuple[VariantEvaluation, ...],
) -> tuple[VariantEvaluation, ...]:
    rankable = [
        (index, evaluation)
        for index, evaluation in enumerate(evaluations)
        if evaluation.status == "completed"
        and evaluation.feasible
        and evaluation.objective_score is not None
    ]
    ordered = sorted(
        rankable,
        key=lambda item: (
            -float(item[1].objective_score),
            item[1].variant_id,
            "" if item[1].source_label is None else item[1].source_label,
            item[0],
        ),
    )
    rank_map = {index: position + 1 for position, (index, _) in enumerate(ordered)}
    return tuple(
        replace(evaluation, rank=rank_map.get(index))
        for index, evaluation in enumerate(evaluations)
    )


def evaluate_experiment_result(
    result: ExperimentExecutionResult,
    objective: ObjectiveConfig,
) -> tuple[VariantEvaluation, ...]:
    if not isinstance(result, ExperimentExecutionResult):
        raise TypeError("result must be an ExperimentExecutionResult instance.")
    if not isinstance(objective, ObjectiveConfig):
        raise TypeError("objective must be an ObjectiveConfig instance.")

    cv_summary_map = {summary.variant_id: summary for summary in result.cv_summaries}
    holdout_summary_map = {
        summary.variant_id: dict(summary.metrics) for summary in result.holdout_summaries
    }
    skipped_variant_ids = set(result.skipped_variant_ids)

    evaluations: list[VariantEvaluation] = []
    for variant in result.experiment.variants:
        summary = cv_summary_map.get(variant.variant_id)
        holdout_metrics = holdout_summary_map.get(variant.variant_id)
        if summary is None:
            status: VariantStatus = (
                "skipped" if variant.variant_id in skipped_variant_ids else "failed"
            )
            evaluations.append(
                VariantEvaluation(
                    variant_id=variant.variant_id,
                    status=status,
                    cv_metrics={},
                    holdout_metrics=holdout_metrics,
                    feasible=False,
                    runtime_seconds=0.0,
                    source_label=variant.variant_id,
                    contract_version=result.contract_version,
                )
            )
            continue

        objective_value, objective_score, violations = evaluate_objective(
            summary.cv_metrics,
            objective,
        )
        feasible = not violations
        if summary.status != "completed":
            objective_score = None
        evaluations.append(
            VariantEvaluation(
                variant_id=summary.variant_id,
                status=summary.status,
                cv_metrics=summary.cv_metrics,
                holdout_metrics=holdout_metrics,
                objective_value=objective_value,
                objective_score=objective_score,
                feasible=feasible,
                constraint_violations=violations,
                runtime_seconds=summary.runtime_seconds,
                source_label=summary.variant_id,
                contract_version=result.contract_version,
            )
        )

    return _rank_evaluations(tuple(evaluations))


def _best_evaluation(
    evaluations: tuple[VariantEvaluation, ...],
) -> VariantEvaluation | None:
    ranked = [evaluation for evaluation in evaluations if evaluation.rank == 1]
    return ranked[0] if ranked else None


def run_search_experiment(
    experiment: ExperimentSpec,
    data: pd.DataFrame,
    *,
    objective: ObjectiveConfig,
    runner_fn: RunnerFn = run_experiment,
    runner_kwargs: dict[str, Any] | None = None,
) -> SearchExecutionResult:
    experiment = _coerce_experiment_spec(experiment)
    if not isinstance(experiment, _current_experiment_spec_class()):
        raise TypeError("experiment must be an ExperimentSpec instance.")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not callable(runner_fn):
        raise TypeError("runner_fn must be callable.")

    runner_result = runner_fn(
        experiment,
        data,
        **_normalize_runner_kwargs(runner_kwargs),
    )
    if not isinstance(runner_result, ExperimentExecutionResult):
        raise TypeError(
            "runner_fn must return an ExperimentExecutionResult instance."
        )

    evaluations = evaluate_experiment_result(runner_result, objective)
    best = _best_evaluation(evaluations)
    return SearchExecutionResult(
        mode=experiment.search.mode,
        objective=objective,
        evaluations=evaluations,
        best_variant_id=None if best is None else best.variant_id,
        best_objective_value=None if best is None else best.objective_value,
        experiment_results=(runner_result,),
        contract_version=experiment.contract_version,
    )


def build_family_search_experiment(
    *,
    name: str,
    base_backtest_spec: BacktestSpec,
    entry_families: tuple[Any, ...] | list[Any],
    exit_families: tuple[Any, ...] | list[Any],
    risk_families: tuple[Any, ...] | list[Any],
    folds: tuple[FoldSpec, ...] | list[FoldSpec],
    search: SearchConfig,
    pruning: PruningConfig | None = None,
    outputs: OutputConfig | None = None,
    holdout: HoldoutSpec | None = None,
    deep_dive: DeepDiveConfig | None = None,
) -> ExperimentSpec:
    _require_non_empty("name", name)
    if not isinstance(base_backtest_spec, BacktestSpec):
        raise TypeError("base_backtest_spec must be a BacktestSpec instance.")
    if not isinstance(search, SearchConfig):
        raise TypeError("search must be a SearchConfig instance.")
    if search.mode not in {"grid", "random"}:
        raise ValueError(
            "build_family_search_experiment only supports grid and random modes."
        )

    variants = generate_variants(
        base_backtest_spec=base_backtest_spec,
        entry_families=entry_families,
        exit_families=exit_families,
        risk_families=risk_families,
        search=search,
    )
    return ExperimentSpec(
        name=name,
        variants=variants,
        folds=tuple(folds),
        search=search,
        pruning=PruningConfig() if pruning is None else pruning,
        outputs=OutputConfig() if outputs is None else outputs,
        holdout=holdout,
        deep_dive=deep_dive,
        contract_version=base_backtest_spec.contract_version,
    )


@dataclass(frozen=True, slots=True)
class GridSearchAdapter:
    objective: ObjectiveConfig
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.objective, ObjectiveConfig):
            raise TypeError("objective must be an ObjectiveConfig instance.")
        _require_contract_version("contract_version", self.contract_version)

    def build_experiment(self, **kwargs: Any) -> ExperimentSpec:
        search = kwargs.get("search")
        if not isinstance(search, SearchConfig):
            raise TypeError("search must be provided as a SearchConfig instance.")
        if search.mode != "grid":
            raise ValueError("GridSearchAdapter requires SearchConfig(mode='grid').")
        return build_family_search_experiment(**kwargs)

    def run(
        self,
        *,
        data: pd.DataFrame,
        runner_fn: RunnerFn = run_experiment,
        runner_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SearchExecutionResult:
        experiment = self.build_experiment(**kwargs)
        return run_search_experiment(
            experiment,
            data,
            objective=self.objective,
            runner_fn=runner_fn,
            runner_kwargs=runner_kwargs,
        )


@dataclass(frozen=True, slots=True)
class RandomSearchAdapter:
    objective: ObjectiveConfig
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.objective, ObjectiveConfig):
            raise TypeError("objective must be an ObjectiveConfig instance.")
        _require_contract_version("contract_version", self.contract_version)

    def build_experiment(self, **kwargs: Any) -> ExperimentSpec:
        search = kwargs.get("search")
        if not isinstance(search, SearchConfig):
            raise TypeError("search must be provided as a SearchConfig instance.")
        if search.mode != "random":
            raise ValueError("RandomSearchAdapter requires SearchConfig(mode='random').")
        return build_family_search_experiment(**kwargs)

    def run(
        self,
        *,
        data: pd.DataFrame,
        runner_fn: RunnerFn = run_experiment,
        runner_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SearchExecutionResult:
        experiment = self.build_experiment(**kwargs)
        return run_search_experiment(
            experiment,
            data,
            objective=self.objective,
            runner_fn=runner_fn,
            runner_kwargs=runner_kwargs,
        )


def _load_optuna_module(optuna_module: object | None) -> Any:
    if optuna_module is not None:
        return optuna_module
    try:
        return importlib.import_module("optuna")
    except ImportError as exc:
        raise ImportError(
            "optuna integration is optional and requires the 'optuna' package."
        ) from exc


def _build_single_variant_experiment(
    base_experiment: ExperimentSpec,
    variant: VariantSpec,
    *,
    trial_name: str,
    remaining_runtime_seconds: int | None,
) -> ExperimentSpec:
    search = SearchConfig(
        mode="optuna",
        max_variants=1,
        max_runtime_seconds=remaining_runtime_seconds,
        random_seed=base_experiment.search.random_seed,
        contract_version=base_experiment.contract_version,
    )
    return ExperimentSpec(
        name=f"{base_experiment.name}__{trial_name}",
        variants=(variant,),
        folds=base_experiment.folds,
        search=search,
        pruning=base_experiment.pruning,
        outputs=base_experiment.outputs,
        holdout=base_experiment.holdout,
        deep_dive=base_experiment.deep_dive,
        contract_version=base_experiment.contract_version,
    )


@dataclass(frozen=True, slots=True)
class OptunaSearchAdapter:
    objective: ObjectiveConfig
    optuna_module: object | None = None
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.objective, ObjectiveConfig):
            raise TypeError("objective must be an ObjectiveConfig instance.")
        _require_contract_version("contract_version", self.contract_version)

    def run(
        self,
        base_experiment: ExperimentSpec,
        data: pd.DataFrame,
        *,
        variant_factory: VariantFactory,
        runner_fn: RunnerFn = run_experiment,
        runner_kwargs: dict[str, Any] | None = None,
        time_fn: TimeFn = time.perf_counter,
    ) -> SearchExecutionResult:
        base_experiment = _coerce_experiment_spec(base_experiment)
        if not isinstance(base_experiment, _current_experiment_spec_class()):
            raise TypeError("base_experiment must be an ExperimentSpec instance.")
        if base_experiment.search.mode != "optuna":
            raise ValueError("OptunaSearchAdapter requires SearchConfig(mode='optuna').")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not callable(variant_factory):
            raise TypeError("variant_factory must be callable.")
        if not callable(runner_fn):
            raise TypeError("runner_fn must be callable.")
        if not callable(time_fn):
            raise TypeError("time_fn must be callable.")

        optuna = _load_optuna_module(self.optuna_module)
        if not hasattr(optuna, "create_study"):
            raise TypeError("optuna module must expose create_study.")

        runner_kwargs_dict = _normalize_runner_kwargs(runner_kwargs)
        trial_records: list[OptunaTrialRecord] = []
        experiment_results: list[ExperimentExecutionResult] = []
        raw_evaluations: list[VariantEvaluation] = []

        direction = self.objective.direction
        study_name = f"{base_experiment.name}__optuna"
        study = optuna.create_study(direction=direction, study_name=study_name)
        started_at = float(time_fn())

        def objective_fn(trial: Any) -> float:
            trial_number = int(getattr(trial, "number", len(trial_records)))
            trial_name = f"trial_{trial_number:05d}"
            variant = variant_factory(trial)
            if not isinstance(variant, VariantSpec):
                raise TypeError("variant_factory must return a VariantSpec instance.")

            elapsed = float(time_fn()) - started_at
            remaining_runtime_seconds: int | None = None
            if base_experiment.search.max_runtime_seconds is not None:
                remaining = float(base_experiment.search.max_runtime_seconds) - elapsed
                remaining_runtime_seconds = max(int(math.ceil(remaining)), 1)

            trial_experiment = _build_single_variant_experiment(
                base_experiment,
                variant,
                trial_name=trial_name,
                remaining_runtime_seconds=remaining_runtime_seconds,
            )
            runner_result = runner_fn(
                trial_experiment,
                data,
                **runner_kwargs_dict,
            )
            if not isinstance(runner_result, ExperimentExecutionResult):
                raise TypeError(
                    "runner_fn must return an ExperimentExecutionResult instance."
                )

            evaluations = evaluate_experiment_result(runner_result, self.objective)
            evaluation = replace(evaluations[0], source_label=trial_name)
            experiment_results.append(runner_result)
            raw_evaluations.append(evaluation)
            trial_records.append(
                OptunaTrialRecord(
                    trial_number=trial_number,
                    trial_name=trial_name,
                    variant_id=evaluation.variant_id,
                    experiment_name=trial_experiment.name,
                    objective_value=evaluation.objective_value,
                    objective_score=evaluation.objective_score,
                    feasible=evaluation.feasible,
                    status=evaluation.status,
                    runtime_seconds=runner_result.runtime_summary.total_runtime_seconds,
                    contract_version=base_experiment.contract_version,
                )
            )

            if hasattr(trial, "set_user_attr"):
                trial.set_user_attr("variant_id", evaluation.variant_id)
                trial.set_user_attr("trial_name", trial_name)
                trial.set_user_attr("experiment_name", trial_experiment.name)

            if (
                evaluation.status != "completed"
                or not evaluation.feasible
                or evaluation.objective_value is None
            ):
                return _worst_objective_value(direction)
            return float(evaluation.objective_value)

        optimize_kwargs: dict[str, Any] = {}
        if base_experiment.search.max_variants is not None:
            optimize_kwargs["n_trials"] = base_experiment.search.max_variants
        if base_experiment.search.max_runtime_seconds is not None:
            optimize_kwargs["timeout"] = base_experiment.search.max_runtime_seconds
        study.optimize(objective_fn, **optimize_kwargs)

        evaluations = _rank_evaluations(tuple(raw_evaluations))
        best = _best_evaluation(evaluations)
        return SearchExecutionResult(
            mode="optuna",
            objective=self.objective,
            evaluations=evaluations,
            best_variant_id=None if best is None else best.variant_id,
            best_objective_value=None if best is None else best.objective_value,
            experiment_results=tuple(experiment_results),
            trial_records=tuple(trial_records),
            study=study,
            contract_version=base_experiment.contract_version,
        )


def _search_summary_frames(
    run_config: SearchRunConfig,
    search_result: SearchExecutionResult,
    *,
    runtime_summary: str,
    stopping_reason: str,
) -> dict[str, pd.DataFrame]:
    run_summary = pd.DataFrame(
        [
            {
                "experiment_name": run_config.experiment.name,
                "search_mode": search_result.mode,
                "objective_mode": search_result.objective.mode,
                "objective_direction": search_result.objective.direction,
                "best_variant_id": search_result.best_variant_id,
                "best_objective_value": search_result.best_objective_value,
                "evaluation_count": len(search_result.evaluations),
                "trial_count": len(search_result.trial_records),
                "runtime_summary": runtime_summary,
                "stopping_reason": stopping_reason,
            }
        ]
    )
    variant_summary = pd.DataFrame(
        [
            {
                "variant_id": evaluation.variant_id,
                "status": evaluation.status,
                "rank": evaluation.rank,
                "objective_value": evaluation.objective_value,
                "objective_score": evaluation.objective_score,
                "feasible": evaluation.feasible,
                "constraint_violations": "|".join(evaluation.constraint_violations),
                "runtime_seconds": evaluation.runtime_seconds,
                "cv_metrics": serialize_manifest(evaluation.cv_metrics),
                "holdout_metrics": (
                    None
                    if evaluation.holdout_metrics is None
                    else serialize_manifest(evaluation.holdout_metrics)
                ),
                "source_label": evaluation.source_label,
            }
            for evaluation in search_result.evaluations
        ]
    )
    trial_summary = pd.DataFrame(
        [
            {
                "trial_number": record.trial_number,
                "trial_name": record.trial_name,
                "variant_id": record.variant_id,
                "experiment_name": record.experiment_name,
                "objective_value": record.objective_value,
                "objective_score": record.objective_score,
                "feasible": record.feasible,
                "status": record.status,
                "runtime_seconds": record.runtime_seconds,
            }
            for record in search_result.trial_records
        ]
    )
    config_rows = [
        {"section": "experiment", "key": "name", "value": run_config.experiment.name},
        {"section": "search", "key": "mode", "value": run_config.experiment.search.mode},
        {
            "section": "search",
            "key": "max_variants",
            "value": run_config.experiment.search.max_variants,
        },
        {
            "section": "search",
            "key": "max_runtime_seconds",
            "value": run_config.experiment.search.max_runtime_seconds,
        },
        {
            "section": "search",
            "key": "max_parallel_variants",
            "value": run_config.experiment.search.max_parallel_variants,
        },
        {
            "section": "search",
            "key": "random_seed",
            "value": run_config.experiment.search.random_seed,
        },
        {
            "section": "objective",
            "key": "mode",
            "value": run_config.objective.mode,
        },
        {
            "section": "objective",
            "key": "direction",
            "value": run_config.objective.direction,
        },
        {
            "section": "objective",
            "key": "metric_name",
            "value": run_config.objective.metric_name,
        },
        {
            "section": "objective",
            "key": "composite_weights",
            "value": serialize_manifest(run_config.objective.composite_weights),
        },
        {
            "section": "objective",
            "key": "constraints",
            "value": serialize_manifest(run_config.objective.constraints),
        },
    ]
    return {
        "run_summary": run_summary,
        "variant_summary": variant_summary,
        "trial_summary": trial_summary,
        "config": pd.DataFrame(config_rows),
    }


def _export_search_summary_workbook(
    run_config: SearchRunConfig,
    search_result: SearchExecutionResult,
    path: Path,
    *,
    runtime_summary: str,
    stopping_reason: str,
) -> Path:
    frames = _search_summary_frames(
        run_config,
        search_result,
        runtime_summary=runtime_summary,
        stopping_reason=stopping_reason,
    )
    if path.parent != Path():
        path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in ("run_summary", "variant_summary", "trial_summary", "config"):
            frames[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
    return path


def _derive_stopping_reason(search_result: SearchExecutionResult) -> str:
    if search_result.mode in {"grid", "random"} and search_result.experiment_results:
        experiment_result = search_result.experiment_results[0]
        skip_reasons = {record.reason for record in experiment_result.skip_records}
        if "runtime_budget_reached" in skip_reasons:
            return "runtime_budget_reached"
        if "max_variants_reached" in skip_reasons:
            return "max_variants_reached"
        return "completed_planned_search"

    if search_result.mode == "optuna" and search_result.experiment_results:
        experiment = search_result.experiment_results[0].experiment
        if (
            experiment.search.max_variants is not None
            and len(search_result.trial_records) >= experiment.search.max_variants
        ):
            return "max_variants_reached"
        if experiment.search.max_runtime_seconds is not None:
            return "max_runtime_seconds_reached_or_study_completed"
        return "study_completed"

    return "completed"


def _runtime_summary_text(search_result: SearchExecutionResult) -> str:
    if search_result.mode in {"grid", "random"} and search_result.experiment_results:
        runtime = search_result.experiment_results[0].runtime_summary
        return (
            f"mode={search_result.mode} total_runtime_seconds={runtime.total_runtime_seconds:.3f} "
            f"estimated_remaining_seconds={runtime.estimated_remaining_seconds:.3f} "
            f"completed_variants={len(search_result.experiment_results[0].completed_variant_ids)} "
            f"pruned_variants={len(search_result.experiment_results[0].pruned_variant_ids)} "
            f"failed_variants={len(search_result.experiment_results[0].failed_variant_ids)} "
            f"skipped_variants={len(search_result.experiment_results[0].skipped_variant_ids)}"
        )

    total_runtime_seconds = sum(
        result.runtime_summary.total_runtime_seconds
        for result in search_result.experiment_results
    )
    completed_count = sum(
        1 for evaluation in search_result.evaluations if evaluation.status == "completed"
    )
    return (
        f"mode={search_result.mode} total_runtime_seconds={total_runtime_seconds:.3f} "
        f"completed_variants={completed_count} "
        f"trial_count={len(search_result.trial_records)}"
    )


def _export_selected_deep_dives(
    run_config: SearchRunConfig,
    search_result: SearchExecutionResult,
    data: pd.DataFrame,
    output_dir: Path,
    *,
    runner_fn: RunnerFn,
    runner_kwargs: dict[str, Any] | None,
) -> tuple[DeepDiveArtifactSet, ...]:
    deep_dive = run_config.experiment.deep_dive
    if deep_dive is None:
        return ()

    artifacts: list[DeepDiveArtifactSet] = []
    deep_dive_root = output_dir / "deep_dive"
    for experiment_result in search_result.experiment_results:
        variant_ids = tuple(
            variant_id
            for variant_id in deep_dive.selected_variant_ids
            if variant_id in {variant.variant_id for variant in experiment_result.experiment.variants}
        )
        if not variant_ids:
            continue
        deep_dive_result = _ensure_deep_dive_results(
            experiment_result,
            data,
            variant_ids=variant_ids,
            runner_fn=runner_fn,
            runner_kwargs=runner_kwargs,
        )
        artifacts.extend(
            export_deep_dive_artifacts(
                deep_dive_result,
                data,
                deep_dive_root,
                selected_variant_ids=variant_ids,
                selected_folds=deep_dive.selected_folds,
                include_holdout=deep_dive.include_holdout,
                generate_trade_log=deep_dive.generate_trade_log,
                generate_equity_plot=deep_dive.generate_equity_plot,
                generate_price_plot=deep_dive.generate_price_plot,
            )
        )
    return tuple(artifacts)


def _build_deep_dive_experiment(
    experiment_result: ExperimentExecutionResult,
    *,
    variant_ids: tuple[str, ...],
) -> ExperimentSpec:
    variants = tuple(
        variant
        for variant in experiment_result.experiment.variants
        if variant.variant_id in set(variant_ids)
    )
    if not variants:
        raise ValueError("deep-dive rerun requires at least one matching variant.")
    return replace(
        experiment_result.experiment,
        variants=variants,
        search=replace(
            experiment_result.experiment.search,
            max_variants=len(variants),
            max_runtime_seconds=None,
            max_parallel_variants=1,
        ),
    )


def _ensure_deep_dive_results(
    experiment_result: ExperimentExecutionResult,
    data: pd.DataFrame,
    *,
    variant_ids: tuple[str, ...],
    runner_fn: RunnerFn,
    runner_kwargs: dict[str, Any] | None,
) -> ExperimentExecutionResult:
    if experiment_result.run_results:
        return experiment_result

    rerun_kwargs = _normalize_runner_kwargs(runner_kwargs)
    rerun_kwargs["retain_run_results"] = True
    deep_dive_experiment = _build_deep_dive_experiment(
        experiment_result,
        variant_ids=variant_ids,
    )
    rerun_result = runner_fn(
        deep_dive_experiment,
        data,
        **rerun_kwargs,
    )
    if not isinstance(rerun_result, ExperimentExecutionResult):
        raise TypeError(
            "runner_fn must return an ExperimentExecutionResult instance."
        )
    return rerun_result


def _build_reproducibility_manifest(
    run_config: SearchRunConfig,
    search_result: SearchExecutionResult,
    *,
    summary_workbook_path: Path,
    deep_dive_artifacts: tuple[DeepDiveArtifactSet, ...],
    runtime_summary: str,
    stopping_reason: str,
) -> dict[str, Any]:
    manifest_payload = {
        "run_config": run_config,
        "search_mode": search_result.mode,
        "best_variant_id": search_result.best_variant_id,
        "best_objective_value": search_result.best_objective_value,
        "stopping_reason": stopping_reason,
        "runtime_summary": runtime_summary,
        "summary_workbook_path": str(summary_workbook_path),
        "experiment_run_ids": [
            result.run_id for result in search_result.experiment_results
        ],
        "variant_evaluations": [
            {
                "variant_id": evaluation.variant_id,
                "status": evaluation.status,
                "rank": evaluation.rank,
                "objective_value": evaluation.objective_value,
                "objective_score": evaluation.objective_score,
                "feasible": evaluation.feasible,
                "constraint_violations": evaluation.constraint_violations,
                "runtime_seconds": evaluation.runtime_seconds,
                "source_label": evaluation.source_label,
            }
            for evaluation in search_result.evaluations
        ],
        "trial_records": [
            {
                "trial_number": record.trial_number,
                "trial_name": record.trial_name,
                "variant_id": record.variant_id,
                "experiment_name": record.experiment_name,
                "objective_value": record.objective_value,
                "objective_score": record.objective_score,
                "feasible": record.feasible,
                "status": record.status,
                "runtime_seconds": record.runtime_seconds,
            }
            for record in search_result.trial_records
        ],
        "deep_dive_artifacts": [
            {
                "variant_id": artifact.target.variant_id,
                "phase": artifact.target.phase,
                "label": artifact.target.label,
                "target_dir": str(artifact.target_dir),
                "equity_plot_path": None
                if artifact.equity_plot_path is None
                else str(artifact.equity_plot_path),
                "price_plot_path": None
                if artifact.price_plot_path is None
                else str(artifact.price_plot_path),
                "trade_log_path": None
                if artifact.trade_log_path is None
                else str(artifact.trade_log_path),
            }
            for artifact in deep_dive_artifacts
        ],
    }
    return json.loads(serialize_manifest(manifest_payload))


def run_search_entrypoint(
    config: SearchRunConfig | ExperimentSpec | dict[str, Any] | str | Path,
    data: pd.DataFrame,
    *,
    objective: ObjectiveConfig | None = None,
    variant_factory: VariantFactory | None = None,
    runner_fn: RunnerFn = run_experiment,
    runner_kwargs: dict[str, Any] | None = None,
    optuna_module: object | None = None,
    time_fn: TimeFn = time.perf_counter,
    output_dir: str | Path | None = None,
    write_manifest: bool | None = None,
    verbose: bool = True,
) -> SearchEntrypointResult:
    run_config = load_search_run_config(config, objective=objective)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not callable(runner_fn):
        raise TypeError("runner_fn must be callable.")
    if not callable(time_fn):
        raise TypeError("time_fn must be callable.")

    root_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(run_config.experiment.outputs.output_dir)
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    if run_config.experiment.search.mode in {"grid", "random"}:
        search_result = run_search_experiment(
            run_config.experiment,
            data,
            objective=run_config.objective,
            runner_fn=runner_fn,
            runner_kwargs=runner_kwargs,
        )
    else:
        if variant_factory is None:
            raise TypeError(
                "variant_factory must be provided for optuna search entrypoints."
            )
        adapter = OptunaSearchAdapter(
            run_config.objective,
            optuna_module=optuna_module,
            contract_version=run_config.contract_version,
        )
        search_result = adapter.run(
            run_config.experiment,
            data,
            variant_factory=variant_factory,
            runner_fn=runner_fn,
            runner_kwargs=runner_kwargs,
            time_fn=time_fn,
        )

    runtime_summary = _runtime_summary_text(search_result)
    stopping_reason = _derive_stopping_reason(search_result)
    summary_workbook_path = root_dir / run_config.experiment.outputs.summary_excel_name
    if len(search_result.experiment_results) == 1 and search_result.mode in {"grid", "random"}:
        export_summary_workbook(search_result.experiment_results[0], summary_workbook_path)
    else:
        _export_search_summary_workbook(
            run_config,
            search_result,
            summary_workbook_path,
            runtime_summary=runtime_summary,
            stopping_reason=stopping_reason,
        )

    deep_dive_artifacts = _export_selected_deep_dives(
        run_config,
        search_result,
        data,
        root_dir,
        runner_fn=runner_fn,
        runner_kwargs=runner_kwargs,
    )
    reproducibility_manifest = _build_reproducibility_manifest(
        run_config,
        search_result,
        summary_workbook_path=summary_workbook_path,
        deep_dive_artifacts=deep_dive_artifacts,
        runtime_summary=runtime_summary,
        stopping_reason=stopping_reason,
    )

    should_write_manifest = (
        run_config.experiment.outputs.write_run_manifests
        if write_manifest is None
        else bool(write_manifest)
    )
    manifest_path: Path | None = None
    if should_write_manifest:
        manifest_path = root_dir / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(reproducibility_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if verbose:
        print(runtime_summary)
        print(f"stopping_reason={stopping_reason}")

    return SearchEntrypointResult(
        run_config=run_config,
        search_result=search_result,
        summary_workbook_path=summary_workbook_path,
        deep_dive_artifacts=deep_dive_artifacts,
        runtime_summary=runtime_summary,
        stopping_reason=stopping_reason,
        reproducibility_manifest=reproducibility_manifest,
        manifest_path=manifest_path,
        contract_version=run_config.contract_version,
    )


__all__ = [
    "GridSearchAdapter",
    "MetricConstraint",
    "ObjectiveConfig",
    "OptunaSearchAdapter",
    "OptunaTrialRecord",
    "RandomSearchAdapter",
    "SearchExecutionResult",
    "SearchEntrypointResult",
    "SearchRunConfig",
    "VariantEvaluation",
    "build_family_search_experiment",
    "evaluate_experiment_result",
    "evaluate_objective",
    "load_search_run_config",
    "run_search_experiment",
    "run_search_entrypoint",
    "serialize_search_run_config",
]
