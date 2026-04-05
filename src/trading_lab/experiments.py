from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Literal

from trading_lab.contracts import (
    DEFAULT_CONTRACT_VERSION,
    BacktestSpec,
    NodeContract,
    validate_node_compatibility,
)

SearchMode = Literal["grid", "random", "optuna"]
ParameterGrid = dict[str, tuple[Any, ...] | list[Any]]
NodeFamilyDefinition = tuple[NodeContract, ParameterGrid]


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


def _require_int(name: str, value: object, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")


def _require_number(
    name: str,
    value: object,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, got {type(value).__name__}.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value}.")
    if positive and numeric <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}.")
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


def _normalize_metric_thresholds(
    value: dict[str, float] | None,
) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("early_metric_thresholds must be a dict.")

    normalized: dict[str, float] = {}
    for key, threshold in value.items():
        _require_non_empty("early_metric_thresholds key", key)
        normalized[key] = _require_number(
            f"early_metric_thresholds[{key!r}]",
            threshold,
        )
    return dict(sorted(normalized.items()))


def _normalize_parameter_grid(
    parameter_grid: ParameterGrid | None,
) -> dict[str, tuple[Any, ...]]:
    if parameter_grid is None:
        return {}
    if not isinstance(parameter_grid, dict):
        raise TypeError("parameter_grid must be a dict.")

    normalized: dict[str, tuple[Any, ...]] = {}
    for key in sorted(parameter_grid):
        _require_non_empty("parameter_grid key", key)
        values = parameter_grid[key]
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
            raise TypeError(
                "parameter grid values must be list or tuple candidate sequences."
            )
        candidate_values = tuple(values)
        if not candidate_values:
            raise ValueError(
                f"parameter grid for {key!r} must contain at least one value."
            )
        for candidate_value in candidate_values:
            _stable_json_value(candidate_value)
        normalized[key] = candidate_values
    return normalized


def _stable_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot serialize non-finite float {value!r}.")
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_info.name: _stable_json_value(getattr(value, field_info.name))
            for field_info in fields(value)
        }
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError("manifest serialization requires string dict keys.")
            normalized[key] = _stable_json_value(value[key])
        return normalized
    if isinstance(value, (tuple, list)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_stable_json_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )
    raise TypeError(
        f"unsupported value for manifest serialization: {type(value).__name__}."
    )


def serialize_manifest(value: Any) -> str:
    """Return a deterministic JSON representation for experiment manifests."""

    return json.dumps(
        _stable_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def expand_parameter_grid(
    parameter_grid: ParameterGrid | None,
) -> tuple[dict[str, Any], ...]:
    normalized_grid = _normalize_parameter_grid(parameter_grid)
    if not normalized_grid:
        return ({},)

    parameter_names = tuple(sorted(normalized_grid))
    combinations: list[dict[str, Any]] = []
    for parameter_values in product(
        *(normalized_grid[name] for name in parameter_names)
    ):
        combinations.append(
            {
                parameter_name: parameter_value
                for parameter_name, parameter_value in zip(
                    parameter_names,
                    parameter_values,
                    strict=True,
                )
            }
        )
    return tuple(combinations)


def _family_sort_key(
    family: NodeFamilyDefinition,
) -> str:
    contract, parameter_grid = family
    return serialize_manifest(
        {
            "name": contract.name,
            "kind": contract.kind,
            "version": contract.spec.version,
            "manifest": contract.manifest,
            "parameter_grid": _normalize_parameter_grid(parameter_grid),
        }
    )


def _normalize_families(
    kind: Literal["entry", "exit", "risk"],
    families: tuple[NodeFamilyDefinition, ...] | list[NodeFamilyDefinition],
) -> tuple[NodeFamilyDefinition, ...]:
    normalized: list[NodeFamilyDefinition] = []
    for family in tuple(families):
        if not isinstance(family, tuple) or len(family) != 2:
            raise TypeError(
                "family definitions must be (NodeContract, parameter_grid) tuples."
            )
        contract, parameter_grid = family
        if not isinstance(contract, NodeContract):
            raise TypeError("family definitions must include NodeContract objects.")
        if contract.kind != kind:
            raise ValueError(
                f"{kind} family definitions must use {kind} node contracts."
            )
        normalized.append((contract, _normalize_parameter_grid(parameter_grid)))
    if not normalized:
        raise ValueError(f"{kind}_families must not be empty.")
    return tuple(sorted(normalized, key=_family_sort_key))


def _merge_parameters(
    base_parameters: dict[str, Any],
    variant_parameters: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base_parameters)
    merged.update(variant_parameters)
    return {key: merged[key] for key in sorted(merged)}


def parameterize_node_contract(
    contract: NodeContract,
    parameters: dict[str, Any] | None,
) -> NodeContract:
    if not isinstance(contract, NodeContract):
        raise TypeError("contract must be a NodeContract instance.")
    normalized_variant_parameters = {
        key: value
        for key, value in sorted((parameters or {}).items())
    }
    for key in normalized_variant_parameters:
        _require_non_empty("parameters key", key)
        _stable_json_value(normalized_variant_parameters[key])

    base_parameters = contract.manifest.get("parameters", {})
    if not isinstance(base_parameters, dict):
        raise TypeError("node manifest 'parameters' must be a dict.")

    merged_parameters = _merge_parameters(base_parameters, normalized_variant_parameters)
    manifest = dict(contract.manifest)
    manifest["parameters"] = merged_parameters
    manifest["parameter_lineage"] = {
        "base_parameters": {key: base_parameters[key] for key in sorted(base_parameters)},
        "variant_parameters": normalized_variant_parameters,
    }

    return NodeContract(
        spec=contract.spec,
        input_contract_version=contract.input_contract_version,
        output_contract_version=contract.output_contract_version,
        metric_dependencies=contract.metric_dependencies,
        manifest=manifest,
    )


def expand_node_family(
    contract: NodeContract,
    parameter_grid: ParameterGrid | None,
) -> tuple[NodeContract, ...]:
    if not isinstance(contract, NodeContract):
        raise TypeError("contract must be a NodeContract instance.")
    parameter_sets = expand_parameter_grid(parameter_grid)
    return tuple(
        parameterize_node_contract(contract, parameter_set)
        for parameter_set in parameter_sets
    )


def label_fold(fold: "FoldSpec") -> str:
    if not isinstance(fold, FoldSpec):
        raise TypeError("fold must be a FoldSpec instance.")
    if fold.label is not None:
        return fold.label
    return f"fold_{fold.fold_index:02d}"


def generate_variant_id(
    backtest_spec: BacktestSpec,
    entry_contract: NodeContract,
    exit_contract: NodeContract,
    risk_contract: NodeContract,
) -> str:
    if not isinstance(backtest_spec, BacktestSpec):
        raise TypeError("backtest_spec must be a BacktestSpec instance.")
    if not isinstance(entry_contract, NodeContract):
        raise TypeError("entry_contract must be a NodeContract instance.")
    if not isinstance(exit_contract, NodeContract):
        raise TypeError("exit_contract must be a NodeContract instance.")
    if not isinstance(risk_contract, NodeContract):
        raise TypeError("risk_contract must be a NodeContract instance.")

    manifest = {
        "backtest_spec": backtest_spec,
        "entry_contract": entry_contract,
        "exit_contract": exit_contract,
        "risk_contract": risk_contract,
    }
    digest = hashlib.sha256(serialize_manifest(manifest).encode("utf-8")).hexdigest()
    return f"variant_{digest[:16]}"


def _select_variants_for_search(
    variants: tuple["VariantSpec", ...],
    search: "SearchConfig",
) -> tuple["VariantSpec", ...]:
    if not isinstance(search, SearchConfig):
        raise TypeError("search must be a SearchConfig instance.")

    limit = search.max_variants
    if limit is None or limit >= len(variants):
        return variants
    if search.mode == "grid":
        return variants[:limit]

    random_seed = 0 if search.random_seed is None else search.random_seed
    rng = random.Random(random_seed)
    selected_indices = rng.sample(range(len(variants)), k=limit)
    return tuple(variants[index] for index in selected_indices)


def generate_variants(
    *,
    base_backtest_spec: BacktestSpec,
    entry_families: tuple[NodeFamilyDefinition, ...] | list[NodeFamilyDefinition],
    exit_families: tuple[NodeFamilyDefinition, ...] | list[NodeFamilyDefinition],
    risk_families: tuple[NodeFamilyDefinition, ...] | list[NodeFamilyDefinition],
    search: "SearchConfig",
) -> tuple["VariantSpec", ...]:
    if not isinstance(base_backtest_spec, BacktestSpec):
        raise TypeError("base_backtest_spec must be a BacktestSpec instance.")

    normalized_entry_families = _normalize_families("entry", entry_families)
    normalized_exit_families = _normalize_families("exit", exit_families)
    normalized_risk_families = _normalize_families("risk", risk_families)

    expanded_entry_contracts = tuple(
        contract
        for family_contract, parameter_grid in normalized_entry_families
        for contract in expand_node_family(family_contract, parameter_grid)
    )
    expanded_exit_contracts = tuple(
        contract
        for family_contract, parameter_grid in normalized_exit_families
        for contract in expand_node_family(family_contract, parameter_grid)
    )
    expanded_risk_contracts = tuple(
        contract
        for family_contract, parameter_grid in normalized_risk_families
        for contract in expand_node_family(family_contract, parameter_grid)
    )

    variants: list[VariantSpec] = []
    for entry_contract, exit_contract, risk_contract in product(
        expanded_entry_contracts,
        expanded_exit_contracts,
        expanded_risk_contracts,
    ):
        backtest_spec = replace(
            base_backtest_spec,
            entry_node=entry_contract.name,
            exit_node=exit_contract.name,
            risk_node=risk_contract.name,
        )
        variants.append(
            VariantSpec(
                backtest_spec=backtest_spec,
                entry_contract=entry_contract,
                exit_contract=exit_contract,
                risk_contract=risk_contract,
            )
        )

    return _select_variants_for_search(tuple(variants), search)


@dataclass(frozen=True, slots=True)
class FoldSpec:
    fold_index: int
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_int("fold_index", self.fold_index, minimum=0)
        _require_datetime("train_start", self.train_start)
        _require_datetime("train_end", self.train_end)
        _require_datetime("validation_start", self.validation_start)
        _require_datetime("validation_end", self.validation_end)
        _require_contract_version("contract_version", self.contract_version)
        if self.label is not None:
            _require_non_empty("label", self.label)
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict.")
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.train_start >= self.train_end:
            raise ValueError("train_start must be earlier than train_end.")
        if self.validation_start >= self.validation_end:
            raise ValueError(
                "validation_start must be earlier than validation_end."
            )
        if self.train_end > self.validation_start:
            raise ValueError(
                "train_end must be earlier than or equal to validation_start."
            )

    @property
    def fold_label(self) -> str:
        return label_fold(self)


@dataclass(frozen=True, slots=True)
class HoldoutSpec:
    start: datetime
    end: datetime
    label: str = "holdout"
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_datetime("start", self.start)
        _require_datetime("end", self.end)
        _require_non_empty("label", self.label)
        _require_contract_version("contract_version", self.contract_version)
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.start >= self.end:
            raise ValueError("start must be earlier than end.")


@dataclass(frozen=True, slots=True)
class SearchConfig:
    mode: SearchMode
    max_variants: int | None = None
    max_runtime_seconds: int | None = None
    random_seed: int | None = None
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.mode not in {"grid", "random", "optuna"}:
            raise ValueError(f"unsupported search mode: {self.mode!r}.")
        if self.max_variants is None and self.max_runtime_seconds is None:
            raise ValueError(
                "search must define at least one bound: max_variants or "
                "max_runtime_seconds."
            )
        if self.max_variants is not None:
            _require_int("max_variants", self.max_variants, minimum=1)
        if self.max_runtime_seconds is not None:
            _require_int(
                "max_runtime_seconds",
                self.max_runtime_seconds,
                minimum=1,
            )
        if self.random_seed is not None:
            _require_int("random_seed", self.random_seed)
        _require_contract_version("contract_version", self.contract_version)


@dataclass(frozen=True, slots=True)
class PruningConfig:
    stop_on_zero_equity: bool = True
    stop_on_invalid_numeric_state: bool = True
    min_trades: int | None = None
    max_drawdown_threshold: float | None = None
    early_metric_thresholds: dict[str, float] = field(default_factory=dict)
    early_min_trades: int | None = None
    early_min_bars: int | None = None
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_bool("stop_on_zero_equity", self.stop_on_zero_equity)
        _require_bool(
            "stop_on_invalid_numeric_state",
            self.stop_on_invalid_numeric_state,
        )
        if self.min_trades is not None:
            _require_int("min_trades", self.min_trades, minimum=0)
        if self.max_drawdown_threshold is not None:
            threshold = _require_number(
                "max_drawdown_threshold",
                self.max_drawdown_threshold,
            )
            if threshold > 0.0:
                raise ValueError("max_drawdown_threshold must be <= 0.0.")
        if self.early_min_trades is not None:
            _require_int("early_min_trades", self.early_min_trades, minimum=1)
        if self.early_min_bars is not None:
            _require_int("early_min_bars", self.early_min_bars, minimum=1)
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(
            self,
            "early_metric_thresholds",
            _normalize_metric_thresholds(self.early_metric_thresholds),
        )


@dataclass(frozen=True, slots=True)
class OutputConfig:
    output_dir: str = "outputs"
    export_summary_excel: bool = True
    summary_excel_name: str = "experiment_summary.xlsx"
    write_run_manifests: bool = True
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("output_dir", self.output_dir)
        _require_bool("export_summary_excel", self.export_summary_excel)
        _require_non_empty("summary_excel_name", self.summary_excel_name)
        _require_bool("write_run_manifests", self.write_run_manifests)
        _require_contract_version("contract_version", self.contract_version)


@dataclass(frozen=True, slots=True)
class DeepDiveConfig:
    selected_variant_ids: tuple[str, ...]
    selected_folds: tuple[str, ...] = ()
    include_holdout: bool = False
    generate_trade_log: bool = True
    generate_equity_plot: bool = True
    generate_price_plot: bool = True
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_variant_ids",
            _normalize_string_tuple(
                "selected_variant_ids",
                self.selected_variant_ids,
            ),
        )
        object.__setattr__(
            self,
            "selected_folds",
            _normalize_string_tuple("selected_folds", self.selected_folds),
        )
        _require_bool("include_holdout", self.include_holdout)
        _require_bool("generate_trade_log", self.generate_trade_log)
        _require_bool("generate_equity_plot", self.generate_equity_plot)
        _require_bool("generate_price_plot", self.generate_price_plot)
        _require_contract_version("contract_version", self.contract_version)
        if not self.selected_variant_ids:
            raise ValueError("selected_variant_ids must not be empty.")
        if not self.selected_folds and not self.include_holdout:
            raise ValueError(
                "deep dive selection must include at least one fold or holdout."
            )


@dataclass(frozen=True, slots=True)
class VariantSpec:
    backtest_spec: BacktestSpec
    entry_contract: NodeContract
    exit_contract: NodeContract
    risk_contract: NodeContract
    variant_id: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.backtest_spec, BacktestSpec):
            raise TypeError("backtest_spec must be a BacktestSpec instance.")
        if not isinstance(self.entry_contract, NodeContract):
            raise TypeError("entry_contract must be a NodeContract instance.")
        if not isinstance(self.exit_contract, NodeContract):
            raise TypeError("exit_contract must be a NodeContract instance.")
        if not isinstance(self.risk_contract, NodeContract):
            raise TypeError("risk_contract must be a NodeContract instance.")
        if self.entry_contract.kind != "entry":
            raise ValueError("entry_contract must be an entry node contract.")
        if self.exit_contract.kind != "exit":
            raise ValueError("exit_contract must be an exit node contract.")
        if self.risk_contract.kind != "risk":
            raise ValueError("risk_contract must be a risk node contract.")
        if not isinstance(self.description, str):
            raise TypeError(
                f"description must be a string, got {type(self.description).__name__}."
            )
        object.__setattr__(self, "tags", _normalize_string_tuple("tags", self.tags))
        _require_contract_version("contract_version", self.contract_version)

        if self.backtest_spec.contract_version != self.contract_version:
            raise ValueError(
                "backtest_spec.contract_version must match variant contract_version."
            )
        if (
            self.backtest_spec.engine_capabilities.contract_version
            != self.backtest_spec.contract_version
        ):
            raise ValueError(
                "engine capabilities contract version must match backtest_spec."
            )
        if self.backtest_spec.entry_node != self.entry_contract.name:
            raise ValueError(
                "backtest_spec.entry_node must match entry_contract.spec.name."
            )
        if self.backtest_spec.exit_node != self.exit_contract.name:
            raise ValueError(
                "backtest_spec.exit_node must match exit_contract.spec.name."
            )
        if self.backtest_spec.risk_node != self.risk_contract.name:
            raise ValueError(
                "backtest_spec.risk_node must match risk_contract.spec.name."
            )

        validate_node_compatibility(
            self.entry_contract,
            self.backtest_spec.engine_capabilities,
        )
        validate_node_compatibility(
            self.exit_contract,
            self.backtest_spec.engine_capabilities,
        )
        validate_node_compatibility(
            self.risk_contract,
            self.backtest_spec.engine_capabilities,
        )

        expected_variant_id = generate_variant_id(
            backtest_spec=self.backtest_spec,
            entry_contract=self.entry_contract,
            exit_contract=self.exit_contract,
            risk_contract=self.risk_contract,
        )
        if self.variant_id:
            _require_non_empty("variant_id", self.variant_id)
            if self.variant_id != expected_variant_id:
                raise ValueError(
                    "variant_id must match the deterministic manifest-derived value."
                )
        else:
            object.__setattr__(self, "variant_id", expected_variant_id)


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    variants: tuple[VariantSpec, ...]
    folds: tuple[FoldSpec, ...]
    search: SearchConfig
    pruning: PruningConfig = field(default_factory=PruningConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    holdout: HoldoutSpec | None = None
    deep_dive: DeepDiveConfig | None = None
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("name", self.name)
        _require_contract_version("contract_version", self.contract_version)
        if not isinstance(self.search, SearchConfig):
            raise TypeError("search must be a SearchConfig instance.")
        if not isinstance(self.pruning, PruningConfig):
            raise TypeError("pruning must be a PruningConfig instance.")
        if not isinstance(self.outputs, OutputConfig):
            raise TypeError("outputs must be an OutputConfig instance.")

        normalized_variants = tuple(self.variants)
        normalized_folds = tuple(self.folds)
        object.__setattr__(self, "variants", normalized_variants)
        object.__setattr__(self, "folds", normalized_folds)

        if not normalized_variants:
            raise ValueError("variants must not be empty.")
        if not normalized_folds:
            raise ValueError("folds must not be empty.")

        variant_ids: set[str] = set()
        for variant in normalized_variants:
            if not isinstance(variant, VariantSpec):
                raise TypeError("variants must contain VariantSpec objects only.")
            if variant.variant_id in variant_ids:
                raise ValueError(f"duplicate variant_id {variant.variant_id!r}.")
            variant_ids.add(variant.variant_id)

        fold_labels: set[str] = set()
        for fold in normalized_folds:
            if not isinstance(fold, FoldSpec):
                raise TypeError("folds must contain FoldSpec objects only.")
            fold_name = label_fold(fold)
            if fold_name in fold_labels:
                raise ValueError(f"duplicate fold label {fold_name!r}.")
            fold_labels.add(fold_name)

        if self.holdout is not None:
            if not isinstance(self.holdout, HoldoutSpec):
                raise TypeError("holdout must be a HoldoutSpec instance.")
            if self.holdout.label in fold_labels:
                raise ValueError("holdout label must not overlap fold labels.")

        if self.deep_dive is not None:
            if not isinstance(self.deep_dive, DeepDiveConfig):
                raise TypeError("deep_dive must be a DeepDiveConfig instance.")
            unknown_variants = sorted(
                set(self.deep_dive.selected_variant_ids) - variant_ids
            )
            if unknown_variants:
                raise ValueError(
                    f"deep_dive references unknown variant_ids {unknown_variants!r}."
                )
            unknown_folds = sorted(set(self.deep_dive.selected_folds) - fold_labels)
            if unknown_folds:
                raise ValueError(
                    f"deep_dive references unknown fold labels {unknown_folds!r}."
                )
            if self.deep_dive.include_holdout and self.holdout is None:
                raise ValueError(
                    "deep_dive cannot include holdout when no holdout is configured."
                )


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    experiment: ExperimentSpec
    run_id: str
    completed_variant_ids: tuple[str, ...] = ()
    completed_fold_labels: tuple[str, ...] = ()
    holdout_executed: bool = False
    artifacts: dict[str, Any] = field(default_factory=dict)
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, ExperimentSpec):
            raise TypeError("experiment must be an ExperimentSpec instance.")
        _require_non_empty("run_id", self.run_id)
        object.__setattr__(
            self,
            "completed_variant_ids",
            _normalize_string_tuple(
                "completed_variant_ids",
                self.completed_variant_ids,
            ),
        )
        object.__setattr__(
            self,
            "completed_fold_labels",
            _normalize_string_tuple(
                "completed_fold_labels",
                self.completed_fold_labels,
            ),
        )
        _require_bool("holdout_executed", self.holdout_executed)
        if not isinstance(self.artifacts, dict):
            raise TypeError("artifacts must be a dict.")
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(self, "artifacts", dict(self.artifacts))

        known_variant_ids = {variant.variant_id for variant in self.experiment.variants}
        unknown_variant_ids = sorted(
            set(self.completed_variant_ids) - known_variant_ids
        )
        if unknown_variant_ids:
            raise ValueError(
                "completed_variant_ids contain unknown variants: "
                f"{unknown_variant_ids!r}."
            )

        known_fold_labels = {label_fold(fold) for fold in self.experiment.folds}
        unknown_fold_labels = sorted(
            set(self.completed_fold_labels) - known_fold_labels
        )
        if unknown_fold_labels:
            raise ValueError(
                "completed_fold_labels contain unknown folds: "
                f"{unknown_fold_labels!r}."
            )
        if self.holdout_executed and self.experiment.holdout is None:
            raise ValueError(
                "holdout_executed cannot be True when experiment.holdout is None."
            )


__all__ = [
    "DeepDiveConfig",
    "ExperimentRunResult",
    "ExperimentSpec",
    "FoldSpec",
    "HoldoutSpec",
    "NodeFamilyDefinition",
    "OutputConfig",
    "ParameterGrid",
    "PruningConfig",
    "SearchConfig",
    "SearchMode",
    "VariantSpec",
    "expand_node_family",
    "expand_parameter_grid",
    "generate_variants",
    "generate_variant_id",
    "label_fold",
    "parameterize_node_contract",
    "serialize_manifest",
]
