from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from itertools import islice
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd

PACKAGE_VERSION = "0.1.0"
DEFAULT_CONTRACT_VERSION = "1.0"

PositionSide = Literal["flat", "long", "short"]
NodeKind = Literal["entry", "exit", "risk"]
ActionType = Literal[
    "enter_long",
    "enter_short",
    "close",
    "hold",
    "partial_exit",
    "scale_in",
    "increase",
    "reduce",
    "set_stop",
]
EntryAction = Literal["none", "enter_long", "enter_short"]
ExitAction = Literal["none", "exit"]
FillRule = Literal["next_bar_open"]
OrderType = Literal["market"]

V1_SUPPORTED_ACTION_TYPES: tuple[ActionType, ...] = (
    "enter_long",
    "enter_short",
    "close",
    "hold",
)
DEFAULT_SUPPORTED_ACTION_TYPES: tuple[ActionType, ...] = (
    "enter_long",
    "enter_short",
    "close",
    "hold",
    "partial_exit",
    "scale_in",
    "increase",
    "reduce",
)
RESERVED_ACTION_TYPES: tuple[ActionType, ...] = ("set_stop",)
KNOWN_ACTION_TYPES: tuple[ActionType, ...] = (
    DEFAULT_SUPPORTED_ACTION_TYPES + RESERVED_ACTION_TYPES
)

V1_SUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "market_orders",
    "next_bar_open_fills",
    "single_position_per_symbol",
    "portfolio_view",
    "simple_fees",
    "simple_slippage",
)
DEFAULT_SUPPORTED_CAPABILITIES: tuple[str, ...] = V1_SUPPORTED_CAPABILITIES + (
    "multiple_action_requests",
    "partial_exit",
    "scale_in",
    "increase_reduce_requests",
    "multiple_entries",
    "multiple_lots_per_symbol",
    "lot_level_accounting",
)
RESERVED_ENGINE_CAPABILITIES: tuple[str, ...] = (
    "set_stop_requests",
    "same_bar_reverse",
    "stop_order",
    "limit_order",
    "bracket_order",
    "target_position_mode",
    "portfolio_allocator",
)


class CompatibilityError(ValueError):
    """Raised when node requirements exceed current engine capabilities."""


class UnsupportedEngineFeatureError(CompatibilityError):
    """Raised when a node requires behavior the current engine does not support."""


class UnsupportedNodeRequirementError(UnsupportedEngineFeatureError):
    """Raised when node capability requirements imply an engine feature request."""


class UnsupportedNodeActionError(UnsupportedEngineFeatureError):
    """Raised when a node emits an action the current engine cannot execute."""


class SetupCompatibilityError(CompatibilityError):
    """Raised when a backtest configuration is incompatible before the event loop starts."""


class NodeOutputValidationError(CompatibilityError):
    """Raised when node outputs violate declared contracts in strict validation mode."""


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    code: str
    message: str
    feature: str | None = None
    requires_engine_change: bool = False


@dataclass(frozen=True, slots=True)
class CompatibilityAudit:
    node_name: str
    node_kind: NodeKind
    engine_contract_version: str
    node_contract_version: str
    supported: bool
    issues: tuple[CompatibilityIssue, ...] = ()
    required_engine_changes: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        if self.supported:
            return (
                f"Node {self.node_name!r} ({self.node_kind}) is compatible with "
                f"engine contract version {self.engine_contract_version!r}."
            )

        parts = [issue.message for issue in self.issues]
        message = (
            f"Node {self.node_name!r} ({self.node_kind}) is incompatible with the "
            f"active engine capabilities: {'; '.join(parts)}"
        )
        if self.required_engine_changes:
            message += (
                f" Required engine changes: {list(self.required_engine_changes)!r}. "
                "This is an engine feature request, not just a new node."
            )
        return message

    def raise_for_errors(self) -> None:
        if self.supported:
            return

        if any(issue.code == "unsupported_action" for issue in self.issues):
            raise UnsupportedNodeActionError(self.summary)
        if any(issue.requires_engine_change for issue in self.issues):
            raise UnsupportedNodeRequirementError(self.summary)
        raise CompatibilityError(self.summary)


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


def _normalize_tuple(name: str, value: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    normalized = tuple(value)
    for item in normalized:
        _require_non_empty(name, item)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")
    return normalized


def _normalize_action_types(
    value: tuple[ActionType, ...] | list[ActionType],
) -> tuple[ActionType, ...]:
    normalized = tuple(value)
    supported = set(KNOWN_ACTION_TYPES)
    if not normalized:
        raise ValueError("emitted_action_types must not be empty.")
    if any(action not in supported for action in normalized):
        raise ValueError(f"unsupported action type in {normalized!r}.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("emitted_action_types must not contain duplicates.")
    return normalized


def _allowed_actions_for_kind(kind: NodeKind) -> tuple[ActionType, ...]:
    if kind == "entry":
        return ("enter_long", "enter_short", "hold", "scale_in", "increase")
    if kind == "exit":
        return ("close", "hold", "partial_exit", "reduce", "set_stop")
    return KNOWN_ACTION_TYPES


def _legacy_default_actions(kind: NodeKind) -> tuple[ActionType, ...]:
    if kind == "entry":
        return ("enter_long", "enter_short", "hold")
    if kind == "exit":
        return ("close", "hold")
    return ("enter_long", "enter_short", "close", "hold")


def _engine_change_marker(feature: str) -> str:
    return feature if feature.startswith("action:") else f"capability:{feature}"


@dataclass(frozen=True, slots=True)
class ActionRequest:
    action_type: ActionType = "hold"
    units: float | None = None
    stop_price: float | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    position_id: str | None = None
    parent_position_id: str | None = None
    lot_id: str | None = None
    entry_tag: str | None = None
    exit_tag: str | None = None
    risk_tag: str | None = None
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.action_type not in KNOWN_ACTION_TYPES:
            raise ValueError(f"unsupported action_type: {self.action_type!r}.")
        _require_contract_version("contract_version", self.contract_version)
        if self.units is not None:
            _require_number("units", self.units, non_negative=True)
        if self.stop_price is not None:
            _require_number("stop_price", self.stop_price, positive=True)
        if self.action_type == "hold":
            if self.units not in (None, 0.0):
                raise ValueError("hold actions cannot request non-zero units.")
            if self.stop_price is not None:
                raise ValueError("hold actions cannot set stop prices.")
        elif self.action_type == "close":
            if self.units not in (None, 0.0):
                raise ValueError("close actions cannot request partial units.")
            if self.stop_price is not None:
                raise ValueError("close actions cannot set stop prices.")
        elif self.action_type in {"partial_exit", "reduce"}:
            if self.units is None or float(self.units) <= 0.0:
                raise ValueError(
                    "partial reduction actions must request positive units."
                )
            if self.stop_price is not None:
                raise ValueError(
                    "partial reduction actions cannot set stop prices."
                )
        elif self.action_type == "set_stop":
            if self.stop_price is None:
                raise ValueError("set_stop actions require stop_price.")
            if self.units not in (None, 0.0):
                raise ValueError("set_stop actions cannot request trade units.")
        else:
            if self.units is not None and float(self.units) <= 0.0:
                raise ValueError(
                    "entry-style actions must request positive units when provided."
                )
            if self.stop_price is not None:
                raise ValueError("entry-style actions cannot set stop prices.")
        if not isinstance(self.reason, str):
            raise TypeError(f"reason must be a string, got {type(self.reason).__name__}.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_active(self) -> bool:
        return self.action_type != "hold"

    @property
    def is_entry(self) -> bool:
        return self.action_type in {"enter_long", "enter_short", "scale_in", "increase"}

    @property
    def is_close(self) -> bool:
        return self.action_type == "close"

    @property
    def is_exit(self) -> bool:
        return self.action_type in {"close", "partial_exit", "reduce"}

    @property
    def is_partial_exit(self) -> bool:
        return self.action_type == "partial_exit"

    @property
    def is_scale_in(self) -> bool:
        return self.action_type == "scale_in"

    @property
    def is_increase(self) -> bool:
        return self.action_type == "increase"

    @property
    def is_reduce(self) -> bool:
        return self.action_type == "reduce"

    @property
    def is_stop_request(self) -> bool:
        return self.action_type == "set_stop"

    @property
    def is_reserved_future_action(self) -> bool:
        return self.action_type in RESERVED_ACTION_TYPES

    @property
    def side(self) -> Literal["long", "short"] | None:
        if self.action_type == "enter_long":
            return "long"
        if self.action_type == "enter_short":
            return "short"
        return None


@dataclass(frozen=True, slots=True)
class ActionBatch:
    requests: tuple[ActionRequest, ...] = ()
    contract_version: str = DEFAULT_CONTRACT_VERSION
    _active_requests: tuple[ActionRequest, ...] = field(
        init=False,
        repr=False,
        default=(),
    )
    _primary_request: ActionRequest = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _require_contract_version("contract_version", self.contract_version)
        normalized_requests = tuple(self.requests)
        for request in normalized_requests:
            if not isinstance(request, ActionRequest):
                raise TypeError("requests must contain ActionRequest objects only.")
            if request.contract_version != self.contract_version:
                raise ValueError(
                    "all requests in an ActionBatch must share the batch contract_version."
                )
        active_requests = tuple(
            request for request in normalized_requests if request.is_active
        )
        object.__setattr__(self, "requests", normalized_requests)
        object.__setattr__(self, "_active_requests", active_requests)
        object.__setattr__(
            self,
            "_primary_request",
            active_requests[0]
            if active_requests
            else ActionRequest(contract_version=self.contract_version),
        )

    @property
    def active_requests(self) -> tuple[ActionRequest, ...]:
        return self._active_requests

    @property
    def is_active(self) -> bool:
        return bool(self._active_requests)

    @property
    def primary_request(self) -> ActionRequest:
        return self._primary_request

    @property
    def action_types(self) -> tuple[ActionType, ...]:
        return tuple(request.action_type for request in self._active_requests)

    @property
    def action_type(self) -> ActionType:
        return self.primary_request.action_type

    @property
    def is_entry(self) -> bool:
        return any(request.is_entry for request in self.active_requests)

    @property
    def is_close(self) -> bool:
        return any(request.is_close for request in self.active_requests)

    @property
    def is_exit(self) -> bool:
        return any(request.is_exit for request in self.active_requests)

    @property
    def is_partial_exit(self) -> bool:
        return any(request.is_partial_exit for request in self.active_requests)

    @property
    def is_scale_in(self) -> bool:
        return any(request.is_scale_in for request in self.active_requests)

    @property
    def is_reduce(self) -> bool:
        return any(request.is_reduce for request in self.active_requests)

    @property
    def is_stop_request(self) -> bool:
        return any(request.is_stop_request for request in self.active_requests)

    @property
    def reason(self) -> str:
        return self.primary_request.reason

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.primary_request.metadata)

    @property
    def side(self) -> Literal["long", "short"] | None:
        return self.primary_request.side

    def with_request(self, request: ActionRequest) -> "ActionBatch":
        if not isinstance(request, ActionRequest):
            raise TypeError("request must be an ActionRequest instance.")
        return ActionBatch(
            requests=self.requests + (request,),
            contract_version=self.contract_version,
        )


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    contract_version: str = DEFAULT_CONTRACT_VERSION
    supported_action_types: tuple[ActionType, ...] = DEFAULT_SUPPORTED_ACTION_TYPES
    supported_capabilities: tuple[str, ...] = DEFAULT_SUPPORTED_CAPABILITIES
    supports_multiple_entries: bool = True
    supports_partial_exits: bool = True
    supports_richer_risk_actions: bool = False
    supports_lot_level_accounting: bool = True
    supports_metric_dependency_checks: bool = False
    supports_node_capability_checks: bool = True

    def __post_init__(self) -> None:
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(
            self,
            "supported_action_types",
            _normalize_action_types(self.supported_action_types),
        )
        object.__setattr__(
            self,
            "supported_capabilities",
            _normalize_tuple("supported_capabilities", self.supported_capabilities),
        )
        for field_name in (
            "supports_multiple_entries",
            "supports_partial_exits",
            "supports_richer_risk_actions",
            "supports_lot_level_accounting",
            "supports_metric_dependency_checks",
            "supports_node_capability_checks",
        ):
            _require_bool(field_name, getattr(self, field_name))

    def capability_set(self) -> frozenset[str]:
        names = set(self.supported_capabilities)
        if self.supports_multiple_entries:
            names.add("multiple_entries")
        if self.supports_partial_exits:
            names.add("partial_exits")
        if self.supports_richer_risk_actions:
            names.add("richer_risk_actions")
        if self.supports_lot_level_accounting:
            names.add("lot_level_accounting")
        if self.supports_metric_dependency_checks:
            names.add("metric_dependency_checks")
        if self.supports_node_capability_checks:
            names.add("node_capability_checks")
        names.update(f"action:{action}" for action in self.supported_action_types)
        return frozenset(names)


@dataclass(frozen=True, slots=True)
class NodeSpec:
    name: str
    kind: NodeKind
    version: str = "0.1.0"
    contract_version: str = DEFAULT_CONTRACT_VERSION
    required_capabilities: tuple[str, ...] = ()
    emitted_action_types: tuple[ActionType, ...] = ("hold",)
    required_history: int = 1
    requires_portfolio_view: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        _require_non_empty("name", self.name)
        if self.kind not in {"entry", "exit", "risk"}:
            raise ValueError(f"unsupported node kind: {self.kind!r}.")
        _require_non_empty("version", self.version)
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(
            self,
            "required_capabilities",
            _normalize_tuple("required_capabilities", self.required_capabilities),
        )
        normalized_actions = _normalize_action_types(self.emitted_action_types)
        allowed = set(_allowed_actions_for_kind(self.kind))
        if not set(normalized_actions).issubset(allowed):
            raise ValueError(
                f"{self.kind} nodes may only emit {tuple(sorted(allowed))}, "
                f"got {normalized_actions!r}."
            )
        object.__setattr__(self, "emitted_action_types", normalized_actions)
        _require_int("required_history", self.required_history, minimum=1)
        _require_bool("requires_portfolio_view", self.requires_portfolio_view)
        if not isinstance(self.description, str):
            raise TypeError(
                f"description must be a string, got {type(self.description).__name__}."
            )


@dataclass(frozen=True, slots=True)
class NodeContract:
    spec: NodeSpec
    input_contract_version: str = DEFAULT_CONTRACT_VERSION
    output_contract_version: str = DEFAULT_CONTRACT_VERSION
    metric_dependencies: tuple[str, ...] = ()
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, NodeSpec):
            raise TypeError("spec must be a NodeSpec instance.")
        _require_contract_version(
            "input_contract_version", self.input_contract_version
        )
        _require_contract_version(
            "output_contract_version", self.output_contract_version
        )
        object.__setattr__(
            self,
            "metric_dependencies",
            _normalize_tuple("metric_dependencies", self.metric_dependencies),
        )
        if not isinstance(self.manifest, dict):
            raise TypeError("manifest must be a dict.")
        normalized_manifest = dict(self.manifest)
        module_name = normalized_manifest.get("module")
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError("manifest must define a non-empty 'module' string.")
        parameters = normalized_manifest.get("parameters")
        if not isinstance(parameters, dict):
            raise TypeError("manifest must define 'parameters' as a dict.")
        object.__setattr__(self, "manifest", normalized_manifest)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def kind(self) -> NodeKind:
        return self.spec.kind

    def validate_compatibility(self, capabilities: EngineCapabilities) -> None:
        audit_node_compatibility(self, capabilities).raise_for_errors()


def audit_node_compatibility(
    node_contract: NodeContract,
    capabilities: EngineCapabilities,
) -> CompatibilityAudit:
    if not isinstance(node_contract, NodeContract):
        raise TypeError("node_contract must be a NodeContract instance.")
    if not isinstance(capabilities, EngineCapabilities):
        raise TypeError("capabilities must be an EngineCapabilities instance.")

    issues: list[CompatibilityIssue] = []
    required_engine_changes: set[str] = set()

    if node_contract.input_contract_version != capabilities.contract_version:
        issues.append(
            CompatibilityIssue(
                code="input_contract_version_mismatch",
                feature="input_contract_version",
                message=(
                    "node input contract version "
                    f"{node_contract.input_contract_version!r} does not match "
                    f"engine contract version {capabilities.contract_version!r}."
                ),
                requires_engine_change=True,
            )
        )
        required_engine_changes.add(
            _engine_change_marker("input_contract_version")
        )

    if node_contract.output_contract_version != capabilities.contract_version:
        issues.append(
            CompatibilityIssue(
                code="output_contract_version_mismatch",
                feature="output_contract_version",
                message=(
                    "node output contract version "
                    f"{node_contract.output_contract_version!r} does not match "
                    f"engine contract version {capabilities.contract_version!r}."
                ),
                requires_engine_change=True,
            )
        )
        required_engine_changes.add(
            _engine_change_marker("output_contract_version")
        )

    if node_contract.spec.contract_version != capabilities.contract_version:
        issues.append(
            CompatibilityIssue(
                code="spec_contract_version_mismatch",
                feature="spec.contract_version",
                message=(
                    "node spec contract version "
                    f"{node_contract.spec.contract_version!r} does not match "
                    f"engine contract version {capabilities.contract_version!r}."
                ),
                requires_engine_change=True,
            )
        )
        required_engine_changes.add(_engine_change_marker("spec.contract_version"))

    capability_set = capabilities.capability_set()
    missing_capabilities = set(node_contract.spec.required_capabilities) - capability_set
    if node_contract.spec.requires_portfolio_view and "portfolio_view" not in capability_set:
        missing_capabilities.add("portfolio_view")
    if node_contract.metric_dependencies and not capabilities.supports_metric_dependency_checks:
        missing_capabilities.add("metric_dependency_checks")

    for capability in sorted(missing_capabilities):
        issues.append(
            CompatibilityIssue(
                code="missing_capability",
                feature=capability,
                message=(
                    f"node requires capability {capability!r}, but the active engine "
                    "capabilities do not declare it."
                ),
                requires_engine_change=True,
            )
        )
        required_engine_changes.add(_engine_change_marker(capability))

    unsupported_actions = sorted(
        set(node_contract.spec.emitted_action_types) - set(capabilities.supported_action_types)
    )
    for action in unsupported_actions:
        action_suffix = (
            " The action is reserved for a later engine phase."
            if action in RESERVED_ACTION_TYPES
            else ""
        )
        issues.append(
            CompatibilityIssue(
                code="unsupported_action",
                feature=action,
                message=(
                    f"node can emit action {action!r}, but the active engine only "
                    f"supports actions {list(capabilities.supported_action_types)!r}."
                    f"{action_suffix}"
                ),
                requires_engine_change=True,
            )
        )
        required_engine_changes.add(_engine_change_marker(f"action:{action}"))

    return CompatibilityAudit(
        node_name=node_contract.name,
        node_kind=node_contract.kind,
        engine_contract_version=capabilities.contract_version,
        node_contract_version=node_contract.spec.contract_version,
        supported=not issues,
        issues=tuple(issues),
        required_engine_changes=tuple(sorted(required_engine_changes)),
    )


def validate_node_compatibility(
    node_contract: NodeContract,
    capabilities: EngineCapabilities,
) -> None:
    audit_node_compatibility(node_contract, capabilities).raise_for_errors()


def legacy_node_contract(name: str, kind: NodeKind) -> NodeContract:
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind=kind,
            version="legacy",
            contract_version=DEFAULT_CONTRACT_VERSION,
            emitted_action_types=_legacy_default_actions(kind),
        ),
        manifest={
            "legacy_inferred": True,
            "module": "trading_lab.legacy",
            "parameters": {},
        },
    )


@dataclass(frozen=True, slots=True)
class Bar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_datetime("timestamp", self.timestamp)
        _require_non_empty("symbol", self.symbol)
        open_price = _require_number("open", self.open, positive=True)
        high_price = _require_number("high", self.high, positive=True)
        low_price = _require_number("low", self.low, positive=True)
        close_price = _require_number("close", self.close, positive=True)
        _require_number("volume", self.volume, non_negative=True)
        _require_contract_version("contract_version", self.contract_version)

        if low_price > high_price:
            raise ValueError("low cannot exceed high.")
        if not low_price <= open_price <= high_price:
            raise ValueError("open must fall within [low, high].")
        if not low_price <= close_price <= high_price:
            raise ValueError("close must fall within [low, high].")


@dataclass(frozen=True, slots=True, eq=False)
class BarHistorySeries(Sequence[Bar]):
    bars: tuple[Bar, ...]
    symbol: str = field(init=False)
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        normalized_bars = tuple(self.bars)
        object.__setattr__(self, "bars", normalized_bars)
        _require_contract_version("contract_version", self.contract_version)
        if not normalized_bars:
            raise ValueError("bars must not be empty.")

        first_symbol: str | None = None
        for bar in normalized_bars:
            if not isinstance(bar, Bar):
                raise TypeError("bars must contain Bar objects only.")
            if first_symbol is None:
                first_symbol = bar.symbol
            elif bar.symbol != first_symbol:
                raise ValueError("all bars in a BarHistorySeries must share the same symbol.")

        assert first_symbol is not None
        object.__setattr__(self, "symbol", first_symbol)

    def __getitem__(self, index: int | slice) -> Bar | tuple[Bar, ...]:
        return self.bars[index]

    def __iter__(self) -> Iterator[Bar]:
        return iter(self.bars)

    def __len__(self) -> int:
        return len(self.bars)

    def window(self, stop: int) -> "BarHistoryWindow":
        return BarHistoryWindow(
            series=self,
            stop=stop,
            contract_version=self.contract_version,
        )

    def index_of_timestamp(self, timestamp: datetime) -> int | None:
        return _timestamp_index_map(self).get(timestamp)


@dataclass(frozen=True, slots=True)
class BarHistoryWindow(Sequence[Bar]):
    series: BarHistorySeries
    stop: int
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.series, BarHistorySeries):
            raise TypeError("series must be a BarHistorySeries instance.")
        _require_int("stop", self.stop, minimum=1)
        _require_contract_version("contract_version", self.contract_version)
        if self.contract_version != self.series.contract_version:
            raise ValueError("contract_version must match series.contract_version.")
        if self.stop > len(self.series):
            raise ValueError("stop must not exceed the length of series.")

    @property
    def symbol(self) -> str:
        return self.series.symbol

    @property
    def end_index(self) -> int:
        return self.stop - 1

    @property
    def bars(self) -> tuple[Bar, ...]:
        return self.series.bars

    def __getitem__(self, index: int | slice) -> Bar | tuple[Bar, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.stop)
            return self.series.bars[start:stop:step]

        normalized_index = index if index >= 0 else self.stop + index
        if normalized_index < 0 or normalized_index >= self.stop:
            raise IndexError("history index out of range.")
        return self.series.bars[normalized_index]

    def __iter__(self) -> Iterator[Bar]:
        return islice(self.series.bars, self.stop)

    def __len__(self) -> int:
        return self.stop

    def index_of_timestamp(self, timestamp: datetime) -> int | None:
        index = self.series.index_of_timestamp(timestamp)
        if index is None or index >= self.stop:
            return None
        return index


@lru_cache(maxsize=128)
def _timestamp_index_map(series: BarHistorySeries) -> dict[datetime, int]:
    return {bar.timestamp: index for index, bar in enumerate(series.bars)}


@dataclass(frozen=True, slots=True)
class InstrumentMeta:
    symbol: str
    price_increment: float
    quantity_increment: float
    contract_multiplier: float = 1.0
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("symbol", self.symbol)
        _require_number("price_increment", self.price_increment, positive=True)
        _require_number("quantity_increment", self.quantity_increment, positive=True)
        _require_number("contract_multiplier", self.contract_multiplier, positive=True)
        _require_contract_version("contract_version", self.contract_version)


@dataclass(frozen=True, slots=True)
class CostAssumptions:
    fee_rate: float = 0.0
    fee_per_unit: float = 0.0
    slippage_bps: float = 0.0
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_number("fee_rate", self.fee_rate, non_negative=True)
        _require_number("fee_per_unit", self.fee_per_unit, non_negative=True)
        _require_number("slippage_bps", self.slippage_bps, non_negative=True)
        _require_contract_version("contract_version", self.contract_version)


@dataclass(frozen=True, slots=True)
class PositionState:
    symbol: str
    side: PositionSide = "flat"
    quantity: float = 0.0
    entry_price: float | None = None
    entry_time: datetime | None = None
    position_id: str | None = None
    parent_position_id: str | None = None
    lot_id: str | None = None
    entry_tag: str | None = None
    exit_tag: str | None = None
    risk_tag: str | None = None
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("symbol", self.symbol)
        quantity = _require_number("quantity", self.quantity, non_negative=True)
        _require_contract_version("contract_version", self.contract_version)

        if self.side == "flat":
            if quantity != 0.0:
                raise ValueError("flat positions must have zero quantity.")
            if self.entry_price is not None or self.entry_time is not None:
                raise ValueError("flat positions cannot carry entry state.")
            return

        if self.side not in {"long", "short"}:
            raise ValueError(f"unsupported side: {self.side!r}.")
        if quantity <= 0.0:
            raise ValueError("open positions must have positive quantity.")
        if self.entry_price is None or self.entry_time is None:
            raise ValueError("open positions require entry_price and entry_time.")

        _require_number("entry_price", self.entry_price, positive=True)
        _require_datetime("entry_time", self.entry_time)

    @property
    def is_flat(self) -> bool:
        return self.side == "flat"

    @property
    def is_long(self) -> bool:
        return self.side == "long"

    @property
    def is_short(self) -> bool:
        return self.side == "short"


@dataclass(frozen=True, slots=True)
class PortfolioState:
    cash: float
    equity: float
    positions: tuple[PositionState, ...] = ()
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_number("cash", self.cash)
        _require_number("equity", self.equity)
        _require_contract_version("contract_version", self.contract_version)

        normalized_positions = tuple(self.positions)
        object.__setattr__(self, "positions", normalized_positions)

        seen_symbols: set[str] = set()
        for position in normalized_positions:
            if not isinstance(position, PositionState):
                raise TypeError("positions must contain PositionState objects only.")
            if position.symbol in seen_symbols:
                raise ValueError(f"duplicate position for symbol {position.symbol!r}.")
            seen_symbols.add(position.symbol)

    def get_position(self, symbol: str) -> PositionState | None:
        for position in self.positions:
            if position.symbol == symbol:
                return position
        return None


@dataclass(frozen=True, slots=True)
class SessionInfo:
    bar_index: int
    bars_total: int
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_int("bar_index", self.bar_index, minimum=0)
        _require_int("bars_total", self.bars_total, minimum=1)
        _require_contract_version("contract_version", self.contract_version)
        if self.bar_index >= self.bars_total:
            raise ValueError("bar_index must be less than bars_total.")

    @property
    def is_first_bar(self) -> bool:
        return self.bar_index == 0

    @property
    def is_last_bar(self) -> bool:
        return self.bar_index == self.bars_total - 1


@dataclass(frozen=True, slots=True)
class DecisionContext:
    bar: Bar
    history: Sequence[Bar]
    instrument: InstrumentMeta
    costs: CostAssumptions
    position: PositionState
    portfolio: PortfolioState
    session: SessionInfo
    engine_capabilities: EngineCapabilities = field(default_factory=EngineCapabilities)
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.bar, Bar):
            raise TypeError(f"bar must be a Bar, got {type(self.bar).__name__}.")

        if isinstance(self.history, BarHistorySeries):
            normalized_history: Sequence[Bar] = self.history.window(len(self.history))
        elif isinstance(self.history, BarHistoryWindow):
            normalized_history = self.history
        else:
            normalized_history = tuple(self.history)
        object.__setattr__(self, "history", normalized_history)

        if not isinstance(self.instrument, InstrumentMeta):
            raise TypeError("instrument must be an InstrumentMeta instance.")
        if not isinstance(self.costs, CostAssumptions):
            raise TypeError("costs must be a CostAssumptions instance.")
        if not isinstance(self.position, PositionState):
            raise TypeError("position must be a PositionState instance.")
        if not isinstance(self.portfolio, PortfolioState):
            raise TypeError("portfolio must be a PortfolioState instance.")
        if not isinstance(self.session, SessionInfo):
            raise TypeError("session must be a SessionInfo instance.")
        if not isinstance(self.engine_capabilities, EngineCapabilities):
            raise TypeError("engine_capabilities must be an EngineCapabilities instance.")
        _require_contract_version("contract_version", self.contract_version)

        if self.bar.symbol != self.instrument.symbol:
            raise ValueError("bar.symbol must match instrument.symbol.")
        if self.position.symbol != self.instrument.symbol:
            raise ValueError("position.symbol must match instrument.symbol.")

        if isinstance(normalized_history, BarHistoryWindow):
            if normalized_history.symbol != self.instrument.symbol:
                raise ValueError("all history bars must match instrument.symbol.")
        else:
            for history_bar in normalized_history:
                if not isinstance(history_bar, Bar):
                    raise TypeError("history must contain Bar objects only.")
                if history_bar.symbol != self.instrument.symbol:
                    raise ValueError("all history bars must match instrument.symbol.")

        if len(normalized_history) > 0 and normalized_history[-1] != self.bar:
            raise ValueError("history must end with the current bar.")

        portfolio_position = self.portfolio.get_position(self.instrument.symbol)
        if portfolio_position is not None and portfolio_position != self.position:
            raise ValueError(
                "portfolio position for instrument.symbol must match position."
            )


@dataclass(frozen=True, slots=True)
class EntryIntent:
    action: EntryAction = "none"
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.action not in {"none", "enter_long", "enter_short"}:
            raise ValueError(f"unsupported entry action: {self.action!r}.")
        _require_contract_version("contract_version", self.contract_version)
        if not isinstance(self.reason, str):
            raise TypeError(f"reason must be a string, got {type(self.reason).__name__}.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.action != "none" and not self.reason.strip():
            raise ValueError("reason must be provided when the intent is active.")

    @property
    def is_active(self) -> bool:
        return self.action != "none"

    @property
    def side(self) -> Literal["long", "short"] | None:
        if self.action == "enter_long":
            return "long"
        if self.action == "enter_short":
            return "short"
        return None

    def as_action_request(self) -> ActionRequest:
        return ActionRequest(
            action_type=self.action if self.action != "none" else "hold",
            reason=self.reason,
            metadata=self.metadata,
            contract_version=self.contract_version,
        )


@dataclass(frozen=True, slots=True)
class ExitIntent:
    action: ExitAction = "none"
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.action not in {"none", "exit"}:
            raise ValueError(f"unsupported exit action: {self.action!r}.")
        _require_contract_version("contract_version", self.contract_version)
        if not isinstance(self.reason, str):
            raise TypeError(f"reason must be a string, got {type(self.reason).__name__}.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict.")
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.action == "exit" and not self.reason.strip():
            raise ValueError("reason must be provided when the intent is active.")

    @property
    def is_active(self) -> bool:
        return self.action == "exit"

    def as_action_request(self) -> ActionRequest:
        return ActionRequest(
            action_type="close" if self.action == "exit" else "hold",
            reason=self.reason,
            metadata=self.metadata,
            contract_version=self.contract_version,
        )


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allow_entry: bool = False
    entry_quantity: float = 0.0
    allow_exit: bool = True
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_bool("allow_entry", self.allow_entry)
        _require_bool("allow_exit", self.allow_exit)
        quantity = _require_number(
            "entry_quantity", self.entry_quantity, non_negative=True
        )
        _require_contract_version("contract_version", self.contract_version)
        if not isinstance(self.reason, str):
            raise TypeError(f"reason must be a string, got {type(self.reason).__name__}.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict.")
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.allow_entry and quantity <= 0.0:
            raise ValueError("allow_entry=True requires entry_quantity > 0.")
        if not self.allow_entry and quantity != 0.0:
            raise ValueError("allow_entry=False requires entry_quantity == 0.")


@dataclass(frozen=True, slots=True)
class BacktestSpec:
    name: str
    instrument: InstrumentMeta
    entry_node: str
    exit_node: str
    risk_node: str
    initial_cash: float
    costs: CostAssumptions = field(default_factory=CostAssumptions)
    allow_short: bool = True
    fill_rule: FillRule = "next_bar_open"
    order_type: OrderType = "market"
    strict_node_output_validation: bool = False
    random_seed: int | None = None
    engine_capabilities: EngineCapabilities = field(default_factory=EngineCapabilities)
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_non_empty("name", self.name)
        if not isinstance(self.instrument, InstrumentMeta):
            raise TypeError("instrument must be an InstrumentMeta instance.")
        _require_non_empty("entry_node", self.entry_node)
        _require_non_empty("exit_node", self.exit_node)
        _require_non_empty("risk_node", self.risk_node)
        _require_number("initial_cash", self.initial_cash, positive=True)
        if not isinstance(self.costs, CostAssumptions):
            raise TypeError("costs must be a CostAssumptions instance.")
        _require_bool("allow_short", self.allow_short)
        _require_bool(
            "strict_node_output_validation", self.strict_node_output_validation
        )
        if self.random_seed is not None:
            _require_int("random_seed", self.random_seed)
        if not isinstance(self.engine_capabilities, EngineCapabilities):
            raise TypeError("engine_capabilities must be an EngineCapabilities instance.")
        _require_contract_version("contract_version", self.contract_version)
        if self.fill_rule != "next_bar_open":
            raise ValueError("fill_rule must be 'next_bar_open' in v1.")
        if self.order_type != "market":
            raise ValueError("order_type must be 'market' in v1.")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    spec: BacktestSpec
    decision_log: pd.DataFrame
    order_log: pd.DataFrame
    fill_log: pd.DataFrame
    trade_ledger: pd.DataFrame
    equity_curve: pd.DataFrame
    artifacts: dict[str, Any] = field(default_factory=dict)
    contract_version: str = DEFAULT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.spec, BacktestSpec):
            raise TypeError("spec must be a BacktestSpec instance.")
        for field_name in (
            "decision_log",
            "order_log",
            "fill_log",
            "trade_ledger",
            "equity_curve",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"{field_name} must be a pandas DataFrame.")
        if not isinstance(self.artifacts, dict):
            raise TypeError("artifacts must be a dict.")
        _require_contract_version("contract_version", self.contract_version)
        object.__setattr__(self, "artifacts", dict(self.artifacts))


@runtime_checkable
class EntryNode(Protocol):
    def __call__(self, ctx: DecisionContext) -> ActionRequest | ActionBatch:
        ...


@runtime_checkable
class ExitNode(Protocol):
    def __call__(self, ctx: DecisionContext) -> ActionRequest | ActionBatch:
        ...


@runtime_checkable
class RiskNode(Protocol):
    def __call__(
        self,
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> ActionRequest | ActionBatch | RiskDecision:
        ...


__all__ = [
    "ActionBatch",
    "ActionRequest",
    "ActionType",
    "BacktestResult",
    "BacktestSpec",
    "Bar",
    "BarHistorySeries",
    "BarHistoryWindow",
    "CompatibilityAudit",
    "CompatibilityError",
    "CompatibilityIssue",
    "CostAssumptions",
    "DEFAULT_CONTRACT_VERSION",
    "DecisionContext",
    "EngineCapabilities",
    "EntryAction",
    "EntryIntent",
    "EntryNode",
    "ExitAction",
    "ExitIntent",
    "ExitNode",
    "FillRule",
    "InstrumentMeta",
    "KNOWN_ACTION_TYPES",
    "NodeContract",
    "NodeKind",
    "NodeSpec",
    "OrderType",
    "PACKAGE_VERSION",
    "PortfolioState",
    "PositionSide",
    "PositionState",
    "RESERVED_ACTION_TYPES",
    "RESERVED_ENGINE_CAPABILITIES",
    "RiskDecision",
    "RiskNode",
    "SessionInfo",
    "SetupCompatibilityError",
    "UnsupportedEngineFeatureError",
    "UnsupportedNodeActionError",
    "UnsupportedNodeRequirementError",
    "V1_SUPPORTED_ACTION_TYPES",
    "V1_SUPPORTED_CAPABILITIES",
    "NodeOutputValidationError",
    "audit_node_compatibility",
    "DEFAULT_SUPPORTED_ACTION_TYPES",
    "DEFAULT_SUPPORTED_CAPABILITIES",
    "legacy_node_contract",
    "validate_node_compatibility",
]
