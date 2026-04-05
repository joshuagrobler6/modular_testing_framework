from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from collections.abc import Callable
from typing import Literal, TypeVar, overload

from trading_lab.contracts import (
    CompatibilityAudit,
    EngineCapabilities,
    EntryNode,
    ExitNode,
    NodeContract,
    NodeKind,
    NodeSpec,
    RiskNode,
    CompatibilityError,
    audit_node_compatibility,
    validate_node_compatibility,
)

NodeCallable = EntryNode | ExitNode | RiskNode
NodeManifest = NodeContract | NodeSpec
NodeT = TypeVar("NodeT", bound=Callable[..., object])

_EXPECTED_SIGNATURES: dict[NodeKind, tuple[str, ...]] = {
    "entry": ("ctx",),
    "exit": ("ctx",),
    "risk": ("ctx", "entry_intent", "exit_intent"),
}


def _require_kind(kind: str) -> NodeKind:
    if kind not in _EXPECTED_SIGNATURES:
        raise ValueError(f"unsupported node kind: {kind!r}.")
    return kind


def _require_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError(f"name must be a string, got {type(name).__name__}.")
    if not name.strip():
        raise ValueError("name must be a non-empty string.")


def _validate_callable_signature(kind: NodeKind, node: Callable[..., object]) -> None:
    try:
        signature = inspect.signature(node)
    except (TypeError, ValueError) as exc:
        raise TypeError("node must expose an inspectable callable signature.") from exc

    parameters = tuple(signature.parameters.values())
    expected_names = _EXPECTED_SIGNATURES[kind]

    if len(parameters) != len(expected_names):
        raise TypeError(
            f"{kind} nodes must accept exactly {len(expected_names)} arguments: "
            f"{expected_names}."
        )

    actual_names = tuple(parameter.name for parameter in parameters)
    if actual_names != expected_names:
        raise TypeError(
            f"{kind} nodes must use parameters {expected_names}, got {actual_names}."
        )

    for parameter in parameters:
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(
                f"{kind} nodes cannot use variadic or keyword-only parameters."
            )
        if parameter.default is not inspect.Parameter.empty:
            raise TypeError(f"{kind} nodes cannot define defaulted parameters.")


def _normalize_contract(
    *,
    kind: NodeKind,
    name: str,
    contract: NodeManifest,
) -> NodeContract:
    if isinstance(contract, NodeSpec):
        normalized = NodeContract(spec=contract)
    elif isinstance(contract, NodeContract):
        normalized = contract
    else:
        raise TypeError(
            "node registration requires an explicit NodeContract or NodeSpec manifest."
        )

    if normalized.kind != kind:
        raise ValueError(
            f"node contract kind {normalized.kind!r} does not match registration "
            f"kind {kind!r}."
        )
    if normalized.name != name:
        raise ValueError(
            f"node contract name {normalized.name!r} does not match registration "
            f"name {name!r}."
        )
    return normalized


@dataclass(frozen=True, slots=True)
class RegisteredNode:
    kind: NodeKind
    name: str
    node: NodeCallable
    contract: NodeContract


class NodeRegistry:
    def __init__(
        self,
        *,
        autoload_package: str | None = None,
        engine_capabilities: EngineCapabilities | None = None,
        enforce_compatibility: bool = True,
    ) -> None:
        self._nodes: dict[NodeKind, dict[str, RegisteredNode]] = {
            "entry": {},
            "exit": {},
            "risk": {},
        }
        self._autoload_package = autoload_package
        self._autoload_attempted = False
        self._engine_capabilities = engine_capabilities or EngineCapabilities()
        self._enforce_compatibility = enforce_compatibility

    @property
    def engine_capabilities(self) -> EngineCapabilities:
        return self._engine_capabilities

    def _autoload(self) -> None:
        if self._autoload_attempted or self._autoload_package is None:
            return

        self._autoload_attempted = True
        package = import_module(self._autoload_package)
        load_all = getattr(package, "load_all_nodes", None)
        if callable(load_all):
            load_all()

    def _validate_registration_compatibility(self, contract: NodeContract) -> None:
        if not self._enforce_compatibility:
            return
        validate_node_compatibility(contract, self._engine_capabilities)

    def register(
        self,
        kind: NodeKind,
        name: str,
        node: NodeT,
        contract: NodeManifest,
    ) -> NodeT:
        resolved_kind = _require_kind(kind)
        _require_name(name)
        if not callable(node):
            raise TypeError("node must be callable.")
        _validate_callable_signature(resolved_kind, node)

        normalized_contract = _normalize_contract(
            kind=resolved_kind,
            name=name,
            contract=contract,
        )
        self._validate_registration_compatibility(normalized_contract)

        bucket = self._nodes[resolved_kind]
        if name in bucket:
            raise ValueError(f"{resolved_kind} node {name!r} is already registered.")

        bucket[name] = RegisteredNode(
            kind=resolved_kind,
            name=name,
            node=node,
            contract=normalized_contract,
        )
        return node

    def entry(
        self,
        name: str,
        *,
        contract: NodeManifest,
    ) -> Callable[[NodeT], NodeT]:
        def decorator(node: NodeT) -> NodeT:
            return self.register("entry", name, node, contract)

        return decorator

    def exit(
        self,
        name: str,
        *,
        contract: NodeManifest,
    ) -> Callable[[NodeT], NodeT]:
        def decorator(node: NodeT) -> NodeT:
            return self.register("exit", name, node, contract)

        return decorator

    def risk(
        self,
        name: str,
        *,
        contract: NodeManifest,
    ) -> Callable[[NodeT], NodeT]:
        def decorator(node: NodeT) -> NodeT:
            return self.register("risk", name, node, contract)

        return decorator

    def validate(
        self,
        kind: NodeKind,
        name: str,
        *,
        capabilities: EngineCapabilities | None = None,
    ) -> None:
        registration = self.registration(kind, name)
        effective_capabilities = capabilities or self._engine_capabilities
        validate_node_compatibility(registration.contract, effective_capabilities)

    def audit(
        self,
        kind: NodeKind,
        name: str,
        *,
        capabilities: EngineCapabilities | None = None,
    ) -> CompatibilityAudit:
        registration = self.registration(kind, name)
        effective_capabilities = capabilities or self._engine_capabilities
        return audit_node_compatibility(registration.contract, effective_capabilities)

    def audit_all(
        self,
        *,
        capabilities: EngineCapabilities | None = None,
    ) -> tuple[CompatibilityAudit, ...]:
        self._autoload()
        effective_capabilities = capabilities or self._engine_capabilities
        audits: list[CompatibilityAudit] = []
        for kind in ("entry", "exit", "risk"):
            for name in sorted(self._nodes[kind]):
                audits.append(
                    audit_node_compatibility(
                        self._nodes[kind][name].contract,
                        effective_capabilities,
                    )
                )
        return tuple(audits)

    def registration(self, kind: NodeKind, name: str) -> RegisteredNode:
        resolved_kind = _require_kind(kind)
        _require_name(name)
        self._autoload()
        try:
            return self._nodes[resolved_kind][name]
        except KeyError as exc:
            raise KeyError(
                f"{resolved_kind} node {name!r} is not registered."
            ) from exc

    def resolve_contract(self, kind: NodeKind, name: str) -> NodeContract:
        return self.registration(kind, name).contract

    @overload
    def resolve(
        self,
        kind: Literal["entry"],
        name: str,
        *,
        capabilities: EngineCapabilities | None = None,
    ) -> EntryNode:
        ...

    @overload
    def resolve(
        self,
        kind: Literal["exit"],
        name: str,
        *,
        capabilities: EngineCapabilities | None = None,
    ) -> ExitNode:
        ...

    @overload
    def resolve(
        self,
        kind: Literal["risk"],
        name: str,
        *,
        capabilities: EngineCapabilities | None = None,
    ) -> RiskNode:
        ...

    def resolve(
        self,
        kind: NodeKind,
        name: str,
        *,
        capabilities: EngineCapabilities | None = None,
    ) -> NodeCallable:
        registration = self.registration(kind, name)
        effective_capabilities = capabilities or self._engine_capabilities
        if self._enforce_compatibility:
            validate_node_compatibility(registration.contract, effective_capabilities)
        return registration.node

    def available(self, kind: NodeKind) -> tuple[str, ...]:
        resolved_kind = _require_kind(kind)
        self._autoload()
        return tuple(sorted(self._nodes[resolved_kind]))


registry = NodeRegistry(autoload_package="trading_lab.nodes")


def register_entry(
    name: str,
    node: EntryNode,
    contract: NodeManifest,
) -> EntryNode:
    return registry.register("entry", name, node, contract)


def register_exit(
    name: str,
    node: ExitNode,
    contract: NodeManifest,
) -> ExitNode:
    return registry.register("exit", name, node, contract)


def register_risk(
    name: str,
    node: RiskNode,
    contract: NodeManifest,
) -> RiskNode:
    return registry.register("risk", name, node, contract)


def entry(
    name: str,
    *,
    contract: NodeManifest,
) -> Callable[[NodeT], NodeT]:
    return registry.entry(name, contract=contract)


def exit(
    name: str,
    *,
    contract: NodeManifest,
) -> Callable[[NodeT], NodeT]:
    return registry.exit(name, contract=contract)


def risk(
    name: str,
    *,
    contract: NodeManifest,
) -> Callable[[NodeT], NodeT]:
    return registry.risk(name, contract=contract)


@overload
def resolve(
    kind: Literal["entry"],
    name: str,
    *,
    capabilities: EngineCapabilities | None = None,
) -> EntryNode:
    ...


@overload
def resolve(
    kind: Literal["exit"],
    name: str,
    *,
    capabilities: EngineCapabilities | None = None,
) -> ExitNode:
    ...


@overload
def resolve(
    kind: Literal["risk"],
    name: str,
    *,
    capabilities: EngineCapabilities | None = None,
) -> RiskNode:
    ...


def resolve(
    kind: NodeKind,
    name: str,
    *,
    capabilities: EngineCapabilities | None = None,
) -> NodeCallable:
    return registry.resolve(kind, name, capabilities=capabilities)


def resolve_contract(kind: NodeKind, name: str) -> NodeContract:
    return registry.resolve_contract(kind, name)


def audit(
    kind: NodeKind,
    name: str,
    *,
    capabilities: EngineCapabilities | None = None,
) -> CompatibilityAudit:
    return registry.audit(kind, name, capabilities=capabilities)


def audit_all(
    *,
    capabilities: EngineCapabilities | None = None,
) -> tuple[CompatibilityAudit, ...]:
    return registry.audit_all(capabilities=capabilities)


def validate(
    kind: NodeKind,
    name: str,
    *,
    capabilities: EngineCapabilities | None = None,
) -> None:
    registry.validate(kind, name, capabilities=capabilities)


def resolve_entry(
    name: str,
    *,
    capabilities: EngineCapabilities | None = None,
) -> EntryNode:
    return registry.resolve("entry", name, capabilities=capabilities)


def resolve_exit(
    name: str,
    *,
    capabilities: EngineCapabilities | None = None,
) -> ExitNode:
    return registry.resolve("exit", name, capabilities=capabilities)


def resolve_risk(
    name: str,
    *,
    capabilities: EngineCapabilities | None = None,
) -> RiskNode:
    return registry.resolve("risk", name, capabilities=capabilities)


__all__ = [
    "CompatibilityAudit",
    "CompatibilityError",
    "NodeCallable",
    "NodeManifest",
    "NodeRegistry",
    "RegisteredNode",
    "audit",
    "audit_all",
    "entry",
    "exit",
    "register_entry",
    "register_exit",
    "register_risk",
    "registry",
    "resolve",
    "resolve_contract",
    "resolve_entry",
    "resolve_exit",
    "resolve_risk",
    "risk",
    "validate",
]
