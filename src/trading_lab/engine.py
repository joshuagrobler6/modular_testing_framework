from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from trading_lab.contracts import (
    ActionBatch,
    ActionRequest,
    BacktestResult,
    BacktestSpec,
    Bar,
    CompatibilityAudit,
    CompatibilityError,
    CostAssumptions,
    DecisionContext,
    EngineCapabilities,
    EntryIntent,
    ExitIntent,
    KNOWN_ACTION_TYPES,
    NodeContract,
    NodeOutputValidationError,
    PACKAGE_VERSION,
    PortfolioState,
    PositionState,
    RESERVED_ACTION_TYPES,
    RiskDecision,
    SessionInfo,
    SetupCompatibilityError,
    UnsupportedNodeActionError,
    UnsupportedNodeRequirementError,
    audit_node_compatibility,
    validate_node_compatibility,
)
from trading_lab.data import validate_ohlcv
from trading_lab.registry import NodeRegistry, registry as default_registry
from trading_lab.schemas import (
    DecisionLogSchema,
    EquityCurveSchema,
    FillLogSchema,
    OrderLogSchema,
    RESERVED_LEDGER_COLUMNS,
    TradeLedgerSchema,
)

OrderSide = Literal["buy", "sell"]
TradeSide = Literal["long", "short"]
ResolvedAction = Literal[
    "hold",
    "submit_entry_long",
    "submit_entry_short",
    "submit_scale_in",
    "submit_partial_exit",
    "submit_reduce",
    "submit_exit",
    "blocked_entry",
    "blocked_exit",
    "blocked_scale_in",
    "blocked_reduce",
    "blocked_stop_request",
]

_RUNTIME_CAPABILITIES = EngineCapabilities()
_KNOWN_ACTIONS = frozenset(KNOWN_ACTION_TYPES)
_ALLOWED_ACTIONS_BY_KIND = {
    "entry": frozenset(("enter_long", "enter_short", "hold", "scale_in", "increase")),
    "exit": frozenset(("close", "hold", "partial_exit", "reduce", "set_stop")),
    "risk": frozenset(KNOWN_ACTION_TYPES),
}

_DECISION_BASE_COLUMNS = [
    "ts",
    "symbol",
    "entry_action",
    "exit_action",
    "risk_approved",
    "target_units",
    "resolved_action",
    "reason",
    "metadata",
]
_ORDER_BASE_COLUMNS = [
    "order_id",
    "ts_submitted",
    "symbol",
    "side",
    "qty",
    "order_type",
    "price_reference",
    "status",
]
_FILL_BASE_COLUMNS = [
    "fill_id",
    "order_id",
    "ts_fill",
    "symbol",
    "side",
    "qty",
    "fill_price",
    "fees",
    "slippage",
    "gross_notional",
]
_TRADE_BASE_COLUMNS = [
    "trade_id",
    "symbol",
    "side",
    "entry_ts",
    "exit_ts",
    "entry_price",
    "exit_price",
    "qty",
    "gross_pnl",
    "net_pnl",
    "mfe",
    "mae",
    "exit_efficiency",
    "bars_held",
    "exit_reason",
    "fees",
]
_EQUITY_BASE_COLUMNS = [
    "ts",
    "cash",
    "equity",
    "realized_pnl",
    "unrealized_pnl",
    "gross_exposure",
    "net_exposure",
    "drawdown",
]

_DECISION_LOG_COLUMNS = _DECISION_BASE_COLUMNS + list(RESERVED_LEDGER_COLUMNS)
_ORDER_LOG_COLUMNS = _ORDER_BASE_COLUMNS + list(RESERVED_LEDGER_COLUMNS)
_FILL_LOG_COLUMNS = _FILL_BASE_COLUMNS + list(RESERVED_LEDGER_COLUMNS)
_TRADE_LEDGER_COLUMNS = _TRADE_BASE_COLUMNS + list(RESERVED_LEDGER_COLUMNS)
_EQUITY_CURVE_COLUMNS = _EQUITY_BASE_COLUMNS + list(RESERVED_LEDGER_COLUMNS)


@dataclass(frozen=True, slots=True)
class _PendingOrder:
    order_id: str
    ts_submitted: datetime
    target_ts: datetime
    symbol: str
    side: OrderSide
    qty: float
    price_reference: float
    action_type: str
    reason: str
    position_id: str | None
    parent_position_id: str | None
    lot_id: str | None
    entry_tag: str | None
    exit_tag: str | None
    risk_tag: str | None
    node_version: str | None
    contract_version: str


@dataclass(frozen=True, slots=True)
class _OpenLot:
    lot_id: str
    position_id: str
    parent_position_id: str | None
    symbol: str
    side: TradeSide
    entry_ts: datetime
    entry_bar_index: int
    entry_price: float
    original_qty: float
    remaining_qty: float
    remaining_entry_fees: float
    entry_tag: str | None
    risk_tag: str | None
    entry_node_version: str | None
    max_price_seen: float
    min_price_seen: float
    contract_version: str


@dataclass(frozen=True, slots=True)
class _ResolvedDecision:
    resolved_action: ResolvedAction
    risk_approved: bool
    target_units: float
    reason: str
    resolver_status: str
    rejection_reason: str | None
    pending_order: _PendingOrder | None
    accepted_requests: tuple[ActionRequest, ...] = ()
    rejected_actions: tuple["_RejectedAction", ...] = ()


@dataclass(frozen=True, slots=True)
class _ActionCandidate:
    source: Literal["entry", "exit", "risk"]
    request: ActionRequest
    node_version: str | None


@dataclass(frozen=True, slots=True)
class _RejectedAction:
    request: ActionRequest
    reason: str


@dataclass(frozen=True, slots=True)
class BacktestSetupAudit:
    entry: CompatibilityAudit
    exit: CompatibilityAudit
    risk: CompatibilityAudit

    @property
    def audits(self) -> tuple[CompatibilityAudit, CompatibilityAudit, CompatibilityAudit]:
        return (self.entry, self.exit, self.risk)

    @property
    def supported(self) -> bool:
        return all(audit.supported for audit in self.audits)

    @property
    def summary(self) -> str:
        if self.supported:
            return "Backtest setup is compatible with the active engine capabilities."
        return "Backtest setup is incompatible:\n- " + "\n- ".join(
            audit.summary for audit in self.audits if not audit.supported
        )

    def raise_for_errors(self) -> None:
        if self.supported:
            return
        raise SetupCompatibilityError(self.summary)


def _require_spec(spec: object) -> BacktestSpec:
    if not isinstance(spec, BacktestSpec):
        raise TypeError("spec must be a BacktestSpec instance.")
    return spec


def _build_bar(row: pd.Series) -> Bar:
    return Bar(
        timestamp=row["ts"],
        symbol=row["symbol"],
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
    )


def _build_portfolio(
    symbol: str,
    cash: float,
    equity: float,
    position: PositionState,
) -> PortfolioState:
    if position.symbol != symbol:
        raise ValueError("portfolio symbol must match position symbol.")
    return PortfolioState(cash=cash, equity=equity, positions=(position,))


def _mark_to_market(
    position: PositionState,
    cash: float,
    close_price: float,
    multiplier: float,
) -> tuple[float, float, float, float]:
    if position.is_flat:
        return 0.0, cash, 0.0, 0.0

    exposure = position.quantity * close_price * multiplier
    if position.is_long:
        net_exposure = exposure
        unrealized_pnl = (
            (close_price - float(position.entry_price))
            * position.quantity
            * multiplier
        )
    else:
        net_exposure = -exposure
        unrealized_pnl = (
            (float(position.entry_price) - close_price)
            * position.quantity
            * multiplier
        )

    equity = cash + net_exposure
    return unrealized_pnl, equity, abs(net_exposure), net_exposure


def _slipped_fill_price(
    open_price: float,
    side: OrderSide,
    costs: CostAssumptions,
) -> float:
    slippage_multiplier = costs.slippage_bps / 10_000.0
    fill_price = (
        open_price * (1.0 + slippage_multiplier)
        if side == "buy"
        else open_price * (1.0 - slippage_multiplier)
    )
    if fill_price <= 0.0:
        raise ValueError("slippage produced a non-positive fill price.")
    return fill_price


def _fill_costs(
    side: OrderSide,
    qty: float,
    open_price: float,
    fill_price: float,
    multiplier: float,
    costs: CostAssumptions,
) -> tuple[float, float, float]:
    gross_notional = fill_price * qty * multiplier
    fees = gross_notional * costs.fee_rate + qty * costs.fee_per_unit
    if side == "buy":
        slippage = (fill_price - open_price) * qty * multiplier
    else:
        slippage = (open_price - fill_price) * qty * multiplier
    return gross_notional, fees, slippage


def _next_order_id(index: int) -> str:
    return f"ord-{index:06d}"


def _next_fill_id(index: int) -> str:
    return f"fill-{index:06d}"


def _next_trade_id(index: int) -> str:
    return f"trade-{index:06d}"


def _next_position_id(index: int) -> str:
    return f"pos-{index:06d}"


def _next_lot_id(index: int) -> str:
    return f"lot-{index:06d}"


def _is_entry_like_action(action_type: str) -> bool:
    return action_type in {"enter_long", "enter_short", "scale_in", "increase"}


def _as_dataframe(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def _jsonable(value: object) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _stable_json_dumps(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _sha256_hex(value: object) -> str:
    return hashlib.sha256(_stable_json_dumps(value).encode("utf-8")).hexdigest()


def _serialize_costs(costs: CostAssumptions) -> dict[str, object]:
    return {
        "fee_rate": costs.fee_rate,
        "fee_per_unit": costs.fee_per_unit,
        "slippage_bps": costs.slippage_bps,
        "contract_version": costs.contract_version,
    }


def _serialize_engine_capabilities(capabilities: EngineCapabilities) -> dict[str, object]:
    return {
        "contract_version": capabilities.contract_version,
        "supported_action_types": list(capabilities.supported_action_types),
        "supported_capabilities": list(capabilities.supported_capabilities),
        "supports_multiple_entries": capabilities.supports_multiple_entries,
        "supports_partial_exits": capabilities.supports_partial_exits,
        "supports_richer_risk_actions": capabilities.supports_richer_risk_actions,
        "supports_lot_level_accounting": capabilities.supports_lot_level_accounting,
        "supports_metric_dependency_checks": capabilities.supports_metric_dependency_checks,
        "supports_node_capability_checks": capabilities.supports_node_capability_checks,
    }


def _serialize_spec(spec: BacktestSpec) -> dict[str, object]:
    instrument = spec.instrument
    return {
        "name": spec.name,
        "instrument": {
            "symbol": instrument.symbol,
            "price_increment": instrument.price_increment,
            "quantity_increment": instrument.quantity_increment,
            "contract_multiplier": instrument.contract_multiplier,
            "contract_version": instrument.contract_version,
        },
        "entry_node": spec.entry_node,
        "exit_node": spec.exit_node,
        "risk_node": spec.risk_node,
        "initial_cash": spec.initial_cash,
        "costs": _serialize_costs(spec.costs),
        "allow_short": spec.allow_short,
        "fill_rule": spec.fill_rule,
        "order_type": spec.order_type,
        "strict_node_output_validation": spec.strict_node_output_validation,
        "random_seed": spec.random_seed,
        "engine_capabilities": _serialize_engine_capabilities(spec.engine_capabilities),
        "contract_version": spec.contract_version,
    }


def _serialize_node_contract(contract: NodeContract) -> dict[str, object]:
    return {
        "name": contract.name,
        "kind": contract.kind,
        "spec": {
            "name": contract.spec.name,
            "kind": contract.spec.kind,
            "version": contract.spec.version,
            "contract_version": contract.spec.contract_version,
            "required_capabilities": list(contract.spec.required_capabilities),
            "emitted_action_types": list(contract.spec.emitted_action_types),
            "required_history": contract.spec.required_history,
            "requires_portfolio_view": contract.spec.requires_portfolio_view,
            "description": contract.spec.description,
        },
        "input_contract_version": contract.input_contract_version,
        "output_contract_version": contract.output_contract_version,
        "metric_dependencies": list(contract.metric_dependencies),
        "manifest": _jsonable(contract.manifest),
    }


def _data_hash(df: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(df, index=True)
    return hashlib.sha256(row_hashes.to_numpy().tobytes()).hexdigest()


def _data_summary(df: pd.DataFrame, timeframe: str) -> dict[str, object]:
    if df.empty:
        return {
            "symbol": None,
            "timeframe": timeframe,
            "rows": 0,
            "start_ts": None,
            "end_ts": None,
        }
    return {
        "symbol": str(df["symbol"].iloc[0]),
        "timeframe": timeframe,
        "rows": int(len(df)),
        "start_ts": pd.Timestamp(df["ts"].iloc[0]).isoformat(),
        "end_ts": pd.Timestamp(df["ts"].iloc[-1]).isoformat(),
    }


def _git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _build_run_artifacts(
    *,
    spec: BacktestSpec,
    data: pd.DataFrame,
    timeframe: str,
    entry_contract: NodeContract,
    exit_contract: NodeContract,
    risk_contract: NodeContract,
) -> tuple[str, dict[str, object]]:
    spec_snapshot = _serialize_spec(spec)
    node_snapshots = {
        "entry": _serialize_node_contract(entry_contract),
        "exit": _serialize_node_contract(exit_contract),
        "risk": _serialize_node_contract(risk_contract),
    }
    spec_hash = _sha256_hex(spec_snapshot)
    data_hash = _data_hash(data)
    git_commit = _git_commit()
    node_versions = {
        "entry": entry_contract.spec.version,
        "exit": exit_contract.spec.version,
        "risk": risk_contract.spec.version,
    }
    run_fingerprint = {
        "spec_hash": spec_hash,
        "data_hash": data_hash,
        "contract_version": spec.contract_version,
        "package_version": PACKAGE_VERSION,
        "git_commit": git_commit,
        "nodes": node_snapshots,
    }
    run_id = f"run-{_sha256_hex(run_fingerprint)[:16]}"
    return (
        run_id,
        {
            "run_manifest": {
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "package_version": PACKAGE_VERSION,
                "git_commit": git_commit,
                "spec_hash": spec_hash,
                "data_hash": data_hash,
                "contract_version": spec.contract_version,
                "random_seed": spec.random_seed,
                "spec": spec_snapshot,
                "data": _data_summary(data, timeframe),
                "engine": {
                    "fill_rule": spec.fill_rule,
                    "order_type": spec.order_type,
                    "allow_short": spec.allow_short,
                    "strict_node_output_validation": spec.strict_node_output_validation,
                    "capabilities": _serialize_engine_capabilities(spec.engine_capabilities),
                },
                "nodes": node_snapshots,
                "node_versions": node_versions,
            }
        },
    )


def _reserved_values(
    *,
    run_id: str | None = None,
    position_id: str | None = None,
    parent_position_id: str | None = None,
    lot_id: str | None = None,
    entry_tag: str | None = None,
    exit_tag: str | None = None,
    risk_tag: str | None = None,
    resolver_status: str | None = None,
    rejection_reason: str | None = None,
    node_version: str | None = None,
    contract_version: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "position_id": position_id,
        "parent_position_id": parent_position_id,
        "lot_id": lot_id,
        "entry_tag": entry_tag,
        "exit_tag": exit_tag,
        "risk_tag": risk_tag,
        "resolver_status": resolver_status,
        "rejection_reason": rejection_reason,
        "node_version": node_version,
        "contract_version": contract_version,
    }


def _entry_action_label(entry_batch: ActionBatch) -> str:
    for request in entry_batch.active_requests:
        if request.action_type in {"enter_long", "enter_short", "scale_in", "increase"}:
            return request.action_type
    return "none"


def _exit_action_label(exit_batch: ActionBatch) -> str:
    for request in exit_batch.active_requests:
        if request.action_type == "close":
            return "exit"
        if request.action_type in {"partial_exit", "reduce", "set_stop"}:
            return request.action_type
    return "none"


def _resolve_entry_order_side(
    request: ActionRequest,
    position: PositionState,
) -> OrderSide:
    if request.action_type == "enter_long":
        return "buy"
    if request.action_type == "enter_short":
        return "sell"
    if request.action_type in {"scale_in", "increase"}:
        if position.is_long:
            return "buy"
        if position.is_short:
            return "sell"
    raise ValueError("entry request must imply a valid order side.")


def _resolve_exit_order_side(position: PositionState) -> OrderSide:
    if position.is_long:
        return "sell"
    if position.is_short:
        return "buy"
    raise ValueError("cannot resolve an exit order side for a flat position.")


def _validate_input_frame(spec: BacktestSpec, data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    validated = validate_ohlcv(data)
    symbols = tuple(validated["symbol"].drop_duplicates())
    if symbols != (spec.instrument.symbol,):
        raise ValueError(
            "data must contain exactly one symbol matching spec.instrument.symbol."
        )

    timeframes = tuple(validated["timeframe"].drop_duplicates())
    if len(timeframes) != 1:
        raise ValueError("data must contain exactly one timeframe for v1 backtests.")

    return validated.reset_index(drop=True), timeframes[0]


def _validate_outputs(
    decision_log: pd.DataFrame,
    order_log: pd.DataFrame,
    fill_log: pd.DataFrame,
    trade_ledger: pd.DataFrame,
    equity_curve: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        DecisionLogSchema.validate(decision_log, lazy=True),
        OrderLogSchema.validate(order_log, lazy=True),
        FillLogSchema.validate(fill_log, lazy=True),
        TradeLedgerSchema.validate(trade_ledger, lazy=True),
        EquityCurveSchema.validate(equity_curve, lazy=True),
    )


def _effective_engine_capabilities(spec: BacktestSpec) -> EngineCapabilities:
    declared = spec.engine_capabilities
    if declared.contract_version != _RUNTIME_CAPABILITIES.contract_version:
        raise SetupCompatibilityError(
            "spec.engine_capabilities.contract_version is not supported by the v1 "
            "engine. This requires engine changes."
        )

    unsupported = sorted(
        declared.capability_set() - _RUNTIME_CAPABILITIES.capability_set()
    )
    if unsupported:
        raise SetupCompatibilityError(
            "spec declares capabilities unsupported by the v1 engine: "
            f"{unsupported}. This requires engine changes."
        )

    return declared


def audit_backtest_setup(
    spec: BacktestSpec,
    *,
    node_registry: NodeRegistry | None = None,
) -> BacktestSetupAudit:
    resolved_spec = _require_spec(spec)
    effective_capabilities = _effective_engine_capabilities(resolved_spec)
    active_registry = node_registry or default_registry

    entry_contract = active_registry.resolve_contract("entry", resolved_spec.entry_node)
    exit_contract = active_registry.resolve_contract("exit", resolved_spec.exit_node)
    risk_contract = active_registry.resolve_contract("risk", resolved_spec.risk_node)

    return BacktestSetupAudit(
        entry=audit_node_compatibility(entry_contract, effective_capabilities),
        exit=audit_node_compatibility(exit_contract, effective_capabilities),
        risk=audit_node_compatibility(risk_contract, effective_capabilities),
    )


def _node_versions(
    entry_contract: NodeContract,
    exit_contract: NodeContract,
    risk_contract: NodeContract,
) -> dict[str, str]:
    return {
        "entry": entry_contract.spec.version,
        "exit": exit_contract.spec.version,
        "risk": risk_contract.spec.version,
    }


def _validate_action_request(
    kind: Literal["entry", "exit", "risk"],
    request: ActionRequest,
    *,
    capabilities: EngineCapabilities,
    contract: NodeContract,
) -> ActionRequest:
    action_type = request.action_type
    if action_type not in _KNOWN_ACTIONS:
        raise NodeOutputValidationError(
            f"{kind} node emitted unknown action {action_type!r}."
        )
    if action_type not in contract.spec.emitted_action_types:
        raise NodeOutputValidationError(
            f"{kind} node emitted action {action_type!r} outside its declared manifest."
        )
    if action_type not in _ALLOWED_ACTIONS_BY_KIND[kind]:
        raise NodeOutputValidationError(
            f"{kind} node cannot emit action {action_type!r} in the v1 engine."
        )
    if action_type not in capabilities.supported_action_types:
        suffix = (
            " This is an engine feature request, not just a new node."
            if action_type in RESERVED_ACTION_TYPES
            else ""
        )
        raise UnsupportedNodeActionError(
            f"{kind} node emitted action {action_type!r}, which is not supported by "
            f"the active engine capabilities.{suffix}"
        )
    if kind == "risk" and request.is_active:
        raise UnsupportedNodeRequirementError(
            "risk nodes emitted an action request, but richer risk actions are not "
            "supported by the v1 engine. This is an engine feature request, not just "
            "a new node."
        )
    return request


def _validate_action_batch(
    kind: Literal["entry", "exit", "risk"],
    batch: ActionBatch,
    *,
    capabilities: EngineCapabilities,
    contract: NodeContract,
) -> ActionBatch:
    validated_requests = tuple(
        _validate_action_request(
            kind,
            request,
            capabilities=capabilities,
            contract=contract,
        )
        for request in batch.requests
    )
    return ActionBatch(
        requests=validated_requests,
        contract_version=batch.contract_version,
    )


def _coerce_entry_output(
    value: object,
    *,
    capabilities: EngineCapabilities,
    contract: NodeContract,
) -> ActionBatch:
    if isinstance(value, EntryIntent):
        batch = ActionBatch(
            requests=(value.as_action_request(),),
            contract_version=value.contract_version,
        )
    elif isinstance(value, ActionBatch):
        batch = value
    elif isinstance(value, ActionRequest):
        batch = ActionBatch(
            requests=(value,),
            contract_version=value.contract_version,
        )
    else:
        raise TypeError("entry node must return EntryIntent, ActionRequest, or ActionBatch.")
    return _validate_action_batch(
        "entry",
        batch,
        capabilities=capabilities,
        contract=contract,
    )


def _coerce_exit_output(
    value: object,
    *,
    capabilities: EngineCapabilities,
    contract: NodeContract,
) -> ActionBatch:
    if isinstance(value, ExitIntent):
        batch = ActionBatch(
            requests=(value.as_action_request(),),
            contract_version=value.contract_version,
        )
    elif isinstance(value, ActionBatch):
        batch = value
    elif isinstance(value, ActionRequest):
        batch = ActionBatch(
            requests=(value,),
            contract_version=value.contract_version,
        )
    else:
        raise TypeError("exit node must return ExitIntent, ActionRequest, or ActionBatch.")
    return _validate_action_batch(
        "exit",
        batch,
        capabilities=capabilities,
        contract=contract,
    )


def _coerce_risk_output(
    value: object,
    *,
    capabilities: EngineCapabilities,
    contract: NodeContract,
) -> tuple[RiskDecision, ActionBatch]:
    if isinstance(value, RiskDecision):
        return value, ActionBatch(contract_version=value.contract_version)
    if isinstance(value, ActionBatch):
        batch = value
    elif isinstance(value, ActionRequest):
        batch = ActionBatch(
            requests=(value,),
            contract_version=value.contract_version,
        )
    else:
        raise TypeError("risk node must return RiskDecision, ActionRequest, or ActionBatch.")
    validated_batch = _validate_action_batch(
        "risk",
        batch,
        capabilities=capabilities,
        contract=contract,
    )
    primary_request = validated_batch.primary_request
    return (
        RiskDecision(
            reason=primary_request.reason,
            metadata=primary_request.metadata,
            contract_version=primary_request.contract_version,
        ),
        validated_batch,
    )


def _validate_node_output_strict(
    *,
    kind: Literal["entry", "exit", "risk"],
    raw_output: object,
    contract: NodeContract,
    ctx: DecisionContext,
    batch: ActionBatch | None = None,
    risk_decision: RiskDecision | None = None,
) -> None:
    output_contract_version = getattr(raw_output, "contract_version", None)
    if output_contract_version != contract.output_contract_version:
        raise NodeOutputValidationError(
            f"{kind} node {contract.name!r} returned contract_version "
            f"{output_contract_version!r}, expected {contract.output_contract_version!r}."
        )

    if len(ctx.history) >= contract.spec.required_history:
        return

    if kind in {"entry", "exit"} and batch is not None and batch.is_active:
        raise NodeOutputValidationError(
            f"{kind} node {contract.name!r} emitted active action "
            f"{batch.primary_request.action_type!r} "
            f"before required_history={contract.spec.required_history} bars were available."
        )

    if kind == "risk":
        risk_request_active = batch is not None and batch.is_active
        if risk_request_active or (risk_decision is not None and risk_decision.allow_entry):
            raise NodeOutputValidationError(
                f"risk node {contract.name!r} emitted an active decision before "
                f"required_history={contract.spec.required_history} bars were available."
            )


def _serialize_request(request: ActionRequest) -> dict[str, object]:
    return {
        "action_type": request.action_type,
        "units": request.units,
        "stop_price": request.stop_price,
        "reason": request.reason,
        "metadata": request.metadata,
    }


def _serialize_rejected_action(rejected_action: _RejectedAction) -> dict[str, object]:
    return {
        "action_type": rejected_action.request.action_type,
        "units": rejected_action.request.units,
        "stop_price": rejected_action.request.stop_price,
        "reason": rejected_action.request.reason,
        "metadata": rejected_action.request.metadata,
        "rejection_reason": rejected_action.reason,
    }


def _join_reasons(requests: tuple[ActionRequest, ...]) -> str:
    parts: list[str] = []
    for request in requests:
        reason = request.reason.strip()
        if reason and reason not in parts:
            parts.append(reason)
    return " | ".join(parts)


def _merge_rejection_reason(
    rejected_actions: tuple[_RejectedAction, ...],
) -> str | None:
    reasons: list[str] = []
    for rejected_action in rejected_actions:
        if rejected_action.reason and rejected_action.reason not in reasons:
            reasons.append(rejected_action.reason)
    if not reasons:
        return None
    return " | ".join(reasons)


def _build_accepted_resolution(
    *,
    resolved_action: ResolvedAction,
    risk_approved: bool,
    target_units: float,
    reason: str,
    pending_order: _PendingOrder,
    accepted_requests: tuple[ActionRequest, ...],
    rejected_actions: tuple[_RejectedAction, ...] = (),
) -> _ResolvedDecision:
    return _ResolvedDecision(
        resolved_action=resolved_action,
        risk_approved=risk_approved,
        target_units=target_units,
        reason=reason,
        resolver_status=(
            "accepted_with_rejections" if rejected_actions else "accepted"
        ),
        rejection_reason=_merge_rejection_reason(rejected_actions),
        pending_order=pending_order,
        accepted_requests=accepted_requests,
        rejected_actions=rejected_actions,
    )


def _build_rejected_resolution(
    *,
    resolved_action: ResolvedAction,
    risk_approved: bool,
    target_units: float,
    reason: str,
    rejected_actions: tuple[_RejectedAction, ...],
    accepted_requests: tuple[ActionRequest, ...] = (),
) -> _ResolvedDecision:
    rejection_reason = _merge_rejection_reason(rejected_actions) or reason
    return _ResolvedDecision(
        resolved_action=resolved_action,
        risk_approved=risk_approved,
        target_units=target_units,
        reason=reason,
        resolver_status="rejected",
        rejection_reason=rejection_reason,
        pending_order=None,
        accepted_requests=accepted_requests,
        rejected_actions=rejected_actions,
    )


def _request_candidates(
    source: Literal["entry", "exit", "risk"],
    batch: ActionBatch,
    node_version: str | None,
) -> tuple[_ActionCandidate, ...]:
    return tuple(
        _ActionCandidate(source=source, request=request, node_version=node_version)
        for request in batch.active_requests
    )


def _decision_metadata(
    *,
    timeframe: str,
    position: PositionState,
    bar: Bar,
    entry_batch: ActionBatch,
    exit_batch: ActionBatch,
    risk_decision: RiskDecision,
    risk_batch: ActionBatch,
    next_bar: Bar | None,
    resolved: _ResolvedDecision,
) -> dict[str, object]:
    pending_order = resolved.pending_order
    return {
        "timeframe": timeframe,
        "bar_close": bar.close,
        "position_side": position.side,
        "entry_reason": entry_batch.primary_request.reason,
        "exit_reason": exit_batch.primary_request.reason,
        "risk_reason": risk_decision.reason,
        "risk_action": risk_batch.primary_request.action_type if risk_batch.is_active else None,
        "entry_requests": [
            _serialize_request(request) for request in entry_batch.active_requests
        ],
        "exit_requests": [
            _serialize_request(request) for request in exit_batch.active_requests
        ],
        "risk_requests": [
            _serialize_request(request) for request in risk_batch.active_requests
        ],
        "accepted_requests": [
            _serialize_request(request) for request in resolved.accepted_requests
        ],
        "rejected_requests": [
            _serialize_rejected_action(rejected_action)
            for rejected_action in resolved.rejected_actions
        ],
        "next_bar_ts": next_bar.timestamp if next_bar is not None else None,
        "pending_order_id": pending_order.order_id if pending_order is not None else None,
    }


def _entry_quantity_from_requests(
    requests: tuple[ActionRequest, ...],
    risk_decision: RiskDecision,
) -> tuple[float, str | None]:
    if not requests:
        return 0.0, None
    unsized_requests = [request for request in requests if request.units is None]
    if not unsized_requests:
        return (
            float(sum(float(request.units) for request in requests if request.units is not None)),
            None,
        )
    if len(requests) == 1 and len(unsized_requests) == 1:
        if risk_decision.allow_entry and risk_decision.entry_quantity > 0.0:
            return float(risk_decision.entry_quantity), None
        return 0.0, "risk_did_not_supply_entry_quantity"
    return 0.0, "ambiguous_multi_request_quantity"


def _resolve_decision(
    *,
    bar: Bar,
    next_bar: Bar | None,
    position: PositionState,
    entry_batch: ActionBatch,
    exit_batch: ActionBatch,
    risk_decision: RiskDecision,
    spec: BacktestSpec,
    order_index: int,
    position_index: int,
    lot_index: int,
    entry_node_version: str,
    exit_node_version: str,
) -> _ResolvedDecision:
    can_submit_order = next_bar is not None
    entry_candidates = _request_candidates("entry", entry_batch, entry_node_version)
    exit_candidates = _request_candidates("exit", exit_batch, exit_node_version)

    if exit_candidates:
        rejected_actions = tuple(
            _RejectedAction(request=candidate.request, reason="exit_precedence")
            for candidate in entry_candidates
        )
        stop_candidates = tuple(
            candidate for candidate in exit_candidates if candidate.request.is_stop_request
        )
        actionable_exit_candidates = tuple(
            candidate for candidate in exit_candidates if not candidate.request.is_stop_request
        )
        rejected_actions += tuple(
            _RejectedAction(
                request=candidate.request,
                reason="stop_requests_require_stop_order_engine_support",
            )
            for candidate in stop_candidates
        )

        if not actionable_exit_candidates:
            return _build_rejected_resolution(
                resolved_action="blocked_stop_request",
                risk_approved=False,
                target_units=position.quantity,
                reason="stop_requests_require_stop_order_engine_support",
                rejected_actions=rejected_actions,
            )

        risk_approved = bool(risk_decision.allow_exit)
        if not can_submit_order:
            return _build_rejected_resolution(
                resolved_action="blocked_exit",
                risk_approved=risk_approved,
                target_units=position.quantity,
                reason="no_next_bar_for_fill",
                rejected_actions=rejected_actions
                + tuple(
                    _RejectedAction(
                        request=candidate.request,
                        reason="no_next_bar_for_fill",
                    )
                    for candidate in actionable_exit_candidates
                ),
            )
        if position.is_flat:
            return _build_rejected_resolution(
                resolved_action="blocked_exit",
                risk_approved=risk_approved,
                target_units=0.0,
                reason="no_open_position_to_exit",
                rejected_actions=rejected_actions
                + tuple(
                    _RejectedAction(
                        request=candidate.request,
                        reason="no_open_position_to_exit",
                    )
                    for candidate in actionable_exit_candidates
                ),
            )
        if not risk_decision.allow_exit:
            rejection_reason = risk_decision.reason or "exit_rejected"
            return _build_rejected_resolution(
                resolved_action="blocked_exit",
                risk_approved=False,
                target_units=position.quantity,
                reason=rejection_reason,
                rejected_actions=rejected_actions
                + tuple(
                    _RejectedAction(
                        request=candidate.request,
                        reason=rejection_reason,
                    )
                    for candidate in actionable_exit_candidates
                ),
            )

        close_candidates = tuple(
            candidate for candidate in actionable_exit_candidates if candidate.request.is_close
        )
        reduction_candidates = tuple(
            candidate
            for candidate in actionable_exit_candidates
            if candidate.request.action_type in {"partial_exit", "reduce"}
        )
        if close_candidates:
            accepted_requests = tuple(candidate.request for candidate in close_candidates)
            rejected_actions += tuple(
                _RejectedAction(
                    request=candidate.request,
                    reason="full_close_takes_precedence",
                )
                for candidate in reduction_candidates
            )
            reason = _join_reasons(accepted_requests) or "close_position"
            return _build_accepted_resolution(
                resolved_action="submit_exit",
                risk_approved=True,
                target_units=0.0,
                reason=reason,
                pending_order=_PendingOrder(
                    order_id=_next_order_id(order_index),
                    ts_submitted=bar.timestamp,
                    target_ts=next_bar.timestamp,
                    symbol=spec.instrument.symbol,
                    side=_resolve_exit_order_side(position),
                    qty=position.quantity,
                    price_reference=bar.close,
                    action_type="close",
                    reason=reason,
                    position_id=position.position_id,
                    parent_position_id=position.parent_position_id,
                    lot_id=None,
                    entry_tag=position.entry_tag,
                    exit_tag=close_candidates[0].request.exit_tag,
                    risk_tag=position.risk_tag,
                    node_version=exit_node_version,
                    contract_version=spec.contract_version,
                ),
                accepted_requests=accepted_requests,
                rejected_actions=rejected_actions,
            )

        requested_qty = float(
            sum(
                float(candidate.request.units)
                for candidate in reduction_candidates
                if candidate.request.units is not None
            )
        )
        if requested_qty <= 0.0:
            rejection_reason = "reduction_requests_require_positive_units"
            return _build_rejected_resolution(
                resolved_action="blocked_reduce",
                risk_approved=True,
                target_units=position.quantity,
                reason=rejection_reason,
                rejected_actions=rejected_actions
                + tuple(
                    _RejectedAction(
                        request=candidate.request,
                        reason=rejection_reason,
                    )
                    for candidate in reduction_candidates
                ),
            )
        if requested_qty > position.quantity:
            rejection_reason = "requested_reduce_exceeds_position"
            return _build_rejected_resolution(
                resolved_action="blocked_reduce",
                risk_approved=True,
                target_units=position.quantity,
                reason=rejection_reason,
                rejected_actions=rejected_actions
                + tuple(
                    _RejectedAction(
                        request=candidate.request,
                        reason=rejection_reason,
                    )
                    for candidate in reduction_candidates
                ),
            )

        accepted_requests = tuple(candidate.request for candidate in reduction_candidates)
        reason = _join_reasons(accepted_requests) or "reduce_position"
        is_full_close = requested_qty == position.quantity
        resolved_action: ResolvedAction = (
            "submit_exit"
            if is_full_close
            else (
                "submit_partial_exit"
                if any(request.is_partial_exit for request in accepted_requests)
                else "submit_reduce"
            )
        )
        return _build_accepted_resolution(
            resolved_action=resolved_action,
            risk_approved=True,
            target_units=max(position.quantity - requested_qty, 0.0),
            reason=reason,
            pending_order=_PendingOrder(
                order_id=_next_order_id(order_index),
                ts_submitted=bar.timestamp,
                target_ts=next_bar.timestamp,
                symbol=spec.instrument.symbol,
                side=_resolve_exit_order_side(position),
                    qty=requested_qty,
                    price_reference=bar.close,
                    action_type="close" if is_full_close else accepted_requests[0].action_type,
                    reason=reason,
                    position_id=position.position_id,
                    parent_position_id=position.parent_position_id,
                    lot_id=None,
                entry_tag=position.entry_tag,
                exit_tag=accepted_requests[0].exit_tag,
                risk_tag=position.risk_tag,
                node_version=exit_node_version,
                contract_version=spec.contract_version,
            ),
            accepted_requests=accepted_requests,
            rejected_actions=rejected_actions,
        )

    if entry_candidates:
        risk_approved = bool(risk_decision.allow_entry)
        explicit_entry_candidates = tuple(
            candidate
            for candidate in entry_candidates
            if candidate.request.action_type in {"enter_long", "enter_short"}
        )
        adjustment_candidates = tuple(
            candidate
            for candidate in entry_candidates
            if candidate.request.action_type in {"scale_in", "increase"}
        )
        blocked_action: ResolvedAction = (
            "blocked_scale_in" if adjustment_candidates and not explicit_entry_candidates else "blocked_entry"
        )

        if not can_submit_order:
            return _build_rejected_resolution(
                resolved_action=blocked_action,
                risk_approved=risk_approved,
                target_units=position.quantity,
                reason="no_next_bar_for_fill",
                rejected_actions=tuple(
                    _RejectedAction(
                        request=candidate.request,
                        reason="no_next_bar_for_fill",
                    )
                    for candidate in entry_candidates
                ),
            )
        if not risk_decision.allow_entry:
            rejection_reason = risk_decision.reason or "entry_rejected"
            return _build_rejected_resolution(
                resolved_action=blocked_action,
                risk_approved=False,
                target_units=position.quantity,
                reason=rejection_reason,
                rejected_actions=tuple(
                    _RejectedAction(
                        request=candidate.request,
                        reason=rejection_reason,
                    )
                    for candidate in entry_candidates
                ),
            )

        if position.is_flat:
            if not explicit_entry_candidates:
                rejection_reason = "position_adjustment_requires_open_position"
                return _build_rejected_resolution(
                    resolved_action="blocked_scale_in",
                    risk_approved=True,
                    target_units=0.0,
                    reason=rejection_reason,
                    rejected_actions=tuple(
                        _RejectedAction(
                            request=candidate.request,
                            reason=rejection_reason,
                        )
                        for candidate in entry_candidates
                    ),
                )

            directions = {
                candidate.request.action_type for candidate in explicit_entry_candidates
            }
            if len(directions) > 1:
                rejection_reason = "conflicting_entry_directions"
                return _build_rejected_resolution(
                    resolved_action="blocked_entry",
                    risk_approved=True,
                    target_units=0.0,
                    reason=rejection_reason,
                    rejected_actions=tuple(
                        _RejectedAction(
                            request=candidate.request,
                            reason=rejection_reason,
                        )
                        for candidate in entry_candidates
                    ),
                )
            if (
                explicit_entry_candidates[0].request.action_type == "enter_short"
                and not spec.allow_short
            ):
                rejection_reason = "short_entries_disabled"
                return _build_rejected_resolution(
                    resolved_action="blocked_entry",
                    risk_approved=True,
                    target_units=0.0,
                    reason=rejection_reason,
                    rejected_actions=tuple(
                        _RejectedAction(
                            request=candidate.request,
                            reason=rejection_reason,
                        )
                        for candidate in entry_candidates
                    ),
                )

            accepted_requests = tuple(
                candidate.request for candidate in explicit_entry_candidates + adjustment_candidates
            )
            requested_qty, quantity_error = _entry_quantity_from_requests(
                accepted_requests,
                risk_decision,
            )
            if quantity_error is not None or requested_qty <= 0.0:
                rejection_reason = quantity_error or "entry_quantity_not_approved"
                return _build_rejected_resolution(
                    resolved_action="blocked_entry",
                    risk_approved=True,
                    target_units=0.0,
                    reason=rejection_reason,
                    rejected_actions=tuple(
                        _RejectedAction(
                            request=request,
                            reason=rejection_reason,
                        )
                        for request in accepted_requests
                    ),
                )

            primary_request = explicit_entry_candidates[0].request
            side = _resolve_entry_order_side(primary_request, position)
            resolved_action = (
                "submit_entry_long" if side == "buy" else "submit_entry_short"
            )
            reason = _join_reasons(accepted_requests) or "open_position"
            return _build_accepted_resolution(
                resolved_action=resolved_action,
                risk_approved=True,
                target_units=requested_qty,
                reason=reason,
                pending_order=_PendingOrder(
                    order_id=_next_order_id(order_index),
                    ts_submitted=bar.timestamp,
                    target_ts=next_bar.timestamp,
                    symbol=spec.instrument.symbol,
                    side=side,
                    qty=requested_qty,
                    price_reference=bar.close,
                    action_type=primary_request.action_type,
                    reason=reason,
                    position_id=primary_request.position_id or _next_position_id(position_index),
                    parent_position_id=primary_request.parent_position_id,
                    lot_id=primary_request.lot_id or _next_lot_id(lot_index),
                    entry_tag=primary_request.entry_tag,
                    exit_tag=None,
                    risk_tag=primary_request.risk_tag,
                    node_version=entry_node_version,
                    contract_version=spec.contract_version,
                ),
                accepted_requests=accepted_requests,
            )

        conflicting_direction = any(
            (position.is_long and candidate.request.action_type == "enter_short")
            or (position.is_short and candidate.request.action_type == "enter_long")
            for candidate in explicit_entry_candidates
        )
        if conflicting_direction:
            rejection_reason = "same_bar_reverse_not_supported"
            return _build_rejected_resolution(
                resolved_action="blocked_entry",
                risk_approved=True,
                target_units=position.quantity,
                reason=rejection_reason,
                rejected_actions=tuple(
                    _RejectedAction(
                        request=candidate.request,
                        reason=rejection_reason,
                    )
                    for candidate in entry_candidates
                ),
            )
        if explicit_entry_candidates:
            rejection_reason = "use_scale_in_or_increase_for_open_position"
            return _build_rejected_resolution(
                resolved_action="blocked_entry",
                risk_approved=True,
                target_units=position.quantity,
                reason=rejection_reason,
                rejected_actions=tuple(
                    _RejectedAction(
                        request=candidate.request,
                        reason=rejection_reason,
                    )
                    for candidate in entry_candidates
                ),
            )

        accepted_requests = tuple(candidate.request for candidate in adjustment_candidates)
        requested_qty, quantity_error = _entry_quantity_from_requests(
            accepted_requests,
            risk_decision,
        )
        if quantity_error is not None or requested_qty <= 0.0:
            rejection_reason = quantity_error or "entry_quantity_not_approved"
            return _build_rejected_resolution(
                resolved_action="blocked_scale_in",
                risk_approved=True,
                target_units=position.quantity,
                reason=rejection_reason,
                rejected_actions=tuple(
                    _RejectedAction(
                        request=request,
                        reason=rejection_reason,
                    )
                    for request in accepted_requests
                ),
            )

        primary_request = accepted_requests[0]
        reason = _join_reasons(accepted_requests) or "scale_in"
        return _build_accepted_resolution(
            resolved_action="submit_scale_in",
            risk_approved=True,
            target_units=position.quantity + requested_qty,
            reason=reason,
            pending_order=_PendingOrder(
                order_id=_next_order_id(order_index),
                ts_submitted=bar.timestamp,
                target_ts=next_bar.timestamp,
                symbol=spec.instrument.symbol,
                side=_resolve_entry_order_side(primary_request, position),
                qty=requested_qty,
                price_reference=bar.close,
                action_type=primary_request.action_type,
                reason=reason,
                position_id=position.position_id,
                parent_position_id=position.parent_position_id,
                lot_id=primary_request.lot_id or _next_lot_id(lot_index),
                entry_tag=primary_request.entry_tag or position.entry_tag,
                exit_tag=None,
                risk_tag=primary_request.risk_tag or position.risk_tag,
                node_version=entry_node_version,
                contract_version=spec.contract_version,
            ),
            accepted_requests=accepted_requests,
        )

    return _ResolvedDecision(
        resolved_action="hold",
        risk_approved=False,
        target_units=0.0,
        reason=risk_decision.reason,
        resolver_status="noop",
        rejection_reason=None,
        pending_order=None,
        accepted_requests=(),
        rejected_actions=(),
    )


def _position_from_open_lots(
    *,
    spec: BacktestSpec,
    open_lots: tuple[_OpenLot, ...],
) -> PositionState:
    if not open_lots:
        return PositionState(
            symbol=spec.instrument.symbol,
            contract_version=spec.contract_version,
        )

    total_qty = sum(lot.remaining_qty for lot in open_lots)
    weighted_entry = (
        sum(lot.entry_price * lot.remaining_qty for lot in open_lots) / total_qty
    )
    entry_time = min(lot.entry_ts for lot in open_lots)
    first_lot = open_lots[0]
    lot_id = first_lot.lot_id if len(open_lots) == 1 else None
    entry_tags = {lot.entry_tag for lot in open_lots}
    risk_tags = {lot.risk_tag for lot in open_lots}
    return PositionState(
        symbol=spec.instrument.symbol,
        side=first_lot.side,
        quantity=total_qty,
        entry_price=weighted_entry,
        entry_time=entry_time,
        position_id=first_lot.position_id,
        parent_position_id=first_lot.parent_position_id,
        lot_id=lot_id,
        entry_tag=next(iter(entry_tags)) if len(entry_tags) == 1 else None,
        risk_tag=next(iter(risk_tags)) if len(risk_tags) == 1 else None,
        contract_version=spec.contract_version,
    )


def _update_open_lot_paths(
    open_lots: tuple[_OpenLot, ...],
    bar: Bar,
) -> tuple[_OpenLot, ...]:
    updated_lots: list[_OpenLot] = []
    for lot in open_lots:
        updated_lots.append(
            _OpenLot(
                lot_id=lot.lot_id,
                position_id=lot.position_id,
                parent_position_id=lot.parent_position_id,
                symbol=lot.symbol,
                side=lot.side,
                entry_ts=lot.entry_ts,
                entry_bar_index=lot.entry_bar_index,
                entry_price=lot.entry_price,
                original_qty=lot.original_qty,
                remaining_qty=lot.remaining_qty,
                remaining_entry_fees=lot.remaining_entry_fees,
                entry_tag=lot.entry_tag,
                risk_tag=lot.risk_tag,
                entry_node_version=lot.entry_node_version,
                max_price_seen=max(lot.max_price_seen, float(bar.high)),
                min_price_seen=min(lot.min_price_seen, float(bar.low)),
                contract_version=lot.contract_version,
            )
        )
    return tuple(updated_lots)


def _build_trade_row_from_lot_exit(
    *,
    spec: BacktestSpec,
    run_id: str,
    trade_id: str,
    lot: _OpenLot,
    exit_qty: float,
    exit_ts: datetime,
    exit_price: float,
    exit_fees: float,
    exit_reason: str,
    exit_tag: str | None,
    exit_node_version: str | None,
    bar_index: int,
) -> tuple[dict[str, object], _OpenLot | None, float]:
    multiplier = spec.instrument.contract_multiplier
    if lot.side == "long":
        gross_pnl = (exit_price - lot.entry_price) * exit_qty * multiplier
        mfe = (lot.max_price_seen - lot.entry_price) * exit_qty * multiplier
        mae = (lot.min_price_seen - lot.entry_price) * exit_qty * multiplier
        realized_move = exit_price - lot.entry_price
        favorable_span = lot.max_price_seen - lot.entry_price
    else:
        gross_pnl = (lot.entry_price - exit_price) * exit_qty * multiplier
        mfe = (lot.entry_price - lot.min_price_seen) * exit_qty * multiplier
        mae = (lot.entry_price - lot.max_price_seen) * exit_qty * multiplier
        realized_move = lot.entry_price - exit_price
        favorable_span = lot.entry_price - lot.min_price_seen

    entry_fee_alloc = (
        lot.remaining_entry_fees * (exit_qty / lot.remaining_qty)
        if lot.remaining_qty > 0.0
        else 0.0
    )
    net_pnl = gross_pnl - entry_fee_alloc - exit_fees
    exit_efficiency = realized_move / favorable_span if favorable_span > 0.0 else 0.0
    remaining_qty = lot.remaining_qty - exit_qty
    remaining_entry_fees = max(lot.remaining_entry_fees - entry_fee_alloc, 0.0)

    trade_row = {
        "trade_id": trade_id,
        "symbol": lot.symbol,
        "side": lot.side,
        "entry_ts": lot.entry_ts,
        "exit_ts": exit_ts,
        "entry_price": lot.entry_price,
        "exit_price": exit_price,
        "qty": exit_qty,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "mfe": mfe,
        "mae": mae,
        "exit_efficiency": exit_efficiency,
        "bars_held": max(bar_index - lot.entry_bar_index, 1),
        "exit_reason": exit_reason,
        "fees": entry_fee_alloc + exit_fees,
        **_reserved_values(
            run_id=run_id,
            position_id=lot.position_id,
            parent_position_id=lot.parent_position_id,
            lot_id=lot.lot_id,
            entry_tag=lot.entry_tag,
            exit_tag=exit_tag,
            risk_tag=lot.risk_tag,
            resolver_status="closed",
            rejection_reason=None,
            node_version=(
                f"entry:{lot.entry_node_version}|"
                f"exit:{exit_node_version}"
            ),
            contract_version=spec.contract_version,
        ),
    }

    if remaining_qty <= 0.0:
        return trade_row, None, net_pnl

    return (
        trade_row,
        _OpenLot(
            lot_id=lot.lot_id,
            position_id=lot.position_id,
            parent_position_id=lot.parent_position_id,
            symbol=lot.symbol,
            side=lot.side,
            entry_ts=lot.entry_ts,
            entry_bar_index=lot.entry_bar_index,
            entry_price=lot.entry_price,
            original_qty=lot.original_qty,
            remaining_qty=remaining_qty,
            remaining_entry_fees=remaining_entry_fees,
            entry_tag=lot.entry_tag,
            risk_tag=lot.risk_tag,
            entry_node_version=lot.entry_node_version,
            max_price_seen=lot.max_price_seen,
            min_price_seen=lot.min_price_seen,
            contract_version=lot.contract_version,
        ),
        net_pnl,
    )


def _apply_fill_to_state(
    *,
    spec: BacktestSpec,
    run_id: str,
    bar_index: int,
    fill_ts: datetime,
    fill_price: float,
    gross_notional: float,
    fees: float,
    pending_order: _PendingOrder,
    position: PositionState,
    open_lots: tuple[_OpenLot, ...],
    cash: float,
    trade_index: int,
) -> tuple[
    float,
    float,
    PositionState,
    tuple[_OpenLot, ...],
    list[dict[str, object]],
    int,
]:
    realized_pnl_delta = 0.0
    trade_rows: list[dict[str, object]] = []

    if _is_entry_like_action(pending_order.action_type):
        trade_side: TradeSide = "long" if pending_order.side == "buy" else "short"
        if open_lots and open_lots[0].side != trade_side:
            raise ValueError("entry-like fills cannot reverse an open lot inventory.")
        if pending_order.lot_id is None:
            raise ValueError("entry-like fills require an activated lot_id.")

        cash = (
            cash - gross_notional - fees
            if pending_order.side == "buy"
            else cash + gross_notional - fees
        )
        new_lot = _OpenLot(
            lot_id=pending_order.lot_id,
            position_id=pending_order.position_id or "missing-position-id",
            parent_position_id=pending_order.parent_position_id,
            symbol=spec.instrument.symbol,
            side=trade_side,
            entry_ts=fill_ts,
            entry_bar_index=bar_index,
            entry_price=fill_price,
            original_qty=pending_order.qty,
            remaining_qty=pending_order.qty,
            remaining_entry_fees=fees,
            entry_tag=pending_order.entry_tag,
            risk_tag=pending_order.risk_tag,
            entry_node_version=pending_order.node_version,
            max_price_seen=fill_price,
            min_price_seen=fill_price,
            contract_version=spec.contract_version,
        )
        open_lots = open_lots + (new_lot,)
        position = _position_from_open_lots(spec=spec, open_lots=open_lots)
        return cash, realized_pnl_delta, position, open_lots, trade_rows, trade_index

    if not open_lots:
        raise ValueError("exit-like fills require open lots.")
    if pending_order.qty > position.quantity:
        raise ValueError("reduction fills cannot exceed the open position quantity.")
    if position.is_long and pending_order.side != "sell":
        raise ValueError("long positions must be reduced with sell orders.")
    if position.is_short and pending_order.side != "buy":
        raise ValueError("short positions must be reduced with buy orders.")

    cash = (
        cash + gross_notional - fees
        if pending_order.side == "sell"
        else cash - gross_notional - fees
    )

    remaining_exit_qty = pending_order.qty
    updated_open_lots: list[_OpenLot] = []

    for lot in open_lots:
        if remaining_exit_qty <= 0.0:
            updated_open_lots.append(lot)
            continue

        exit_qty = min(lot.remaining_qty, remaining_exit_qty)
        exit_fee_alloc = fees * (exit_qty / pending_order.qty)
        trade_row, remaining_lot, lot_net_pnl = _build_trade_row_from_lot_exit(
            spec=spec,
            run_id=run_id,
            trade_id=_next_trade_id(trade_index),
            lot=lot,
            exit_qty=exit_qty,
            exit_ts=fill_ts,
            exit_price=fill_price,
            exit_fees=exit_fee_alloc,
            exit_reason=pending_order.reason,
            exit_tag=pending_order.exit_tag,
            exit_node_version=pending_order.node_version,
            bar_index=bar_index,
        )
        trade_rows.append(trade_row)
        realized_pnl_delta += lot_net_pnl
        trade_index += 1
        remaining_exit_qty -= exit_qty
        if remaining_lot is not None:
            updated_open_lots.append(remaining_lot)

    if remaining_exit_qty > 1e-12:
        raise ValueError("exit fill allocation left unconsumed quantity.")

    open_lots = tuple(updated_open_lots)
    position = _position_from_open_lots(spec=spec, open_lots=open_lots)
    return cash, realized_pnl_delta, position, open_lots, trade_rows, trade_index


def run_backtest(
    spec: BacktestSpec,
    data: pd.DataFrame,
    *,
    node_registry: NodeRegistry | None = None,
) -> BacktestResult:
    resolved_spec = _require_spec(spec)
    validated_data, timeframe = _validate_input_frame(resolved_spec, data)
    active_registry = node_registry or default_registry
    setup_audit = audit_backtest_setup(resolved_spec, node_registry=active_registry)
    setup_audit.raise_for_errors()
    effective_capabilities = _effective_engine_capabilities(resolved_spec)

    entry_contract = active_registry.resolve_contract("entry", resolved_spec.entry_node)
    exit_contract = active_registry.resolve_contract("exit", resolved_spec.exit_node)
    risk_contract = active_registry.resolve_contract("risk", resolved_spec.risk_node)

    validate_node_compatibility(entry_contract, effective_capabilities)
    validate_node_compatibility(exit_contract, effective_capabilities)
    validate_node_compatibility(risk_contract, effective_capabilities)

    entry_node = active_registry.resolve(
        "entry",
        resolved_spec.entry_node,
        capabilities=effective_capabilities,
    )
    exit_node = active_registry.resolve(
        "exit",
        resolved_spec.exit_node,
        capabilities=effective_capabilities,
    )
    risk_node = active_registry.resolve(
        "risk",
        resolved_spec.risk_node,
        capabilities=effective_capabilities,
    )

    node_versions = _node_versions(entry_contract, exit_contract, risk_contract)
    node_version_summary = (
        f"entry:{node_versions['entry']}|exit:{node_versions['exit']}|"
        f"risk:{node_versions['risk']}"
    )
    run_id, artifacts = _build_run_artifacts(
        spec=resolved_spec,
        data=validated_data,
        timeframe=timeframe,
        entry_contract=entry_contract,
        exit_contract=exit_contract,
        risk_contract=risk_contract,
    )

    bars = [_build_bar(row) for _, row in validated_data.iterrows()]
    decision_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    fill_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    cash = resolved_spec.initial_cash
    realized_pnl = 0.0
    running_peak_equity = resolved_spec.initial_cash
    position = PositionState(
        symbol=resolved_spec.instrument.symbol,
        contract_version=resolved_spec.contract_version,
    )
    pending_order: _PendingOrder | None = None
    open_lots: tuple[_OpenLot, ...] = ()
    order_index = 1
    fill_index = 1
    position_index = 1
    lot_index = 1
    trade_index = 1

    for bar_index, bar in enumerate(bars):
        if pending_order is not None:
            if pending_order.target_ts != bar.timestamp:
                raise ValueError("pending order target timestamp does not match the bar.")

            fill_price = _slipped_fill_price(bar.open, pending_order.side, resolved_spec.costs)
            gross_notional, fees, slippage = _fill_costs(
                pending_order.side,
                pending_order.qty,
                bar.open,
                fill_price,
                resolved_spec.instrument.contract_multiplier,
                resolved_spec.costs,
            )

            order_rows.append(
                {
                    "order_id": pending_order.order_id,
                    "ts_submitted": pending_order.ts_submitted,
                    "symbol": pending_order.symbol,
                    "side": pending_order.side,
                    "qty": pending_order.qty,
                    "order_type": resolved_spec.order_type,
                    "price_reference": pending_order.price_reference,
                    "status": "filled",
                    **_reserved_values(
                        run_id=run_id,
                        position_id=pending_order.position_id,
                        parent_position_id=pending_order.parent_position_id,
                        lot_id=pending_order.lot_id,
                        entry_tag=pending_order.entry_tag,
                        exit_tag=pending_order.exit_tag,
                        risk_tag=pending_order.risk_tag,
                        resolver_status="accepted",
                        rejection_reason=None,
                        node_version=pending_order.node_version,
                        contract_version=resolved_spec.contract_version,
                    ),
                }
            )
            fill_rows.append(
                {
                    "fill_id": _next_fill_id(fill_index),
                    "order_id": pending_order.order_id,
                    "ts_fill": bar.timestamp,
                    "symbol": pending_order.symbol,
                    "side": pending_order.side,
                    "qty": pending_order.qty,
                    "fill_price": fill_price,
                    "fees": fees,
                    "slippage": slippage,
                    "gross_notional": gross_notional,
                    **_reserved_values(
                        run_id=run_id,
                        position_id=pending_order.position_id,
                        parent_position_id=pending_order.parent_position_id,
                        lot_id=pending_order.lot_id,
                        entry_tag=pending_order.entry_tag,
                        exit_tag=pending_order.exit_tag,
                        risk_tag=pending_order.risk_tag,
                        resolver_status="filled",
                        rejection_reason=None,
                        node_version=pending_order.node_version,
                        contract_version=resolved_spec.contract_version,
                    ),
                }
            )
            fill_index += 1

            cash, realized_pnl_delta, position, open_lots, new_trade_rows, trade_index = _apply_fill_to_state(
                spec=resolved_spec,
                run_id=run_id,
                bar_index=bar_index,
                fill_ts=bar.timestamp,
                fill_price=fill_price,
                gross_notional=gross_notional,
                fees=fees,
                pending_order=pending_order,
                position=position,
                open_lots=open_lots,
                cash=cash,
                trade_index=trade_index,
            )
            realized_pnl += realized_pnl_delta
            trade_rows.extend(new_trade_rows)

            pending_order = None

        open_lots = _update_open_lot_paths(open_lots, bar)
        position = _position_from_open_lots(spec=resolved_spec, open_lots=open_lots)

        unrealized_pnl, equity, gross_exposure, net_exposure = _mark_to_market(
            position,
            cash,
            bar.close,
            resolved_spec.instrument.contract_multiplier,
        )
        running_peak_equity = max(running_peak_equity, equity)
        drawdown = (
            (equity / running_peak_equity) - 1.0 if running_peak_equity > 0.0 else 0.0
        )

        ctx = DecisionContext(
            bar=bar,
            history=tuple(bars[: bar_index + 1]),
            instrument=resolved_spec.instrument,
            costs=resolved_spec.costs,
            position=position,
            portfolio=_build_portfolio(resolved_spec.instrument.symbol, cash, equity, position),
            session=SessionInfo(
                bar_index=bar_index,
                bars_total=len(bars),
                contract_version=resolved_spec.contract_version,
            ),
            engine_capabilities=effective_capabilities,
            contract_version=resolved_spec.contract_version,
        )

        entry_output = entry_node(ctx)
        entry_batch = _coerce_entry_output(
            entry_output,
            capabilities=effective_capabilities,
            contract=entry_contract,
        )
        exit_output = exit_node(ctx)
        exit_batch = _coerce_exit_output(
            exit_output,
            capabilities=effective_capabilities,
            contract=exit_contract,
        )
        risk_output = risk_node(
            ctx,
            entry_batch,
            exit_batch,
        )
        risk_decision, risk_batch = _coerce_risk_output(
            risk_output,
            capabilities=effective_capabilities,
            contract=risk_contract,
        )

        if resolved_spec.strict_node_output_validation:
            _validate_node_output_strict(
                kind="entry",
                raw_output=entry_output,
                contract=entry_contract,
                ctx=ctx,
                batch=entry_batch,
            )
            _validate_node_output_strict(
                kind="exit",
                raw_output=exit_output,
                contract=exit_contract,
                ctx=ctx,
                batch=exit_batch,
            )
            _validate_node_output_strict(
                kind="risk",
                raw_output=risk_output,
                contract=risk_contract,
                ctx=ctx,
                batch=risk_batch,
                risk_decision=risk_decision,
            )

        next_bar = bars[bar_index + 1] if bar_index < len(bars) - 1 else None
        resolved = _resolve_decision(
            bar=bar,
            next_bar=next_bar,
            position=position,
            entry_batch=entry_batch,
            exit_batch=exit_batch,
            risk_decision=risk_decision,
            spec=resolved_spec,
            order_index=order_index,
            position_index=position_index,
            lot_index=lot_index,
            entry_node_version=node_versions["entry"],
            exit_node_version=node_versions["exit"],
        )

        if resolved.pending_order is not None:
            if position.is_flat and _is_entry_like_action(resolved.pending_order.action_type):
                position_index += 1
            if _is_entry_like_action(resolved.pending_order.action_type):
                lot_index += 1
            order_index += 1
            pending_order = resolved.pending_order

        decision_rows.append(
            {
                "ts": bar.timestamp,
                "symbol": resolved_spec.instrument.symbol,
                "entry_action": _entry_action_label(entry_batch),
                "exit_action": _exit_action_label(exit_batch),
                "risk_approved": resolved.risk_approved,
                "target_units": resolved.target_units,
                "resolved_action": resolved.resolved_action,
                "reason": resolved.reason,
                "metadata": _decision_metadata(
                    timeframe=timeframe,
                    position=position,
                    bar=bar,
                    entry_batch=entry_batch,
                    exit_batch=exit_batch,
                    risk_decision=risk_decision,
                    risk_batch=risk_batch,
                    next_bar=next_bar,
                    resolved=resolved,
                ),
                **_reserved_values(
                    run_id=run_id,
                    position_id=position.position_id or entry_batch.primary_request.position_id,
                    parent_position_id=(
                        position.parent_position_id or entry_batch.primary_request.parent_position_id
                    ),
                    lot_id=position.lot_id or entry_batch.primary_request.lot_id,
                    entry_tag=position.entry_tag or entry_batch.primary_request.entry_tag,
                    exit_tag=position.exit_tag or exit_batch.primary_request.exit_tag,
                    risk_tag=position.risk_tag or entry_batch.primary_request.risk_tag,
                    resolver_status=resolved.resolver_status,
                    rejection_reason=resolved.rejection_reason,
                    node_version=node_version_summary,
                    contract_version=resolved_spec.contract_version,
                ),
            }
        )
        equity_rows.append(
            {
                "ts": bar.timestamp,
                "cash": cash,
                "equity": equity,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "drawdown": drawdown,
                **_reserved_values(
                    run_id=run_id,
                    position_id=position.position_id,
                    parent_position_id=position.parent_position_id,
                    lot_id=position.lot_id,
                    entry_tag=position.entry_tag,
                    exit_tag=position.exit_tag,
                    risk_tag=position.risk_tag,
                    resolver_status="mark_to_market",
                    rejection_reason=None,
                    node_version=None,
                    contract_version=resolved_spec.contract_version,
                ),
            }
        )

    decision_log, order_log, fill_log, trade_ledger, equity_curve = _validate_outputs(
        _as_dataframe(decision_rows, _DECISION_LOG_COLUMNS),
        _as_dataframe(order_rows, _ORDER_LOG_COLUMNS),
        _as_dataframe(fill_rows, _FILL_LOG_COLUMNS),
        _as_dataframe(trade_rows, _TRADE_LEDGER_COLUMNS),
        _as_dataframe(equity_rows, _EQUITY_CURVE_COLUMNS),
    )

    return BacktestResult(
        spec=resolved_spec,
        decision_log=decision_log.reset_index(drop=True),
        order_log=order_log.reset_index(drop=True),
        fill_log=fill_log.reset_index(drop=True),
        trade_ledger=trade_ledger.reset_index(drop=True),
        equity_curve=equity_curve.reset_index(drop=True),
        artifacts=artifacts,
        contract_version=resolved_spec.contract_version,
    )


__all__ = ["BacktestSetupAudit", "audit_backtest_setup", "run_backtest"]
