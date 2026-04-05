from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_lab.contracts import (  # noqa: E402
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
    EntryNode,
    ExitNode,
    InstrumentMeta,
    NodeOutputValidationError,
    NodeContract,
    NodeSpec,
    PortfolioState,
    PositionState,
    UnsupportedNodeActionError,
    UnsupportedNodeRequirementError,
    RiskDecision,
    RiskNode,
    SessionInfo,
    audit_node_compatibility,
    validate_node_compatibility,
)
from trading_lab.registry import NodeRegistry  # noqa: E402


def _make_bar(offset: int = 0) -> Bar:
    timestamp = datetime(2024, 1, 1, 9, 30) + timedelta(minutes=offset)
    return Bar(
        timestamp=timestamp,
        symbol="TEST",
        open=100.0 + offset,
        high=101.0 + offset,
        low=99.0 + offset,
        close=100.5 + offset,
        volume=1_000.0,
    )


def _make_context() -> DecisionContext:
    instrument = InstrumentMeta(
        symbol="TEST",
        price_increment=0.01,
        quantity_increment=1.0,
    )
    costs = CostAssumptions(fee_rate=0.001, fee_per_unit=0.0, slippage_bps=5.0)
    position = PositionState(symbol="TEST")
    portfolio = PortfolioState(cash=10_000.0, equity=10_000.0, positions=(position,))
    bar = _make_bar()
    session = SessionInfo(bar_index=0, bars_total=1)
    return DecisionContext(
        bar=bar,
        history=(bar,),
        instrument=instrument,
        costs=costs,
        position=position,
        portfolio=portfolio,
        session=session,
    )


def _contract(
    name: str,
    kind: str,
    emitted_action_types: tuple[str, ...],
    *,
    required_history: int = 1,
    required_capabilities: tuple[str, ...] = (),
    requires_portfolio_view: bool = False,
    metric_dependencies: tuple[str, ...] = (),
    manifest: dict[str, object] | None = None,
) -> NodeContract:
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind=kind,  # type: ignore[arg-type]
            emitted_action_types=emitted_action_types,  # type: ignore[arg-type]
            required_history=required_history,
            required_capabilities=required_capabilities,
            requires_portfolio_view=requires_portfolio_view,
        ),
        metric_dependencies=metric_dependencies,
        manifest=manifest or {"module": "tests.test_contracts", "parameters": {}},
    )


@pytest.mark.parametrize(
    ("instance", "field_name", "new_value"),
    [
        (_make_bar(), "close", 0.0),
        (ActionRequest(), "reason", "updated"),
        (
            BacktestSpec(
                name="phase-1",
                instrument=InstrumentMeta(
                    symbol="TEST",
                    price_increment=0.01,
                    quantity_increment=1.0,
                ),
                entry_node="entry_sma_cross",
                exit_node="exit_time_stop",
                risk_node="risk_fixed_fraction",
                initial_cash=10_000.0,
            ),
            "entry_node",
            "other_entry",
        ),
    ],
)
def test_frozen_dataclasses_reject_mutation(
    instance: object, field_name: str, new_value: object
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field_name, new_value)


def test_registry_registers_and_resolves_with_explicit_node_contract() -> None:
    registry = NodeRegistry()
    contract = NodeContract(
        spec=NodeSpec(
            name="sample_entry",
            kind="entry",
            version="1.0.0",
            emitted_action_types=("enter_long", "hold"),
            required_history=20,
            description="Example entry node.",
        ),
        manifest={
            "owner": "tests",
            "module": "tests.test_contracts",
            "parameters": {},
        },
    )

    @registry.entry("sample_entry", contract=contract)
    def sample_entry(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="enter_long", units=1.0, reason="cross_up")

    resolved = registry.resolve("entry", "sample_entry")
    resolved_contract = registry.resolve_contract("entry", "sample_entry")

    assert resolved is sample_entry
    assert registry.available("entry") == ("sample_entry",)
    assert resolved_contract is contract
    assert resolved_contract.spec.required_history == 20
    assert resolved_contract.manifest["owner"] == "tests"


def test_registry_rejects_duplicate_names_and_invalid_signatures() -> None:
    registry = NodeRegistry()

    def sample_entry(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="enter_long", units=1.0, reason="cross")

    registry.register(
        "entry",
        "sample",
        sample_entry,
        _contract("sample", "entry", ("enter_long", "hold")),
    )

    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            "entry",
            "sample",
            sample_entry,
            _contract("sample", "entry", ("enter_long", "hold")),
        )

    def bad_entry(context: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="enter_long", units=1.0, reason="bad-name")

    with pytest.raises(TypeError, match="parameters"):
        registry.register(
            "entry",
            "bad",
            bad_entry,
            _contract("bad", "entry", ("enter_long", "hold")),
        )


def test_registry_requires_manifest_name_and_kind_match_registration() -> None:
    registry = NodeRegistry()

    def sample_exit(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="close", reason="time_stop")

    with pytest.raises(ValueError, match="does not match registration name"):
        registry.register(
            "exit",
            "time_stop",
            sample_exit,
            NodeContract(
                spec=NodeSpec(
                    name="different_name",
                    kind="exit",
                    emitted_action_types=("close",),
                ),
                manifest={"module": "tests.test_contracts", "parameters": {}},
            ),
        )

    with pytest.raises(ValueError, match="does not match registration kind"):
        registry.register(
            "exit",
            "time_stop",
            sample_exit,
            NodeContract(
                spec=NodeSpec(
                    name="time_stop",
                    kind="entry",
                    emitted_action_types=("enter_long", "hold"),
                ),
                manifest={"module": "tests.test_contracts", "parameters": {}},
            ),
        )


def test_registry_requires_explicit_contract_manifest() -> None:
    registry = NodeRegistry()

    def sample_risk(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        return RiskDecision(
            allow_entry=entry_intent.is_entry and not exit_intent.is_active,
            entry_quantity=1.0 if entry_intent.is_entry and not exit_intent.is_active else 0.0,
            reason="explicit-only",
        )

    with pytest.raises(TypeError, match="explicit NodeContract or NodeSpec manifest"):
        registry.register("risk", "legacy_risk", sample_risk, None)  # type: ignore[arg-type]


def test_registry_rejects_missing_manifest_metadata() -> None:
    registry = NodeRegistry(enforce_compatibility=False)

    def sample_entry(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    with pytest.raises(ValueError, match="non-empty 'module'"):
        registry.register(
            "entry",
            "missing_module",
            sample_entry,
            NodeContract(
                spec=NodeSpec(
                    name="missing_module",
                    kind="entry",
                    emitted_action_types=("hold",),
                ),
                manifest={"parameters": {}},
            ),
        )

    with pytest.raises(TypeError, match="'parameters' as a dict"):
        registry.register(
            "entry",
            "missing_parameters",
            sample_entry,
            NodeContract(
                spec=NodeSpec(
                    name="missing_parameters",
                    kind="entry",
                    emitted_action_types=("hold",),
                ),
                manifest={"module": "tests.test_contracts"},
            ),
        )


def test_compatibility_failures_are_raised_for_unsupported_features() -> None:
    capabilities = EngineCapabilities(
        supported_capabilities=("market_orders", "next_bar_open_fills"),
        supports_metric_dependency_checks=False,
    )
    registry = NodeRegistry(engine_capabilities=capabilities)

    def portfolio_entry(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="enter_long", units=1.0, reason="rebalance")

    with pytest.raises(CompatibilityError, match="portfolio_view"):
        registry.register(
            "entry",
            "portfolio_entry",
            portfolio_entry,
            NodeContract(
                spec=NodeSpec(
                    name="portfolio_entry",
                    kind="entry",
                    emitted_action_types=("enter_long", "hold"),
                    requires_portfolio_view=True,
                ),
                manifest={"module": "tests.test_contracts", "parameters": {}},
            ),
        )

    with pytest.raises(CompatibilityError, match="metric_dependency_checks"):
        registry.register(
            "entry",
            "metrics_aware_entry",
            portfolio_entry,
            NodeContract(
                spec=NodeSpec(
                    name="metrics_aware_entry",
                    kind="entry",
                    emitted_action_types=("enter_long", "hold"),
                ),
                metric_dependencies=("equity_curve.drawdown",),
                manifest={"module": "tests.test_contracts", "parameters": {}},
            ),
        )


def test_reserved_future_actions_are_audited_as_engine_feature_requests() -> None:
    contract = NodeContract(
        spec=NodeSpec(
            name="future_set_stop",
            kind="exit",
            emitted_action_types=("set_stop", "hold"),
        ),
        manifest={"module": "tests.test_contracts", "parameters": {}},
    )
    audit = audit_node_compatibility(contract, EngineCapabilities())

    assert isinstance(audit, CompatibilityAudit)
    assert audit.supported is False
    assert "action:set_stop" in audit.required_engine_changes
    assert "engine feature request" in audit.summary

    with pytest.raises(UnsupportedNodeActionError, match="engine feature request"):
        validate_node_compatibility(contract, EngineCapabilities())


def test_action_batch_tracks_active_requests_and_contract_versions() -> None:
    batch = ActionBatch(
        requests=(
            ActionRequest(action_type="hold"),
            ActionRequest(action_type="enter_long", units=1.0, reason="open"),
            ActionRequest(action_type="increase", units=0.5, reason="add"),
        )
    )

    assert batch.is_active is True
    assert batch.primary_request.action_type == "enter_long"
    assert batch.action_types == ("enter_long", "increase")

    with pytest.raises(ValueError, match="share the batch contract_version"):
        ActionBatch(
            requests=(
                ActionRequest(action_type="hold", contract_version="1.0"),
                ActionRequest(
                    action_type="enter_long",
                    units=1.0,
                    reason="bad-version",
                    contract_version="2.0",
                ),
            ),
            contract_version="1.0",
        )


def test_action_request_rejects_unknown_action_type_immediately() -> None:
    with pytest.raises(ValueError, match="unsupported action_type"):
        ActionRequest(action_type="explode", reason="bad-action")  # type: ignore[arg-type]


def test_registry_audit_explains_when_node_requires_engine_changes() -> None:
    registry = NodeRegistry(enforce_compatibility=False)

    def allocator_entry(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    registry.register(
        "entry",
        "allocator_entry",
        allocator_entry,
        NodeContract(
            spec=NodeSpec(
                name="allocator_entry",
                kind="entry",
                emitted_action_types=("hold",),
                required_capabilities=("portfolio_allocator",),
            ),
            manifest={"module": "tests.test_contracts", "parameters": {}},
        ),
    )

    audit = registry.audit("entry", "allocator_entry")

    assert audit.supported is False
    assert "capability:portfolio_allocator" in audit.required_engine_changes
    assert "engine feature request" in audit.summary

    with pytest.raises(UnsupportedNodeRequirementError, match="portfolio_allocator"):
        registry.validate("entry", "allocator_entry")


def test_node_spec_rejects_actions_not_supported_for_kind() -> None:
    with pytest.raises(ValueError, match="exit nodes may only emit"):
        NodeSpec(
            name="bad_exit",
            kind="exit",
            emitted_action_types=("enter_long",),
        )


def test_protocol_compatible_callable_objects_satisfy_node_interfaces() -> None:
    ctx = _make_context()

    class ExampleEntry:
        def __call__(self, ctx: DecisionContext) -> ActionRequest:
            return ActionRequest(action_type="enter_long", units=1.0, reason="entry")

    class ExampleExit:
        def __call__(self, ctx: DecisionContext) -> ActionRequest:
            return ActionRequest(action_type="hold")

    class ExampleRisk:
        def __call__(
            self,
            ctx: DecisionContext,
            entry_intent: ActionBatch,
            exit_intent: ActionBatch,
        ) -> RiskDecision:
            if entry_intent.is_entry and not exit_intent.is_active:
                return RiskDecision(
                    allow_entry=True,
                    entry_quantity=10.0,
                    reason="fixed_fraction",
                )
            return RiskDecision()

    entry_node = ExampleEntry()
    exit_node = ExampleExit()
    risk_node = ExampleRisk()

    assert isinstance(entry_node, EntryNode)
    assert isinstance(exit_node, ExitNode)
    assert isinstance(risk_node, RiskNode)

    entry_request = entry_node(ctx)
    exit_request = exit_node(ctx)
    risk_decision = risk_node(
        ctx,
        ActionBatch(requests=(entry_request,)),
        ActionBatch(requests=(exit_request,)),
    )

    assert entry_request.action_type == "enter_long"
    assert exit_request.action_type == "hold"
    assert risk_decision.allow_entry is True
    assert risk_decision.entry_quantity == pytest.approx(10.0)


def test_backtest_result_artifacts_are_validated_and_copied() -> None:
    spec = BacktestSpec(
        name="artifacts-test",
        instrument=InstrumentMeta(
            symbol="TEST",
            price_increment=0.01,
            quantity_increment=1.0,
        ),
        entry_node="entry_sma_cross",
        exit_node="exit_time_stop",
        risk_node="risk_fixed_fraction",
        initial_cash=10_000.0,
    )
    artifacts = {"summary": {"trades": 1}}

    result = BacktestResult(
        spec=spec,
        decision_log=pd.DataFrame(),
        order_log=pd.DataFrame(),
        fill_log=pd.DataFrame(),
        trade_ledger=pd.DataFrame(),
        equity_curve=pd.DataFrame(),
        artifacts=artifacts,
    )

    assert result.artifacts == artifacts
    assert result.artifacts is not artifacts

    with pytest.raises(TypeError, match="artifacts must be a dict"):
        BacktestResult(
            spec=spec,
            decision_log=pd.DataFrame(),
            order_log=pd.DataFrame(),
            fill_log=pd.DataFrame(),
            trade_ledger=pd.DataFrame(),
            equity_curve=pd.DataFrame(),
            artifacts=["not", "a", "dict"],  # type: ignore[arg-type]
        )


def test_backtest_spec_validates_random_seed_when_provided() -> None:
    spec = BacktestSpec(
        name="seeded",
        instrument=InstrumentMeta(
            symbol="TEST",
            price_increment=0.01,
            quantity_increment=1.0,
        ),
        entry_node="entry_sma_cross",
        exit_node="exit_time_stop",
        risk_node="risk_fixed_fraction",
        initial_cash=10_000.0,
        random_seed=42,
    )

    assert spec.random_seed == 42

    with pytest.raises(TypeError, match="random_seed must be an int"):
        BacktestSpec(
            name="bad-seed",
            instrument=InstrumentMeta(
                symbol="TEST",
                price_increment=0.01,
                quantity_increment=1.0,
            ),
            entry_node="entry_sma_cross",
            exit_node="exit_time_stop",
            risk_node="risk_fixed_fraction",
            initial_cash=10_000.0,
            random_seed=3.14,  # type: ignore[arg-type]
        )
