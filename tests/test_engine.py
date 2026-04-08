from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import trading_lab.engine as engine_module  # noqa: E402
from trading_lab.contracts import (  # noqa: E402
    ActionBatch,
    ActionRequest,
    BacktestSpec,
    CompatibilityError,
    CostAssumptions,
    DecisionContext,
    EngineCapabilities,
    EntryIntent,
    ExitIntent,
    InstrumentMeta,
    NodeOutputValidationError,
    NodeContract,
    NodeSpec,
    PACKAGE_VERSION,
    RiskDecision,
    SetupCompatibilityError,
)
from trading_lab.engine import audit_backtest_setup, prepare_backtest_data, run_backtest  # noqa: E402
from trading_lab.nodes.entry_sma_cross import (  # noqa: E402
    SMACrossEntryNode,
    build_entry_sma_cross_contract,
)
from trading_lab.nodes.exit_no_progress import (  # noqa: E402
    NoProgressExitNode,
    build_exit_no_progress_contract,
)
from trading_lab.nodes.exit_time_stop import (  # noqa: E402
    TimeStopExitNode,
    build_exit_time_stop_contract,
)
from trading_lab.nodes.risk_fixed_fraction import (  # noqa: E402
    FixedFractionRiskNode,
    build_risk_fixed_fraction_contract,
)
from trading_lab.registry import (  # noqa: E402
    NodeRegistry,
    resolve_contract,
    resolve_entry,
    resolve_exit,
    resolve_risk,
)


def _make_ohlcv(opens: list[float], *, symbol: str = "TEST") -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:31:00", periods=len(opens), freq="1min")
    rows = []
    for ts, open_price in zip(timestamps, opens):
        rows.append(
            {
                "ts": ts,
                "symbol": symbol,
                "timeframe": "1m",
                "open": open_price,
                "high": open_price + 1.0,
                "low": open_price - 1.0,
                "close": open_price,
                "volume": 1_000.0,
            }
        )
    return pd.DataFrame(rows)


def _make_spec(
    entry_name: str,
    exit_name: str,
    risk_name: str,
    *,
    costs: CostAssumptions | None = None,
    allow_short: bool = True,
    engine_capabilities: EngineCapabilities | None = None,
    random_seed: int | None = None,
) -> BacktestSpec:
    return BacktestSpec(
        name="engine-test",
        instrument=InstrumentMeta(
            symbol="TEST",
            price_increment=0.01,
            quantity_increment=1.0,
        ),
        entry_node=entry_name,
        exit_node=exit_name,
        risk_node=risk_name,
        initial_cash=10_000.0,
        costs=costs or CostAssumptions(),
        allow_short=allow_short,
        random_seed=random_seed,
        engine_capabilities=engine_capabilities or EngineCapabilities(),
    )


def _contract(
    name: str,
    kind: str,
    emitted_action_types: tuple[str, ...],
    *,
    required_history: int = 1,
    required_capabilities: tuple[str, ...] = (),
    requires_portfolio_view: bool = False,
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
        manifest={"module": "tests.test_engine", "parameters": {}},
    )


def test_next_bar_open_execution_timing_and_trade_ledger_for_long() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "entry_long_once",
        contract=_contract("entry_long_once", "entry", ("enter_long", "hold")),
    )
    def entry_long_once(ctx: DecisionContext) -> EntryIntent:
        if ctx.session.bar_index == 0 and ctx.position.is_flat:
            return EntryIntent(action="enter_long", reason="open-long")
        return EntryIntent()

    @registry.exit(
        "exit_after_two_bars",
        contract=_contract("exit_after_two_bars", "exit", ("close", "hold")),
    )
    def exit_after_two_bars(ctx: DecisionContext) -> ExitIntent:
        if ctx.session.bar_index == 2 and ctx.position.is_long:
            return ExitIntent(action="exit", reason="time-exit")
        return ExitIntent()

    @registry.risk(
        "risk_one_unit",
        contract=_contract("risk_one_unit", "risk", ("hold",)),
    )
    def risk_one_unit(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_active:
            return RiskDecision(
                allow_entry=True,
                entry_quantity=1.0,
                reason="one-unit",
            )
        return RiskDecision()

    result = run_backtest(
        _make_spec("entry_long_once", "exit_after_two_bars", "risk_one_unit"),
        _make_ohlcv([100.0, 101.0, 102.0, 103.0, 104.0]),
        node_registry=registry,
    )

    assert list(result.order_log["ts_submitted"]) == [
        pd.Timestamp("2024-01-02 09:31:00"),
        pd.Timestamp("2024-01-02 09:33:00"),
    ]
    assert list(result.fill_log["ts_fill"]) == [
        pd.Timestamp("2024-01-02 09:32:00"),
        pd.Timestamp("2024-01-02 09:34:00"),
    ]
    assert list(result.fill_log["fill_price"]) == [pytest.approx(101.0), pytest.approx(103.0)]
    assert list(result.decision_log["resolved_action"]) == [
        "submit_entry_long",
        "hold",
        "submit_exit",
        "hold",
        "hold",
    ]

    trade = result.trade_ledger.iloc[0]
    assert trade["side"] == "long"
    assert trade["entry_ts"] == pd.Timestamp("2024-01-02 09:32:00")
    assert trade["exit_ts"] == pd.Timestamp("2024-01-02 09:34:00")
    assert trade["qty"] == pytest.approx(1.0)
    assert trade["gross_pnl"] == pytest.approx(2.0)
    assert trade["net_pnl"] == pytest.approx(2.0)
    assert trade["bars_held"] == 2
    assert trade["exit_reason"] == "time-exit"


def test_no_progress_exit_uses_position_entry_time_to_close_trade() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "entry_long_once",
        contract=_contract("entry_long_once", "entry", ("enter_long", "hold")),
    )
    def entry_long_once(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 0 and ctx.position.is_flat:
            return ActionRequest(action_type="enter_long", units=1.0, reason="open")
        return ActionRequest(action_type="hold")

    registry.register(
        "exit",
        "exit_no_progress_test",
        NoProgressExitNode(evaluation_bars=2, min_open_profit_points=0.0),
        build_exit_no_progress_contract(
            name="exit_no_progress_test",
            evaluation_bars=2,
            min_open_profit_points=0.0,
        ),
    )

    @registry.risk(
        "risk_one_unit",
        contract=_contract("risk_one_unit", "risk", ("hold",)),
    )
    def risk_one_unit(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_active:
            return RiskDecision(
                allow_entry=True,
                entry_quantity=1.0,
                reason="one-unit",
            )
        return RiskDecision()

    result = run_backtest(
        _make_spec("entry_long_once", "exit_no_progress_test", "risk_one_unit"),
        _make_ohlcv([100.0, 100.0, 99.0, 98.0, 97.0]),
        node_registry=registry,
    )

    assert len(result.trade_ledger) == 1
    trade = result.trade_ledger.iloc[0]
    assert trade["entry_ts"] == pd.Timestamp("2024-01-02 09:32:00")
    assert trade["exit_ts"] == pd.Timestamp("2024-01-02 09:34:00")
    assert trade["bars_held"] == 2
    assert "no_progress" in trade["exit_reason"]


def test_run_backtest_can_skip_output_validation(monkeypatch) -> None:
    registry = NodeRegistry()

    @registry.entry(
        "entry_long_once",
        contract=_contract("entry_long_once", "entry", ("enter_long", "hold")),
    )
    def entry_long_once(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 0 and ctx.position.is_flat:
            return ActionRequest(action_type="enter_long", units=1.0, reason="open")
        return ActionRequest()

    @registry.exit(
        "exit_after_one_bar",
        contract=_contract("exit_after_one_bar", "exit", ("close", "hold")),
    )
    def exit_after_one_bar(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 1 and ctx.position.is_long:
            return ActionRequest(action_type="close", reason="time-exit")
        return ActionRequest()

    @registry.risk(
        "risk_one_unit",
        contract=_contract("risk_one_unit", "risk", ("hold",)),
    )
    def risk_one_unit(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_active:
            return RiskDecision(
                allow_entry=True,
                entry_quantity=1.0,
                reason="one-unit",
            )
        return RiskDecision()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("_validate_outputs should be skipped when validate_outputs=False")

    monkeypatch.setattr(engine_module, "_validate_outputs", fail_if_called)

    result = run_backtest(
        _make_spec("entry_long_once", "exit_after_one_bar", "risk_one_unit"),
        _make_ohlcv([100.0, 101.0, 102.0]),
        node_registry=registry,
        validate_outputs=False,
    )

    assert len(result.trade_ledger) == 1
    assert list(result.decision_log["resolved_action"]) == [
        "submit_entry_long",
        "submit_exit",
        "hold",
    ]


def test_run_backtest_can_disable_heavy_decision_request_details() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "entry_long_once",
        contract=_contract("entry_long_once", "entry", ("enter_long", "hold")),
    )
    def entry_long_once(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 0 and ctx.position.is_flat:
            return ActionRequest(action_type="enter_long", units=1.0, reason="open")
        return ActionRequest()

    @registry.exit(
        "exit_after_one_bar",
        contract=_contract("exit_after_one_bar", "exit", ("close", "hold")),
    )
    def exit_after_one_bar(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 1 and ctx.position.is_long:
            return ActionRequest(action_type="close", reason="time-exit")
        return ActionRequest()

    @registry.risk(
        "risk_one_unit",
        contract=_contract("risk_one_unit", "risk", ("hold",)),
    )
    def risk_one_unit(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_active:
            return RiskDecision(
                allow_entry=True,
                entry_quantity=1.0,
                reason="one-unit",
            )
        return RiskDecision()

    result = run_backtest(
        _make_spec("entry_long_once", "exit_after_one_bar", "risk_one_unit"),
        _make_ohlcv([100.0, 101.0, 102.0]),
        node_registry=registry,
        capture_decision_details=False,
    )

    first_metadata = result.decision_log.iloc[0]["metadata"]
    assert first_metadata["timeframe"] == "1m"
    assert first_metadata["entry_reason"] == "open"
    assert "entry_requests" not in first_metadata
    assert "rejected_requests" not in first_metadata


def test_run_backtest_accepts_prepared_backtest_data() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "entry_long_once",
        contract=_contract("entry_long_once", "entry", ("enter_long", "hold")),
    )
    def entry_long_once(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 0 and ctx.position.is_flat:
            return ActionRequest(action_type="enter_long", units=1.0, reason="open")
        return ActionRequest()

    @registry.exit(
        "exit_after_one_bar",
        contract=_contract("exit_after_one_bar", "exit", ("close", "hold")),
    )
    def exit_after_one_bar(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 1 and ctx.position.is_long:
            return ActionRequest(action_type="close", reason="time-exit")
        return ActionRequest()

    @registry.risk(
        "risk_one_unit",
        contract=_contract("risk_one_unit", "risk", ("hold",)),
    )
    def risk_one_unit(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_active:
            return RiskDecision(
                allow_entry=True,
                entry_quantity=1.0,
                reason="one-unit",
            )
        return RiskDecision()

    dataset = _make_ohlcv([100.0, 101.0, 102.0])
    spec = _make_spec("entry_long_once", "exit_after_one_bar", "risk_one_unit")
    prepared = prepare_backtest_data(spec, dataset)

    raw_result = run_backtest(spec, dataset, node_registry=registry)
    prepared_result = run_backtest(spec, prepared, node_registry=registry)

    pd.testing.assert_frame_equal(raw_result.decision_log, prepared_result.decision_log)
    pd.testing.assert_frame_equal(raw_result.order_log, prepared_result.order_log)
    pd.testing.assert_frame_equal(raw_result.fill_log, prepared_result.fill_log)
    pd.testing.assert_frame_equal(raw_result.trade_ledger, prepared_result.trade_ledger)
    pd.testing.assert_frame_equal(raw_result.equity_curve, prepared_result.equity_curve)


def test_run_manifest_and_run_id_are_stamped_for_reproducibility() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "entry_once_manifest",
        contract=_contract("entry_once_manifest", "entry", ("enter_long", "hold")),
    )
    def entry_once_manifest(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 0 and ctx.position.is_flat:
            return ActionRequest(action_type="enter_long", units=1.0, reason="open")
        return ActionRequest(action_type="hold")

    @registry.exit(
        "exit_once_manifest",
        contract=_contract("exit_once_manifest", "exit", ("close", "hold")),
    )
    def exit_once_manifest(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 2 and ctx.position.is_long:
            return ActionRequest(action_type="close", reason="close")
        return ActionRequest(action_type="hold")

    @registry.risk(
        "risk_manifest",
        contract=_contract("risk_manifest", "risk", ("hold",)),
    )
    def risk_manifest(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_entry:
            return RiskDecision(allow_entry=True, entry_quantity=1.0, reason="approve")
        return RiskDecision()

    result = run_backtest(
        _make_spec(
            "entry_once_manifest",
            "exit_once_manifest",
            "risk_manifest",
            costs=CostAssumptions(fee_rate=0.001, fee_per_unit=0.25, slippage_bps=5.0),
            random_seed=7,
        ),
        _make_ohlcv([100.0, 101.0, 102.0, 103.0, 104.0]),
        node_registry=registry,
    )

    run_manifest = result.artifacts["run_manifest"]
    run_id = run_manifest["run_id"]

    assert isinstance(run_id, str)
    assert run_id.startswith("run-")
    assert run_manifest["package_version"] == PACKAGE_VERSION
    assert isinstance(run_manifest["spec_hash"], str)
    assert isinstance(run_manifest["data_hash"], str)
    assert run_manifest["random_seed"] == 7
    assert run_manifest["spec"]["costs"]["slippage_bps"] == pytest.approx(5.0)
    assert run_manifest["spec"]["costs"]["fee_per_unit"] == pytest.approx(0.25)
    assert run_manifest["nodes"]["entry"]["manifest"]["module"] == "tests.test_engine"
    assert run_manifest["nodes"]["risk"]["manifest"]["parameters"] == {}
    assert run_manifest["node_versions"]["entry"] == "0.1.0"

    for ledger in (
        result.decision_log,
        result.order_log,
        result.fill_log,
        result.trade_ledger,
        result.equity_curve,
    ):
        assert set(ledger["run_id"]) == {run_id}


def test_short_path_and_equity_curve_updates() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "entry_short_once",
        contract=_contract("entry_short_once", "entry", ("enter_short", "hold")),
    )
    def entry_short_once(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 0 and ctx.position.is_flat:
            return ActionRequest(action_type="enter_short", units=1.0, reason="open-short")
        return ActionRequest(action_type="hold")

    @registry.exit(
        "exit_short_after_two_bars",
        contract=_contract("exit_short_after_two_bars", "exit", ("close", "hold")),
    )
    def exit_short_after_two_bars(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 2 and ctx.position.is_short:
            return ActionRequest(action_type="close", reason="cover-short")
        return ActionRequest(action_type="hold")

    @registry.risk(
        "risk_one_unit_short",
        contract=_contract("risk_one_unit_short", "risk", ("hold",)),
    )
    def risk_one_unit_short(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_entry:
            return RiskDecision(
                allow_entry=True,
                entry_quantity=1.0,
                reason="one-unit",
            )
        return RiskDecision()

    result = run_backtest(
        _make_spec(
            "entry_short_once",
            "exit_short_after_two_bars",
            "risk_one_unit_short",
        ),
        _make_ohlcv([105.0, 104.0, 103.0, 102.0, 101.0]),
        node_registry=registry,
    )

    trade = result.trade_ledger.iloc[0]
    assert trade["side"] == "short"
    assert trade["gross_pnl"] == pytest.approx(2.0)
    assert trade["net_pnl"] == pytest.approx(2.0)

    equity = result.equity_curve.set_index("ts")
    assert equity.loc[pd.Timestamp("2024-01-02 09:31:00"), "equity"] == pytest.approx(
        10_000.0
    )
    assert equity.loc[
        pd.Timestamp("2024-01-02 09:32:00"), "gross_exposure"
    ] == pytest.approx(104.0)
    assert equity.loc[
        pd.Timestamp("2024-01-02 09:32:00"), "net_exposure"
    ] == pytest.approx(-104.0)
    assert equity.loc[
        pd.Timestamp("2024-01-02 09:33:00"), "unrealized_pnl"
    ] == pytest.approx(1.0)
    assert equity.loc[
        pd.Timestamp("2024-01-02 09:34:00"), "realized_pnl"
    ] == pytest.approx(2.0)
    assert equity.loc[pd.Timestamp("2024-01-02 09:34:00"), "gross_exposure"] == 0.0
    assert equity.loc[pd.Timestamp("2024-01-02 09:34:00"), "equity"] == pytest.approx(
        10_002.0
    )


def test_fees_and_slippage_reduce_trade_pnl() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "entry_long_costs",
        contract=_contract("entry_long_costs", "entry", ("enter_long", "hold")),
    )
    def entry_long_costs(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 0 and ctx.position.is_flat:
            return ActionRequest(action_type="enter_long", units=1.0, reason="cost-long")
        return ActionRequest(action_type="hold")

    @registry.exit(
        "exit_long_costs",
        contract=_contract("exit_long_costs", "exit", ("close", "hold")),
    )
    def exit_long_costs(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 2 and ctx.position.is_long:
            return ActionRequest(action_type="close", reason="cost-exit")
        return ActionRequest(action_type="hold")

    @registry.risk(
        "risk_costs",
        contract=_contract("risk_costs", "risk", ("hold",)),
    )
    def risk_costs(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_entry:
            return RiskDecision(
                allow_entry=True,
                entry_quantity=1.0,
                reason="one-unit",
            )
        return RiskDecision()

    base_data = _make_ohlcv([100.0, 101.0, 102.0, 103.0, 104.0])
    zero_cost = run_backtest(
        _make_spec("entry_long_costs", "exit_long_costs", "risk_costs"),
        base_data,
        node_registry=registry,
    )
    with_costs = run_backtest(
        _make_spec(
            "entry_long_costs",
            "exit_long_costs",
            "risk_costs",
            costs=CostAssumptions(fee_rate=0.001, slippage_bps=50.0),
        ),
        base_data,
        node_registry=registry,
    )

    assert with_costs.fill_log.iloc[0]["fill_price"] > zero_cost.fill_log.iloc[0]["fill_price"]
    assert with_costs.fill_log.iloc[1]["fill_price"] < zero_cost.fill_log.iloc[1]["fill_price"]
    assert with_costs.fill_log["slippage"].sum() > 0.0
    assert with_costs.trade_ledger.iloc[0]["fees"] > 0.0
    assert with_costs.trade_ledger.iloc[0]["net_pnl"] < zero_cost.trade_ledger.iloc[0][
        "net_pnl"
    ]


def test_exit_precedence_blocks_same_bar_reversal() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "entry_reverse_later",
        contract=_contract(
            "entry_reverse_later",
            "entry",
            ("enter_long", "enter_short", "hold"),
        ),
    )
    def entry_reverse_later(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 0 and ctx.position.is_flat:
            return ActionRequest(action_type="enter_long", units=1.0, reason="initial-long")
        if ctx.session.bar_index >= 2:
            return ActionRequest(action_type="enter_short", units=1.0, reason="reverse-short")
        return ActionRequest(action_type="hold")

    @registry.exit(
        "exit_reverse_bar",
        contract=_contract("exit_reverse_bar", "exit", ("close", "hold")),
    )
    def exit_reverse_bar(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 2 and ctx.position.is_long:
            return ActionRequest(action_type="close", reason="exit-now")
        return ActionRequest(action_type="hold")

    @registry.risk(
        "risk_reverse",
        contract=_contract("risk_reverse", "risk", ("hold",)),
    )
    def risk_reverse(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_entry:
            return RiskDecision(
                allow_entry=True,
                entry_quantity=1.0,
                reason="one-unit",
            )
        return RiskDecision()

    result = run_backtest(
        _make_spec("entry_reverse_later", "exit_reverse_bar", "risk_reverse"),
        _make_ohlcv([100.0, 101.0, 102.0, 103.0, 104.0]),
        node_registry=registry,
    )

    decisions = result.decision_log.set_index("ts")
    assert decisions.loc[pd.Timestamp("2024-01-02 09:33:00"), "resolved_action"] == "submit_exit"
    assert decisions.loc[pd.Timestamp("2024-01-02 09:33:00"), "reason"] == "exit-now"
    assert decisions.loc[pd.Timestamp("2024-01-02 09:34:00"), "resolved_action"] == "submit_entry_short"
    assert decisions.loc[pd.Timestamp("2024-01-02 09:34:00"), "reason"] == "reverse-short"
    assert len(
        result.order_log[
            result.order_log["ts_submitted"] == pd.Timestamp("2024-01-02 09:33:00")
        ]
    ) == 1


def test_action_batches_support_scale_in_and_partial_exit() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "batch_entry",
        contract=NodeContract(
            spec=NodeSpec(
                name="batch_entry",
                kind="entry",
                emitted_action_types=("enter_long", "scale_in", "hold"),
            ),
            manifest={"module": "tests.test_engine", "parameters": {}},
        ),
    )
    def batch_entry(ctx: DecisionContext) -> ActionBatch:
        if ctx.session.bar_index == 0 and ctx.position.is_flat:
            return ActionBatch(
                requests=(
                    ActionRequest(action_type="enter_long", units=1.0, reason="open"),
                )
            )
        if ctx.session.bar_index == 1 and ctx.position.is_long:
            return ActionBatch(
                requests=(
                    ActionRequest(action_type="scale_in", units=1.0, reason="add"),
                )
            )
        return ActionBatch()

    @registry.exit(
        "batch_exit",
        contract=NodeContract(
            spec=NodeSpec(
                name="batch_exit",
                kind="exit",
                emitted_action_types=("partial_exit", "close", "hold"),
            ),
            manifest={"module": "tests.test_engine", "parameters": {}},
        ),
    )
    def batch_exit(ctx: DecisionContext) -> ActionBatch:
        if ctx.session.bar_index == 2 and ctx.position.is_long:
            return ActionBatch(
                requests=(
                    ActionRequest(action_type="partial_exit", units=0.5, reason="trim"),
                )
            )
        if ctx.session.bar_index == 3 and ctx.position.is_long:
            return ActionBatch(
                requests=(ActionRequest(action_type="close", reason="flatten"),)
            )
        return ActionBatch()

    @registry.risk(
        "batch_risk",
        contract=_contract("batch_risk", "risk", ("hold",)),
    )
    def batch_risk(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_active:
            return RiskDecision(allow_entry=True, entry_quantity=1.0, reason="approved")
        if exit_intent.is_active:
            return RiskDecision(allow_exit=True, reason="approved-exit")
        return RiskDecision()

    result = run_backtest(
        _make_spec("batch_entry", "batch_exit", "batch_risk"),
        _make_ohlcv([100.0, 101.0, 102.0, 104.0, 105.0, 106.0]),
        node_registry=registry,
    )

    assert list(result.decision_log["resolved_action"])[:4] == [
        "submit_entry_long",
        "submit_scale_in",
        "submit_partial_exit",
        "submit_exit",
    ]
    assert list(result.order_log["qty"]) == [
        pytest.approx(1.0),
        pytest.approx(1.0),
        pytest.approx(0.5),
        pytest.approx(1.5),
    ]
    assert list(result.order_log.loc[result.order_log["side"] == "buy", "lot_id"]) == [
        "lot-000001",
        "lot-000002",
    ]
    assert result.trade_ledger["position_id"].nunique() == 1
    assert list(result.trade_ledger["lot_id"]) == [
        "lot-000001",
        "lot-000001",
        "lot-000002",
    ]
    assert list(result.trade_ledger["qty"]) == [
        pytest.approx(0.5),
        pytest.approx(0.5),
        pytest.approx(1.0),
    ]
    assert list(result.trade_ledger["entry_price"]) == [
        pytest.approx(101.0),
        pytest.approx(101.0),
        pytest.approx(102.0),
    ]
    assert list(result.trade_ledger["exit_price"]) == [
        pytest.approx(104.0),
        pytest.approx(105.0),
        pytest.approx(105.0),
    ]
    assert result.trade_ledger["gross_pnl"].sum() == pytest.approx(6.5)
    assert result.trade_ledger["net_pnl"].sum() == pytest.approx(6.5)
    assert list(result.trade_ledger["mfe"]) == [
        pytest.approx(1.0),
        pytest.approx(2.0),
        pytest.approx(3.0),
    ]
    assert list(result.trade_ledger["mae"]) == [
        pytest.approx(-0.5),
        pytest.approx(-0.5),
        pytest.approx(-1.0),
    ]
    equity = result.equity_curve.set_index("ts")
    assert equity.loc[pd.Timestamp("2024-01-02 09:34:00"), "realized_pnl"] == pytest.approx(
        1.5
    )


def test_full_close_takes_precedence_over_partial_exit_within_batch() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "open_once",
        contract=_contract("open_once", "entry", ("enter_long", "hold")),
    )
    def open_once(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 0 and ctx.position.is_flat:
            return ActionRequest(action_type="enter_long", units=1.0, reason="open")
        return ActionRequest(action_type="hold")

    @registry.exit(
        "close_conflict",
        contract=NodeContract(
            spec=NodeSpec(
                name="close_conflict",
                kind="exit",
                emitted_action_types=("partial_exit", "close", "hold"),
            ),
            manifest={"module": "tests.test_engine", "parameters": {}},
        ),
    )
    def close_conflict(ctx: DecisionContext) -> ActionBatch:
        if ctx.session.bar_index == 2 and ctx.position.is_long:
            return ActionBatch(
                requests=(
                    ActionRequest(action_type="partial_exit", units=0.5, reason="trim"),
                    ActionRequest(action_type="close", reason="flatten"),
                )
            )
        return ActionBatch()

    @registry.risk(
        "always_allow",
        contract=_contract("always_allow", "risk", ("hold",)),
    )
    def always_allow(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_active:
            return RiskDecision(allow_entry=True, entry_quantity=1.0, reason="enter-ok")
        if exit_intent.is_active:
            return RiskDecision(allow_exit=True, reason="exit-ok")
        return RiskDecision()

    result = run_backtest(
        _make_spec("open_once", "close_conflict", "always_allow"),
        _make_ohlcv([100.0, 101.0, 102.0, 103.0, 104.0]),
        node_registry=registry,
    )

    decision = result.decision_log.set_index("ts").loc[pd.Timestamp("2024-01-02 09:33:00")]
    assert decision["resolved_action"] == "submit_exit"
    assert decision["resolver_status"] == "accepted_with_rejections"
    assert "full_close_takes_precedence" in str(decision["rejection_reason"])
    assert result.order_log.iloc[1]["qty"] == pytest.approx(1.0)
    rejected = decision["metadata"]["rejected_requests"]
    assert rejected[0]["action_type"] == "partial_exit"
    assert rejected[0]["rejection_reason"] == "full_close_takes_precedence"


def test_over_reduction_requests_are_rejected_without_creating_orders() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "open_and_hold",
        contract=_contract("open_and_hold", "entry", ("enter_long", "hold")),
    )
    def open_and_hold(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 0 and ctx.position.is_flat:
            return ActionRequest(action_type="enter_long", units=1.0, reason="open")
        return ActionRequest(action_type="hold")

    @registry.exit(
        "reduce_too_much",
        contract=NodeContract(
            spec=NodeSpec(
                name="reduce_too_much",
                kind="exit",
                emitted_action_types=("partial_exit", "hold"),
            ),
            manifest={"module": "tests.test_engine", "parameters": {}},
        ),
    )
    def reduce_too_much(ctx: DecisionContext) -> ActionBatch:
        if ctx.session.bar_index == 1 and ctx.position.is_long:
            return ActionBatch(
                requests=(
                    ActionRequest(
                        action_type="partial_exit",
                        units=2.0,
                        reason="too-much",
                    ),
                )
            )
        return ActionBatch()

    @registry.risk(
        "allow_entry_only",
        contract=_contract("allow_entry_only", "risk", ("hold",)),
    )
    def allow_entry_only(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_active:
            return RiskDecision(allow_entry=True, entry_quantity=1.0, reason="enter-ok")
        return RiskDecision(allow_exit=True, reason="exit-ok")

    result = run_backtest(
        _make_spec("open_and_hold", "reduce_too_much", "allow_entry_only"),
        _make_ohlcv([100.0, 101.0, 102.0, 103.0]),
        node_registry=registry,
    )

    decisions = result.decision_log.set_index("ts")
    reduce_decision = decisions.loc[pd.Timestamp("2024-01-02 09:32:00")]
    assert reduce_decision["resolved_action"] == "blocked_reduce"
    assert "requested_reduce_exceeds_position" in str(reduce_decision["rejection_reason"])
    assert len(result.order_log) == 1


def test_unsupported_action_is_rejected_clearly() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "bad_entry_action",
        contract=_contract("bad_entry_action", "entry", ("enter_long", "hold")),
    )
    def bad_entry_action(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="close", reason="illegal-close")

    @registry.exit(
        "noop_exit",
        contract=_contract("noop_exit", "exit", ("close", "hold")),
    )
    def noop_exit(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    @registry.risk(
        "noop_risk",
        contract=_contract("noop_risk", "risk", ("hold",)),
    )
    def noop_risk(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        return RiskDecision()

    with pytest.raises(
        NodeOutputValidationError,
        match="outside its declared manifest",
    ):
        run_backtest(
            _make_spec("bad_entry_action", "noop_exit", "noop_risk"),
            _make_ohlcv([100.0, 101.0, 102.0]),
            node_registry=registry,
        )


def test_compatibility_checks_block_invalid_node_engine_combinations() -> None:
    registry = NodeRegistry(enforce_compatibility=False)

    @registry.entry(
        "requires_portfolio_allocator",
        contract=NodeContract(
            spec=NodeSpec(
                name="requires_portfolio_allocator",
                kind="entry",
                emitted_action_types=("enter_long", "hold"),
                required_capabilities=("portfolio_allocator",),
            ),
            manifest={"module": "tests.test_engine", "parameters": {}},
        ),
    )
    def requires_portfolio_allocator(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    @registry.exit(
        "noop_exit",
        contract=_contract("noop_exit", "exit", ("close", "hold")),
    )
    def noop_exit(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    @registry.risk(
        "noop_risk",
        contract=_contract("noop_risk", "risk", ("hold",)),
    )
    def noop_risk(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        return RiskDecision()

    with pytest.raises(
        SetupCompatibilityError,
        match="portfolio_allocator",
    ):
        run_backtest(
            _make_spec("requires_portfolio_allocator", "noop_exit", "noop_risk"),
            _make_ohlcv([100.0, 101.0, 102.0]),
            node_registry=registry,
        )


def test_setup_rejects_declared_partial_exit_when_engine_capabilities_disable_it() -> None:
    registry = NodeRegistry(enforce_compatibility=False)

    @registry.entry(
        "hold_entry",
        contract=_contract("hold_entry", "entry", ("hold",)),
    )
    def hold_entry(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    @registry.exit(
        "declares_partial_exit",
        contract=NodeContract(
            spec=NodeSpec(
                name="declares_partial_exit",
                kind="exit",
                emitted_action_types=("partial_exit", "hold"),
            ),
            manifest={"module": "tests.test_engine", "parameters": {}},
        ),
    )
    def declares_partial_exit(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    @registry.risk(
        "hold_risk",
        contract=_contract("hold_risk", "risk", ("hold",)),
    )
    def hold_risk(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        return RiskDecision()

    limited_actions = EngineCapabilities(
        supported_action_types=("enter_long", "enter_short", "close", "hold")
    )
    spec = _make_spec(
        "hold_entry",
        "declares_partial_exit",
        "hold_risk",
        engine_capabilities=limited_actions,
    )

    audit = audit_backtest_setup(spec, node_registry=registry)
    assert audit.supported is False
    assert "partial_exit" in audit.summary

    with pytest.raises(SetupCompatibilityError, match="partial_exit"):
        run_backtest(
            spec,
            _make_ohlcv([100.0, 101.0, 102.0]),
            node_registry=registry,
        )


def test_builtin_nodes_are_resolved_by_registry_with_manifest_metadata() -> None:
    entry_node = resolve_entry("entry_sma_cross")
    exit_node = resolve_exit("exit_time_stop")
    risk_node = resolve_risk("risk_fixed_fraction")

    assert isinstance(entry_node, SMACrossEntryNode)
    assert isinstance(exit_node, TimeStopExitNode)
    assert isinstance(risk_node, FixedFractionRiskNode)

    entry_contract = resolve_contract("entry", "entry_sma_cross")
    exit_contract = resolve_contract("exit", "exit_time_stop")
    risk_contract = resolve_contract("risk", "risk_fixed_fraction")

    assert entry_contract.spec.required_history == 4
    assert entry_contract.spec.required_capabilities == ("single_position_per_symbol",)
    assert entry_contract.spec.emitted_action_types == (
        "enter_long",
        "enter_short",
        "hold",
    )
    assert entry_contract.manifest["module"] == "trading_lab.nodes.entry_sma_cross"
    assert entry_contract.manifest["parameters"]["slow_window"] == 3

    assert exit_contract.spec.required_history == 2
    assert exit_contract.spec.emitted_action_types == ("close", "hold")
    assert exit_contract.manifest["parameters"]["hold_bars"] == 2

    assert risk_contract.spec.requires_portfolio_view is True
    assert risk_contract.spec.required_capabilities == (
        "portfolio_view",
        "single_position_per_symbol",
    )
    assert risk_contract.spec.emitted_action_types == ("hold",)
    assert risk_contract.manifest["parameters"]["capital_fraction"] == pytest.approx(0.1)


def test_runtime_fails_immediately_when_node_constructs_unknown_action_type() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "unknown_action_entry",
        contract=_contract("unknown_action_entry", "entry", ("enter_long", "hold")),
    )
    def unknown_action_entry(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 0:
            return ActionRequest(action_type="explode", reason="bad-action")  # type: ignore[arg-type]
        return ActionRequest(action_type="hold")

    @registry.exit(
        "noop_exit",
        contract=_contract("noop_exit", "exit", ("close", "hold")),
    )
    def noop_exit(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    @registry.risk(
        "noop_risk",
        contract=_contract("noop_risk", "risk", ("hold",)),
    )
    def noop_risk(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        return RiskDecision()

    with pytest.raises(ValueError, match="unsupported action_type"):
        run_backtest(
            _make_spec("unknown_action_entry", "noop_exit", "noop_risk"),
            _make_ohlcv([100.0, 101.0, 102.0]),
            node_registry=registry,
        )


def test_builtin_nodes_run_end_to_end_through_engine() -> None:
    result = run_backtest(
        _make_spec("entry_sma_cross", "exit_time_stop", "risk_fixed_fraction"),
        _make_ohlcv([10.0, 9.0, 8.0, 11.0, 12.0, 13.0, 14.0]),
    )

    trade = result.trade_ledger.iloc[0]
    assert trade["side"] == "long"
    assert trade["entry_ts"] == pd.Timestamp("2024-01-02 09:35:00")
    assert trade["exit_ts"] == pd.Timestamp("2024-01-02 09:37:00")
    assert trade["qty"] == pytest.approx(90.0)
    assert trade["gross_pnl"] == pytest.approx(180.0)
    assert trade["net_pnl"] == pytest.approx(180.0)
    assert trade["exit_reason"] == "time_stop hold_bars=2"
    assert result.decision_log["reason"].eq("sma_cross_up fast=2 slow=3").any()


def test_builtin_node_contract_helpers_can_be_registered_explicitly() -> None:
    registry = NodeRegistry()
    registry.register(
        "entry",
        "custom_sma_entry",
        SMACrossEntryNode(fast_window=2, slow_window=3, allow_short_signals=False),
        contract=build_entry_sma_cross_contract(
            name="custom_sma_entry",
            fast_window=2,
            slow_window=3,
            allow_short_signals=False,
        ),
    )
    registry.register(
        "exit",
        "custom_time_stop",
        TimeStopExitNode(hold_bars=2),
        contract=build_exit_time_stop_contract(name="custom_time_stop", hold_bars=2),
    )
    registry.register(
        "risk",
        "custom_fixed_fraction",
        FixedFractionRiskNode(capital_fraction=0.25, stop_loss_pct=0.05, max_holding_bars=2),
        contract=build_risk_fixed_fraction_contract(
            name="custom_fixed_fraction",
            capital_fraction=0.25,
            stop_loss_pct=0.05,
            max_holding_bars=2,
        ),
    )

    result = run_backtest(
        _make_spec("custom_sma_entry", "custom_time_stop", "custom_fixed_fraction"),
        _make_ohlcv([10.0, 9.0, 8.0, 11.0, 12.0, 13.0, 14.0]),
        node_registry=registry,
    )

    trade = result.trade_ledger.iloc[0]
    assert trade["qty"] == pytest.approx(227.0)
    assert trade["gross_pnl"] == pytest.approx(454.0)
    assert trade["net_pnl"] == pytest.approx(454.0)

    risk_contract = registry.resolve_contract("risk", "custom_fixed_fraction")
    assert risk_contract.manifest["parameters"]["stop_loss_pct"] == pytest.approx(0.05)
    assert risk_contract.manifest["parameters"]["max_holding_bars"] == 2


def test_builtin_unsupported_node_requirements_fail_loudly() -> None:
    limited_capabilities = EngineCapabilities(
        supported_capabilities=(
            "market_orders",
            "next_bar_open_fills",
            "single_position_per_symbol",
            "simple_fees",
            "simple_slippage",
        )
    )
    registry = NodeRegistry(engine_capabilities=limited_capabilities)

    with pytest.raises(CompatibilityError, match="portfolio_view"):
        registry.register(
            "risk",
            "risk_fixed_fraction_limited",
            FixedFractionRiskNode(),
            contract=build_risk_fixed_fraction_contract(
                name="risk_fixed_fraction_limited"
            ),
        )


def test_setup_audit_explains_when_nodes_require_engine_changes() -> None:
    registry = NodeRegistry(enforce_compatibility=False)

    @registry.entry(
        "future_partial_exit_entry",
        contract=NodeContract(
            spec=NodeSpec(
                name="future_partial_exit_entry",
                kind="entry",
                emitted_action_types=("hold",),
                required_capabilities=("portfolio_allocator",),
            ),
            manifest={"module": "tests.test_engine", "parameters": {}},
        ),
    )
    def future_partial_exit_entry(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    @registry.exit(
        "noop_exit",
        contract=_contract("noop_exit", "exit", ("close", "hold")),
    )
    def noop_exit(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    @registry.risk(
        "noop_risk",
        contract=_contract("noop_risk", "risk", ("hold",)),
    )
    def noop_risk(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        return RiskDecision()

    spec = _make_spec("future_partial_exit_entry", "noop_exit", "noop_risk")
    audit = audit_backtest_setup(spec, node_registry=registry)

    assert audit.supported is False
    assert "portfolio_allocator" in audit.summary
    assert "engine feature request" in audit.summary

    with pytest.raises(SetupCompatibilityError, match="engine feature request"):
        run_backtest(
            spec,
            _make_ohlcv([100.0, 101.0, 102.0]),
            node_registry=registry,
        )


def test_setup_reports_older_node_contract_versions_clearly() -> None:
    registry = NodeRegistry(enforce_compatibility=False)

    @registry.entry(
        "old_contract_entry",
        contract=NodeContract(
            spec=NodeSpec(
                name="old_contract_entry",
                kind="entry",
                contract_version="0.9",
                emitted_action_types=("hold",),
            ),
            input_contract_version="0.9",
            output_contract_version="0.9",
            manifest={"module": "tests.test_engine", "parameters": {}},
        ),
    )
    def old_contract_entry(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    @registry.exit(
        "noop_exit",
        contract=_contract("noop_exit", "exit", ("close", "hold")),
    )
    def noop_exit(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    @registry.risk(
        "noop_risk",
        contract=_contract("noop_risk", "risk", ("hold",)),
    )
    def noop_risk(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        return RiskDecision()

    spec = _make_spec("old_contract_entry", "noop_exit", "noop_risk")
    audit = audit_backtest_setup(spec, node_registry=registry)

    assert audit.supported is False
    assert "0.9" in audit.summary
    assert "contract version" in audit.summary

    with pytest.raises(SetupCompatibilityError, match="0.9"):
        run_backtest(
            spec,
            _make_ohlcv([100.0, 101.0, 102.0]),
            node_registry=registry,
        )


def test_strict_node_output_validation_catches_required_history_violations() -> None:
    registry = NodeRegistry()

    @registry.entry(
        "premature_entry",
        contract=NodeContract(
            spec=NodeSpec(
                name="premature_entry",
                kind="entry",
                emitted_action_types=("enter_long", "hold"),
                required_history=5,
            ),
            manifest={"module": "tests.test_engine", "parameters": {}},
        ),
    )
    def premature_entry(ctx: DecisionContext) -> ActionRequest:
        if ctx.session.bar_index == 0:
            return ActionRequest(action_type="enter_long", units=1.0, reason="too-early")
        return ActionRequest(action_type="hold")

    @registry.exit(
        "noop_exit",
        contract=_contract("noop_exit", "exit", ("close", "hold")),
    )
    def noop_exit(ctx: DecisionContext) -> ActionRequest:
        return ActionRequest(action_type="hold")

    @registry.risk(
        "risk_one_unit",
        contract=_contract("risk_one_unit", "risk", ("hold",)),
    )
    def risk_one_unit(
        ctx: DecisionContext,
        entry_intent: ActionBatch,
        exit_intent: ActionBatch,
    ) -> RiskDecision:
        if entry_intent.is_entry:
            return RiskDecision(allow_entry=True, entry_quantity=1.0, reason="one-unit")
        return RiskDecision()

    spec = BacktestSpec(
        name="strict-node-output",
        instrument=InstrumentMeta(
            symbol="TEST",
            price_increment=0.01,
            quantity_increment=1.0,
        ),
        entry_node="premature_entry",
        exit_node="noop_exit",
        risk_node="risk_one_unit",
        initial_cash=10_000.0,
        strict_node_output_validation=True,
    )

    with pytest.raises(NodeOutputValidationError, match="required_history=5"):
        run_backtest(
            spec,
            _make_ohlcv([100.0, 101.0, 102.0, 103.0]),
            node_registry=registry,
        )
