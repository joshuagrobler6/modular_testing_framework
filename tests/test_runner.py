from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import trading_lab.runner as runner_module  # noqa: E402
from trading_lab.contracts import (  # noqa: E402
    BacktestResult,
    BacktestSpec,
    CostAssumptions,
    InstrumentMeta,
    NodeContract,
    NodeSpec,
)
from trading_lab.engine import PreparedBacktestData  # noqa: E402
from trading_lab.experiments import (  # noqa: E402
    ExperimentSpec,
    FoldSpec,
    HoldoutSpec,
    PruningConfig,
    SearchConfig,
    VariantSpec,
)
from trading_lab.runner import run_experiment  # noqa: E402


class FakeClock:
    def __init__(self, runtimes: list[float]) -> None:
        self.current = 0.0
        self._runtimes = list(runtimes)
        self._pending_runtime: float | None = None

    def __call__(self) -> float:
        if self._pending_runtime is None:
            self._pending_runtime = self._runtimes.pop(0) if self._runtimes else 0.0
            return self.current
        self.current += self._pending_runtime
        self._pending_runtime = None
        return self.current


def _instrument() -> InstrumentMeta:
    return InstrumentMeta(
        symbol="TEST",
        price_increment=0.01,
        quantity_increment=1.0,
    )


def _node_contract(name: str, kind: str) -> NodeContract:
    emitted_actions = {
        "entry": ("enter_long", "hold"),
        "exit": ("close", "hold"),
        "risk": ("hold",),
    }[kind]
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind=kind,  # type: ignore[arg-type]
            emitted_action_types=emitted_actions,  # type: ignore[arg-type]
            required_history=2,
        ),
        manifest={"module": "tests.test_runner", "parameters": {}},
    )


def _variant(name: str) -> VariantSpec:
    entry_name = f"{name}_entry"
    exit_name = f"{name}_exit"
    risk_name = f"{name}_risk"
    return VariantSpec(
        backtest_spec=BacktestSpec(
            name=name,
            instrument=_instrument(),
            entry_node=entry_name,
            exit_node=exit_name,
            risk_node=risk_name,
            initial_cash=100_000.0,
            costs=CostAssumptions(fee_rate=0.001),
        ),
        entry_contract=_node_contract(entry_name, "entry"),
        exit_contract=_node_contract(exit_name, "exit"),
        risk_contract=_node_contract(risk_name, "risk"),
    )


def _data() -> pd.DataFrame:
    rows = []
    for day, price in enumerate(range(100, 106), start=1):
        ts = pd.Timestamp(datetime(2024, 1, day))
        rows.append(
            {
                "ts": ts,
                "symbol": "TEST",
                "timeframe": "1D",
                "open": float(price),
                "high": float(price + 1),
                "low": float(price - 1),
                "close": float(price + 0.5),
                "volume": 1_000.0,
            }
        )
    return pd.DataFrame(rows)


def _experiment(
    variants: tuple[VariantSpec, ...],
    *,
    max_variants: int | None = None,
    max_runtime_seconds: int | None = None,
    max_parallel_variants: int = 1,
    holdout: bool = False,
    single_fold: bool = False,
    pruning: PruningConfig | None = None,
) -> ExperimentSpec:
    folds = [
        FoldSpec(
            fold_index=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 1, 2),
            validation_start=datetime(2024, 1, 2),
            validation_end=datetime(2024, 1, 3),
            label="fold_a",
        )
    ]
    if not single_fold:
        folds.append(
            FoldSpec(
                fold_index=1,
                train_start=datetime(2024, 1, 2),
                train_end=datetime(2024, 1, 3),
                validation_start=datetime(2024, 1, 3),
                validation_end=datetime(2024, 1, 4),
                label="fold_b",
            )
        )
    search_kwargs: dict[str, int] = {}
    if max_variants is not None:
        search_kwargs["max_variants"] = max_variants
    if max_runtime_seconds is not None:
        search_kwargs["max_runtime_seconds"] = max_runtime_seconds
    if not search_kwargs:
        search_kwargs["max_variants"] = len(variants)
    return ExperimentSpec(
        name="runner-phase",
        variants=variants,
        folds=tuple(folds),
        holdout=(
            HoldoutSpec(
                start=datetime(2024, 1, 4),
                end=datetime(2024, 1, 5),
                label="holdout",
            )
            if holdout
            else None
        ),
        search=SearchConfig(
            mode="grid",
            max_parallel_variants=max_parallel_variants,
            **search_kwargs,
        ),
        pruning=pruning or PruningConfig(),
    )


def _parallel_backtest_fn(
    spec: BacktestSpec,
    data_slice: pd.DataFrame,
    *,
    node_registry=None,
) -> BacktestResult:
    label = _label_from_slice(data_slice)
    time.sleep(0.2)
    return BacktestResult(
        spec=spec,
        decision_log=pd.DataFrame(),
        order_log=pd.DataFrame(),
        fill_log=pd.DataFrame(),
        trade_ledger=pd.DataFrame(),
        equity_curve=pd.DataFrame({"equity": [100_000.0]}),
        artifacts={
            "run_manifest": {"run_id": f"{spec.name}-{label}"},
            "stub_metrics": {"trade_count": 1, "net_pnl": 1.0, "gross_pnl": 1.0},
            "worker_pid": os.getpid(),
        },
    )


def _label_from_slice(data_slice: pd.DataFrame) -> str:
    ts = pd.Timestamp(data_slice["ts"].min())
    mapping = {
        pd.Timestamp("2024-01-02"): "fold_a",
        pd.Timestamp("2024-01-03"): "fold_b",
        pd.Timestamp("2024-01-04"): "holdout",
    }
    return mapping[ts]


def _stub_result(
    spec: BacktestSpec,
    *,
    run_id: str,
    metrics: dict[str, object],
    final_equity: float = 100_000.0,
    equity_curve: pd.DataFrame | None = None,
) -> BacktestResult:
    return BacktestResult(
        spec=spec,
        decision_log=pd.DataFrame(),
        order_log=pd.DataFrame(),
        fill_log=pd.DataFrame(),
        trade_ledger=pd.DataFrame(),
        equity_curve=(
            equity_curve.copy()
            if equity_curve is not None
            else pd.DataFrame({"equity": [final_equity]})
        ),
        artifacts={"run_manifest": {"run_id": run_id}, "stub_metrics": metrics},
    )


def _metrics_fn(result: BacktestResult) -> dict[str, object]:
    return dict(result.artifacts["stub_metrics"])  # type: ignore[index]


def test_runner_executes_folds_in_order_and_runs_holdout_after_cv() -> None:
    dataset = _data()
    variants = (_variant("variant_a"), _variant("variant_b"))
    experiment = _experiment(variants, holdout=True)
    calls: list[str] = []

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        label = _label_from_slice(data_slice)
        calls.append(f"{spec.name}:{label}")
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{label}",
            metrics={"trade_count": 1, "net_pnl": 1.0, "gross_pnl": 1.0},
        )

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([1.0] * 6),
    )

    assert calls == [
        "variant_a:fold_a",
        "variant_a:fold_b",
        "variant_a:holdout",
        "variant_b:fold_a",
        "variant_b:fold_b",
        "variant_b:holdout",
    ]
    assert [summary.label for summary in result.fold_summaries] == [
        "fold_a",
        "fold_b",
        "fold_a",
        "fold_b",
    ]
    assert [summary.label for summary in result.holdout_summaries] == [
        "holdout",
        "holdout",
    ]


def test_holdout_metrics_are_excluded_from_cv_ranking() -> None:
    dataset = _data()
    variant_a = _variant("variant_a")
    variant_b = _variant("variant_b")
    experiment = _experiment((variant_a, variant_b), holdout=True)

    metrics_map = {
        ("variant_a", "fold_a"): {"trade_count": 1, "net_pnl": 10.0, "gross_pnl": 10.0},
        ("variant_a", "fold_b"): {"trade_count": 1, "net_pnl": 10.0, "gross_pnl": 10.0},
        ("variant_a", "holdout"): {"trade_count": 1, "net_pnl": -100.0, "gross_pnl": -100.0},
        ("variant_b", "fold_a"): {"trade_count": 1, "net_pnl": 5.0, "gross_pnl": 5.0},
        ("variant_b", "fold_b"): {"trade_count": 1, "net_pnl": 5.0, "gross_pnl": 5.0},
        ("variant_b", "holdout"): {"trade_count": 1, "net_pnl": 1_000.0, "gross_pnl": 1_000.0},
    }

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        label = _label_from_slice(data_slice)
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{label}",
            metrics=metrics_map[(spec.name, label)],
        )

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([1.0] * 6),
    )

    assert [ranking.variant_id for ranking in result.cv_rankings] == [
        variant_a.variant_id,
        variant_b.variant_id,
    ]
    holdout_by_variant = {
        summary.variant_id: summary.metrics["net_pnl"] for summary in result.holdout_summaries
    }
    assert holdout_by_variant[variant_a.variant_id] == -100.0
    assert holdout_by_variant[variant_b.variant_id] == 1_000.0


def test_runtime_budget_stops_launching_new_variants() -> None:
    dataset = _data()
    variants = (_variant("variant_a"), _variant("variant_b"), _variant("variant_c"))
    experiment = _experiment(variants, max_runtime_seconds=5, single_fold=True)

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{_label_from_slice(data_slice)}",
            metrics={"trade_count": 1, "net_pnl": 1.0, "gross_pnl": 1.0},
        )

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([3.0, 3.0, 3.0, 3.0]),
    )

    assert result.completed_variant_ids == (
        variants[0].variant_id,
        variants[1].variant_id,
    )
    assert result.skipped_variant_ids == (variants[2].variant_id,)
    assert result.skip_records[0].reason == "runtime_budget_reached"
    assert result.runtime_summary.total_runtime_seconds == pytest.approx(6.0)
    assert result.runtime_summary.progress[0].estimated_remaining_seconds == pytest.approx(6.0)


def test_max_variants_cap_stops_after_deterministic_prefix() -> None:
    dataset = _data()
    variants = (_variant("variant_a"), _variant("variant_b"), _variant("variant_c"))
    experiment = _experiment(variants, max_variants=2)

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{_label_from_slice(data_slice)}",
            metrics={"trade_count": 1, "net_pnl": 1.0, "gross_pnl": 1.0},
        )

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([1.0, 1.0, 1.0, 1.0]),
    )

    assert result.completed_variant_ids == (
        variants[0].variant_id,
        variants[1].variant_id,
    )
    assert result.skipped_variant_ids == (variants[2].variant_id,)
    assert result.skip_records[0].reason == "max_variants_reached"


def test_runtime_budget_stopping_is_reproducible() -> None:
    dataset = _data()
    variants = (_variant("variant_a"), _variant("variant_b"), _variant("variant_c"))
    experiment = _experiment(variants, max_runtime_seconds=5, single_fold=True)

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{_label_from_slice(data_slice)}",
            metrics={"trade_count": 1, "net_pnl": 1.0, "gross_pnl": 1.0},
        )

    first = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([3.0, 3.0, 3.0, 3.0]),
    )
    second = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([3.0, 3.0, 3.0, 3.0]),
    )

    assert first.completed_variant_ids == second.completed_variant_ids
    assert first.skipped_variant_ids == second.skipped_variant_ids
    assert [record.reason for record in first.skip_records] == [
        record.reason for record in second.skip_records
    ]


def test_zero_equity_stop_prunes_variant_and_logs_reason() -> None:
    dataset = _data()
    variant_a = _variant("variant_a")
    variant_b = _variant("variant_b")
    experiment = _experiment(
        (variant_a, variant_b),
        pruning=PruningConfig(stop_on_zero_equity=True),
    )

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        label = _label_from_slice(data_slice)
        final_equity = 0.0 if (spec.name, label) == ("variant_a", "fold_a") else 100_000.0
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{label}",
            metrics={"trade_count": 1, "net_pnl": -5.0, "gross_pnl": -5.0},
            final_equity=final_equity,
        )

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([1.0, 1.0, 1.0]),
    )

    assert result.pruned_variant_ids == (variant_a.variant_id,)
    assert result.completed_variant_ids == (variant_b.variant_id,)
    assert result.cv_summaries[0].status == "pruned"
    assert result.prune_records[0].reason == "stop_on_zero_equity:final_equity=0.0"
    assert result.prune_records[0].stage == "after_fold"


def test_min_trades_pruning_happens_after_cv_before_holdout() -> None:
    dataset = _data()
    variant = _variant("variant_a")
    experiment = _experiment(
        (variant,),
        holdout=True,
        pruning=PruningConfig(min_trades=3),
    )
    calls: list[str] = []

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        label = _label_from_slice(data_slice)
        calls.append(label)
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{label}",
            metrics={"trade_count": 1, "net_pnl": 1.0, "gross_pnl": 1.0},
        )

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([1.0, 1.0]),
    )

    assert calls == ["fold_a", "fold_b"]
    assert result.pruned_variant_ids == (variant.variant_id,)
    assert result.holdout_summaries == ()
    assert result.prune_records[0].reason == "min_trades:trade_count=2<3"
    assert result.prune_records[0].stage == "after_cv"


def test_drawdown_threshold_pruning_uses_clear_reason() -> None:
    dataset = _data()
    variant = _variant("variant_a")
    experiment = _experiment(
        (variant,),
        pruning=PruningConfig(max_drawdown_threshold=-0.2),
    )

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{_label_from_slice(data_slice)}",
            metrics={
                "trade_count": 1,
                "net_pnl": -10.0,
                "gross_pnl": -10.0,
                "max_drawdown": -0.25,
            },
        )

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([1.0]),
    )

    assert result.pruned_variant_ids == (variant.variant_id,)
    assert result.prune_records[0].reason == "max_drawdown_threshold:max_drawdown=-0.25<=-0.2"


def test_invalid_numeric_state_prunes_with_explicit_reason() -> None:
    dataset = _data()
    variant = _variant("variant_a")
    experiment = _experiment(
        (variant,),
        pruning=PruningConfig(stop_on_invalid_numeric_state=True),
    )

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{_label_from_slice(data_slice)}",
            metrics={"trade_count": 1, "net_pnl": float("nan"), "gross_pnl": 1.0},
        )

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([1.0]),
    )

    assert result.pruned_variant_ids == (variant.variant_id,)
    assert result.prune_records[0].reason == "stop_on_invalid_numeric_state:metrics.net_pnl"


def test_early_metric_thresholds_can_prune_after_minimum_trade_or_bar_gate() -> None:
    dataset = _data()
    variant = _variant("variant_a")
    experiment = _experiment(
        (variant,),
        pruning=PruningConfig(
            early_metric_thresholds={"net_pnl": 0.0},
            early_min_trades=2,
        ),
    )

    metrics_map = {
        "fold_a": {"trade_count": 1, "net_pnl": -1.0, "gross_pnl": -1.0},
        "fold_b": {"trade_count": 1, "net_pnl": -2.0, "gross_pnl": -2.0},
    }

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        label = _label_from_slice(data_slice)
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{label}",
            metrics=metrics_map[label],
        )

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([1.0, 1.0]),
    )

    assert result.pruned_variant_ids == (variant.variant_id,)
    assert result.prune_records[0].reason == (
        "early_metric_threshold:net_pnl=-3.0<0.0 after trades=2 bars=2"
    )


def test_prune_outcomes_are_reproducible() -> None:
    dataset = _data()
    variants = (_variant("variant_a"), _variant("variant_b"))
    experiment = _experiment(
        variants,
        pruning=PruningConfig(max_drawdown_threshold=-0.2),
    )

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        label = _label_from_slice(data_slice)
        drawdown = -0.25 if spec.name == "variant_a" and label == "fold_a" else -0.05
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{label}",
            metrics={
                "trade_count": 1,
                "net_pnl": 1.0,
                "gross_pnl": 1.0,
                "max_drawdown": drawdown,
            },
        )

    first = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([1.0, 1.0, 1.0]),
    )
    second = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        time_fn=FakeClock([1.0, 1.0, 1.0]),
    )

    assert first.pruned_variant_ids == second.pruned_variant_ids
    assert [record.reason for record in first.prune_records] == [
        record.reason for record in second.prune_records
    ]


def test_parallel_variant_execution_uses_configured_batch_executor_and_preserves_order(
    monkeypatch,
) -> None:
    dataset = _data()
    variants = (
        _variant("variant_a"),
        _variant("variant_b"),
        _variant("variant_c"),
        _variant("variant_d"),
    )
    experiment = _experiment(
        variants,
        single_fold=True,
        max_parallel_variants=2,
    )

    executor_state: dict[str, object] = {"max_workers": None, "submitted": []}

    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakeProcessPoolExecutor:
        def __init__(self, *, max_workers, initializer, initargs):
            executor_state["max_workers"] = max_workers
            self._initializer = initializer
            self._initargs = initargs

        def __enter__(self):
            self._initializer(*self._initargs)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, variant):
            executor_state["submitted"].append(variant.variant_id)
            return _ImmediateFuture(fn(variant))

    monkeypatch.setattr(
        runner_module.concurrent.futures,
        "ProcessPoolExecutor",
        _FakeProcessPoolExecutor,
    )

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=_parallel_backtest_fn,
        metrics_fn=_metrics_fn,
    )

    assert executor_state["max_workers"] == 2
    assert executor_state["submitted"] == [variant.variant_id for variant in variants]
    assert result.completed_variant_ids == tuple(variant.variant_id for variant in variants)
    assert [summary.variant_id for summary in result.cv_summaries] == [
        variant.variant_id for variant in variants
    ]


def test_parallel_variant_execution_rejects_non_picklable_callbacks() -> None:
    dataset = _data()
    variants = (_variant("variant_a"), _variant("variant_b"))
    experiment = _experiment(
        variants,
        single_fold=True,
        max_parallel_variants=2,
    )

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{_label_from_slice(data_slice)}",
            metrics={"trade_count": 1, "net_pnl": 1.0, "gross_pnl": 1.0},
        )

    with pytest.raises(TypeError, match="pickle-safe backtest_fn"):
        run_experiment(
            experiment,
            dataset,
            backtest_fn=backtest_fn,
            metrics_fn=_metrics_fn,
        )


def test_parallel_variant_execution_falls_back_when_process_pool_is_unavailable(
    monkeypatch,
) -> None:
    dataset = _data()
    variants = (_variant("variant_a"), _variant("variant_b"))
    experiment = _experiment(
        variants,
        single_fold=True,
        max_parallel_variants=2,
    )

    class _BrokenProcessPoolExecutor:
        def __init__(self, *args, **kwargs):
            raise PermissionError("blocked")

    monkeypatch.setattr(
        runner_module.concurrent.futures,
        "ProcessPoolExecutor",
        _BrokenProcessPoolExecutor,
    )

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=_parallel_backtest_fn,
        metrics_fn=_metrics_fn,
    )

    assert result.completed_variant_ids == tuple(variant.variant_id for variant in variants)
    assert [summary.variant_id for summary in result.cv_summaries] == [
        variant.variant_id for variant in variants
    ]


def test_run_experiment_can_drop_run_results_for_lightweight_search() -> None:
    dataset = _data()
    variant = _variant("variant_a")
    experiment = _experiment((variant,), single_fold=True)

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=_parallel_backtest_fn,
        metrics_fn=_metrics_fn,
        retain_run_results=False,
    )

    assert result.run_results == {}
    assert result.completed_variant_ids == (variant.variant_id,)
    assert result.fold_summaries[0].variant_id == variant.variant_id


def test_run_experiment_can_prepare_fold_inputs_once_and_reuse_them() -> None:
    dataset = _data()
    variants = (_variant("variant_a"), _variant("variant_b"))
    experiment = _experiment(variants, single_fold=True)
    prepared_calls: list[tuple[str, int]] = []
    seen_input_types: list[str] = []

    def prepared_data_fn(spec: BacktestSpec, data_slice: pd.DataFrame) -> PreparedBacktestData:
        prepared_calls.append((spec.instrument.symbol, len(data_slice)))
        return runner_module.prepare_backtest_data(spec, data_slice)

    def backtest_fn(
        spec: BacktestSpec,
        data_slice,
        *,
        node_registry=None,
    ) -> BacktestResult:
        seen_input_types.append(type(data_slice).__name__)
        assert isinstance(data_slice, PreparedBacktestData)
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{data_slice.timeframe}",
            metrics={"trade_count": 1, "net_pnl": 1.0, "gross_pnl": 1.0},
        )

    result = run_experiment(
        experiment,
        dataset,
        backtest_fn=backtest_fn,
        metrics_fn=_metrics_fn,
        prepare_backtest_inputs=True,
        prepared_data_fn=prepared_data_fn,
    )

    assert prepared_calls == [("TEST", 1)]
    assert seen_input_types == ["PreparedBacktestData", "PreparedBacktestData"]
    assert result.completed_variant_ids == tuple(variant.variant_id for variant in variants)
