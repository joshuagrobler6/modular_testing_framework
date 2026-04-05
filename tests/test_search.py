from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_lab.contracts import (  # noqa: E402
    BacktestResult,
    BacktestSpec,
    CostAssumptions,
    InstrumentMeta,
    NodeContract,
    NodeSpec,
)
from trading_lab.experiments import (  # noqa: E402
    DeepDiveConfig,
    ExperimentSpec,
    FoldSpec,
    HoldoutSpec,
    OutputConfig,
    SearchConfig,
    VariantSpec,
)
from trading_lab.runner import run_experiment  # noqa: E402
from trading_lab.search import (  # noqa: E402
    GridSearchAdapter,
    MetricConstraint,
    ObjectiveConfig,
    OptunaSearchAdapter,
    RandomSearchAdapter,
    evaluate_objective,
    run_search_entrypoint,
    run_search_experiment,
    serialize_search_run_config,
    SearchRunConfig,
)


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


class FakeTrial:
    def __init__(self, number: int) -> None:
        self.number = number
        self.user_attrs: dict[str, object] = {}

    def suggest_int(self, name: str, low: int, high: int) -> int:
        span = max(high - low, 0)
        return low + min(self.number, span)

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value


class FakeStudy:
    def __init__(self, direction: str, study_name: str) -> None:
        self.direction = direction
        self.study_name = study_name
        self.optimize_calls: list[dict[str, object]] = []
        self.trials: list[FakeTrial] = []

    def optimize(self, objective, n_trials=None, timeout=None) -> None:
        self.optimize_calls.append({"n_trials": n_trials, "timeout": timeout})
        total = 0 if n_trials is None else int(n_trials)
        for trial_number in range(total):
            trial = FakeTrial(trial_number)
            trial.value = objective(trial)
            self.trials.append(trial)


class FakeOptunaModule:
    def __init__(self) -> None:
        self.studies: list[FakeStudy] = []

    def create_study(self, *, direction: str, study_name: str) -> FakeStudy:
        study = FakeStudy(direction=direction, study_name=study_name)
        self.studies.append(study)
        return study


def _instrument() -> InstrumentMeta:
    return InstrumentMeta(
        symbol="TEST",
        price_increment=0.01,
        quantity_increment=1.0,
    )


def _backtest_spec(
    *,
    name: str = "variant-template",
    entry_node: str = "entry_alpha",
    exit_node: str = "exit_beta",
    risk_node: str = "risk_gamma",
) -> BacktestSpec:
    return BacktestSpec(
        name=name,
        instrument=_instrument(),
        entry_node=entry_node,
        exit_node=exit_node,
        risk_node=risk_node,
        initial_cash=100_000.0,
        costs=CostAssumptions(fee_rate=0.001),
    )


def _node_contract(
    name: str,
    kind: str,
    emitted_action_types: tuple[str, ...],
    *,
    parameters: dict[str, object] | None = None,
) -> NodeContract:
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind=kind,  # type: ignore[arg-type]
            emitted_action_types=emitted_action_types,  # type: ignore[arg-type]
            required_history=2,
        ),
        manifest={
            "module": "tests.test_search",
            "parameters": parameters or {},
        },
    )


def _variant(
    name: str,
    *,
    entry_parameters: dict[str, object] | None = None,
    exit_parameters: dict[str, object] | None = None,
    risk_parameters: dict[str, object] | None = None,
) -> VariantSpec:
    entry_name = f"{name}_entry"
    exit_name = f"{name}_exit"
    risk_name = f"{name}_risk"
    return VariantSpec(
        backtest_spec=_backtest_spec(
            name=name,
            entry_node=entry_name,
            exit_node=exit_name,
            risk_node=risk_name,
        ),
        entry_contract=_node_contract(
            entry_name,
            "entry",
            ("enter_long", "enter_short", "hold"),
            parameters=entry_parameters,
        ),
        exit_contract=_node_contract(
            exit_name,
            "exit",
            ("close", "hold"),
            parameters=exit_parameters,
        ),
        risk_contract=_node_contract(
            risk_name,
            "risk",
            ("hold",),
            parameters=risk_parameters,
        ),
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


def _folds() -> tuple[FoldSpec, ...]:
    return (
        FoldSpec(
            fold_index=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 1, 2),
            validation_start=datetime(2024, 1, 2),
            validation_end=datetime(2024, 1, 3),
            label="fold_a",
        ),
        FoldSpec(
            fold_index=1,
            train_start=datetime(2024, 1, 2),
            train_end=datetime(2024, 1, 3),
            validation_start=datetime(2024, 1, 3),
            validation_end=datetime(2024, 1, 4),
            label="fold_b",
        ),
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
) -> BacktestResult:
    label = run_id.split("-")[-1]
    ts_map = {
        "fold_a": pd.Timestamp("2024-01-02 00:00:00"),
        "fold_b": pd.Timestamp("2024-01-03 00:00:00"),
        "holdout": pd.Timestamp("2024-01-04 00:00:00"),
    }
    ts = ts_map.get(label, pd.Timestamp("2024-01-02 00:00:00"))
    net_pnl = float(metrics.get("net_pnl", 0.0))
    return BacktestResult(
        spec=spec,
        decision_log=pd.DataFrame(),
        order_log=pd.DataFrame(),
        fill_log=pd.DataFrame(),
        trade_ledger=pd.DataFrame(
            [
                {
                    "trade_id": f"{run_id}-trade-1",
                    "symbol": spec.instrument.symbol,
                    "side": "long",
                    "entry_ts": ts,
                    "exit_ts": ts,
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "qty": 1.0,
                    "gross_pnl": net_pnl,
                    "net_pnl": net_pnl,
                    "mfe": net_pnl,
                    "mae": 0.0,
                    "exit_efficiency": 1.0,
                    "bars_held": 1,
                    "exit_reason": "stub-exit",
                    "fees": 0.0,
                }
            ]
        ),
        equity_curve=pd.DataFrame(
            [
                {
                    "ts": ts,
                    "cash": 100_000.0 + net_pnl,
                    "equity": 100_000.0 + net_pnl,
                    "realized_pnl": net_pnl,
                    "unrealized_pnl": 0.0,
                    "gross_exposure": 0.0,
                    "net_exposure": 0.0,
                    "drawdown": 0.0,
                }
            ]
        ),
        artifacts={"run_manifest": {"run_id": run_id}, "stub_metrics": metrics},
    )


def _metrics_fn(result: BacktestResult) -> dict[str, object]:
    return dict(result.artifacts["stub_metrics"])  # type: ignore[index]


def test_grid_search_adapter_builds_deterministic_cartesian_variants() -> None:
    adapter = GridSearchAdapter(ObjectiveConfig.single_metric("net_pnl"))
    base_spec = _backtest_spec(
        entry_node="placeholder_entry",
        exit_node="placeholder_exit",
        risk_node="placeholder_risk",
    )

    experiment = adapter.build_experiment(
        name="grid-search",
        base_backtest_spec=base_spec,
        entry_families=[
            (
                _node_contract(
                    "entry_alpha",
                    "entry",
                    ("enter_long", "enter_short", "hold"),
                    parameters={"family": "sma"},
                ),
                {"fast": (5, 10), "slow": (20,)},
            )
        ],
        exit_families=[
            (
                _node_contract(
                    "exit_beta",
                    "exit",
                    ("close", "hold"),
                    parameters={"family": "time_stop"},
                ),
                {"bars": (3, 5)},
            )
        ],
        risk_families=[
            (
                _node_contract(
                    "risk_gamma",
                    "risk",
                    ("hold",),
                    parameters={"family": "fixed_fraction"},
                ),
                {"risk_fraction": (0.01, 0.02)},
            )
        ],
        folds=_folds(),
        search=SearchConfig(mode="grid", max_variants=100),
    )

    assert len(experiment.variants) == 8
    assert experiment.variants[0].entry_contract.manifest["parameters"] == {
        "family": "sma",
        "fast": 5,
        "slow": 20,
    }
    assert experiment.variants[0].exit_contract.manifest["parameters"]["bars"] == 3
    assert experiment.variants[1].risk_contract.manifest["parameters"]["risk_fraction"] == 0.02
    assert [variant.entry_contract.manifest["parameters"]["fast"] for variant in experiment.variants[:4]] == [
        5,
        5,
        5,
        5,
    ]


def test_random_search_adapter_is_reproducible_with_fixed_seed() -> None:
    adapter = RandomSearchAdapter(ObjectiveConfig.single_metric("net_pnl"))
    base_spec = _backtest_spec(
        entry_node="placeholder_entry",
        exit_node="placeholder_exit",
        risk_node="placeholder_risk",
    )
    kwargs = dict(
        name="random-search",
        base_backtest_spec=base_spec,
        entry_families=[
            (
                _node_contract(
                    "entry_alpha",
                    "entry",
                    ("enter_long", "enter_short", "hold"),
                    parameters={"family": "sma"},
                ),
                {"fast": (5, 10, 15), "slow": (20, 30)},
            )
        ],
        exit_families=[
            (_node_contract("exit_beta", "exit", ("close", "hold")), {"bars": (3, 5)})
        ],
        risk_families=[
            (
                _node_contract("risk_gamma", "risk", ("hold",)),
                {"risk_fraction": (0.01, 0.02)},
            )
        ],
        folds=_folds(),
        search=SearchConfig(mode="random", max_variants=4, random_seed=17),
    )

    first = adapter.build_experiment(**kwargs)
    second = adapter.build_experiment(**kwargs)

    assert [variant.variant_id for variant in first.variants] == [
        variant.variant_id for variant in second.variants
    ]


def test_objective_config_supports_composite_and_constrained_scoring() -> None:
    objective = ObjectiveConfig.composite(
        {"net_pnl": 1.0, "sharpe": 2.0},
        constraints=(
            MetricConstraint("trade_count", minimum=3),
            MetricConstraint("max_drawdown", minimum=-0.2),
        ),
    )

    passing_value, passing_score, passing_violations = evaluate_objective(
        {
            "net_pnl": 10.0,
            "sharpe": 1.5,
            "trade_count": 4,
            "max_drawdown": -0.1,
        },
        objective,
    )
    failing_value, failing_score, failing_violations = evaluate_objective(
        {
            "net_pnl": 10.0,
            "sharpe": 1.5,
            "trade_count": 2,
            "max_drawdown": -0.3,
        },
        objective,
    )

    assert passing_value == 13.0
    assert passing_score == 13.0
    assert passing_violations == ()
    assert failing_value == 13.0
    assert failing_score is None
    assert failing_violations == ("trade_count<3.0", "max_drawdown<-0.2")


def test_search_objective_uses_cv_only_and_ignores_holdout() -> None:
    dataset = _data()
    variant_a = _variant("variant_a")
    variant_b = _variant("variant_b")
    experiment = ExperimentSpec(
        name="search-objective",
        variants=(variant_a, variant_b),
        folds=_folds(),
        holdout=HoldoutSpec(
            start=datetime(2024, 1, 4),
            end=datetime(2024, 1, 5),
            label="holdout",
        ),
        search=SearchConfig(mode="grid", max_variants=2),
    )
    objective = ObjectiveConfig.single_metric("net_pnl")
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

    result = run_search_experiment(
        experiment,
        dataset,
        objective=objective,
        runner_fn=run_experiment,
        runner_kwargs={
            "backtest_fn": backtest_fn,
            "metrics_fn": _metrics_fn,
            "time_fn": FakeClock([1.0] * 6),
        },
    )

    assert result.best_variant_id == variant_a.variant_id
    holdout_by_variant = {
        evaluation.variant_id: evaluation.holdout_metrics["net_pnl"]  # type: ignore[index]
        for evaluation in result.evaluations
        if evaluation.holdout_metrics is not None
    }
    assert holdout_by_variant[variant_a.variant_id] == -100.0
    assert holdout_by_variant[variant_b.variant_id] == 1_000.0


def test_optuna_adapter_runs_trials_and_preserves_cv_lineage() -> None:
    fake_optuna = FakeOptunaModule()
    adapter = OptunaSearchAdapter(
        ObjectiveConfig.single_metric("net_pnl"),
        optuna_module=fake_optuna,
    )
    base_experiment = ExperimentSpec(
        name="optuna-search",
        variants=(_variant("template_variant"),),
        folds=(_folds()[0],),
        search=SearchConfig(mode="optuna", max_variants=2, max_runtime_seconds=30),
    )
    dataset = _data()

    def variant_factory(trial: FakeTrial) -> VariantSpec:
        fast = trial.suggest_int("fast", 5, 6)
        return _variant(
            f"trial_variant_{trial.number}",
            entry_parameters={"family": "sma", "fast": fast, "slow": 20},
            exit_parameters={"bars": 3},
            risk_parameters={"risk_fraction": 0.01},
        )

    def backtest_fn(
        spec: BacktestSpec,
        data_slice: pd.DataFrame,
        *,
        node_registry=None,
    ) -> BacktestResult:
        fast = spec.entry_node
        variant_number = int(fast.removeprefix("trial_variant_").split("_")[0])
        net_pnl = 10.0 + float(variant_number)
        label = _label_from_slice(data_slice)
        return _stub_result(
            spec,
            run_id=f"{spec.name}-{label}",
            metrics={"trade_count": 1, "net_pnl": net_pnl, "gross_pnl": net_pnl},
        )

    result = adapter.run(
        base_experiment,
        dataset,
        variant_factory=variant_factory,
        runner_fn=run_experiment,
        runner_kwargs={
            "backtest_fn": backtest_fn,
            "metrics_fn": _metrics_fn,
            "time_fn": FakeClock([1.0] * 2),
        },
        time_fn=FakeClock([0.0] * 10),
    )

    assert len(fake_optuna.studies) == 1
    study = fake_optuna.studies[0]
    assert study.optimize_calls == [{"n_trials": 2, "timeout": 30}]
    assert len(result.trial_records) == 2
    assert result.trial_records[0].trial_name == "trial_00000"
    assert result.trial_records[1].trial_name == "trial_00001"
    assert all("variant_id" in trial.user_attrs for trial in study.trials)
    assert result.best_variant_id == result.trial_records[-1].variant_id


def test_search_entrypoint_runs_from_experiment_spec_and_writes_summary_outputs(
    tmp_path: Path,
    capsys,
) -> None:
    dataset = _data()
    variant_a = _variant("variant_a")
    variant_b = _variant("variant_b")
    experiment = ExperimentSpec(
        name="entrypoint-grid",
        variants=(variant_a, variant_b),
        folds=_folds(),
        search=SearchConfig(mode="grid", max_variants=2),
    )
    objective = ObjectiveConfig.single_metric("net_pnl")
    metrics_map = {
        ("variant_a", "fold_a"): {"trade_count": 1, "net_pnl": 5.0, "gross_pnl": 5.0},
        ("variant_a", "fold_b"): {"trade_count": 1, "net_pnl": 5.0, "gross_pnl": 5.0},
        ("variant_b", "fold_a"): {"trade_count": 1, "net_pnl": 3.0, "gross_pnl": 3.0},
        ("variant_b", "fold_b"): {"trade_count": 1, "net_pnl": 3.0, "gross_pnl": 3.0},
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

    result = run_search_entrypoint(
        experiment,
        dataset,
        objective=objective,
        runner_fn=run_experiment,
        runner_kwargs={
            "backtest_fn": backtest_fn,
            "metrics_fn": _metrics_fn,
            "time_fn": FakeClock([1.0] * 4),
        },
        output_dir=tmp_path / "entrypoint_grid",
    )

    captured = capsys.readouterr()
    assert "mode=grid" in captured.out
    assert "stopping_reason=completed_planned_search" in captured.out
    assert result.summary_workbook_path.exists()
    assert result.manifest_path is not None and result.manifest_path.exists()
    assert result.search_result.best_variant_id == variant_a.variant_id
    assert result.stopping_reason == "completed_planned_search"
    assert result.reproducibility_manifest["best_variant_id"] == variant_a.variant_id


def test_search_entrypoint_accepts_serialized_config_and_emits_deep_dive(
    tmp_path: Path,
) -> None:
    dataset = _data()
    variant = _variant("variant_a")
    objective = ObjectiveConfig.single_metric("net_pnl")
    experiment = ExperimentSpec(
        name="entrypoint-serialized",
        variants=(variant,),
        folds=_folds(),
        holdout=HoldoutSpec(
            start=datetime(2024, 1, 4),
            end=datetime(2024, 1, 5),
            label="holdout",
        ),
        search=SearchConfig(mode="grid", max_variants=1),
        outputs=OutputConfig(output_dir=str(tmp_path / "serialized_entrypoint")),
        deep_dive=DeepDiveConfig(
            selected_variant_ids=(variant.variant_id,),
            selected_folds=("fold_a",),
        ),
    )
    serialized = serialize_search_run_config(
        SearchRunConfig(experiment=experiment, objective=objective)
    )

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
            metrics={"trade_count": 1, "net_pnl": 4.0, "gross_pnl": 4.0},
        )

    result = run_search_entrypoint(
        serialized,
        dataset,
        runner_fn=run_experiment,
        runner_kwargs={
            "backtest_fn": backtest_fn,
            "metrics_fn": _metrics_fn,
            "time_fn": FakeClock([1.0] * 3),
        },
        verbose=False,
    )

    assert result.summary_workbook_path.exists()
    assert result.manifest_path is not None and result.manifest_path.exists()
    assert len(result.deep_dive_artifacts) == 1
    artifact = result.deep_dive_artifacts[0]
    assert artifact.target.variant_id == variant.variant_id
    assert artifact.target.label == "fold_a"
    assert artifact.target_dir.exists()
    assert artifact.equity_plot_path is not None and artifact.equity_plot_path.exists()
    assert artifact.price_plot_path is not None and artifact.price_plot_path.exists()
    assert artifact.trade_log_path is not None and artifact.trade_log_path.exists()
    assert result.reproducibility_manifest["summary_workbook_path"] == str(
        result.summary_workbook_path
    )
