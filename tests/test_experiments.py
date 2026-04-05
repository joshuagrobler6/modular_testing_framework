from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_lab.contracts import (  # noqa: E402
    BacktestSpec,
    CostAssumptions,
    InstrumentMeta,
    NodeContract,
    NodeSpec,
)
from trading_lab.experiments import (  # noqa: E402
    DeepDiveConfig,
    ExperimentRunResult,
    ExperimentSpec,
    FoldSpec,
    HoldoutSpec,
    OutputConfig,
    PruningConfig,
    SearchConfig,
    VariantSpec,
    expand_parameter_grid,
    generate_variants,
    generate_variant_id,
    label_fold,
    parameterize_node_contract,
    serialize_manifest,
)


def _instrument() -> InstrumentMeta:
    return InstrumentMeta(
        symbol="TEST",
        price_increment=0.01,
        quantity_increment=1.0,
    )


def _backtest_spec(
    *,
    entry_node: str = "entry_alpha",
    exit_node: str = "exit_beta",
    risk_node: str = "risk_gamma",
) -> BacktestSpec:
    return BacktestSpec(
        name="variant-template",
        instrument=_instrument(),
        entry_node=entry_node,
        exit_node=exit_node,
        risk_node=risk_node,
        initial_cash=100_000.0,
        costs=CostAssumptions(fee_rate=0.001, slippage_bps=5.0),
    )


def _node_contract(
    name: str,
    kind: str,
    emitted_action_types: tuple[str, ...],
    *,
    version: str = "1.0.0",
    parameters: dict[str, object] | None = None,
) -> NodeContract:
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind=kind,  # type: ignore[arg-type]
            version=version,
            emitted_action_types=emitted_action_types,  # type: ignore[arg-type]
            required_history=20,
        ),
        manifest={
            "module": "tests.test_experiments",
            "parameters": parameters or {},
        },
    )


def _variant(
    *,
    entry_parameters: dict[str, object] | None = None,
    exit_parameters: dict[str, object] | None = None,
    risk_parameters: dict[str, object] | None = None,
    description: str = "",
    tags: tuple[str, ...] = (),
) -> VariantSpec:
    backtest_spec = _backtest_spec()
    return VariantSpec(
        backtest_spec=backtest_spec,
        entry_contract=_node_contract(
            "entry_alpha",
            "entry",
            ("enter_long", "enter_short", "hold"),
            parameters=entry_parameters,
        ),
        exit_contract=_node_contract(
            "exit_beta",
            "exit",
            ("close", "hold"),
            parameters=exit_parameters,
        ),
        risk_contract=_node_contract(
            "risk_gamma",
            "risk",
            ("hold",),
            parameters=risk_parameters,
        ),
        description=description,
        tags=tags,
    )


def test_variant_id_is_deterministic_and_derived_from_reproducible_inputs() -> None:
    variant_a = _variant(
        entry_parameters={"fast": 5, "slow": 20},
        exit_parameters={"bars": 7},
        risk_parameters={"risk_fraction": 0.02, "max_holding_bars": 10},
        description="first label",
        tags=("baseline",),
    )
    variant_b = _variant(
        entry_parameters={"slow": 20, "fast": 5},
        exit_parameters={"bars": 7},
        risk_parameters={"max_holding_bars": 10, "risk_fraction": 0.02},
        description="renamed label",
        tags=("renamed", "metadata"),
    )

    assert variant_a.variant_id == variant_b.variant_id
    assert variant_a.variant_id.startswith("variant_")
    assert (
        generate_variant_id(
            variant_a.backtest_spec,
            variant_a.entry_contract,
            variant_a.exit_contract,
            variant_a.risk_contract,
        )
        == variant_a.variant_id
    )


def test_fold_and_holdout_are_explicitly_separated() -> None:
    fold_a = FoldSpec(
        fold_index=0,
        train_start=datetime(2024, 1, 1),
        train_end=datetime(2024, 2, 1),
        validation_start=datetime(2024, 2, 1),
        validation_end=datetime(2024, 3, 1),
    )
    fold_b = FoldSpec(
        fold_index=1,
        train_start=datetime(2024, 3, 1),
        train_end=datetime(2024, 4, 1),
        validation_start=datetime(2024, 4, 1),
        validation_end=datetime(2024, 5, 1),
        label="fold_custom",
    )
    holdout = HoldoutSpec(
        start=datetime(2024, 5, 1),
        end=datetime(2024, 6, 1),
        label="holdout",
    )
    experiment = ExperimentSpec(
        name="cv-with-holdout",
        variants=(_variant(),),
        folds=(fold_a, fold_b),
        holdout=holdout,
        search=SearchConfig(mode="grid", max_variants=4, max_runtime_seconds=600),
    )

    assert label_fold(fold_a) == "fold_00"
    assert label_fold(fold_b) == "fold_custom"
    assert experiment.holdout is holdout
    assert holdout.label not in {label_fold(fold) for fold in experiment.folds}


def test_config_validation_rejects_invalid_search_and_deep_dive_inputs() -> None:
    with pytest.raises(ValueError, match="at least one bound"):
        SearchConfig(mode="grid")

    with pytest.raises(ValueError, match="max_variants must be >= 1"):
        SearchConfig(mode="random", max_variants=0, max_runtime_seconds=60)

    with pytest.raises(ValueError, match="selected_variant_ids must not be empty"):
        DeepDiveConfig(selected_variant_ids=(), selected_folds=("fold_00",))

    with pytest.raises(ValueError, match="include at least one fold or holdout"):
        DeepDiveConfig(
            selected_variant_ids=("variant_abc",),
            selected_folds=(),
            include_holdout=False,
        )


def test_variant_and_experiment_specs_validate_consistency() -> None:
    backtest_spec = _backtest_spec(entry_node="entry_alpha")
    wrong_entry_contract = _node_contract(
        "entry_other",
        "entry",
        ("enter_long", "hold"),
    )

    with pytest.raises(ValueError, match="entry_node must match"):
        VariantSpec(
            backtest_spec=backtest_spec,
            entry_contract=wrong_entry_contract,
            exit_contract=_node_contract("exit_beta", "exit", ("close", "hold")),
            risk_contract=_node_contract("risk_gamma", "risk", ("hold",)),
        )

    fold = FoldSpec(
        fold_index=0,
        train_start=datetime(2024, 1, 1),
        train_end=datetime(2024, 2, 1),
        validation_start=datetime(2024, 2, 1),
        validation_end=datetime(2024, 3, 1),
        label="shared_label",
    )
    with pytest.raises(ValueError, match="holdout label must not overlap"):
        ExperimentSpec(
            name="bad-holdout-overlap",
            variants=(_variant(),),
            folds=(fold,),
            holdout=HoldoutSpec(
                start=datetime(2024, 3, 1),
                end=datetime(2024, 4, 1),
                label="shared_label",
            ),
            search=SearchConfig(mode="grid", max_variants=2),
        )


def test_manifest_serialization_is_stable_across_dict_ordering() -> None:
    left = {
        "parameters": {"slow": 20, "fast": 5},
        "module": "tests.test_experiments",
        "ts": datetime(2024, 1, 1, 9, 30),
        "enabled": True,
    }
    right = {
        "enabled": True,
        "ts": datetime(2024, 1, 1, 9, 30),
        "module": "tests.test_experiments",
        "parameters": {"fast": 5, "slow": 20},
    }

    serialized = serialize_manifest(left)

    assert serialized == serialize_manifest(right)
    assert (
        serialized
        == '{"enabled":true,"module":"tests.test_experiments","parameters":{"fast":5,"slow":20},"ts":"2024-01-01T09:30:00"}'
    )


def test_experiment_run_result_tracks_known_variant_and_fold_ids_only() -> None:
    variant = _variant()
    experiment = ExperimentSpec(
        name="run-summary",
        variants=(variant,),
        folds=(
            FoldSpec(
                fold_index=0,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 2, 1),
                validation_start=datetime(2024, 2, 1),
                validation_end=datetime(2024, 3, 1),
            ),
        ),
        search=SearchConfig(mode="grid", max_variants=1),
    )

    result = ExperimentRunResult(
        experiment=experiment,
        run_id="exp-run-001",
        completed_variant_ids=(variant.variant_id,),
        completed_fold_labels=("fold_00",),
    )

    assert result.completed_variant_ids == (variant.variant_id,)
    assert result.completed_fold_labels == ("fold_00",)

    with pytest.raises(ValueError, match="unknown variants"):
        ExperimentRunResult(
            experiment=experiment,
            run_id="exp-run-002",
            completed_variant_ids=("variant_missing",),
        )


def test_experiment_specs_are_frozen() -> None:
    search = SearchConfig(mode="grid", max_variants=1)
    fold = FoldSpec(
        fold_index=0,
        train_start=datetime(2024, 1, 1),
        train_end=datetime(2024, 2, 1),
        validation_start=datetime(2024, 2, 1),
        validation_end=datetime(2024, 3, 1),
    )

    with pytest.raises(FrozenInstanceError):
        search.max_variants = 2

    with pytest.raises(FrozenInstanceError):
        fold.label = "other"


def test_deep_dive_must_reference_known_variants_and_available_targets() -> None:
    variant = _variant()
    fold = FoldSpec(
        fold_index=0,
        train_start=datetime(2024, 1, 1),
        train_end=datetime(2024, 2, 1),
        validation_start=datetime(2024, 2, 1),
        validation_end=datetime(2024, 3, 1),
    )

    with pytest.raises(ValueError, match="unknown variant_ids"):
        ExperimentSpec(
            name="bad-deep-dive-variant",
            variants=(variant,),
            folds=(fold,),
            holdout=HoldoutSpec(
                start=datetime(2024, 3, 1),
                end=datetime(2024, 4, 1),
            ),
            search=SearchConfig(mode="grid", max_variants=1),
            deep_dive=DeepDiveConfig(
                selected_variant_ids=("variant_missing",),
                selected_folds=("fold_00",),
            ),
        )

    with pytest.raises(ValueError, match="unknown fold labels"):
        ExperimentSpec(
            name="bad-deep-dive-fold",
            variants=(variant,),
            folds=(fold,),
            search=SearchConfig(mode="grid", max_variants=1),
            deep_dive=DeepDiveConfig(
                selected_variant_ids=(variant.variant_id,),
                selected_folds=("fold_99",),
            ),
        )

    with pytest.raises(ValueError, match="no holdout is configured"):
        ExperimentSpec(
            name="bad-deep-dive-holdout",
            variants=(variant,),
            folds=(fold,),
            search=SearchConfig(mode="grid", max_variants=1),
            deep_dive=DeepDiveConfig(
                selected_variant_ids=(variant.variant_id,),
                include_holdout=True,
            ),
        )


def test_output_and_pruning_configs_validate_their_inputs() -> None:
    config = PruningConfig(
        stop_on_zero_equity=True,
        min_trades=5,
        early_metric_thresholds={"net_pnl": 0.0, "sharpe": -1.0},
    )

    assert config.early_metric_thresholds == {"net_pnl": 0.0, "sharpe": -1.0}

    with pytest.raises(TypeError, match="early_metric_thresholds must be a dict"):
        PruningConfig(early_metric_thresholds=[("net_pnl", 0.0)])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="output_dir must be a non-empty string"):
        OutputConfig(output_dir="")


def test_expand_parameter_grid_is_cartesian_and_deterministic() -> None:
    grid = {"slow": (20, 50), "fast": (5, 10)}

    combinations = expand_parameter_grid(grid)

    assert combinations == (
        {"fast": 5, "slow": 20},
        {"fast": 5, "slow": 50},
        {"fast": 10, "slow": 20},
        {"fast": 10, "slow": 50},
    )


def test_generate_variants_produces_expected_cartesian_count_and_order() -> None:
    base_spec = _backtest_spec(entry_node="placeholder_entry", exit_node="placeholder_exit", risk_node="placeholder_risk")
    entry_fast = _node_contract(
        "entry_fast",
        "entry",
        ("enter_long", "enter_short", "hold"),
        parameters={"family": "fast"},
    )
    entry_slow = _node_contract(
        "entry_slow",
        "entry",
        ("enter_long", "enter_short", "hold"),
        parameters={"family": "slow"},
    )
    exit_time = _node_contract(
        "exit_time",
        "exit",
        ("close", "hold"),
        parameters={"family": "time"},
    )
    risk_fixed = _node_contract(
        "risk_fixed",
        "risk",
        ("hold",),
        parameters={"family": "fixed"},
    )

    variants = generate_variants(
        base_backtest_spec=base_spec,
        entry_families=[
            (entry_slow, {"slow": (50,), "fast": (10,)}),
            (entry_fast, {"slow": (20,), "fast": (5, 10)}),
        ],
        exit_families=[(exit_time, {"bars": (3, 5)})],
        risk_families=[(risk_fixed, {"risk_fraction": (0.01, 0.02)})],
        search=SearchConfig(mode="grid", max_variants=100),
    )

    assert len(variants) == 12
    assert [variant.entry_contract.name for variant in variants[:8]] == [
        "entry_fast",
        "entry_fast",
        "entry_fast",
        "entry_fast",
        "entry_fast",
        "entry_fast",
        "entry_fast",
        "entry_fast",
    ]
    assert variants[0].entry_contract.manifest["parameters"] == {
        "family": "fast",
        "fast": 5,
        "slow": 20,
    }
    assert [
        (
            variant.exit_contract.manifest["parameters"]["bars"],
            variant.risk_contract.manifest["parameters"]["risk_fraction"],
        )
        for variant in variants[:4]
    ] == [(3, 0.01), (3, 0.02), (5, 0.01), (5, 0.02)]
    assert variants[-1].entry_contract.name == "entry_slow"


def test_grid_mode_max_variants_applies_after_deterministic_ordering() -> None:
    base_spec = _backtest_spec(entry_node="placeholder_entry", exit_node="placeholder_exit", risk_node="placeholder_risk")
    variants = generate_variants(
        base_backtest_spec=base_spec,
        entry_families=[
            (
                _node_contract(
                    "entry_alpha",
                    "entry",
                    ("enter_long", "enter_short", "hold"),
                ),
                {"fast": (5, 10), "slow": (20,)},
            )
        ],
        exit_families=[
            (_node_contract("exit_beta", "exit", ("close", "hold")), {"bars": (3, 5)})
        ],
        risk_families=[
            (_node_contract("risk_gamma", "risk", ("hold",)), {"risk_fraction": (0.01, 0.02)})
        ],
        search=SearchConfig(mode="grid", max_variants=3),
    )

    assert len(variants) == 3
    assert [variant.entry_contract.manifest["parameters"]["fast"] for variant in variants] == [
        5,
        5,
        5,
    ]


def test_randomized_sampling_is_reproducible_with_fixed_seed() -> None:
    base_spec = _backtest_spec(entry_node="placeholder_entry", exit_node="placeholder_exit", risk_node="placeholder_risk")
    entry = _node_contract("entry_alpha", "entry", ("enter_long", "enter_short", "hold"))
    exit_contract = _node_contract("exit_beta", "exit", ("close", "hold"))
    risk_contract = _node_contract("risk_gamma", "risk", ("hold",))

    first = generate_variants(
        base_backtest_spec=base_spec,
        entry_families=[(entry, {"fast": (5, 10, 15), "slow": (20, 30)})],
        exit_families=[(exit_contract, {"bars": (3, 5)})],
        risk_families=[(risk_contract, {"risk_fraction": (0.01, 0.02)})],
        search=SearchConfig(mode="random", max_variants=4, random_seed=17),
    )
    second = generate_variants(
        base_backtest_spec=base_spec,
        entry_families=[(entry, {"slow": (20, 30), "fast": (5, 10, 15)})],
        exit_families=[(exit_contract, {"bars": (3, 5)})],
        risk_families=[(risk_contract, {"risk_fraction": (0.01, 0.02)})],
        search=SearchConfig(mode="random", max_variants=4, random_seed=17),
    )

    assert [variant.variant_id for variant in first] == [
        variant.variant_id for variant in second
    ]


def test_parameter_lineage_is_preserved_inside_variant_contracts() -> None:
    contract = _node_contract(
        "entry_alpha",
        "entry",
        ("enter_long", "enter_short", "hold"),
        parameters={"source": "base", "slow": 50},
    )
    parameterized = parameterize_node_contract(
        contract,
        {"fast": 10, "slow": 20},
    )

    assert parameterized.manifest["parameters"] == {
        "fast": 10,
        "slow": 20,
        "source": "base",
    }
    assert parameterized.manifest["parameter_lineage"] == {
        "base_parameters": {"slow": 50, "source": "base"},
        "variant_parameters": {"fast": 10, "slow": 20},
    }
