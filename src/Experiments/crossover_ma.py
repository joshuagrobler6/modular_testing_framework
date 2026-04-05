from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd


def _bootstrap_repo_src() -> None:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        candidate = parent / "src"
        if (candidate / "trading_lab" / "__init__.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


_bootstrap_repo_src()

from trading_lab import (
    BacktestSpec,
    CostAssumptions,
    ExperimentSpec,
    FoldSpec,
    HoldoutSpec,
    InstrumentMeta,
    NodeRegistry,
    ObjectiveConfig,
    OutputConfig,
    SearchConfig,
    SearchRunConfig,
    VariantSpec,
    run_search_entrypoint,
    validate_ohlcv,
)
from trading_lab.nodes.entry_sma_cross import (
    SMACrossEntryNode,
    build_entry_sma_cross_contract,
)
from trading_lab.nodes.exit_time_stop import (
    TimeStopExitNode,
    build_exit_time_stop_contract,
)
from trading_lab.nodes.risk_fixed_fraction import (
    FixedFractionRiskNode,
    build_risk_fixed_fraction_contract,
)

DATA_PATH = Path("data/demo_bars.csv")


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH, parse_dates=["ts"])
    return validate_ohlcv(frame)


def build_components():
    registry = NodeRegistry()

    entry_3_8 = build_entry_sma_cross_contract(
        name="entry_sma_3_8",
        fast_window=3,
        slow_window=8,
    )
    entry_5_13 = build_entry_sma_cross_contract(
        name="entry_sma_5_13",
        fast_window=5,
        slow_window=13,
    )
    exit_6 = build_exit_time_stop_contract(
        name="exit_time_6",
        hold_bars=6,
    )
    risk_10pct = build_risk_fixed_fraction_contract(
        name="risk_fraction_10pct",
        capital_fraction=0.10,
    )

    registry.register(
        "entry",
        entry_3_8.name,
        SMACrossEntryNode(fast_window=3, slow_window=8),
        entry_3_8,
    )
    registry.register(
        "entry",
        entry_5_13.name,
        SMACrossEntryNode(fast_window=5, slow_window=13),
        entry_5_13,
    )
    registry.register(
        "exit",
        exit_6.name,
        TimeStopExitNode(hold_bars=6),
        exit_6,
    )
    registry.register(
        "risk",
        risk_10pct.name,
        FixedFractionRiskNode(capital_fraction=0.10),
        risk_10pct,
    )

    return registry, (entry_3_8, entry_5_13), exit_6, risk_10pct


def build_run_config(entry_contracts, exit_contract, risk_contract) -> SearchRunConfig:
    instrument = InstrumentMeta(
        symbol="DEMO",
        price_increment=0.01,
        quantity_increment=1.0,
    )

    base_spec = BacktestSpec(
        name="template",
        instrument=instrument,
        entry_node="placeholder_entry",
        exit_node="placeholder_exit",
        risk_node="placeholder_risk",
        initial_cash=100_000.0,
        costs=CostAssumptions(fee_rate=0.0005, slippage_bps=1.0),
    )

    variants = tuple(
        VariantSpec(
            backtest_spec=replace(
                base_spec,
                name=f"{entry_contract.name}__{exit_contract.name}__{risk_contract.name}",
                entry_node=entry_contract.name,
                exit_node=exit_contract.name,
                risk_node=risk_contract.name,
            ),
            entry_contract=entry_contract,
            exit_contract=exit_contract,
            risk_contract=risk_contract,
        )
        for entry_contract in entry_contracts
    )

    experiment = ExperimentSpec(
        name="sma_grid_demo",
        variants=variants,
        folds=(
            FoldSpec(
                fold_index=0,
                train_start=datetime(2024, 1, 1),
                train_end=datetime(2024, 3, 1),
                validation_start=datetime(2024, 3, 1),
                validation_end=datetime(2024, 4, 1),
                label="fold_a",
            ),
            FoldSpec(
                fold_index=1,
                train_start=datetime(2024, 2, 1),
                train_end=datetime(2024, 4, 1),
                validation_start=datetime(2024, 4, 1),
                validation_end=datetime(2024, 5, 1),
                label="fold_b",
            ),
        ),
        holdout=HoldoutSpec(
            start=datetime(2024, 5, 1),
            end=datetime(2024, 6, 1),
            label="holdout",
        ),
        search=SearchConfig(
            mode="grid",
            max_variants=len(variants),
            max_runtime_seconds=300,
            random_seed=17,
        ),
        outputs=OutputConfig(
            output_dir="outputs/sma_grid_demo",
            summary_excel_name="summary.xlsx",
            write_run_manifests=True,
        ),
    )

    objective = ObjectiveConfig.single_metric("net_pnl")
    return SearchRunConfig(experiment=experiment, objective=objective)


def main() -> None:
    data = load_data()
    registry, entry_contracts, exit_contract, risk_contract = build_components()
    run_config = build_run_config(entry_contracts, exit_contract, risk_contract)

    result = run_search_entrypoint(
        run_config,
        data,
        runner_kwargs={"node_registry": registry},
    )

    print("summary workbook:", result.summary_workbook_path)
    print("best variant_id:", result.search_result.best_variant_id)
    print("stopping reason:", result.stopping_reason)


if __name__ == "__main__":
    main()
