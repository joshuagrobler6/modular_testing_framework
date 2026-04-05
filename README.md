# trading-lab

`trading-lab` is a contract-first backtesting framework for trading research. It is built around modular strategy nodes, one central engine, and canonical ledgers that serve as the only durable base for analytics.

## Purpose

- keep entry, exit, and risk logic modular
- keep execution, accounting, and evaluation centralized
- make metrics reproducible from stable ledgers
- preserve anti-leakage behavior by default

## High-Level Architecture

Core modules:

- `contracts.py`: object-level contracts and protocol interfaces
- `schemas.py`: dataframe contracts for OHLCV and ledgers
- `registry.py`: registry-only node discovery
- `engine.py`: central evaluator and simulator
- `analytics.py`: ledger-driven metrics and reports
- `nodes/`: separate entry, exit, and risk modules

Next layer above the engine:

- `experiments.py`: variant construction and orchestration
- `runner.py`: fold and holdout execution
- `search.py`: bounded search and pruning
- `reporting.py`: summary exports and deep dive outputs

Flow:

1. validate OHLCV input
2. build decision context from completed data only
3. call entry, exit, and risk nodes
4. let the engine resolve actions and simulate execution
5. write canonical ledgers
6. derive analytics from ledgers only

Experiment-layer rules:

- the engine is responsible only for simulating one variant and one fold correctly
- experiment orchestration, fold handling, holdout handling, search, pruning, and reporting sit above the engine
- variant combinations are built by the experiment layer, not by nodes
- all concrete strategy combinations must have stable `variant_id` values
- holdout is never used for search ranking or optimizer objectives
- deep dive outputs are selected by `variant_id` and are fold-aware
- deep dive price entry and exit plots should default to one selected fold or holdout, not the full dataset
- search must support both `max_variants` and `max_runtime_seconds`
- pruning decisions must be logged and reproducible

## V1 Scope

Current supported behavior:

- OHLCV data
- market orders only
- next-bar-open fills
- single position per symbol
- no pyramiding
- long and short
- fees and simple slippage
- one central event loop
- canonical ledgers plus ledger-driven analytics

## Canonical Ledgers

Canonical ledgers are the base layer for all future calculations:

- `decision_log`
- `order_log`
- `fill_log`
- `trade_ledger`
- `equity_curve`

Metrics and reports must read these ledgers, not arbitrary engine state. This is the core extensibility rule for future slicing, attribution, robustness analysis, and live-vs-backtest comparison.

## Validation And Leakage Philosophy

Validation is explicit at boundaries:

- frozen dataclasses for object contracts
- `Protocol` interfaces for node call signatures
- `pandera` schemas for OHLCV and ledgers
- ledger validation before result return
- analytics input validation before metric computation

Anti-leakage rules:

- strategy nodes may only access fully completed bars as of the decision timestamp
- higher timeframe context must be as-of aligned
- partially formed bars must never be exposed to nodes
- the engine owns execution timing and fill timing

## How New Nodes Are Added

New strategy logic is added as a new module under `src/trading_lab/nodes/`.

Rules:

- entry, exit, and risk logic stay in separate node modules
- nodes are discovered through the registry, not direct engine imports
- adding a new signal must not require changing engine semantics
- nodes do not generate experiment grids or search combinations
- nodes must consume and emit declared contract objects only
- unsupported capability requirements must fail loudly
- unsupported complexity must not silently degrade into approximate v1 behavior
- if a node requires behavior outside declared engine capabilities, that is an engine feature request, not just a new node

## Future Extension Principles

The initial boundary must already reserve space for:

- multiple entries
- partial exits
- scale-in behavior
- same-bar reversal
- stop, limit, and bracket order workflows
- multiple lots per symbol
- target-position mode and portfolio allocators
- richer risk actions and overlays
- engine capability checks
- node manifests and capability requirements
- metric dependency checks
- future lot-level accounting
- advanced metrics and portfolio analysis

Backward-compatible evolution is preferred. New complexity should be introduced through declared capabilities and contracts, not silent engine assumptions.

Summary comparison metrics are expected to be exported to Excel from the reporting layer, while deep dive analysis remains tied to a selected `variant_id` plus fold or holdout context.

## Running Experiments

The minimal orchestration entrypoint is `run_search_entrypoint(...)`. It accepts either:

- an in-memory `ExperimentSpec` plus an `ObjectiveConfig`
- a serialized `SearchRunConfig` JSON payload containing both `experiment` and `objective`

The entrypoint:

- runs the selected search mode
- writes a summary workbook
- optionally writes deep dive artifacts
- prints a runtime summary and stopping reason
- writes a reproducibility manifest when `outputs.write_run_manifests=True`

### Define An Experiment Run

```python
from datetime import datetime

import pandas as pd

from trading_lab.experiments import (
    DeepDiveConfig,
    ExperimentSpec,
    FoldSpec,
    HoldoutSpec,
    SearchConfig,
)
from trading_lab.search import ObjectiveConfig, run_search_entrypoint

experiment = ExperimentSpec(
    name="sma_grid",
    variants=variants,
    folds=(
        FoldSpec(
            fold_index=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 2, 1),
            validation_start=datetime(2024, 2, 1),
            validation_end=datetime(2024, 3, 1),
            label="fold_a",
        ),
        FoldSpec(
            fold_index=1,
            train_start=datetime(2024, 2, 1),
            train_end=datetime(2024, 3, 1),
            validation_start=datetime(2024, 3, 1),
            validation_end=datetime(2024, 4, 1),
            label="fold_b",
        ),
    ),
    holdout=HoldoutSpec(
        start=datetime(2024, 4, 1),
        end=datetime(2024, 5, 1),
        label="holdout",
    ),
    search=SearchConfig(
        mode="grid",
        max_variants=50,
        max_runtime_seconds=1800,
        random_seed=17,
    ),
    deep_dive=DeepDiveConfig(
        selected_variant_ids=("variant_abc123",),
        selected_folds=("fold_b",),
    ),
)

objective = ObjectiveConfig.single_metric("net_pnl")
result = run_search_entrypoint(experiment, data, objective=objective)
```

### Folds And Holdout

- validation folds live in `experiment.folds`
- holdout is configured separately with `experiment.holdout`
- holdout runs only after cross-validation for a variant is available
- holdout metrics are preserved for reporting, but they are never used for search ranking or optimization objectives

### Search Bounds

Use `SearchConfig` to bound the orchestration layer:

- `max_variants`: stop launching new variants or trials after the configured count
- `max_runtime_seconds`: stop launching new work when the time budget is exhausted
- both bounds are enforced through the runner controls, not by nodes

### Deep Dive By Variant ID

Deep dive selection is explicit and stable:

- choose one or more `variant_id` values in `DeepDiveConfig.selected_variant_ids`
- choose specific validation folds in `selected_folds`
- set `include_holdout=True` to target holdout
- price entry and exit charts default to the selected fold or holdout window, not the full dataset

Artifacts are written under a separate deep-dive folder tree keyed by `variant_id` and fold or holdout label.

### Search First, Deep Dive Later

If you want to rank variants first and only generate deep-dive artifacts after you have chosen stable `variant_id` values, run the search and reporting steps separately.

For real end-to-end parameter sweeps, register each concrete node instance under a distinct name. The experiment layer varies `VariantSpec` and node contracts, while the engine still executes node callables from the registry by name.

```python
from dataclasses import replace
from datetime import datetime
import math

import pandas as pd

from trading_lab import (
    BacktestSpec,
    CostAssumptions,
    ExperimentSpec,
    FoldSpec,
    HoldoutSpec,
    InstrumentMeta,
    NodeRegistry,
    ObjectiveConfig,
    PruningConfig,
    SearchConfig,
    VariantSpec,
    export_deep_dive_artifacts,
    export_summary_workbook,
    run_search_experiment,
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


def build_demo_data() -> pd.DataFrame:
    closes = [
        100.0 + 0.05 * index + 4.0 * math.sin(index / 5.0) + 2.0 * math.sin(index / 11.0)
        for index in range(180)
    ]
    opens = [closes[0], *closes[:-1]]
    highs = [max(open_, close) + 0.50 for open_, close in zip(opens, closes, strict=True)]
    lows = [min(open_, close) - 0.50 for open_, close in zip(opens, closes, strict=True)]
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
            "symbol": "DEMO",
            "timeframe": "1D",
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1_000.0] * len(closes),
        }
    )
    return validate_ohlcv(frame)


def build_registry() -> tuple[NodeRegistry, tuple[object, ...], object, object]:
    registry = NodeRegistry()

    entry_contract_a = build_entry_sma_cross_contract(
        name="entry_sma_3_8",
        fast_window=3,
        slow_window=8,
    )
    entry_contract_b = build_entry_sma_cross_contract(
        name="entry_sma_5_13",
        fast_window=5,
        slow_window=13,
    )
    exit_contract = build_exit_time_stop_contract(
        name="exit_time_6",
        hold_bars=6,
    )
    risk_contract = build_risk_fixed_fraction_contract(
        name="risk_fraction_10pct",
        capital_fraction=0.10,
    )

    registry.register(
        "entry",
        entry_contract_a.name,
        SMACrossEntryNode(fast_window=3, slow_window=8),
        entry_contract_a,
    )
    registry.register(
        "entry",
        entry_contract_b.name,
        SMACrossEntryNode(fast_window=5, slow_window=13),
        entry_contract_b,
    )
    registry.register(
        "exit",
        exit_contract.name,
        TimeStopExitNode(hold_bars=6),
        exit_contract,
    )
    registry.register(
        "risk",
        risk_contract.name,
        FixedFractionRiskNode(capital_fraction=0.10),
        risk_contract,
    )

    return registry, (entry_contract_a, entry_contract_b), exit_contract, risk_contract


def build_variant(base_spec, entry_contract, exit_contract, risk_contract) -> VariantSpec:
    return VariantSpec(
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


data = build_demo_data()
instrument = InstrumentMeta(symbol="DEMO", price_increment=0.01, quantity_increment=1.0)
node_registry, entry_contracts, exit_contract, risk_contract = build_registry()

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
    build_variant(base_spec, entry_contract, exit_contract, risk_contract)
    for entry_contract in entry_contracts
)

experiment = ExperimentSpec(
    name="search_then_deep_dive",
    variants=variants,
    folds=(
        FoldSpec(
            fold_index=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 2, 1),
            validation_start=datetime(2024, 2, 1),
            validation_end=datetime(2024, 3, 15),
            label="fold_a",
        ),
        FoldSpec(
            fold_index=1,
            train_start=datetime(2024, 2, 1),
            train_end=datetime(2024, 3, 15),
            validation_start=datetime(2024, 3, 15),
            validation_end=datetime(2024, 4, 30),
            label="fold_b",
        ),
    ),
    holdout=HoldoutSpec(
        start=datetime(2024, 4, 30),
        end=datetime(2024, 6, 15),
        label="holdout",
    ),
    search=SearchConfig(
        mode="grid",
        max_variants=len(variants),
        max_runtime_seconds=120,
        random_seed=17,
    ),
    pruning=PruningConfig(stop_on_invalid_numeric_state=False),
)

objective = ObjectiveConfig.single_metric("net_pnl")
search_result = run_search_experiment(
    experiment,
    data,
    objective=objective,
    runner_kwargs={"node_registry": node_registry},
)

experiment_result = search_result.experiment_results[0]
summary_path = export_summary_workbook(
    experiment_result,
    "outputs/search_then_deep_dive/experiment_summary.xlsx",
)

print("Summary workbook:", summary_path)
print("Cross-validation rankings:")
for ranking in experiment_result.cv_rankings:
    print(ranking.rank, ranking.variant_id, ranking.metric_name, ranking.metric_value)

if experiment_result.cv_rankings:
    selected_variant_id = experiment_result.cv_rankings[0].variant_id
else:
    selected_variant_id = search_result.best_variant_id
if selected_variant_id is None:
    raise RuntimeError("No ranked variant_id is available for deep dive.")

# Later, after reviewing the workbook or rankings, export deep-dive artifacts
# for the selected variant. Choose either one or more folds, or include holdout.
artifacts = export_deep_dive_artifacts(
    experiment_result,
    data,
    output_dir="outputs/search_then_deep_dive/deep_dive",
    selected_variant_ids=(selected_variant_id,),
    selected_folds=("fold_b",),
    include_holdout=False,
)

for artifact in artifacts:
    print("Deep dive target:", artifact.target.variant_id, artifact.target.phase, artifact.target.label)
    print("  target_dir:", artifact.target_dir)
    print("  equity_plot:", artifact.equity_plot_path)
    print("  price_plot:", artifact.price_plot_path)
    print("  trade_log:", artifact.trade_log_path)
```

If you want the holdout deep dive instead of a validation fold, call `export_deep_dive_artifacts(...)` with `selected_folds=()` and `include_holdout=True`.

### Grid Search

Use `SearchConfig(mode="grid", ...)` when variants are already concrete or when you build them from deterministic family grids before execution.

```python
objective = ObjectiveConfig.single_metric("net_pnl")
result = run_search_entrypoint(experiment, data, objective=objective)
```

### Random Search

Use `SearchConfig(mode="random", random_seed=..., ...)` to sample from the deterministic variant universe reproducibly.

```python
from dataclasses import replace

experiment = replace(experiment, search=SearchConfig(mode="random", max_variants=25, random_seed=17))
result = run_search_entrypoint(experiment, data, objective=objective)
```

### Optuna Search

Optuna remains optional and lightweight. Install the extra if needed:

```bash
pip install .[optuna]
```

Optuna search requires a `variant_factory(trial) -> VariantSpec` because trials create variants dynamically.

```python
from trading_lab.search import OptunaSearchAdapter

def variant_factory(trial):
    fast = trial.suggest_int("fast", 5, 20)
    slow = trial.suggest_int("slow", 30, 80)
    return build_variant(fast=fast, slow=slow)

experiment = ExperimentSpec(
    name="sma_optuna",
    variants=(template_variant,),
    folds=folds,
    search=SearchConfig(mode="optuna", max_variants=50, max_runtime_seconds=1800),
)

result = run_search_entrypoint(
    experiment,
    data,
    objective=ObjectiveConfig.single_metric("net_pnl"),
    variant_factory=variant_factory,
)
```

Optuna objectives use cross-validation aggregate metrics only. Holdout remains excluded from optimization.

### Serialized Run Config

To persist and rerun a search configuration:

```python
from trading_lab.search import SearchRunConfig, serialize_search_run_config

payload = serialize_search_run_config(
    SearchRunConfig(experiment=experiment, objective=objective)
)

result = run_search_entrypoint(payload, data)
```
