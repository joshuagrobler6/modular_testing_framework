from __future__ import annotations

import math
import os
import sys
from dataclasses import replace
from datetime import datetime, timedelta
from functools import partial
from itertools import product
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
    MetricConstraint,
    NodeRegistry,
    ObjectiveConfig,
    OutputConfig,
    PruningConfig,
    SearchConfig,
    SearchRunConfig,
    VariantSpec,
    run_backtest,
    run_search_entrypoint,
    validate_ohlcv,
)
from trading_lab.nodes.entry_zscore_state import (
    ZScoreStateEntryNode,
    build_entry_zscore_state_contract,
)
from trading_lab.nodes.entry_sma_cross import (
    SMACrossEntryNode,
    build_entry_sma_cross_contract,
)
from trading_lab.nodes.z_score_exit_grid_example import build_exit_contracts
from trading_lab.nodes.risk_fixed_fraction import (
    FixedFractionRiskNode,
    build_risk_fixed_fraction_contract,
)

DATA_PATH_CANDIDATES = (
    Path("data/gold_m5_combined.csv"),
    Path("data/z_score_bars.csv"),
)
DEFAULT_OUTPUT_DIR = "outputs/z_score_search"
PRICE_INCREMENT = 0.01
QUANTITY_INCREMENT = 1.0
INITIAL_CASH = 100_000.0
DEFAULT_MAX_RUNTIME_SECONDS = 600
DEFAULT_RUNTIME_SECONDS_PER_VARIANT = 30
DEFAULT_MAX_RUNTIME_CAP_SECONDS = 8 * 60 * 60
DEFAULT_MAX_PARALLEL_VARIANTS = 4
DEFAULT_MAX_TRIALS = 1000
DEFAULT_MIN_TRADES = 20
DEFAULT_EARLY_MIN_FOLD_TRADES = 5
DEFAULT_EARLY_MIN_BARS = 5000

ZSCORE_ENTRY_BASELINE_KINDS = ("ema", "sma", "median")
ZSCORE_ENTRY_BASELINE_LOOKBACKS = (10, 20, 40)
ZSCORE_ENTRY_SCALE_KINDS = ("atr", "residual_std", "residual_mad")
ZSCORE_ENTRY_SCALE_LOOKBACKS = (10, 20)
ZSCORE_ENTRY_Z_THRESHOLDS = (0.5, 0.75, 1.0, 1.25)

ZSCORE_GATED_ENTRY_VARIANTS = (
    {
        "baseline_kind": "ema",
        "baseline_lookback": 20,
        "scale_kind": "residual_std",
        "scale_lookback": 20,
        "z_threshold": 0.75,
        "gradient_threshold": 0.03,
        "gradient_horizon": 3,
        "persistence_bars": 2,
        "require_baseline_slope": True,
        "baseline_slope_horizon": 5,
    },
    {
        "baseline_kind": "ema",
        "baseline_lookback": 20,
        "scale_kind": "residual_std",
        "scale_lookback": 20,
        "z_threshold": 1.0,
        "gradient_threshold": 0.05,
        "gradient_horizon": 3,
        "persistence_bars": 2,
        "require_baseline_slope": True,
        "baseline_slope_horizon": 5,
    },
    {
        "baseline_kind": "sma",
        "baseline_lookback": 20,
        "scale_kind": "residual_std",
        "scale_lookback": 20,
        "z_threshold": 0.75,
        "gradient_threshold": 0.03,
        "gradient_horizon": 3,
        "persistence_bars": 2,
        "require_baseline_slope": True,
        "baseline_slope_horizon": 5,
    },
    {
        "baseline_kind": "sma",
        "baseline_lookback": 20,
        "scale_kind": "residual_std",
        "scale_lookback": 20,
        "z_threshold": 1.0,
        "gradient_threshold": 0.05,
        "gradient_horizon": 3,
        "persistence_bars": 2,
        "require_baseline_slope": True,
        "baseline_slope_horizon": 5,
    },
    {
        "baseline_kind": "ema",
        "baseline_lookback": 40,
        "scale_kind": "atr",
        "scale_lookback": 20,
        "z_threshold": 0.75,
        "gradient_threshold": 0.03,
        "gradient_horizon": 5,
        "persistence_bars": 3,
        "require_baseline_slope": True,
        "baseline_slope_horizon": 10,
    },
    {
        "baseline_kind": "ema",
        "baseline_lookback": 40,
        "scale_kind": "atr",
        "scale_lookback": 20,
        "z_threshold": 1.0,
        "gradient_threshold": 0.05,
        "gradient_horizon": 5,
        "persistence_bars": 3,
        "require_baseline_slope": True,
        "baseline_slope_horizon": 10,
    },
    {
        "baseline_kind": "sma",
        "baseline_lookback": 40,
        "scale_kind": "atr",
        "scale_lookback": 20,
        "z_threshold": 0.75,
        "gradient_threshold": 0.03,
        "gradient_horizon": 5,
        "persistence_bars": 3,
        "require_baseline_slope": True,
        "baseline_slope_horizon": 10,
    },
    {
        "baseline_kind": "sma",
        "baseline_lookback": 40,
        "scale_kind": "atr",
        "scale_lookback": 20,
        "z_threshold": 1.0,
        "gradient_threshold": 0.05,
        "gradient_horizon": 5,
        "persistence_bars": 3,
        "require_baseline_slope": True,
        "baseline_slope_horizon": 10,
    },
    {
        "baseline_kind": "median",
        "baseline_lookback": 20,
        "scale_kind": "residual_mad",
        "scale_lookback": 20,
        "z_threshold": 0.75,
        "gradient_threshold": 0.03,
        "gradient_horizon": 3,
        "persistence_bars": 2,
    },
    {
        "baseline_kind": "median",
        "baseline_lookback": 20,
        "scale_kind": "residual_mad",
        "scale_lookback": 20,
        "z_threshold": 1.0,
        "gradient_threshold": 0.05,
        "gradient_horizon": 3,
        "persistence_bars": 2,
    },
    {
        "baseline_kind": "ema",
        "baseline_lookback": 40,
        "scale_kind": "residual_std",
        "scale_lookback": 20,
        "z_threshold": 1.0,
        "gradient_threshold": 0.05,
        "gradient_horizon": 3,
        "persistence_bars": 2,
        "require_baseline_slope": True,
        "baseline_slope_horizon": 5,
    },
    {
        "baseline_kind": "sma",
        "baseline_lookback": 40,
        "scale_kind": "residual_std",
        "scale_lookback": 20,
        "z_threshold": 1.0,
        "gradient_threshold": 0.05,
        "gradient_horizon": 3,
        "persistence_bars": 2,
        "require_baseline_slope": True,
        "baseline_slope_horizon": 5,
    },
)

MA_CROSS_ENTRY_VARIANTS = (
    {"fast_window": 10, "slow_window": 50, "allow_short_signals": True},
    {"fast_window": 20, "slow_window": 100, "allow_short_signals": True},
    {"fast_window": 50, "slow_window": 200, "allow_short_signals": True},
    {"fast_window": 10, "slow_window": 50, "allow_short_signals": False},
    {"fast_window": 20, "slow_window": 100, "allow_short_signals": False},
    {"fast_window": 50, "slow_window": 200, "allow_short_signals": False},
)


def build_demo_data() -> pd.DataFrame:
    closes: list[float] = []
    for index in range(240):
        drift = 0.03 * index
        seasonal = 3.0 * math.sin(index / 6.0) + 1.5 * math.sin(index / 17.0)
        pulse = 0.0
        if index % 48 in {5, 6, 7}:
            pulse += 3.5
        if index % 65 in {25, 26, 27}:
            pulse -= 4.0
        closes.append(100.0 + drift + seasonal + pulse)

    opens = [closes[0], *closes[:-1]]
    highs = [max(open_, close) + 0.60 for open_, close in zip(opens, closes, strict=True)]
    lows = [min(open_, close) - 0.60 for open_, close in zip(opens, closes, strict=True)]
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


def load_data() -> pd.DataFrame:
    data_path = resolve_data_path()
    if data_path is not None:
        frame = pd.read_csv(data_path, parse_dates=["ts"])
        return validate_ohlcv(frame)
    return build_demo_data()


def resolve_data_path() -> Path | None:
    for path in DATA_PATH_CANDIDATES:
        if path.exists():
            return path
    return None


def build_zscore_entry_parameter_sets() -> tuple[dict[str, object], ...]:
    parameter_sets: list[dict[str, object]] = []
    for (
        baseline_kind,
        baseline_lookback,
        scale_kind,
        scale_lookback,
        z_threshold,
    ) in product(
        ZSCORE_ENTRY_BASELINE_KINDS,
        ZSCORE_ENTRY_BASELINE_LOOKBACKS,
        ZSCORE_ENTRY_SCALE_KINDS,
        ZSCORE_ENTRY_SCALE_LOOKBACKS,
        ZSCORE_ENTRY_Z_THRESHOLDS,
    ):
        parameter_sets.append(
            {
                "baseline_kind": baseline_kind,
                "baseline_lookback": baseline_lookback,
                "scale_kind": scale_kind,
                "scale_lookback": scale_lookback,
                "z_threshold": z_threshold,
            }
        )
    parameter_sets.extend(dict(parameters) for parameters in ZSCORE_GATED_ENTRY_VARIANTS)
    return tuple(parameter_sets)


def build_ma_cross_entry_parameter_sets() -> tuple[dict[str, object], ...]:
    return tuple(dict(parameters) for parameters in MA_CROSS_ENTRY_VARIANTS)


def build_components():
    registry = NodeRegistry()
    entry_contracts = []
    for parameters in build_zscore_entry_parameter_sets():
        name = build_entry_variant_name(parameters)
        contract = build_entry_zscore_state_contract(name=name, **parameters)
        registry.register("entry", name, ZScoreStateEntryNode(**parameters), contract)
        entry_contracts.append(contract)
    for parameters in build_ma_cross_entry_parameter_sets():
        name = build_ma_cross_entry_name(parameters)
        contract = build_entry_sma_cross_contract(name=name, **parameters)
        registry.register("entry", name, SMACrossEntryNode(**parameters), contract)
        entry_contracts.append(contract)

    exit_contracts = tuple(build_exit_contracts(registry))
    risk_contract = build_risk_fixed_fraction_contract(
        name="risk_fraction_05pct",
        capital_fraction=0.05,
        max_holding_bars=12,
    )
    registry.register(
        "risk",
        risk_contract.name,
        FixedFractionRiskNode(capital_fraction=0.05, max_holding_bars=12),
        risk_contract,
    )
    return registry, tuple(entry_contracts), exit_contracts, risk_contract


def build_run_config(data: pd.DataFrame, entry_contracts, exit_contracts, risk_contract) -> SearchRunConfig:
    instrument = infer_instrument(data)
    folds, holdout = build_time_windows(data)
    base_spec = BacktestSpec(
        name="zscore_template",
        instrument=instrument,
        entry_node="placeholder_entry",
        exit_node="placeholder_exit",
        risk_node="placeholder_risk",
        initial_cash=INITIAL_CASH,
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
        for exit_contract in exit_contracts
    )
    experiment = ExperimentSpec(
        name="zscore_signal_quality",
        variants=variants,
        folds=folds,
        holdout=holdout,
        search=SearchConfig(
            mode="optuna",
            max_variants=resolve_max_trials(len(variants)),
            max_runtime_seconds=resolve_max_runtime_seconds(len(variants)),
            max_parallel_variants=1,
            random_seed=17,
        ),
        outputs=OutputConfig(
            output_dir=DEFAULT_OUTPUT_DIR,
            summary_excel_name="zscore_summary.xlsx",
            write_run_manifests=True,
        ),
        pruning=PruningConfig(
            stop_on_invalid_numeric_state=False,
            min_trades=resolve_min_trades(),
            early_min_trades=resolve_early_min_fold_trades(),
            early_min_bars=resolve_early_min_bars(),
            early_metric_thresholds={
                "trade_count": float(resolve_early_min_fold_trades()),
            },
        ),
    )
    return SearchRunConfig(
        experiment=experiment,
        objective=ObjectiveConfig.single_metric(
            "net_pnl",
            constraints=(MetricConstraint("trade_count", minimum=resolve_min_trades()),),
        ),
    )


def build_search_runner_kwargs(registry: NodeRegistry) -> dict[str, object]:
    return {
        "node_registry": registry,
        "backtest_fn": partial(
            run_backtest,
            validate_outputs=False,
            capture_decision_details=False,
        ),
        "retain_run_results": False,
        "prepare_backtest_inputs": True,
    }


def infer_instrument(data: pd.DataFrame) -> InstrumentMeta:
    symbols = tuple(sorted({str(symbol) for symbol in data["symbol"]}))
    if len(symbols) != 1:
        raise ValueError("z_score experiment expects exactly one symbol in the input data.")
    return InstrumentMeta(
        symbol=symbols[0],
        price_increment=PRICE_INCREMENT,
        quantity_increment=QUANTITY_INCREMENT,
    )


def resolve_max_runtime_seconds(variant_count: int | None = None) -> int:
    raw_value = os.getenv("Z_SCORE_MAX_RUNTIME_SECONDS")
    if raw_value is None:
        if variant_count is None:
            return DEFAULT_MAX_RUNTIME_SECONDS
        return max(
            DEFAULT_MAX_RUNTIME_SECONDS,
            min(
                variant_count * DEFAULT_RUNTIME_SECONDS_PER_VARIANT,
                DEFAULT_MAX_RUNTIME_CAP_SECONDS,
            ),
        )

    value = int(raw_value)
    if value <= 0:
        raise ValueError("Z_SCORE_MAX_RUNTIME_SECONDS must be positive.")
    return value


def resolve_max_parallel_variants() -> int:
    raw_value = os.getenv("Z_SCORE_MAX_PARALLEL_VARIANTS")
    if raw_value is not None:
        value = int(raw_value)
        if value <= 0:
            raise ValueError("Z_SCORE_MAX_PARALLEL_VARIANTS must be positive.")
        return value

    cpu_count = os.cpu_count() or 1
    half_cores = max(cpu_count // 2, 1)
    return min(DEFAULT_MAX_PARALLEL_VARIANTS, half_cores)


def resolve_max_trials(variant_count: int) -> int:
    raw_value = os.getenv("Z_SCORE_MAX_TRIALS")
    if raw_value is not None:
        value = int(raw_value)
        if value <= 0:
            raise ValueError("Z_SCORE_MAX_TRIALS must be positive.")
        return min(value, variant_count)
    return min(DEFAULT_MAX_TRIALS, variant_count)


def resolve_min_trades() -> int:
    raw_value = os.getenv("Z_SCORE_MIN_TRADES")
    if raw_value is not None:
        value = int(raw_value)
        if value < 1:
            raise ValueError("Z_SCORE_MIN_TRADES must be >= 1.")
        return value
    return DEFAULT_MIN_TRADES


def resolve_early_min_fold_trades() -> int:
    raw_value = os.getenv("Z_SCORE_EARLY_MIN_FOLD_TRADES")
    if raw_value is not None:
        value = int(raw_value)
        if value < 1:
            raise ValueError("Z_SCORE_EARLY_MIN_FOLD_TRADES must be >= 1.")
        return value
    return DEFAULT_EARLY_MIN_FOLD_TRADES


def resolve_early_min_bars() -> int:
    raw_value = os.getenv("Z_SCORE_EARLY_MIN_BARS")
    if raw_value is not None:
        value = int(raw_value)
        if value < 1:
            raise ValueError("Z_SCORE_EARLY_MIN_BARS must be >= 1.")
        return value
    return DEFAULT_EARLY_MIN_BARS


def build_time_windows(data: pd.DataFrame) -> tuple[tuple[FoldSpec, ...], HoldoutSpec]:
    timestamps = sorted_unique_timestamps(data)
    if len(timestamps) < 40:
        raise ValueError(
            "z_score experiment requires at least 40 unique timestamps to build folds and holdout."
        )

    count = len(timestamps)
    train_b_start = count // 5
    train_a_end = count // 2
    validation_a_end = (count * 7) // 10
    validation_b_end = (count * 17) // 20
    if not 0 < train_b_start < train_a_end < validation_a_end < validation_b_end < count:
        raise ValueError("unable to derive non-overlapping folds from the provided timestamps.")

    folds = (
        FoldSpec(
            fold_index=0,
            train_start=timestamps[0],
            train_end=timestamps[train_a_end],
            validation_start=timestamps[train_a_end],
            validation_end=timestamps[validation_a_end],
            label="fold_a",
        ),
        FoldSpec(
            fold_index=1,
            train_start=timestamps[train_b_start],
            train_end=timestamps[validation_a_end],
            validation_start=timestamps[validation_a_end],
            validation_end=timestamps[validation_b_end],
            label="fold_b",
        ),
    )
    holdout = HoldoutSpec(
        start=timestamps[validation_b_end],
        end=_final_window_end(timestamps),
        label="holdout",
    )
    return folds, holdout


def sorted_unique_timestamps(data: pd.DataFrame) -> tuple[datetime, ...]:
    unique_index = pd.DatetimeIndex(pd.to_datetime(data["ts"])).sort_values().unique()
    return tuple(timestamp.to_pydatetime() for timestamp in unique_index)


def build_entry_variant_name(parameters: dict[str, object]) -> str:
    scale_tokens = {"atr": "atr", "residual_std": "resstd", "residual_mad": "resmad"}
    parts = [
        "entry_zs",
        f"{parameters['baseline_kind']}{int(parameters['baseline_lookback'])}",
        f"{scale_tokens[str(parameters['scale_kind'])]}{int(parameters['scale_lookback'])}",
        f"z{_number_token(float(parameters['z_threshold']))}",
        f"p{int(parameters.get('persistence_bars', 1))}",
    ]
    gradient_threshold = parameters.get("gradient_threshold")
    if gradient_threshold is not None:
        parts.append(f"g{_number_token(float(gradient_threshold))}")
    acceleration_threshold = parameters.get("acceleration_threshold")
    if acceleration_threshold is not None:
        parts.append(f"a{_number_token(float(acceleration_threshold))}")
    if bool(parameters.get("require_baseline_slope", False)):
        parts.append(f"slope{int(parameters.get('baseline_slope_horizon', 1))}")
    if not bool(parameters.get("allow_short_signals", True)):
        parts.append("longonly")
    return "_".join(parts)


def build_ma_cross_entry_name(parameters: dict[str, object]) -> str:
    parts = [
        "entry_ma",
        f"{int(parameters['fast_window'])}x{int(parameters['slow_window'])}",
    ]
    if not bool(parameters.get("allow_short_signals", True)):
        parts.append("longonly")
    return "_".join(parts)


def build_optuna_variant_factory(
    run_config: SearchRunConfig,
    entry_contracts,
    exit_contracts,
    risk_contract,
):
    if not run_config.experiment.variants:
        raise ValueError("run_config must contain at least one template variant.")

    base_spec = replace(
        run_config.experiment.variants[0].backtest_spec,
        name="zscore_optuna_template",
        entry_node="placeholder_entry",
        exit_node="placeholder_exit",
        risk_node=risk_contract.name,
    )
    entry_by_name = {contract.name: contract for contract in entry_contracts}
    exit_by_name = {contract.name: contract for contract in exit_contracts}
    entry_names = tuple(sorted(entry_by_name))
    exit_names = tuple(sorted(exit_by_name))

    def variant_factory(trial):
        entry_name = trial.suggest_categorical("entry_node", entry_names)
        exit_name = trial.suggest_categorical("exit_node", exit_names)
        entry_contract = entry_by_name[entry_name]
        exit_contract = exit_by_name[exit_name]
        return VariantSpec(
            backtest_spec=replace(
                base_spec,
                name=f"{entry_name}__{exit_name}__{risk_contract.name}",
                entry_node=entry_name,
                exit_node=exit_name,
                risk_node=risk_contract.name,
            ),
            entry_contract=entry_contract,
            exit_contract=exit_contract,
            risk_contract=risk_contract,
        )

    return variant_factory


def _number_token(value: float) -> str:
    return f"{int(round(value * 100.0)):03d}"


def _final_window_end(timestamps: tuple[datetime, ...]) -> datetime:
    if len(timestamps) >= 2:
        step = timestamps[-1] - timestamps[-2]
        if step <= timedelta(0):
            step = timedelta(days=1)
    else:
        step = timedelta(days=1)
    return timestamps[-1] + step


def main() -> None:
    data = load_data()
    data_path = resolve_data_path()
    data_source = str(data_path) if data_path is not None else "built-in demo data"
    registry, entry_contracts, exit_contracts, risk_contract = build_components()
    run_config = build_run_config(data, entry_contracts, exit_contracts, risk_contract)
    result = run_search_entrypoint(
        run_config,
        data,
        runner_kwargs=build_search_runner_kwargs(registry),
        variant_factory=build_optuna_variant_factory(
            run_config,
            entry_contracts,
            exit_contracts,
            risk_contract,
        ),
    )
    print("data source:", data_source)
    print("entry variants:", len(entry_contracts))
    print("exit variants:", len(exit_contracts))
    print("total variants:", len(run_config.experiment.variants))
    print("summary workbook:", result.summary_workbook_path)
    print("best variant_id:", result.search_result.best_variant_id)
    print("stopping reason:", result.stopping_reason)


if __name__ == "__main__":
    main()
