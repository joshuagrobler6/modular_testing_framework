from __future__ import annotations

from dataclasses import replace

from trading_lab.experiments import VariantSpec
from trading_lab.nodes.exit_composite_release import (
    CompositeReleaseExitNode,
    build_exit_composite_release_contract,
)
from trading_lab.nodes.exit_macd_release import MacdReleaseExitNode, build_exit_macd_release_contract
from trading_lab.nodes.exit_mfe_giveback import MfeGivebackExitNode, build_exit_mfe_giveback_contract
from trading_lab.nodes.exit_no_progress import NoProgressExitNode, build_exit_no_progress_contract
from trading_lab.nodes.exit_profit_target import ProfitTargetExitNode, build_exit_profit_target_contract
from trading_lab.nodes.exit_time_stop import TimeStopExitNode, build_exit_time_stop_contract
from trading_lab.nodes.exit_trailing_atr import TrailingAtrExitNode, build_exit_trailing_atr_contract
from trading_lab.nodes.exit_zscore_release import ZScoreReleaseExitNode, build_exit_zscore_release_contract


def build_exit_contracts(registry):
    exit_contracts = []

    for hold_bars in (5, 10, 20):
        name = f"exit_time_{hold_bars}"
        contract = build_exit_time_stop_contract(name=name, hold_bars=hold_bars)
        registry.register("exit", name, TimeStopExitNode(hold_bars=hold_bars), contract)
        exit_contracts.append(contract)

    for evaluation_bars, min_profit in ((3, 0.0), (5, 0.0), (8, 0.5)):
        name = f"exit_no_progress_b{evaluation_bars}_p{str(min_profit).replace('.', 'p')}"
        contract = build_exit_no_progress_contract(
            name=name,
            evaluation_bars=evaluation_bars,
            min_open_profit_points=min_profit,
        )
        registry.register(
            "exit",
            name,
            NoProgressExitNode(
                evaluation_bars=evaluation_bars,
                min_open_profit_points=min_profit,
            ),
            contract,
        )
        exit_contracts.append(contract)

    for atr_multiple in (1.5, 2.0, 3.0):
        name = f"exit_profit_target_atr_{str(atr_multiple).replace('.', 'p')}"
        contract = build_exit_profit_target_contract(
            name=name,
            target_kind="atr_from_entry",
            atr_lookback=20,
            atr_multiple=atr_multiple,
        )
        registry.register(
            "exit",
            name,
            ProfitTargetExitNode(
                target_kind="atr_from_entry",
                atr_lookback=20,
                atr_multiple=atr_multiple,
            ),
            contract,
        )
        exit_contracts.append(contract)

    for atr_multiple in (2.0, 3.0, 4.0):
        name = f"exit_trailing_atr_{str(atr_multiple).replace('.', 'p')}"
        contract = build_exit_trailing_atr_contract(
            name=name,
            atr_lookback=20,
            atr_multiple=atr_multiple,
            reference_kind="highest_high",
            activation_bars=2,
        )
        registry.register(
            "exit",
            name,
            TrailingAtrExitNode(
                atr_lookback=20,
                atr_multiple=atr_multiple,
                reference_kind="highest_high",
                activation_bars=2,
            ),
            contract,
        )
        exit_contracts.append(contract)

    for release_kind in ("histogram_cross", "histogram_slope", "macd_signal_cross"):
        name = f"exit_macd_{release_kind}"
        contract = build_exit_macd_release_contract(name=name, release_kind=release_kind, confirm_bars=2)
        registry.register(
            "exit",
            name,
            MacdReleaseExitNode(release_kind=release_kind, confirm_bars=2),
            contract,
        )
        exit_contracts.append(contract)

    for release_kind, threshold in (("threshold_cross", 0.0), ("gradient_flip", 0.05), ("acceleration_flip", 0.05)):
        name = f"exit_z_release_{release_kind}"
        contract = build_exit_zscore_release_contract(
            name=name,
            release_kind=release_kind,
            z_exit_threshold=threshold if release_kind == "threshold_cross" else 0.0,
            gradient_threshold=threshold if release_kind == "gradient_flip" else 0.0,
            acceleration_threshold=threshold if release_kind == "acceleration_flip" else 0.0,
            confirm_bars=2,
        )
        registry.register(
            "exit",
            name,
            ZScoreReleaseExitNode(
                release_kind=release_kind,
                z_exit_threshold=threshold if release_kind == "threshold_cross" else 0.0,
                gradient_threshold=threshold if release_kind == "gradient_flip" else 0.0,
                acceleration_threshold=threshold if release_kind == "acceleration_flip" else 0.0,
                confirm_bars=2,
            ),
            contract,
        )
        exit_contracts.append(contract)

    for giveback_fraction in (0.25, 0.35, 0.50):
        name = f"exit_mfe_giveback_{str(giveback_fraction).replace('.', 'p')}"
        contract = build_exit_mfe_giveback_contract(
            name=name,
            activation_profit_points=1.0,
            giveback_kind="fraction",
            giveback_fraction=giveback_fraction,
        )
        registry.register(
            "exit",
            name,
            MfeGivebackExitNode(
                activation_profit_points=1.0,
                giveback_kind="fraction",
                giveback_fraction=giveback_fraction,
            ),
            contract,
        )
        exit_contracts.append(contract)

    composite_configs = [
        {
            "name": "exit_combo_timeout_macd",
            "max_hold_bars": 20,
            "no_progress_bars": 5,
            "macd_fast_lookback": 12,
            "macd_confirm_bars": 2,
        },
        {
            "name": "exit_combo_trail_zrelease",
            "trailing_atr_lookback": 20,
            "trailing_atr_multiple": 3.0,
            "z_gradient_threshold": 0.05,
            "z_confirm_bars": 2,
        },
        {
            "name": "exit_combo_mfe_macd",
            "macd_fast_lookback": 12,
            "macd_confirm_bars": 2,
            "mfe_activation_profit_points": 1.0,
            "mfe_giveback_fraction": 0.35,
        },
    ]
    for config in composite_configs:
        name = config["name"]
        params = {key: value for key, value in config.items() if key != "name"}
        contract = build_exit_composite_release_contract(name=name, **params)
        registry.register("exit", name, CompositeReleaseExitNode(**params), contract)
        exit_contracts.append(contract)

    return exit_contracts



def build_variants(*, base_spec, entry_contracts, exit_contracts, risk_contract):
    variants = []
    for entry_contract in entry_contracts:
        for exit_contract in exit_contracts:
            variants.append(
                VariantSpec(
                    backtest_spec=replace(
                        base_spec,
                        entry_node=entry_contract.name,
                        exit_node=exit_contract.name,
                        risk_node=risk_contract.name,
                    ),
                    entry_contract=entry_contract,
                    exit_contract=exit_contract,
                    risk_contract=risk_contract,
                )
            )
    return variants
