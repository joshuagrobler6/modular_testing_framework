from __future__ import annotations

from dataclasses import replace
from itertools import product

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

    for hold_bars in (5, 8, 10, 15, 20, 30):
        name = f"exit_time_{hold_bars}"
        contract = build_exit_time_stop_contract(name=name, hold_bars=hold_bars)
        registry.register("exit", name, TimeStopExitNode(hold_bars=hold_bars), contract)
        exit_contracts.append(contract)

    for evaluation_bars, min_profit in product((3, 5, 8, 12), (0.0, 0.25, 0.5)):
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

    for atr_lookback, atr_multiple in product((14, 20), (1.0, 1.5, 2.0, 2.5, 3.0)):
        name = (
            "exit_profit_target_atr_"
            f"lb{atr_lookback}_"
            f"m{str(atr_multiple).replace('.', 'p')}"
        )
        contract = build_exit_profit_target_contract(
            name=name,
            target_kind="atr_from_entry",
            atr_lookback=atr_lookback,
            atr_multiple=atr_multiple,
        )
        registry.register(
            "exit",
            name,
            ProfitTargetExitNode(
                target_kind="atr_from_entry",
                atr_lookback=atr_lookback,
                atr_multiple=atr_multiple,
            ),
            contract,
        )
        exit_contracts.append(contract)

    for target_percent in (0.005, 0.01, 0.015, 0.02):
        name = f"exit_profit_target_pct_{str(target_percent).replace('.', 'p')}"
        contract = build_exit_profit_target_contract(
            name=name,
            target_kind="percent",
            target_percent=target_percent,
        )
        registry.register(
            "exit",
            name,
            ProfitTargetExitNode(
                target_kind="percent",
                target_percent=target_percent,
            ),
            contract,
        )
        exit_contracts.append(contract)

    for atr_lookback, atr_multiple, reference_kind, activation_bars in product(
        (14, 20),
        (1.5, 2.0, 3.0),
        ("highest_high", "highest_close"),
        (1, 2),
    ):
        name = (
            "exit_trailing_atr_"
            f"lb{atr_lookback}_"
            f"m{str(atr_multiple).replace('.', 'p')}_"
            f"{reference_kind}_"
            f"a{activation_bars}"
        )
        contract = build_exit_trailing_atr_contract(
            name=name,
            atr_lookback=atr_lookback,
            atr_multiple=atr_multiple,
            reference_kind=reference_kind,
            activation_bars=activation_bars,
        )
        registry.register(
            "exit",
            name,
            TrailingAtrExitNode(
                atr_lookback=atr_lookback,
                atr_multiple=atr_multiple,
                reference_kind=reference_kind,
                activation_bars=activation_bars,
            ),
            contract,
        )
        exit_contracts.append(contract)

    for release_kind, confirm_bars in product(
        ("histogram_cross", "histogram_slope", "macd_signal_cross"),
        (1, 2, 3),
    ):
        name = f"exit_macd_{release_kind}_c{confirm_bars}"
        contract = build_exit_macd_release_contract(
            name=name,
            release_kind=release_kind,
            confirm_bars=confirm_bars,
        )
        registry.register(
            "exit",
            name,
            MacdReleaseExitNode(release_kind=release_kind, confirm_bars=confirm_bars),
            contract,
        )
        exit_contracts.append(contract)

    for threshold, confirm_bars in product((0.0, 0.25, 0.5), (1, 2)):
        name = f"exit_z_release_threshold_cross_t{str(threshold).replace('.', 'p')}_c{confirm_bars}"
        contract = build_exit_zscore_release_contract(
            name=name,
            release_kind="threshold_cross",
            z_exit_threshold=threshold,
            confirm_bars=confirm_bars,
        )
        registry.register(
            "exit",
            name,
            ZScoreReleaseExitNode(
                release_kind="threshold_cross",
                z_exit_threshold=threshold,
                confirm_bars=confirm_bars,
            ),
            contract,
        )
        exit_contracts.append(contract)

    for threshold, confirm_bars in product((0.03, 0.05, 0.08), (1, 2)):
        name = f"exit_z_release_gradient_flip_t{str(threshold).replace('.', 'p')}_c{confirm_bars}"
        contract = build_exit_zscore_release_contract(
            name=name,
            release_kind="gradient_flip",
            gradient_threshold=threshold,
            confirm_bars=confirm_bars,
        )
        registry.register(
            "exit",
            name,
            ZScoreReleaseExitNode(
                release_kind="gradient_flip",
                gradient_threshold=threshold,
                confirm_bars=confirm_bars,
            ),
            contract,
        )
        exit_contracts.append(contract)

    for threshold, confirm_bars in product((0.03, 0.05, 0.08), (1, 2)):
        name = f"exit_z_release_acceleration_flip_t{str(threshold).replace('.', 'p')}_c{confirm_bars}"
        contract = build_exit_zscore_release_contract(
            name=name,
            release_kind="acceleration_flip",
            acceleration_threshold=threshold,
            confirm_bars=confirm_bars,
        )
        registry.register(
            "exit",
            name,
            ZScoreReleaseExitNode(
                release_kind="acceleration_flip",
                acceleration_threshold=threshold,
                confirm_bars=confirm_bars,
            ),
            contract,
        )
        exit_contracts.append(contract)

    for activation_profit_points, giveback_fraction in product(
        (0.5, 1.0, 1.5),
        (0.25, 0.35, 0.50),
    ):
        name = (
            "exit_mfe_giveback_"
            f"a{str(activation_profit_points).replace('.', 'p')}_"
            f"g{str(giveback_fraction).replace('.', 'p')}"
        )
        contract = build_exit_mfe_giveback_contract(
            name=name,
            activation_profit_points=activation_profit_points,
            giveback_kind="fraction",
            giveback_fraction=giveback_fraction,
        )
        registry.register(
            "exit",
            name,
            MfeGivebackExitNode(
                activation_profit_points=activation_profit_points,
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
        {
            "name": "exit_combo_time_trail_fast",
            "max_hold_bars": 15,
            "trailing_atr_lookback": 14,
            "trailing_atr_multiple": 1.5,
            "trailing_activation_bars": 1,
        },
        {
            "name": "exit_combo_time_trail_slow",
            "max_hold_bars": 30,
            "trailing_atr_lookback": 20,
            "trailing_atr_multiple": 3.0,
            "trailing_activation_bars": 2,
        },
        {
            "name": "exit_combo_target_trail_2r",
            "profit_target_atr_multiple": 2.0,
            "trailing_atr_lookback": 20,
            "trailing_atr_multiple": 2.0,
            "trailing_activation_bars": 2,
        },
        {
            "name": "exit_combo_target_trail_3r",
            "profit_target_atr_multiple": 3.0,
            "trailing_atr_lookback": 20,
            "trailing_atr_multiple": 3.0,
            "trailing_activation_bars": 2,
        },
        {
            "name": "exit_combo_trail_mfe_guard",
            "trailing_atr_lookback": 20,
            "trailing_atr_multiple": 2.5,
            "trailing_activation_bars": 2,
            "mfe_activation_profit_points": 1.0,
            "mfe_giveback_fraction": 0.35,
        },
        {
            "name": "exit_combo_timeout_trail_mfe",
            "max_hold_bars": 30,
            "trailing_atr_lookback": 20,
            "trailing_atr_multiple": 2.5,
            "trailing_activation_bars": 2,
            "mfe_activation_profit_points": 1.5,
            "mfe_giveback_fraction": 0.25,
        },
        {
            "name": "exit_combo_no_progress_trail",
            "no_progress_bars": 8,
            "no_progress_min_profit_points": 0.25,
            "trailing_atr_lookback": 20,
            "trailing_atr_multiple": 2.0,
            "trailing_activation_bars": 2,
        },
        {
            "name": "exit_combo_macd_zrelease",
            "macd_fast_lookback": 12,
            "macd_confirm_bars": 2,
            "z_gradient_threshold": 0.05,
            "z_confirm_bars": 2,
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
