from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_zscore_script() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    for module_name in list(sys.modules):
        if module_name == "trading_lab" or module_name.startswith("trading_lab."):
            sys.modules.pop(module_name, None)
    script_path = repo_root / "src" / "Experiments" / "z_score.py"
    return runpy.run_path(str(script_path), run_name="zscore_test")


def test_zscore_experiment_registers_expected_entry_variants() -> None:
    namespace = _load_zscore_script()

    registry, entry_contracts, exit_contract, risk_contract = namespace["build_components"]()

    assert len(entry_contracts) == 12
    assert len({contract.name for contract in entry_contracts}) == 12
    assert entry_contracts[0].name.startswith("entry_zs_")
    assert registry.resolve("entry", entry_contracts[-1].name) is not None
    assert exit_contract.name == "exit_time_10"
    assert risk_contract.name == "risk_fraction_05pct"


def test_zscore_experiment_builds_variants_and_runs_search(tmp_path) -> None:
    namespace = _load_zscore_script()
    run_search_entrypoint = namespace["run_search_entrypoint"]

    data = namespace["build_demo_data"]()
    registry, entry_contracts, exit_contract, risk_contract = namespace["build_components"]()
    run_config = namespace["build_run_config"](
        data,
        entry_contracts[2:4],
        exit_contract,
        risk_contract,
    )

    result = run_search_entrypoint(
        run_config,
        data,
        runner_kwargs={"node_registry": registry},
        output_dir=tmp_path,
        write_manifest=False,
        verbose=False,
    )

    assert len(run_config.experiment.variants) == 2
    assert run_config.experiment.holdout is not None
    assert result.summary_workbook_path.exists()
    assert result.search_result.best_variant_id in {
        variant.variant_id for variant in result.run_config.experiment.variants
    }


def test_zscore_experiment_respects_runtime_env_override(monkeypatch) -> None:
    namespace = _load_zscore_script()
    monkeypatch.setenv("Z_SCORE_MAX_RUNTIME_SECONDS", "1800")

    data = namespace["build_demo_data"]()
    registry, entry_contracts, exit_contract, risk_contract = namespace["build_components"]()
    run_config = namespace["build_run_config"](
        data,
        entry_contracts[:1],
        exit_contract,
        risk_contract,
    )

    assert registry.resolve("entry", entry_contracts[0].name) is not None
    assert run_config.experiment.search.max_runtime_seconds == 1800
