from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _FakeTrial:
    def __init__(self, number: int) -> None:
        self.number = number
        self.user_attrs: dict[str, object] = {}

    def suggest_categorical(self, name: str, choices):
        ordered = tuple(choices)
        return ordered[self.number % len(ordered)]

    def set_user_attr(self, name: str, value: object) -> None:
        self.user_attrs[name] = value


class _FakeStudy:
    def __init__(self) -> None:
        self.trials: list[_FakeTrial] = []
        self.optimize_calls: list[dict[str, object]] = []

    def optimize(self, objective, **kwargs) -> None:
        self.optimize_calls.append(dict(kwargs))
        n_trials = int(kwargs.get("n_trials", 1))
        for number in range(n_trials):
            trial = _FakeTrial(number)
            self.trials.append(trial)
            objective(trial)


class _FakeOptunaModule:
    def __init__(self) -> None:
        self.studies: list[_FakeStudy] = []

    def create_study(self, *, direction: str, study_name: str):
        study = _FakeStudy()
        self.studies.append(study)
        return study


def _load_zscore_script() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "src" / "Experiments" / "z_score.py"
    return runpy.run_path(str(script_path), run_name="zscore_test")


def test_zscore_experiment_registers_expected_entry_variants() -> None:
    namespace = _load_zscore_script()

    registry, entry_contracts, exit_contracts, risk_contract = namespace["build_components"]()

    assert len(entry_contracts) == 234
    assert len({contract.name for contract in entry_contracts}) == 234
    assert entry_contracts[0].name.startswith("entry_zs_")
    assert any(contract.name == "entry_ma_20x100" for contract in entry_contracts)
    assert any(contract.name == "entry_ma_20x100_longonly" for contract in entry_contracts)
    assert registry.resolve("entry", entry_contracts[-1].name) is not None
    assert len(exit_contracts) == 103
    assert len({contract.name for contract in exit_contracts}) == 103
    assert exit_contracts[0].name == "exit_time_5"
    assert any(contract.name == "exit_combo_time_trail_fast" for contract in exit_contracts)
    assert exit_contracts[-1].name == "exit_combo_macd_zrelease"
    assert registry.resolve("exit", exit_contracts[-1].name) is not None
    assert risk_contract.name == "risk_fraction_05pct"


def test_zscore_experiment_builds_variants_and_runs_search(tmp_path) -> None:
    namespace = _load_zscore_script()
    run_search_entrypoint = namespace["run_search_entrypoint"]
    fake_optuna = _FakeOptunaModule()
    namespace["os"].environ["Z_SCORE_MIN_TRADES"] = "1"
    namespace["os"].environ["Z_SCORE_EARLY_MIN_FOLD_TRADES"] = "1"
    namespace["os"].environ["Z_SCORE_EARLY_MIN_BARS"] = "1"
    try:
        data = namespace["build_demo_data"]()
        registry, entry_contracts, exit_contracts, risk_contract = namespace["build_components"]()
        run_config = namespace["build_run_config"](
            data,
            entry_contracts[2:4],
            exit_contracts[:2],
            risk_contract,
        )

        result = run_search_entrypoint(
            run_config,
            data,
            runner_kwargs=namespace["build_search_runner_kwargs"](registry),
            variant_factory=namespace["build_optuna_variant_factory"](
                run_config,
                entry_contracts[2:4],
                exit_contracts[:2],
                risk_contract,
            ),
            optuna_module=fake_optuna,
            output_dir=tmp_path,
            write_manifest=False,
            verbose=False,
        )

        assert len(run_config.experiment.variants) == 4
        assert run_config.experiment.holdout is not None
        assert run_config.experiment.search.mode == "optuna"
        assert result.summary_workbook_path.exists()
        assert result.search_result.best_variant_id in {
            variant.variant_id for variant in result.run_config.experiment.variants
        }
    finally:
        namespace["os"].environ.pop("Z_SCORE_MIN_TRADES", None)
        namespace["os"].environ.pop("Z_SCORE_EARLY_MIN_FOLD_TRADES", None)
        namespace["os"].environ.pop("Z_SCORE_EARLY_MIN_BARS", None)


def test_zscore_experiment_respects_runtime_env_override(monkeypatch) -> None:
    namespace = _load_zscore_script()
    monkeypatch.setenv("Z_SCORE_MAX_RUNTIME_SECONDS", "1800")

    data = namespace["build_demo_data"]()
    registry, entry_contracts, exit_contracts, risk_contract = namespace["build_components"]()
    run_config = namespace["build_run_config"](
        data,
        entry_contracts[:10],
        exit_contracts[:40],
        risk_contract,
    )

    assert registry.resolve("entry", entry_contracts[0].name) is not None
    assert registry.resolve("exit", exit_contracts[0].name) is not None
    assert run_config.experiment.search.max_runtime_seconds == 1800


def test_zscore_experiment_respects_optuna_env_overrides(monkeypatch) -> None:
    namespace = _load_zscore_script()
    monkeypatch.setenv("Z_SCORE_MAX_TRIALS", "321")
    monkeypatch.setenv("Z_SCORE_MIN_TRADES", "25")
    monkeypatch.setenv("Z_SCORE_EARLY_MIN_FOLD_TRADES", "7")
    monkeypatch.setenv("Z_SCORE_EARLY_MIN_BARS", "6000")

    data = namespace["build_demo_data"]()
    registry, entry_contracts, exit_contracts, risk_contract = namespace["build_components"]()
    run_config = namespace["build_run_config"](
        data,
        entry_contracts[:10],
        exit_contracts[:40],
        risk_contract,
    )

    assert registry.resolve("entry", entry_contracts[0].name) is not None
    assert registry.resolve("exit", exit_contracts[0].name) is not None
    assert run_config.experiment.search.max_variants == 321
    assert run_config.experiment.pruning.min_trades == 25
    assert run_config.experiment.pruning.early_min_trades == 7
    assert run_config.experiment.pruning.early_min_bars == 6000
