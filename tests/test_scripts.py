from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pandas as pd


def test_crossover_ma_bootstraps_src_for_direct_execution(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    script_path = src_root / "Experiments" / "crossover_ma.py"

    src_root_resolved = str(src_root.resolve())
    cleaned_path = [
        entry
        for entry in sys.path
        if str(Path(entry or ".").resolve()) != src_root_resolved
    ]
    monkeypatch.setattr(sys, "path", cleaned_path)

    for module_name in list(sys.modules):
        if module_name == "trading_lab" or module_name.startswith("trading_lab."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    runpy.run_path(str(script_path), run_name="__test__")

    assert sys.path[0] == src_root_resolved


def test_save_data_script_builds_canonical_combined_gold_dataset(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "data" / "save_data.py"
    namespace = runpy.run_path(str(script_path), run_name="save_data_test")

    first = pd.DataFrame(
        [
            {
                "time": "2024-01-02 09:30:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "tick_volume": 1000,
                "spread": 10,
                "real_volume": 0,
            },
            {
                "time": "2024-01-02 09:35:00",
                "open": 100.5,
                "high": 101.4,
                "low": 100.1,
                "close": 101.0,
                "tick_volume": 1100,
                "spread": 9,
                "real_volume": 0,
            },
        ]
    )
    second = pd.DataFrame(
        [
            {
                "Unnamed: 0": 7,
                "time": "2024-01-02 09:35:00",
                "open": 100.6,
                "high": 101.8,
                "low": 100.2,
                "close": 101.3,
                "tick_volume": 1150,
                "spread": 8,
                "real_volume": 0,
            },
            {
                "Unnamed: 0": 8,
                "time": "2024-01-02 09:40:00",
                "open": 101.3,
                "high": 102.0,
                "low": 101.0,
                "close": 101.7,
                "tick_volume": 1200,
                "spread": 8,
                "real_volume": 0,
            },
        ]
    )

    first.to_csv(tmp_path / "gold_m5_a.csv", index=False)
    second.to_csv(tmp_path / "gold_m5_b.csv", index=False)

    output_path = tmp_path / "gold_m5_combined.csv"
    combined = namespace["build_canonical_gold_dataset"](
        tmp_path,
        symbol="GOLD",
        timeframe="5m",
        output_path=output_path,
    )

    assert output_path.exists()
    assert list(combined["ts"]) == [
        pd.Timestamp("2024-01-02 09:30:00"),
        pd.Timestamp("2024-01-02 09:35:00"),
        pd.Timestamp("2024-01-02 09:40:00"),
    ]
    assert combined.iloc[1]["close"] == 101.3
    assert set(combined["symbol"]) == {"GOLD"}
    assert set(combined["timeframe"]) == {"5m"}
