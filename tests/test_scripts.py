from __future__ import annotations

import runpy
import sys
from pathlib import Path


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
