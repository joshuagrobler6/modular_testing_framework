from __future__ import annotations

import argparse
import os
import re
import runpy
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def _bootstrap_repo_src() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        candidate = parent / "src"
        if (candidate / "trading_lab" / "__init__.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return parent
    raise RuntimeError("Unable to locate repo src/ directory from data/save_data.py.")


REPO_ROOT = _bootstrap_repo_src()

from trading_lab.data import combine_ohlcv_frames, load_mt5_csv

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


DATA_DIR = REPO_ROOT / "data"
COMBINED_OUTPUT_PATH = DATA_DIR / "gold_m5_combined.csv"
LATEST_RAW_PATH = DATA_DIR / "gold_m5_latest.csv"
LEGACY_CONFIG_PATH = DATA_DIR / "save_data,py"
DEFAULT_SYMBOL = "GOLD"
DEFAULT_TIMEFRAME = "5m"
DEFAULT_FETCH_DAYS = 450


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the latest GOLD M5 data, combine all GOLD exports, and write a canonical dataset.",
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--days", type=int, default=DEFAULT_FETCH_DAYS)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--run-zscore", action="store_true")
    parser.add_argument("--output-path", type=Path, default=COMBINED_OUTPUT_PATH)
    parser.add_argument("--latest-raw-path", type=Path, default=LATEST_RAW_PATH)
    parser.add_argument("--legacy-config-path", type=Path, default=LEGACY_CONFIG_PATH)
    parser.add_argument("--terminal-path")
    parser.add_argument("--login", type=int)
    parser.add_argument("--password")
    parser.add_argument("--server")
    return parser.parse_args()


def collect_gold_csv_paths(data_dir: Path, *, exclude_paths: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    excluded = {path.resolve() for path in exclude_paths}
    candidates = [
        path
        for path in data_dir.iterdir()
        if path.is_file()
        and path.suffix.casefold() == ".csv"
        and path.name.casefold().startswith("gold_m5")
        and path.resolve() not in excluded
    ]
    return tuple(sorted(candidates, key=lambda path: (path.stat().st_mtime, path.name.casefold())))


def load_legacy_mt5_settings(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    patterns = {
        "terminal_path": r"^\s*path\s*=\s*['\"]([^'\"]+)['\"]",
        "login": r"^\s*LOGIN\s*=\s*(\d+)",
        "password": r"^\s*Password\s*=\s*['\"]([^'\"]+)['\"]",
        "server": r"server\s*=\s*['\"]([^'\"]+)['\"]",
    }

    settings: dict[str, object] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is None:
            continue
        value = match.group(1).strip()
        settings[key] = int(value) if key == "login" else value
    return settings


def resolve_mt5_settings(args: argparse.Namespace) -> dict[str, object]:
    legacy = load_legacy_mt5_settings(args.legacy_config_path)

    terminal_path = args.terminal_path or os.getenv("MT5_TERMINAL_PATH") or legacy.get("terminal_path")
    login = args.login or os.getenv("MT5_LOGIN") or legacy.get("login")
    password = args.password or os.getenv("MT5_PASSWORD") or legacy.get("password")
    server = args.server or os.getenv("MT5_SERVER") or legacy.get("server")

    settings: dict[str, object] = {}
    if terminal_path:
        settings["terminal_path"] = str(terminal_path)
    if login is not None:
        settings["login"] = int(login)
    if password:
        settings["password"] = str(password)
    if server:
        settings["server"] = str(server)
    return settings


def fetch_latest_gold_data(
    *,
    symbol: str,
    days: int,
    raw_output_path: Path,
    mt5_settings: dict[str, object],
) -> Path:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 is not installed in this environment.")
    if days <= 0:
        raise ValueError("days must be positive.")

    initialize_kwargs: dict[str, object] = {}
    terminal_path = mt5_settings.get("terminal_path")
    if terminal_path is not None:
        terminal_path = Path(str(terminal_path))
        if not terminal_path.exists():
            raise FileNotFoundError(f"MT5 terminal path does not exist: {terminal_path}")
        initialize_kwargs["path"] = str(terminal_path)

    login = mt5_settings.get("login")
    password = mt5_settings.get("password")
    server = mt5_settings.get("server")
    if login is not None or password is not None or server is not None:
        if login is None or password is None or server is None:
            raise RuntimeError(
                "MT5 login settings are incomplete. Provide login, password, and server together."
            )
        initialize_kwargs["login"] = int(login)
        initialize_kwargs["password"] = str(password)
        initialize_kwargs["server"] = str(server)

    if not mt5.initialize(**initialize_kwargs):
        raise RuntimeError(f"MetaTrader5 initialize failed: {mt5.last_error()}")

    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"MetaTrader5 could not select symbol {symbol!r}: {mt5.last_error()}")

        utc_to = datetime.now(timezone.utc)
        utc_from = utc_to - timedelta(days=days)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, utc_from, utc_to)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MetaTrader5 returned no data for {symbol!r} over the last {days} days.")

        frame = pd.DataFrame(rates)
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True).dt.tz_localize(None)
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(raw_output_path, index=False)
        return raw_output_path
    finally:
        mt5.shutdown()


def build_canonical_gold_dataset(
    data_dir: Path,
    *,
    symbol: str,
    timeframe: str,
    output_path: Path,
    raw_paths: tuple[Path, ...] | None = None,
) -> pd.DataFrame:
    if raw_paths is None:
        raw_paths = collect_gold_csv_paths(data_dir, exclude_paths=(output_path,))

    canonical_frames = [load_mt5_csv(path, symbol=symbol, timeframe=timeframe) for path in raw_paths]
    if not canonical_frames:
        raise FileNotFoundError(
            f"No GOLD CSV files were found in {data_dir}. Expected files such as gold_m5_*.csv."
        )

    combined = combine_ohlcv_frames(canonical_frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined


def run_zscore_experiment() -> None:
    script_path = REPO_ROOT / "src" / "Experiments" / "z_score.py"
    runpy.run_path(str(script_path), run_name="__main__")


def main() -> None:
    args = parse_args()
    mt5_settings = resolve_mt5_settings(args)

    fetch_error: Exception | None = None
    fetched_path: Path | None = None
    if not args.skip_fetch:
        try:
            fetched_path = fetch_latest_gold_data(
                symbol=args.symbol,
                days=args.days,
                raw_output_path=args.latest_raw_path,
                mt5_settings=mt5_settings,
            )
        except Exception as exc:
            fetch_error = exc

    raw_paths = collect_gold_csv_paths(DATA_DIR, exclude_paths=(args.output_path,))
    combined = build_canonical_gold_dataset(
        DATA_DIR,
        symbol=args.symbol,
        timeframe=args.timeframe,
        output_path=args.output_path,
        raw_paths=raw_paths,
    )

    print("raw files combined:", len(raw_paths))
    print("combined rows:", len(combined))
    print("ts range:", combined["ts"].min(), "->", combined["ts"].max())
    print("combined output:", args.output_path)
    if fetched_path is not None:
        print("latest raw fetch:", fetched_path)
    elif fetch_error is not None:
        print("latest raw fetch: unavailable")
        print("fetch warning:", str(fetch_error))
    else:
        print("latest raw fetch: skipped")

    if args.run_zscore:
        run_zscore_experiment()


if __name__ == "__main__":
    main()
