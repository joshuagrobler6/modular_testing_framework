from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from trading_lab.schemas import OHLCVSchema

SymbolTimeframe = tuple[str, str]
DataBundle = Mapping[SymbolTimeframe, pd.DataFrame]

_TIMEFRAME_PATTERN = re.compile(r"^\s*(\d+)\s*([A-Za-z]+)\s*$")
_TIMEFRAME_UNITS = {
    "m": ("m", "min"),
    "min": ("m", "min"),
    "mins": ("m", "min"),
    "minute": ("m", "min"),
    "minutes": ("m", "min"),
    "t": ("m", "min"),
    "h": ("h", "h"),
    "hr": ("h", "h"),
    "hour": ("h", "h"),
    "hours": ("h", "h"),
    "d": ("d", "D"),
    "day": ("d", "D"),
    "days": ("d", "D"),
    "w": ("w", "W"),
    "wk": ("w", "W"),
    "week": ("w", "W"),
    "weeks": ("w", "W"),
}

_OHLCV_COLUMNS = [
    "ts",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def _parse_timeframe(timeframe: str) -> tuple[str, str, pd.Timedelta]:
    if not isinstance(timeframe, str):
        raise TypeError("timeframe must be a string.")

    match = _TIMEFRAME_PATTERN.match(timeframe)
    if match is None:
        raise ValueError(f"unsupported timeframe format: {timeframe!r}.")

    multiple = int(match.group(1))
    raw_unit = match.group(2).lower()

    if multiple <= 0:
        raise ValueError("timeframe multiple must be positive.")
    if raw_unit not in _TIMEFRAME_UNITS:
        raise ValueError(f"unsupported timeframe unit: {raw_unit!r}.")

    label_unit, pandas_unit = _TIMEFRAME_UNITS[raw_unit]
    label = f"{multiple}{label_unit}"

    if pandas_unit == "W":
        delta = pd.Timedelta(weeks=multiple)
    elif pandas_unit == "D":
        delta = pd.Timedelta(days=multiple)
    elif pandas_unit == "h":
        delta = pd.Timedelta(hours=multiple)
    else:
        delta = pd.Timedelta(minutes=multiple)

    return label, f"{multiple}{pandas_unit}", delta


def _require_dataframe(name: str, df: object) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    return df


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _validate_lookback(lookback: int) -> None:
    if isinstance(lookback, bool) or not isinstance(lookback, int):
        raise TypeError("lookback must be an integer.")
    if lookback <= 0:
        raise ValueError("lookback must be positive.")


def _empty_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_OHLCV_COLUMNS)


def _coerce_timestamp_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        timestamps = pd.to_datetime(series, unit="s", utc=True)
    else:
        timestamps = pd.to_datetime(series, utc=True)
    return pd.Series(timestamps).dt.tz_localize(None)


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    validated = OHLCVSchema.validate(_require_dataframe("df", df).copy(), lazy=True)
    return validated.reset_index(drop=True)


def normalize_mt5_ohlcv(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    timestamp_column: str = "time",
    volume_column: str = "tick_volume",
) -> pd.DataFrame:
    source = _require_dataframe("df", df).copy()
    source = source.loc[:, ~source.columns.astype(str).str.startswith("Unnamed:")]

    symbol = _require_text("symbol", symbol)
    timeframe_label, _, _ = _parse_timeframe(_require_text("timeframe", timeframe))

    required_columns = [
        timestamp_column,
        "open",
        "high",
        "low",
        "close",
        volume_column,
    ]
    missing_columns = [column for column in required_columns if column not in source.columns]
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"MT5 OHLCV frame is missing required columns: {missing}.")

    normalized = pd.DataFrame(
        {
            "ts": _coerce_timestamp_series(source[timestamp_column]),
            "symbol": symbol,
            "timeframe": timeframe_label,
            "open": pd.to_numeric(source["open"], errors="raise"),
            "high": pd.to_numeric(source["high"], errors="raise"),
            "low": pd.to_numeric(source["low"], errors="raise"),
            "close": pd.to_numeric(source["close"], errors="raise"),
            "volume": pd.to_numeric(source[volume_column], errors="raise"),
        }
    )
    normalized = normalized.sort_values("ts", kind="stable")
    normalized = normalized.drop_duplicates(["ts", "symbol", "timeframe"], keep="last")
    return validate_ohlcv(normalized[_OHLCV_COLUMNS].reset_index(drop=True))


def load_mt5_csv(path: str | Path, *, symbol: str, timeframe: str) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    return normalize_mt5_ohlcv(frame, symbol=symbol, timeframe=timeframe)


def combine_ohlcv_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    validated_frames = [validate_ohlcv(_require_dataframe("frame", frame)) for frame in frames]
    if not validated_frames:
        return validate_ohlcv(_empty_ohlcv_frame())

    combined = pd.concat(validated_frames, ignore_index=True)
    combined = combined.sort_values(["ts", "symbol", "timeframe"], kind="stable")
    combined = combined.drop_duplicates(["ts", "symbol", "timeframe"], keep="last")
    return validate_ohlcv(combined[_OHLCV_COLUMNS].reset_index(drop=True))


def split_by_symbol_timeframe(df: pd.DataFrame) -> dict[SymbolTimeframe, pd.DataFrame]:
    validated = validate_ohlcv(df)
    bundles: dict[SymbolTimeframe, pd.DataFrame] = {}

    for (symbol, timeframe), group in validated.groupby(
        ["symbol", "timeframe"], sort=False, as_index=False
    ):
        bundles[(symbol, timeframe)] = group.reset_index(drop=True).copy()

    return bundles


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    validated = validate_ohlcv(df)
    if validated.empty:
        return validate_ohlcv(_empty_ohlcv_frame())

    target_label, target_freq, target_delta = _parse_timeframe(timeframe)

    source_timeframes = validated["timeframe"].drop_duplicates().tolist()
    if len(source_timeframes) != 1:
        raise ValueError("resample_ohlcv requires a single source timeframe.")

    _, _, source_delta = _parse_timeframe(source_timeframes[0])
    if target_delta < source_delta:
        raise ValueError("target timeframe must not be finer than the source timeframe.")
    if target_delta.value % source_delta.value != 0:
        raise ValueError("target timeframe must be an integer multiple of source timeframe.")

    expected_rows = target_delta.value // source_delta.value
    if expected_rows == 1:
        result = validated.copy()
        result["timeframe"] = target_label
        return validate_ohlcv(result)

    resampled_frames: list[pd.DataFrame] = []

    for symbol, group in validated.groupby("symbol", sort=False):
        indexed = group.set_index("ts")
        aggregation = indexed.resample(
            target_freq,
            label="right",
            closed="right",
            origin="start_day",
        ).agg(
            {
                "symbol": "last",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        counts = indexed["open"].resample(
            target_freq,
            label="right",
            closed="right",
            origin="start_day",
        ).count()

        complete = aggregation.loc[counts == expected_rows].dropna(subset=["open", "close"])
        if complete.empty:
            continue

        complete = complete.reset_index()
        complete["symbol"] = symbol
        complete["timeframe"] = target_label
        complete = complete[_OHLCV_COLUMNS]
        resampled_frames.append(complete)

    if not resampled_frames:
        return validate_ohlcv(_empty_ohlcv_frame())

    combined = pd.concat(resampled_frames, ignore_index=True)
    return validate_ohlcv(combined)


def get_history_asof(
    data_bundle: DataBundle,
    symbol: str,
    timeframe: str,
    ts: pd.Timestamp,
    lookback: int,
) -> pd.DataFrame:
    _validate_lookback(lookback)
    symbol = _require_text("symbol", symbol)
    timeframe = _require_text("timeframe", timeframe)

    key = (symbol, timeframe)
    if key not in data_bundle:
        raise KeyError(f"missing data bundle for {key!r}.")

    asof_ts = pd.Timestamp(ts)
    source = validate_ohlcv(_require_dataframe("data_bundle entry", data_bundle[key]))
    history = source.loc[source["ts"] <= asof_ts].tail(lookback).reset_index(drop=True)

    if history.empty:
        return validate_ohlcv(_empty_ohlcv_frame())
    return validate_ohlcv(history)


__all__ = [
    "combine_ohlcv_frames",
    "DataBundle",
    "SymbolTimeframe",
    "get_history_asof",
    "load_mt5_csv",
    "normalize_mt5_ohlcv",
    "resample_ohlcv",
    "split_by_symbol_timeframe",
    "validate_ohlcv",
]
