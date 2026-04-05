from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading_lab.data import (  # noqa: E402
    get_history_asof,
    resample_ohlcv,
    split_by_symbol_timeframe,
    validate_ohlcv,
)
from trading_lab.schemas import (  # noqa: E402
    DecisionLogSchema,
    EquityCurveSchema,
    FillLogSchema,
    OHLCVSchema,
    OrderLogSchema,
    RESERVED_LEDGER_COLUMNS,
    TradeLedgerSchema,
)


def _make_ohlcv() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 09:31:00", periods=10, freq="1min")
    rows = []

    for index, ts in enumerate(timestamps):
        base = 100.0 + index
        rows.append(
            {
                "ts": ts,
                "symbol": "TEST",
                "timeframe": "1m",
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.5,
                "volume": 1_000.0 + index,
            }
        )

    return pd.DataFrame(rows)


def test_validate_ohlcv_accepts_valid_frame() -> None:
    validated = validate_ohlcv(_make_ohlcv())

    assert list(validated.columns) == [
        "ts",
        "symbol",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert validated["ts"].is_monotonic_increasing
    assert validated["volume"].min() >= 0.0


@pytest.mark.parametrize(
    "mutator",
    [
        lambda df: df.assign(high=df["close"] - 0.1),
        lambda df: df.assign(low=df["open"] + 0.1),
        lambda df: df.assign(volume=-1.0),
        lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True),
        lambda df: pd.concat([df.iloc[[1]], df.iloc[[0]], df.iloc[2:]], ignore_index=True),
    ],
)
def test_validate_ohlcv_rejects_invalid_frames(mutator) -> None:
    invalid = mutator(_make_ohlcv())

    with pytest.raises((SchemaError, SchemaErrors)):
        validate_ohlcv(invalid)


def test_validate_ohlcv_rejects_missing_required_column() -> None:
    invalid = _make_ohlcv().drop(columns=["timeframe"])

    with pytest.raises((SchemaError, SchemaErrors)):
        validate_ohlcv(invalid)


def test_split_by_symbol_timeframe_returns_grouped_bundle() -> None:
    base = _make_ohlcv()
    other = base.assign(symbol="ALT")
    bundled = split_by_symbol_timeframe(pd.concat([base, other], ignore_index=True))

    assert set(bundled) == {("TEST", "1m"), ("ALT", "1m")}
    assert all(frame["ts"].is_monotonic_increasing for frame in bundled.values())


def test_resample_ohlcv_aggregates_completed_bars_only() -> None:
    resampled = resample_ohlcv(_make_ohlcv(), "5m")

    assert list(resampled["ts"]) == [
        pd.Timestamp("2024-01-02 09:35:00"),
        pd.Timestamp("2024-01-02 09:40:00"),
    ]
    assert list(resampled["timeframe"]) == ["5m", "5m"]

    first = resampled.iloc[0]
    assert first["open"] == pytest.approx(100.0)
    assert first["high"] == pytest.approx(105.0)
    assert first["low"] == pytest.approx(99.0)
    assert first["close"] == pytest.approx(104.5)
    assert first["volume"] == pytest.approx(sum(1_000.0 + i for i in range(5)))


def test_resample_drops_trailing_incomplete_target_bar() -> None:
    incomplete = _make_ohlcv().iloc[:-1].reset_index(drop=True)
    resampled = resample_ohlcv(incomplete, "5m")

    assert list(resampled["ts"]) == [pd.Timestamp("2024-01-02 09:35:00")]


def test_get_history_asof_aligns_completed_higher_timeframe_context() -> None:
    base = _make_ohlcv()
    bundle = split_by_symbol_timeframe(base)
    bundle.update(split_by_symbol_timeframe(resample_ohlcv(base, "5m")))

    lower_history = get_history_asof(
        bundle,
        symbol="TEST",
        timeframe="1m",
        ts=pd.Timestamp("2024-01-02 09:37:00"),
        lookback=3,
    )
    higher_history = get_history_asof(
        bundle,
        symbol="TEST",
        timeframe="5m",
        ts=pd.Timestamp("2024-01-02 09:37:00"),
        lookback=3,
    )

    assert list(lower_history["ts"]) == [
        pd.Timestamp("2024-01-02 09:35:00"),
        pd.Timestamp("2024-01-02 09:36:00"),
        pd.Timestamp("2024-01-02 09:37:00"),
    ]
    assert list(higher_history["ts"]) == [pd.Timestamp("2024-01-02 09:35:00")]


def test_get_history_asof_never_leaks_future_bars() -> None:
    base = _make_ohlcv()
    bundle = split_by_symbol_timeframe(base)
    history = get_history_asof(
        bundle,
        symbol="TEST",
        timeframe="1m",
        ts=pd.Timestamp("2024-01-02 09:37:00"),
        lookback=10,
    )

    assert history["ts"].max() == pd.Timestamp("2024-01-02 09:37:00")
    assert pd.Timestamp("2024-01-02 09:38:00") not in set(history["ts"])


@pytest.mark.parametrize(
    "schema_cls",
    [
        DecisionLogSchema,
        OrderLogSchema,
        FillLogSchema,
        TradeLedgerSchema,
        EquityCurveSchema,
    ],
)
def test_reserved_future_columns_exist_on_all_ledger_schemas(schema_cls) -> None:
    schema_columns = set(schema_cls.to_schema().columns)

    for column in RESERVED_LEDGER_COLUMNS:
        assert column in schema_columns


def test_decision_log_schema_accepts_reserved_columns_and_rejects_bad_values() -> None:
    valid = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2024-01-02 09:35:00"),
                "symbol": "TEST",
                "entry_action": "enter_long",
                "exit_action": "none",
                "risk_approved": True,
                "target_units": 10.0,
                "resolved_action": "submit_entry_long",
                "reason": "sma_cross",
                "metadata": {"fast": 5, "slow": 20},
                "run_id": "run-0000000000000000",
                "position_id": None,
                "parent_position_id": None,
                "lot_id": None,
                "entry_tag": None,
                "exit_tag": None,
                "risk_tag": None,
                "resolver_status": None,
                "rejection_reason": None,
                "node_version": None,
                "contract_version": "1.0",
            }
        ]
    )

    validated = DecisionLogSchema.validate(valid, lazy=True)
    assert list(validated.columns) == list(valid.columns)

    invalid = valid.assign(entry_action="buy_now")
    with pytest.raises((SchemaError, SchemaErrors)):
        DecisionLogSchema.validate(invalid, lazy=True)


def test_ohlcv_schema_columns_match_expected_shape() -> None:
    assert list(OHLCVSchema.to_schema().columns) == [
        "ts",
        "symbol",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
