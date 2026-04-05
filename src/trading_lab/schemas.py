from __future__ import annotations

from typing import Optional

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

RESERVED_LEDGER_COLUMNS = (
    "run_id",
    "position_id",
    "parent_position_id",
    "lot_id",
    "entry_tag",
    "exit_tag",
    "risk_tag",
    "resolver_status",
    "rejection_reason",
    "node_version",
    "contract_version",
)


class OHLCVSchema(pa.DataFrameModel):
    ts: Series[pd.Timestamp]
    symbol: Series[str]
    timeframe: Series[str]
    open: Series[float] = pa.Field(gt=0)
    high: Series[float] = pa.Field(gt=0)
    low: Series[float] = pa.Field(gt=0)
    close: Series[float] = pa.Field(gt=0)
    volume: Series[float] = pa.Field(ge=0)

    class Config:
        strict = True
        coerce = True
        unique = ["ts", "symbol", "timeframe"]

    @pa.dataframe_check
    def monotonic_ts(cls, df: pd.DataFrame) -> bool:
        if df.empty:
            return True
        return bool(
            df.groupby(["symbol", "timeframe"], sort=False)["ts"]
            .apply(lambda series: series.is_monotonic_increasing)
            .all()
        )

    @pa.dataframe_check
    def high_envelopes_bar(cls, df: pd.DataFrame) -> bool:
        if df.empty:
            return True
        return bool((df["high"] >= df[["open", "close", "low"]].max(axis=1)).all())

    @pa.dataframe_check
    def low_envelopes_bar(cls, df: pd.DataFrame) -> bool:
        if df.empty:
            return True
        return bool((df["low"] <= df[["open", "close", "high"]].min(axis=1)).all())


class ReservedLedgerColumns(pa.DataFrameModel):
    run_id: Optional[Series[str]] = pa.Field(nullable=True)
    position_id: Optional[Series[str]] = pa.Field(nullable=True)
    parent_position_id: Optional[Series[str]] = pa.Field(nullable=True)
    lot_id: Optional[Series[str]] = pa.Field(nullable=True)
    entry_tag: Optional[Series[str]] = pa.Field(nullable=True)
    exit_tag: Optional[Series[str]] = pa.Field(nullable=True)
    risk_tag: Optional[Series[str]] = pa.Field(nullable=True)
    resolver_status: Optional[Series[str]] = pa.Field(nullable=True)
    rejection_reason: Optional[Series[str]] = pa.Field(nullable=True)
    node_version: Optional[Series[str]] = pa.Field(nullable=True)
    contract_version: Optional[Series[str]] = pa.Field(nullable=True)


class DecisionLogSchema(ReservedLedgerColumns):
    ts: Series[pd.Timestamp]
    symbol: Series[str]
    entry_action: Series[str] = pa.Field(
        isin=["none", "enter_long", "enter_short", "scale_in", "increase"]
    )
    exit_action: Series[str] = pa.Field(
        isin=["none", "exit", "partial_exit", "reduce", "set_stop"]
    )
    risk_approved: Series[bool]
    target_units: Series[float] = pa.Field(ge=0)
    resolved_action: Series[str] = pa.Field(
        isin=[
            "hold",
            "submit_entry_long",
            "submit_entry_short",
            "submit_scale_in",
            "submit_partial_exit",
            "submit_reduce",
            "submit_exit",
            "blocked_entry",
            "blocked_exit",
            "blocked_scale_in",
            "blocked_reduce",
            "blocked_stop_request",
        ]
    )
    reason: Series[str]
    metadata: Series[object]

    class Config:
        strict = True
        coerce = True


class OrderLogSchema(ReservedLedgerColumns):
    order_id: Series[str]
    ts_submitted: Series[pd.Timestamp]
    symbol: Series[str]
    side: Series[str] = pa.Field(isin=["buy", "sell"])
    qty: Series[float] = pa.Field(gt=0)
    order_type: Series[str] = pa.Field(isin=["market"])
    price_reference: Series[float] = pa.Field(gt=0)
    status: Series[str] = pa.Field(isin=["created", "filled", "cancelled", "rejected"])

    class Config:
        strict = True
        coerce = True
        unique = ["order_id"]


class FillLogSchema(ReservedLedgerColumns):
    fill_id: Series[str]
    order_id: Series[str]
    ts_fill: Series[pd.Timestamp]
    symbol: Series[str]
    side: Series[str] = pa.Field(isin=["buy", "sell"])
    qty: Series[float] = pa.Field(gt=0)
    fill_price: Series[float] = pa.Field(gt=0)
    fees: Series[float] = pa.Field(ge=0)
    slippage: Series[float] = pa.Field(ge=0)
    gross_notional: Series[float]

    class Config:
        strict = True
        coerce = True
        unique = ["fill_id"]


class TradeLedgerSchema(ReservedLedgerColumns):
    trade_id: Series[str]
    symbol: Series[str]
    side: Series[str] = pa.Field(isin=["long", "short"])
    entry_ts: Series[pd.Timestamp]
    exit_ts: Series[pd.Timestamp]
    entry_price: Series[float] = pa.Field(gt=0)
    exit_price: Series[float] = pa.Field(gt=0)
    qty: Series[float] = pa.Field(gt=0)
    gross_pnl: Series[float]
    net_pnl: Series[float]
    mfe: Series[float]
    mae: Series[float]
    exit_efficiency: Series[float]
    bars_held: Series[int] = pa.Field(ge=1)
    exit_reason: Series[str]
    fees: Series[float] = pa.Field(ge=0)

    class Config:
        strict = True
        coerce = True
        unique = ["trade_id"]

    @pa.dataframe_check
    def exit_not_before_entry(cls, df: pd.DataFrame) -> bool:
        if df.empty:
            return True
        return bool((df["exit_ts"] >= df["entry_ts"]).all())


class EquityCurveSchema(ReservedLedgerColumns):
    ts: Series[pd.Timestamp]
    cash: Series[float]
    equity: Series[float]
    realized_pnl: Series[float]
    unrealized_pnl: Series[float]
    gross_exposure: Series[float] = pa.Field(ge=0)
    net_exposure: Series[float]
    drawdown: Series[float] = pa.Field(le=0)

    class Config:
        strict = True
        coerce = True
        unique = ["ts"]

    @pa.dataframe_check
    def monotonic_ts(cls, df: pd.DataFrame) -> bool:
        return bool(df["ts"].is_monotonic_increasing) if not df.empty else True

    @pa.dataframe_check
    def gross_covers_net(cls, df: pd.DataFrame) -> bool:
        if df.empty:
            return True
        return bool((df["gross_exposure"] >= df["net_exposure"].abs()).all())

    @pa.dataframe_check
    def drawdown_matches_equity(cls, df: pd.DataFrame) -> bool:
        if df.empty:
            return True
        expected = df["equity"] / df["equity"].cummax() - 1.0
        return bool((df["drawdown"].round(12) == expected.round(12)).all())


OHLCV = OHLCVSchema
decision_log = DecisionLogSchema
order_log = OrderLogSchema
fill_log = FillLogSchema
trade_ledger = TradeLedgerSchema
equity_curve = EquityCurveSchema

__all__ = [
    "DecisionLogSchema",
    "EquityCurveSchema",
    "FillLogSchema",
    "OHLCV",
    "OHLCVSchema",
    "OrderLogSchema",
    "RESERVED_LEDGER_COLUMNS",
    "TradeLedgerSchema",
    "decision_log",
    "equity_curve",
    "fill_log",
    "order_log",
    "trade_ledger",
]
