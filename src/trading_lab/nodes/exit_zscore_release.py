from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from trading_lab.contracts import (
    ActionRequest,
    Bar,
    BarHistorySeries,
    BarHistoryWindow,
    DecisionContext,
    NodeContract,
    NodeSpec,
)
from trading_lab.nodes.exit_shared import (
    action_request_exit,
    baseline_series,
    diff_series,
    position_direction,
    scale_series,
    zscore_series,
)
from trading_lab.registry import register_exit

DEFAULT_EXIT_ZSCORE_RELEASE_NAME = "exit_zscore_release"
_ALLOWED_BASELINES = {"ema", "sma", "median"}
_ALLOWED_SCALES = {"atr", "residual_std", "residual_mad"}
_ALLOWED_RELEASE_KINDS = {"threshold_cross", "gradient_flip", "acceleration_flip"}


def build_exit_zscore_release_contract(
    *,
    name: str = DEFAULT_EXIT_ZSCORE_RELEASE_NAME,
    baseline_kind: str = "ema",
    baseline_lookback: int = 50,
    scale_kind: str = "atr",
    scale_lookback: int = 20,
    release_kind: str = "threshold_cross",
    z_exit_threshold: float = 0.0,
    gradient_horizon: int = 3,
    gradient_threshold: float = 0.0,
    acceleration_horizon: int = 1,
    acceleration_threshold: float = 0.0,
    confirm_bars: int = 1,
) -> NodeContract:
    return NodeContract(
        spec=NodeSpec(
            name=name,
            kind="exit",
            version="1.0.0",
            required_capabilities=("single_position_per_symbol",),
            emitted_action_types=("close", "hold"),
            required_history=_required_history(
                baseline_lookback=baseline_lookback,
                scale_kind=scale_kind,
                scale_lookback=scale_lookback,
                gradient_horizon=gradient_horizon,
                acceleration_horizon=acceleration_horizon,
                confirm_bars=confirm_bars,
            ),
            description="Z-score release exit using completed bars only.",
        ),
        manifest={
            "family": "zscore_release",
            "module": __name__,
            "class": "ZScoreReleaseExitNode",
            "uses_completed_bars_only": True,
            "parameters": {
                "baseline_kind": baseline_kind,
                "baseline_lookback": baseline_lookback,
                "scale_kind": scale_kind,
                "scale_lookback": scale_lookback,
                "release_kind": release_kind,
                "z_exit_threshold": z_exit_threshold,
                "gradient_horizon": gradient_horizon,
                "gradient_threshold": gradient_threshold,
                "acceleration_horizon": acceleration_horizon,
                "acceleration_threshold": acceleration_threshold,
                "confirm_bars": confirm_bars,
            },
        },
    )


@dataclass(frozen=True, slots=True)
class ZScoreReleaseExitNode:
    baseline_kind: str = "ema"
    baseline_lookback: int = 50
    scale_kind: str = "atr"
    scale_lookback: int = 20
    release_kind: str = "threshold_cross"
    z_exit_threshold: float = 0.0
    gradient_horizon: int = 3
    gradient_threshold: float = 0.0
    acceleration_horizon: int = 1
    acceleration_threshold: float = 0.0
    confirm_bars: int = 1

    def __post_init__(self) -> None:
        if self.baseline_kind not in _ALLOWED_BASELINES:
            raise ValueError(f"baseline_kind must be one of {_ALLOWED_BASELINES}.")
        if self.scale_kind not in _ALLOWED_SCALES:
            raise ValueError(f"scale_kind must be one of {_ALLOWED_SCALES}.")
        if self.release_kind not in _ALLOWED_RELEASE_KINDS:
            raise ValueError(f"release_kind must be one of {_ALLOWED_RELEASE_KINDS}.")
        if self.baseline_lookback <= 0:
            raise ValueError("baseline_lookback must be positive.")
        if self.scale_lookback <= 0:
            raise ValueError("scale_lookback must be positive.")
        if self.gradient_horizon <= 0:
            raise ValueError("gradient_horizon must be positive.")
        if self.acceleration_horizon <= 0:
            raise ValueError("acceleration_horizon must be positive.")
        if self.confirm_bars <= 0:
            raise ValueError("confirm_bars must be positive.")
        if self.z_exit_threshold < 0.0:
            raise ValueError("z_exit_threshold must be non-negative.")
        if self.gradient_threshold < 0.0:
            raise ValueError("gradient_threshold must be non-negative.")
        if self.acceleration_threshold < 0.0:
            raise ValueError("acceleration_threshold must be non-negative.")

    def __call__(self, ctx: DecisionContext) -> ActionRequest:
        direction = position_direction(ctx)
        if direction is None:
            return ActionRequest()

        if isinstance(ctx.history, BarHistoryWindow):
            long_plan, short_plan = _cached_release_plans(
                ctx.history.series,
                self.baseline_kind,
                self.baseline_lookback,
                self.scale_kind,
                self.scale_lookback,
                self.release_kind,
                self.z_exit_threshold,
                self.gradient_horizon,
                self.gradient_threshold,
                self.acceleration_horizon,
                self.acceleration_threshold,
                self.confirm_bars,
            )
            plan = long_plan if direction == "long" else short_plan
            return plan[ctx.history.end_index]

        long_plan, short_plan = _compute_release_plans(
            tuple(ctx.history),
            baseline_kind=self.baseline_kind,
            baseline_lookback=self.baseline_lookback,
            scale_kind=self.scale_kind,
            scale_lookback=self.scale_lookback,
            release_kind=self.release_kind,
            z_exit_threshold=self.z_exit_threshold,
            gradient_horizon=self.gradient_horizon,
            gradient_threshold=self.gradient_threshold,
            acceleration_horizon=self.acceleration_horizon,
            acceleration_threshold=self.acceleration_threshold,
            confirm_bars=self.confirm_bars,
        )
        plan = long_plan if direction == "long" else short_plan
        return plan[-1] if plan else ActionRequest()


@lru_cache(maxsize=48)
def _cached_release_plans(
    series: BarHistorySeries,
    baseline_kind: str,
    baseline_lookback: int,
    scale_kind: str,
    scale_lookback: int,
    release_kind: str,
    z_exit_threshold: float,
    gradient_horizon: int,
    gradient_threshold: float,
    acceleration_horizon: int,
    acceleration_threshold: float,
    confirm_bars: int,
) -> tuple[tuple[ActionRequest, ...], tuple[ActionRequest, ...]]:
    return _compute_release_plans(
        series.bars,
        baseline_kind=baseline_kind,
        baseline_lookback=baseline_lookback,
        scale_kind=scale_kind,
        scale_lookback=scale_lookback,
        release_kind=release_kind,
        z_exit_threshold=z_exit_threshold,
        gradient_horizon=gradient_horizon,
        gradient_threshold=gradient_threshold,
        acceleration_horizon=acceleration_horizon,
        acceleration_threshold=acceleration_threshold,
        confirm_bars=confirm_bars,
    )


def _compute_release_plans(
    bars: Sequence[Bar],
    *,
    baseline_kind: str,
    baseline_lookback: int,
    scale_kind: str,
    scale_lookback: int,
    release_kind: str,
    z_exit_threshold: float,
    gradient_horizon: int,
    gradient_threshold: float,
    acceleration_horizon: int,
    acceleration_threshold: float,
    confirm_bars: int,
) -> tuple[tuple[ActionRequest, ...], tuple[ActionRequest, ...]]:
    if not bars:
        return (), ()

    closes = tuple(float(bar.close) for bar in bars)
    highs = tuple(float(bar.high) for bar in bars)
    lows = tuple(float(bar.low) for bar in bars)
    baseline = baseline_series(closes, baseline_kind, baseline_lookback)
    scale = scale_series(
        closes=closes,
        highs=highs,
        lows=lows,
        baseline=baseline,
        scale_kind=scale_kind,
        scale_lookback=scale_lookback,
    )
    z_series = zscore_series(closes, baseline, scale)
    gradient_series = diff_series(z_series, gradient_horizon, divide_by_horizon=True)
    acceleration_series = diff_series(
        gradient_series,
        acceleration_horizon,
        divide_by_horizon=False,
    )

    long_actions: list[ActionRequest] = []
    short_actions: list[ActionRequest] = []
    for index in range(len(closes)):
        long_actions.append(
            _release_action_at(
                "long",
                index=index,
                release_kind=release_kind,
                z_exit_threshold=z_exit_threshold,
                gradient_threshold=gradient_threshold,
                acceleration_threshold=acceleration_threshold,
                confirm_bars=confirm_bars,
                z_series=z_series,
                gradient_series=gradient_series,
                acceleration_series=acceleration_series,
            )
        )
        short_actions.append(
            _release_action_at(
                "short",
                index=index,
                release_kind=release_kind,
                z_exit_threshold=z_exit_threshold,
                gradient_threshold=gradient_threshold,
                acceleration_threshold=acceleration_threshold,
                confirm_bars=confirm_bars,
                z_series=z_series,
                gradient_series=gradient_series,
                acceleration_series=acceleration_series,
            )
        )
    return tuple(long_actions), tuple(short_actions)


def _release_action_at(
    direction: str,
    *,
    index: int,
    release_kind: str,
    z_exit_threshold: float,
    gradient_threshold: float,
    acceleration_threshold: float,
    confirm_bars: int,
    z_series: Sequence[float | None],
    gradient_series: Sequence[float | None],
    acceleration_series: Sequence[float | None],
) -> ActionRequest:
    if not _should_exit_at(
        direction,
        index=index,
        release_kind=release_kind,
        z_exit_threshold=z_exit_threshold,
        gradient_threshold=gradient_threshold,
        acceleration_threshold=acceleration_threshold,
        confirm_bars=confirm_bars,
        z_series=z_series,
        gradient_series=gradient_series,
        acceleration_series=acceleration_series,
    ):
        return ActionRequest()

    return action_request_exit(
        direction,
        (
            f"zscore_release release_kind={release_kind} "
            f"z={z_series[index]} dz={gradient_series[index]} ddz={acceleration_series[index]}"
        ),
    )


def _should_exit_at(
    direction: str,
    *,
    index: int,
    release_kind: str,
    z_exit_threshold: float,
    gradient_threshold: float,
    acceleration_threshold: float,
    confirm_bars: int,
    z_series: Sequence[float | None],
    gradient_series: Sequence[float | None],
    acceleration_series: Sequence[float | None],
) -> bool:
    start = index - confirm_bars + 1
    if start < 0:
        return False

    if release_kind == "threshold_cross":
        for offset in range(start, index + 1):
            value = z_series[offset]
            if direction == "long":
                if value is None or value > z_exit_threshold:
                    return False
            elif value is None or value < -z_exit_threshold:
                return False
        return True

    if release_kind == "gradient_flip":
        for offset in range(start, index + 1):
            value = gradient_series[offset]
            if direction == "long":
                if value is None or value > -gradient_threshold:
                    return False
            elif value is None or value < gradient_threshold:
                return False
        return True

    for offset in range(start, index + 1):
        value = acceleration_series[offset]
        if direction == "long":
            if value is None or value > -acceleration_threshold:
                return False
        elif value is None or value < acceleration_threshold:
            return False
    return True



def _required_history(
    *,
    baseline_lookback: int,
    scale_kind: str,
    scale_lookback: int,
    gradient_horizon: int,
    acceleration_horizon: int,
    confirm_bars: int,
) -> int:
    baseline_history = baseline_lookback
    scale_history = scale_lookback + 1 if scale_kind == "atr" else baseline_history + scale_lookback - 1
    derivative_history = gradient_horizon + acceleration_horizon
    return max(baseline_history, scale_history) + derivative_history + confirm_bars - 1


DEFAULT_EXIT_ZSCORE_RELEASE = ZScoreReleaseExitNode()
DEFAULT_EXIT_ZSCORE_RELEASE_CONTRACT = build_exit_zscore_release_contract()

register_exit(
    DEFAULT_EXIT_ZSCORE_RELEASE_NAME,
    DEFAULT_EXIT_ZSCORE_RELEASE,
    DEFAULT_EXIT_ZSCORE_RELEASE_CONTRACT,
)
