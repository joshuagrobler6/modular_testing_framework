from __future__ import annotations

from dataclasses import dataclass

from trading_lab.contracts import ActionRequest, DecisionContext, NodeContract, NodeSpec
from trading_lab.nodes.exit_shared import (
    action_request_exit,
    baseline_series,
    closes_from_ctx,
    diff_series,
    highs_from_ctx,
    lows_from_ctx,
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

        closes = closes_from_ctx(ctx)
        highs = highs_from_ctx(ctx)
        lows = lows_from_ctx(ctx)
        baseline = baseline_series(closes, self.baseline_kind, self.baseline_lookback)
        scale = scale_series(
            closes=closes,
            highs=highs,
            lows=lows,
            baseline=baseline,
            scale_kind=self.scale_kind,
            scale_lookback=self.scale_lookback,
        )
        z_series = zscore_series(closes, baseline, scale)
        gradient_series = diff_series(z_series, self.gradient_horizon, divide_by_horizon=True)
        acceleration_series = diff_series(
            gradient_series,
            self.acceleration_horizon,
            divide_by_horizon=False,
        )
        if not self._should_exit(direction, z_series, gradient_series, acceleration_series):
            return ActionRequest()

        return action_request_exit(
            direction,
            (
                f"zscore_release release_kind={self.release_kind} "
                f"z={z_series[-1]} dz={gradient_series[-1]} ddz={acceleration_series[-1]}"
            ),
        )

    def _should_exit(
        self,
        direction: str,
        z_series: list[float | None],
        gradient_series: list[float | None],
        acceleration_series: list[float | None],
    ) -> bool:
        if self.release_kind == "threshold_cross":
            recent = z_series[-self.confirm_bars:]
            if direction == "long":
                return all(value is not None and value <= self.z_exit_threshold for value in recent)
            return all(value is not None and value >= -self.z_exit_threshold for value in recent)

        if self.release_kind == "gradient_flip":
            recent = gradient_series[-self.confirm_bars:]
            if direction == "long":
                return all(value is not None and value <= -self.gradient_threshold for value in recent)
            return all(value is not None and value >= self.gradient_threshold for value in recent)

        recent = acceleration_series[-self.confirm_bars:]
        if direction == "long":
            return all(value is not None and value <= -self.acceleration_threshold for value in recent)
        return all(value is not None and value >= self.acceleration_threshold for value in recent)



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
