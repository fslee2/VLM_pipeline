"""Small Bridge-specific adapters for replaceable VLM clients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Protocol

from robot_workflow.contracts import Waypoint
from robot_workflow.pipeline import MultiViewVisualIntent, ScreenDirection
from robot_workflow.ports import RGBObservation

from .adapter import BridgeSimplerAdapter


@dataclass(frozen=True, slots=True)
class BridgeDeltaAction:
    """Simpler Bridge native-action semantics before waypoint anchoring."""

    dx: float
    dy: float
    dz: float
    rx: float
    ry: float
    rz: float
    gripper: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in self.as_tuple()):
            raise ValueError("Bridge action values must be finite")

    def as_tuple(self) -> tuple[float, ...]:
        return (self.dx, self.dy, self.dz, self.rx, self.ry, self.rz, self.gripper)


class BridgeIntentClient(Protocol):
    def infer_per_view(self, views: Mapping[str, RGBObservation], instruction: str) -> Mapping[str, Any]: ...


class BridgeActionClient(Protocol):
    def predict_delta(self, views: Mapping[str, RGBObservation], instruction: str,
                      intent: MultiViewVisualIntent, prior: Any) -> Sequence[float] | BridgeDeltaAction: ...


class BridgeVisualIntentProvider:
    """Require visual (per-camera pixel-plane) outputs, not robot-axis signs.

    ``infer_per_view`` may return ``{"primary": {"horizontal": ..., ...}}``
    or already-typed ``ScreenDirection`` values. CoTracker direction vectors
    are deliberately not sent through this provider; the compiler consumes
    them deterministically after visual intent inference.
    """

    def __init__(self, client: BridgeIntentClient) -> None:
        self.client = client

    def infer(self, views: Mapping[str, RGBObservation], instruction: str) -> MultiViewVisualIntent:
        result = self.client.infer_per_view(views, instruction)
        raw_directions = result.get("directions") if isinstance(result, Mapping) else None
        if not isinstance(raw_directions, Mapping):
            raise ValueError("Bridge visual-intent response must contain per-view directions")
        directions = {}
        for camera_id, value in raw_directions.items():
            if isinstance(value, ScreenDirection):
                directions[camera_id] = value
            elif isinstance(value, Mapping):
                directions[camera_id] = ScreenDirection(
                    float(value["horizontal"]), float(value["vertical"]),
                )
            else:
                raise ValueError(f"invalid screen direction for {camera_id}")
        return MultiViewVisualIntent(
            directions=directions,
            phase=str(result.get("phase", "")),
            holding=str(result.get("holding", "")),
            evidence=result.get("evidence", {}),
        )


class BridgeWaypointPolicy:
    """Wrap the Bridge policy's local 7-D action as a canonical waypoint."""

    def __init__(self, client: BridgeActionClient, adapter: BridgeSimplerAdapter) -> None:
        self.client = client
        self.adapter = adapter

    def plan(self, views: Mapping[str, RGBObservation], instruction: str,
             intent: MultiViewVisualIntent, prior: Any) -> Waypoint:
        delta = self.client.predict_delta(views, instruction, intent, prior)
        if isinstance(delta, BridgeDeltaAction):
            delta = delta.as_tuple()
        return self.adapter.waypoint_from_delta(delta, phase=intent.phase)
