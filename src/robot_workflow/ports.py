"""Ports implemented by simulator- or hardware-specific adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .contracts import EnvironmentProfile, MotionCommand, Pose, TaskStatus, Waypoint


@dataclass(frozen=True, slots=True)
class RGBObservation:
    """One named camera frame with enough provenance for downstream audits.

    ``rgb`` intentionally has no concrete array type. Backends may return a
    NumPy array, a tensor, or a byte buffer, while the workflow only forwards it
    to the configured visual component.
    """

    camera_id: str
    camera_fingerprint: str
    rgb: Any
    timestamp_ns: int | None = None


@dataclass(frozen=True, slots=True)
class Transition:
    """Result of one or more native environment steps."""

    observation: RGBObservation
    reward: float | None
    terminated: bool
    truncated: bool
    native_steps: int
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.native_steps < 1:
            raise ValueError("a transition must contain at least one native step")


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    """Auditable record of an adapter-owned command expansion."""

    command_kind: str
    native_actions: tuple[Any, ...]
    transition: Transition
    tcp_before: Pose | None = None
    tcp_after: Pose | None = None
    stop_reason: str = ""


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """The only runtime boundary visible to the generic workflow.

    Implementations may wrap Gymnasium, a simulator SDK, or a physical robot.
    They own native action encoding and may access backend internals; callers
    above this boundary may not.
    """

    @property
    def profile(self) -> EnvironmentProfile: ...

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None) -> RGBObservation: ...

    def capture_rgb(self, camera_id: str) -> RGBObservation: ...

    def current_tcp_pose(self) -> Pose: ...

    def execute_motion(self, command: MotionCommand) -> ExecutionTrace: ...

    def execute_waypoint(self, waypoint: Waypoint) -> ExecutionTrace: ...

    def task_status(self, transition: Transition) -> TaskStatus: ...

    def close(self) -> None: ...
