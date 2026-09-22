"""Small backend-independent workflow kernel.

Model calls and CoTracker are intentionally not implemented here. They should
consume the context produced by this module, never the backend environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import CalibrationArtifact, Capability, MotionCommand, TaskStatus, Waypoint
from .ports import EnvironmentAdapter, ExecutionTrace, RGBObservation


class WorkflowPreflightError(RuntimeError):
    """Raised before motion when a selected environment lacks a requirement."""


@dataclass(frozen=True, slots=True)
class WorkflowContext:
    profile_id: str
    camera_id: str
    observation: RGBObservation
    _adapter: EnvironmentAdapter

    def assert_calibration_compatible(self, artifact: CalibrationArtifact) -> None:
        _require(self._adapter, (Capability.CALIBRATION_PROBES,))
        artifact.assert_compatible(self._adapter.profile, self.camera_id)

    def observe(self) -> RGBObservation:
        return self._validate_observation(self._adapter.capture_rgb(self.camera_id))

    def move(self, command: MotionCommand) -> tuple[ExecutionTrace, TaskStatus]:
        if command.frame not in self._adapter.profile.motion.supported_frames:
            raise WorkflowPreflightError(f"motion frame {command.frame.value!r} is not supported")
        _require(self._adapter, (Capability.TASK_EVALUATION,))
        trace = self._adapter.execute_motion(command)
        self._validate_observation(trace.transition.observation)
        return trace, self._adapter.task_status(trace.transition)

    def move_waypoint(self, waypoint: Waypoint) -> tuple[ExecutionTrace, TaskStatus]:
        _require(self._adapter, (Capability.WAYPOINT_EXECUTION, Capability.TASK_EVALUATION))
        spec = self._adapter.profile.waypoint(waypoint.profile_id)
        if waypoint.phase and self._adapter.profile.task_phases and waypoint.phase not in self._adapter.profile.task_phases:
            raise WorkflowPreflightError(f"phase {waypoint.phase!r} is not declared by {self._adapter.profile.id}")
        if spec.frame not in self._adapter.profile.motion.supported_frames:
            raise WorkflowPreflightError(f"waypoint frame {spec.frame.value!r} is unsupported by motion profile")
        trace = self._adapter.execute_waypoint(waypoint)
        self._validate_observation(trace.transition.observation)
        return trace, self._adapter.task_status(trace.transition)

    def close(self) -> None:
        self._adapter.close()

    def _validate_observation(self, observation: RGBObservation) -> RGBObservation:
        camera = self._adapter.profile.camera(self.camera_id)
        if observation.camera_id != camera.id or observation.camera_fingerprint != camera.fingerprint:
            raise WorkflowPreflightError("adapter returned an observation from an unregistered or changed camera")
        return observation


class WorkflowEngine:
    """Preflights a profile and exposes only canonical workflow operations."""

    def __init__(self, adapter: EnvironmentAdapter) -> None:
        self._adapter = adapter

    def prepare(
        self,
        *,
        seed: int | None,
        camera_id: str,
        required_capabilities: Iterable[Capability] = (),
    ) -> WorkflowContext:
        _require(self._adapter, tuple(required_capabilities))
        self._adapter.profile.camera(camera_id)
        initial = self._adapter.reset(seed=seed)
        context = WorkflowContext(self._adapter.profile.id, camera_id, initial, self._adapter)
        context._validate_observation(initial)
        return context

    def validate_motion(self, command: MotionCommand) -> None:
        if command.frame not in self._adapter.profile.motion.supported_frames:
            raise WorkflowPreflightError(f"motion frame {command.frame.value!r} is not supported")


def _require(adapter: EnvironmentAdapter, required: tuple[Capability, ...]) -> None:
    missing = set(required) - set(adapter.profile.capabilities)
    if missing:
        names = ", ".join(sorted(item.value for item in missing))
        raise WorkflowPreflightError(f"{adapter.profile.id} lacks required capabilities: {names}")
