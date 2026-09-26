"""Small backend-independent workflow kernel.

Model calls and CoTracker are intentionally not implemented here. They should
consume the context produced by this module, never the backend environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .contracts import CalibrationArtifact, Capability, MotionCommand, TaskStatus, Waypoint
from .ports import EnvironmentAdapter, ExecutionTrace, MultiViewCaptureAdapter, RGBObservation


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

    def observe_views(self, camera_ids: tuple[str, ...]) -> Mapping[str, RGBObservation]:
        """Capture arbitrary registered cameras at the same robot state."""
        if len(camera_ids) < 2 or len(set(camera_ids)) != len(camera_ids):
            raise WorkflowPreflightError("multi-view capture requires distinct camera ids")
        if self.camera_id not in camera_ids:
            raise WorkflowPreflightError("the prepared primary camera must be in the view set")
        _require(self._adapter, (Capability.MULTI_VIEW_CAPTURE,))
        for camera_id in camera_ids:
            self._adapter.profile.camera(camera_id)
        if not isinstance(self._adapter, MultiViewCaptureAdapter):
            raise WorkflowPreflightError("adapter declares multi-view capture but has no capture_views port")
        frames = self._adapter.capture_views(camera_ids)
        if not isinstance(frames, Mapping):
            raise WorkflowPreflightError("adapter capture_views must return a camera-id mapping")
        if set(frames) != set(camera_ids):
            raise WorkflowPreflightError("adapter returned a different set of camera views")
        state_ids = set()
        for camera_id in camera_ids:
            frame = frames[camera_id]
            camera = self._adapter.profile.camera(camera_id)
            if frame.camera_id != camera.id or frame.camera_fingerprint != camera.fingerprint:
                raise WorkflowPreflightError(f"invalid frame identity for {camera_id}")
            if isinstance(frame.state_id, bool) or not isinstance(frame.state_id, (str, int)):
                raise WorkflowPreflightError("multi-view frame lacks a valid environment state_id")
            state_ids.add(frame.state_id)
        if len(state_ids) != 1:
            raise WorkflowPreflightError("camera frames were captured from different environment states")
        return {camera_id: frames[camera_id] for camera_id in camera_ids}

    def assert_camera_calibration_compatible(self, artifact: CalibrationArtifact, camera_id: str) -> None:
        _require(self._adapter, (Capability.CALIBRATION_PROBES,))
        artifact.assert_compatible(self._adapter.profile, camera_id)

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
