"""Versioned, backend-independent types for manipulation workflows.

This module has no Gymnasium, simulator, vision-model, or NumPy dependency.
It is safe to use for config validation, offline audits, and unit tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Mapping


class Capability(str, Enum):
    """Optional operations an adapter can safely provide."""

    TCP_FEEDBACK = "tcp_feedback"
    WAYPOINT_EXECUTION = "waypoint_execution"
    CALIBRATION_PROBES = "calibration_probes"
    TASK_EVALUATION = "task_evaluation"


class CameraKind(str, Enum):
    NATIVE = "native"
    INJECTED = "injected"


class MotionFrame(str, Enum):
    ROBOT_BASE = "robot_base"
    EE_ALIGNED = "ee_aligned"


class GripperIntent(str, Enum):
    OPEN = "open"
    HOLD = "hold"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class Pose:
    """End-effector pose expressed in a declared reference frame."""

    x: float
    y: float
    z: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def __post_init__(self) -> None:
        _finite_dataclass_values(self, "pose")


@dataclass(frozen=True, slots=True)
class MotionCommand:
    """Semantic robot motion, before an adapter turns it into a native action.

    Exactly one of ``delta`` or ``target`` must be supplied. ``gripper`` uses
    semantic intent, which keeps the opposite Google/WidowX polarities out of
    policies and workflow code.
    """

    frame: MotionFrame
    gripper: GripperIntent = GripperIntent.HOLD
    delta: Pose | None = None
    target: Pose | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if (self.delta is None) == (self.target is None):
            raise ValueError("motion command requires exactly one of delta or target")
        if not isinstance(self.note, str) or len(self.note) > 160:
            raise ValueError("motion note must be a string of at most 160 characters")


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """A reproducible camera source registered by an environment profile."""

    id: str
    kind: CameraKind
    width: int
    height: int
    fingerprint: str
    description: str = ""
    supports_calibration: bool = True

    def __post_init__(self) -> None:
        if not self.id or not self.fingerprint:
            raise ValueError("camera id and fingerprint must be nonempty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")


@dataclass(frozen=True, slots=True)
class MotionSpec:
    """Native-control facts needed by an adapter, never by a policy."""

    id: str
    supported_frames: frozenset[MotionFrame]
    max_translation_m: float
    max_rotation_rad: float
    native_action_shape: tuple[int, ...]
    reference_pose_contract: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.reference_pose_contract:
            raise ValueError("motion id and reference_pose_contract must be nonempty")
        if not self.supported_frames:
            raise ValueError("motion profile must support at least one frame")
        if not _positive_finite(self.max_translation_m) or not _positive_finite(self.max_rotation_rad):
            raise ValueError("motion limits must be positive")
        if not self.native_action_shape or any(value <= 0 for value in self.native_action_shape):
            raise ValueError("native_action_shape must contain positive dimensions")


@dataclass(frozen=True, slots=True)
class WaypointSpec:
    """A named, adapter-owned expansion rule for a canonical waypoint."""

    id: str
    frame: MotionFrame
    position_tolerance_m: float
    rotation_tolerance_rad: float
    max_native_steps: int

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("waypoint id must be nonempty")
        if not _positive_finite(self.position_tolerance_m) or not _positive_finite(self.rotation_tolerance_rad):
            raise ValueError("waypoint tolerances must be positive")
        if not _positive_int(self.max_native_steps):
            raise ValueError("waypoint max_native_steps must be positive")


@dataclass(frozen=True, slots=True)
class Waypoint:
    """A policy-level target expanded by the adapter's registered executor."""

    profile_id: str
    target: Pose
    gripper: GripperIntent = GripperIntent.HOLD
    phase: str = ""

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("waypoint profile_id must be nonempty")
        if not isinstance(self.phase, str) or len(self.phase) > 64:
            raise ValueError("waypoint phase must be a string of at most 64 characters")


@dataclass(frozen=True, slots=True)
class CalibrationRequest:
    """A declared probe protocol, not a calibration result."""

    protocol_version: str
    camera_id: str
    axes: tuple[str, ...]
    displacement_m: float
    repetitions: int

    def __post_init__(self) -> None:
        if not self.protocol_version or not self.camera_id:
            raise ValueError("calibration protocol_version and camera_id must be nonempty")
        if not self.axes or any(axis not in {"x", "y", "z"} for axis in self.axes):
            raise ValueError("calibration axes must be a nonempty subset of x, y, z")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("calibration axes must be unique")
        if not _positive_finite(self.displacement_m) or not _positive_int(self.repetitions):
            raise ValueError("calibration displacement and repetitions must be positive")


@dataclass(frozen=True, slots=True)
class CalibrationSpec:
    """An adapter's registered calibration protocol and probe safety envelope."""

    id: str
    protocol_version: str
    probe_frame: MotionFrame
    supported_axes: frozenset[str]
    max_displacement_m: float
    min_repetitions: int
    artifact_schema_version: str = "calibration.v1"

    def __post_init__(self) -> None:
        if not self.id or not self.protocol_version:
            raise ValueError("calibration id and protocol_version must be nonempty")
        if not self.supported_axes or not self.supported_axes <= {"x", "y", "z"}:
            raise ValueError("supported_axes must be a nonempty subset of x, y, z")
        if not _positive_finite(self.max_displacement_m) or not _positive_int(self.min_repetitions):
            raise ValueError("calibration limits must be positive")
        if not self.artifact_schema_version:
            raise ValueError("artifact_schema_version must be nonempty")


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    """Static registration record for one robot-camera-task scenario."""

    id: str
    suite: str
    task_id: str
    robot_profile_id: str
    cameras: tuple[CameraSpec, ...]
    motion: MotionSpec
    waypoint_profiles: tuple[WaypointSpec, ...] = ()
    calibration: CalibrationSpec | None = None
    capabilities: frozenset[Capability] = frozenset()
    task_phases: frozenset[str] = frozenset()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((self.id, self.suite, self.task_id, self.robot_profile_id)):
            raise ValueError("profile identity fields must be nonempty")
        ids = [camera.id for camera in self.cameras]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("profile must contain uniquely named cameras")
        waypoint_ids = [waypoint.id for waypoint in self.waypoint_profiles]
        if len(set(waypoint_ids)) != len(waypoint_ids):
            raise ValueError("waypoint profile ids must be unique")
        if self.waypoint_profiles and Capability.WAYPOINT_EXECUTION not in self.capabilities:
            raise ValueError("waypoint profiles require waypoint_execution capability")
        if self.calibration is not None and Capability.CALIBRATION_PROBES not in self.capabilities:
            raise ValueError("a calibration profile requires calibration_probes capability")
        if Capability.CALIBRATION_PROBES in self.capabilities and self.calibration is None:
            raise ValueError("calibration_probes capability requires a calibration profile")
        if not all(phase and isinstance(phase, str) for phase in self.task_phases):
            raise ValueError("task phases must be nonempty strings")

    def camera(self, camera_id: str) -> CameraSpec:
        for camera in self.cameras:
            if camera.id == camera_id:
                return camera
        raise KeyError(f"camera {camera_id!r} is not registered for {self.id}")

    def waypoint(self, waypoint_id: str) -> WaypointSpec:
        for waypoint in self.waypoint_profiles:
            if waypoint.id == waypoint_id:
                return waypoint
        raise KeyError(f"waypoint {waypoint_id!r} is not registered for {self.id}")


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    """Compact calibration metadata plus an opaque, backend-specific payload."""

    schema_version: str
    profile_id: str
    robot_profile_id: str
    camera_id: str
    camera_fingerprint: str
    motion_profile_id: str
    reference_pose_contract: str
    protocol_version: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        fields = (
            self.schema_version, self.profile_id, self.robot_profile_id, self.camera_id,
            self.camera_fingerprint, self.motion_profile_id, self.reference_pose_contract,
            self.protocol_version,
        )
        if not all(isinstance(value, str) and value for value in fields):
            raise ValueError("calibration artifact identity fields must be nonempty strings")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    @property
    def compatibility_key(self) -> str:
        data = {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "robot_profile_id": self.robot_profile_id,
            "camera_id": self.camera_id,
            "camera_fingerprint": self.camera_fingerprint,
            "motion_profile_id": self.motion_profile_id,
            "reference_pose_contract": self.reference_pose_contract,
            "protocol_version": self.protocol_version,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def for_profile(
        cls,
        profile: EnvironmentProfile,
        request: CalibrationRequest,
        payload: Mapping[str, Any],
        *,
        schema_version: str | None = None,
    ) -> "CalibrationArtifact":
        camera = profile.camera(request.camera_id)
        calibration = profile.calibration
        if calibration is None:
            raise ValueError(f"profile {profile.id!r} does not register a calibration protocol")
        if not camera.supports_calibration:
            raise ValueError(f"camera {camera.id!r} does not support calibration")
        if request.protocol_version != calibration.protocol_version:
            raise ValueError("calibration request protocol does not match registered profile")
        if not set(request.axes) <= calibration.supported_axes:
            raise ValueError("calibration request contains unsupported probe axes")
        if request.displacement_m > calibration.max_displacement_m:
            raise ValueError("calibration request exceeds registered displacement limit")
        if request.repetitions < calibration.min_repetitions:
            raise ValueError("calibration request has too few repetitions")
        if schema_version is not None and schema_version != calibration.artifact_schema_version:
            raise ValueError("calibration artifact schema does not match registered profile")
        return cls(
            schema_version=calibration.artifact_schema_version if schema_version is None else schema_version,
            profile_id=profile.id,
            robot_profile_id=profile.robot_profile_id,
            camera_id=camera.id,
            camera_fingerprint=camera.fingerprint,
            motion_profile_id=profile.motion.id,
            reference_pose_contract=profile.motion.reference_pose_contract,
            protocol_version=request.protocol_version,
            payload=dict(payload),
        )

    def assert_compatible(self, profile: EnvironmentProfile, camera_id: str) -> None:
        camera = profile.camera(camera_id)
        expected = {
            "profile_id": profile.id,
            "robot_profile_id": profile.robot_profile_id,
            "camera_id": camera.id,
            "camera_fingerprint": camera.fingerprint,
            "motion_profile_id": profile.motion.id,
            "reference_pose_contract": profile.motion.reference_pose_contract,
        }
        if profile.calibration is not None:
            expected["protocol_version"] = profile.calibration.protocol_version
            expected["schema_version"] = profile.calibration.artifact_schema_version
        actual = {key: getattr(self, key) for key in expected}
        mismatches = {key: (actual[key], expected[key]) for key in expected if actual[key] != expected[key]}
        if mismatches:
            details = ", ".join(f"{key}: artifact={old!r}, runtime={new!r}" for key, (old, new) in mismatches.items())
            raise ValueError(f"incompatible calibration artifact ({details})")


@dataclass(frozen=True, slots=True)
class TaskStatus:
    """Post-action evaluation result; diagnostics are never policy input."""

    success: bool
    terminated: bool = False
    truncated: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def _finite_dataclass_values(value: Any, label: str) -> None:
    for name, number in asdict(value).items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f"{label}.{name} must be finite")


def _positive_finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0
