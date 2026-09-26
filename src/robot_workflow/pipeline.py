"""Reusable multi-view visual-intent -> CoTracker prior -> waypoint workflow.

The environment adapter owns capture and execution. A CoTracker backend owns
the one-time probe/tracking job. Neither the VLM nor the policy receives raw
probe tracks or simulator state: only the compiler reads calibrated directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Mapping, Protocol

from .contracts import CalibrationArtifact, CalibrationRequest, EnvironmentProfile, TaskStatus, Waypoint
from .ports import ExecutionTrace, RGBObservation
from .workflow import WorkflowContext, WorkflowPreflightError


AXES = ("x", "y", "z")


@dataclass(frozen=True, slots=True)
class ScreenDirection:
    """Task-desired movement in one image, without a robot-axis label.

    Components follow image coordinates: right/down are positive. The VLM may
    emit coarse -1/0/+1 labels, or a vision component may provide richer
    continuous visual deltas. Coarse labels can discard mixed-axis detail.
    """

    horizontal: float
    vertical: float

    def __post_init__(self) -> None:
        for name in ("horizontal", "vertical"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"screen {name} must be a finite number")

    @classmethod
    def from_labels(cls, horizontal: str, vertical: str) -> "ScreenDirection":
        try:
            return cls({"left": -1, "hold": 0, "right": 1}[horizontal],
                       {"up": -1, "hold": 0, "down": 1}[vertical])
        except KeyError as exc:
            raise ValueError(f"invalid screen direction label: {exc.args[0]!r}") from exc


@dataclass(frozen=True, slots=True)
class MultiViewVisualIntent:
    """VLM output in camera/image semantics; no CoTracker or robot signs."""

    directions: Mapping[str, ScreenDirection]
    phase: str = ""
    holding: str = ""
    evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.directions or any(not key or not isinstance(value, ScreenDirection)
                                      for key, value in self.directions.items()):
            raise ValueError("visual intent requires named screen directions")
        if any(key not in self.directions or not isinstance(value, str)
               for key, value in self.evidence.items()):
            raise ValueError("intent evidence must be text for a named camera")
        object.__setattr__(self, "directions", MappingProxyType(dict(self.directions)))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True, slots=True)
class CompiledDirectionPrior:
    """Only compact robot-base signs and audit metadata reach the policy."""

    signs: Mapping[str, int]
    relative_screen_residual: float
    calibration_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if any(axis not in AXES or type(sign) is not int or sign not in (-1, 0, 1)
               for axis, sign in self.signs.items()):
            raise ValueError("compiled signs must be robot-base x/y/z tri-states")
        if not math.isfinite(self.relative_screen_residual):
            raise ValueError("compiler residual must be finite")
        object.__setattr__(self, "signs", MappingProxyType(dict(self.signs)))
        object.__setattr__(self, "calibration_sha256", MappingProxyType(dict(self.calibration_sha256)))


def _positive_directions(artifact: CalibrationArtifact, axes: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    payload = artifact.payload
    if payload.get("estimator") != "CoTracker3":
        raise ValueError(f"{artifact.camera_id}: calibration estimator must be CoTracker3")
    vectors = payload.get("positive_robot_axis_image_directions")
    if not isinstance(vectors, Mapping):
        raise ValueError(f"{artifact.camera_id}: missing positive robot-axis image directions")
    result = {}
    for axis in axes:
        value = vectors.get(axis)
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"{artifact.camera_id}: +{axis} must be a 2-D vector")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
               for x in value):
            raise ValueError(f"{artifact.camera_id}: +{axis} contains invalid numbers")
        norm = math.hypot(*value)
        if norm <= 1e-9:
            raise ValueError(f"{artifact.camera_id}: +{axis} is degenerate")
        result[axis] = (float(value[0]) / norm, float(value[1]) / norm)
    return result


def _artifact_sha256(artifact: CalibrationArtifact) -> str:
    content = {"compatibility_key": artifact.compatibility_key, "payload": dict(artifact.payload)}
    try:
        serialized = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("CoTracker artifact payload must be finite JSON data") from exc
    return sha256(serialized.encode("utf-8")).hexdigest()


def build_cotracker_direction_artifact(
    profile: EnvironmentProfile, request: CalibrationRequest,
    positive_directions: Mapping[str, tuple[float, float] | list[float]],
    *, source_sha256: str, quality: Mapping[str, object] | None = None,
) -> CalibrationArtifact:
    """Wrap one camera's measured CoTracker result in a portable artifact.

    The CoTracker backend must measure the vectors at the registered camera
    and reference pose. This function binds that result to the profile and
    validates all requested axes; it never calls CoTracker or a simulator.
    """
    if len(source_sha256) != 64 or any(char not in "0123456789abcdef" for char in source_sha256):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    if set(positive_directions) != set(request.axes):
        raise ValueError("CoTracker result must contain exactly the requested axes")
    payload = {
        "estimator": "CoTracker3",
        "positive_robot_axis_image_directions": {
            axis: list(positive_directions[axis]) for axis in request.axes
        },
        "source_sha256": source_sha256,
        "quality": dict(quality or {}),
    }
    artifact = CalibrationArtifact.for_profile(profile, request, payload)
    normalized = _positive_directions(artifact, request.axes)
    payload["positive_robot_axis_image_directions"] = {
        axis: list(normalized[axis]) for axis in request.axes
    }
    artifact = CalibrationArtifact.for_profile(profile, request, payload)
    _artifact_sha256(artifact)
    return artifact


def _solve_normal_equations(rows: list[list[float]], desired: list[float], dimension: int) -> list[float]:
    """Small, dependency-free least-squares solve with rank rejection."""
    gram = [[sum(row[i] * row[j] for row in rows) for j in range(dimension)]
            for i in range(dimension)]
    rhs = [sum(row[i] * target for row, target in zip(rows, desired))
           for i in range(dimension)]
    matrix = [gram[i] + [rhs[i]] for i in range(dimension)]
    scale = max(max(abs(value) for value in row[:dimension]) for row in matrix)
    for col in range(dimension):
        pivot = max(range(col, dimension), key=lambda index: abs(matrix[index][col]))
        if abs(matrix[pivot][col]) <= 1e-8 * scale:
            raise WorkflowPreflightError("CoTracker view geometry cannot determine the requested robot axes")
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        factor = matrix[col][col]
        matrix[col] = [value / factor for value in matrix[col]]
        for index in range(dimension):
            if index == col:
                continue
            multiple = matrix[index][col]
            matrix[index] = [value - multiple * base for value, base in zip(matrix[index], matrix[col])]
    return [row[-1] for row in matrix]


class CoTrackerDirectionCompiler:
    """Compile N camera-space directions to robot-base translation signs.

    Calibration is a one-time, camera-bound artifact. The VLM supplies only
    desired visual directions. Unit vectors give coarse signs, not metres or
    a rotation prior. Rotation requires a separate multi-point-field contract.
    """

    def __init__(self, axes: tuple[str, ...] = ("x", "y", "z"), *,
                 deadband_ratio: float = 0.15, max_relative_residual: float = 0.75) -> None:
        if not axes or len(set(axes)) != len(axes) or any(axis not in AXES for axis in axes):
            raise ValueError("compiler axes must be a unique subset of x/y/z")
        if not 0 <= deadband_ratio < 1 or not 0 < max_relative_residual <= 1:
            raise ValueError("invalid compiler thresholds")
        self.axes = axes
        self.deadband_ratio = deadband_ratio
        self.max_relative_residual = max_relative_residual

    def compile(self, intent: MultiViewVisualIntent,
                artifacts: Mapping[str, CalibrationArtifact]) -> CompiledDirectionPrior:
        if set(intent.directions) != set(artifacts):
            raise WorkflowPreflightError("intent views and CoTracker calibration views differ")
        if len(intent.directions) * 2 < len(self.axes):
            raise WorkflowPreflightError("not enough camera-space dimensions for requested axes")
        rows: list[list[float]] = []
        desired: list[float] = []
        ids: dict[str, str] = {}
        for camera_id in intent.directions:
            artifact = artifacts[camera_id]
            if artifact.camera_id != camera_id:
                raise WorkflowPreflightError("CoTracker artifact is registered under the wrong camera")
            vectors = _positive_directions(artifact, self.axes)
            visual = intent.directions[camera_id]
            rows.extend([[vectors[axis][0] for axis in self.axes],
                         [vectors[axis][1] for axis in self.axes]])
            desired.extend((float(visual.horizontal), float(visual.vertical)))
            ids[camera_id] = _artifact_sha256(artifact)
        coefficients = _solve_normal_equations(rows, desired, len(self.axes))
        residual = math.sqrt(sum((sum(x * c for x, c in zip(row, coefficients)) - target) ** 2
                                 for row, target in zip(rows, desired)))
        desired_norm = math.sqrt(sum(value * value for value in desired))
        relative = residual / desired_norm if desired_norm else 0.0
        if relative > self.max_relative_residual:
            raise WorkflowPreflightError(f"camera-space intents conflict (relative residual {relative:.3f})")
        largest = max((abs(value) for value in coefficients), default=0.0)
        signs = {axis: (0 if largest == 0 or abs(value) <= self.deadband_ratio * largest
                        else 1 if value > 0 else -1)
                 for axis, value in zip(self.axes, coefficients)}
        return CompiledDirectionPrior(signs, relative, ids)


class VisualIntentProvider(Protocol):
    def infer(self, views: Mapping[str, RGBObservation], instruction: str) -> MultiViewVisualIntent: ...


class CoTrackerCalibrationProvider(Protocol):
    """Load saved artifacts or run setup probes once for a camera set."""

    def load_or_calibrate(self, context: WorkflowContext, camera_ids: tuple[str, ...],
                          axes: tuple[str, ...]) -> Mapping[str, CalibrationArtifact]: ...


class WaypointPolicy(Protocol):
    def plan(self, views: Mapping[str, RGBObservation], instruction: str,
             intent: MultiViewVisualIntent, prior: CompiledDirectionPrior) -> Waypoint: ...


@dataclass(frozen=True, slots=True)
class PipelineStepResult:
    intent: MultiViewVisualIntent
    prior: CompiledDirectionPrior
    waypoint: Waypoint
    trace: ExecutionTrace
    status: TaskStatus
    state_id: str | int
    may_infer_again: bool


class MultiViewCoTrackerPipeline:
    """The same workflow for any registered camera set and environment adapter."""

    @classmethod
    def prepare(cls, context: WorkflowContext, *, camera_ids: tuple[str, ...],
                calibration_provider: CoTrackerCalibrationProvider,
                intent_provider: VisualIntentProvider, policy: WaypointPolicy,
                policy_camera_ids: tuple[str, ...] | None = None,
                compiler: CoTrackerDirectionCompiler | None = None) -> "MultiViewCoTrackerPipeline":
        """Resolve calibration once; subsequent steps reuse the same artifacts."""
        selected_compiler = compiler or CoTrackerDirectionCompiler()
        artifacts = calibration_provider.load_or_calibrate(context, camera_ids, selected_compiler.axes)
        return cls(context, camera_ids=camera_ids, artifacts=artifacts,
                   intent_provider=intent_provider, policy=policy,
                   policy_camera_ids=policy_camera_ids, compiler=selected_compiler)

    def __init__(self, context: WorkflowContext, *, camera_ids: tuple[str, ...],
                 artifacts: Mapping[str, CalibrationArtifact], intent_provider: VisualIntentProvider,
                 policy: WaypointPolicy, policy_camera_ids: tuple[str, ...] | None = None,
                 compiler: CoTrackerDirectionCompiler | None = None) -> None:
        if len(camera_ids) < 2 or len(set(camera_ids)) != len(camera_ids) or context.camera_id != camera_ids[0]:
            raise WorkflowPreflightError("pipeline requires distinct registered cameras, primary first")
        if set(artifacts) != set(camera_ids):
            raise WorkflowPreflightError("one CoTracker artifact is required per registered view")
        self.policy_camera_ids = policy_camera_ids or (camera_ids[0],)
        if not self.policy_camera_ids or not set(self.policy_camera_ids) <= set(camera_ids):
            raise WorkflowPreflightError("policy cameras must be selected from the registered views")
        self.compiler = compiler or CoTrackerDirectionCompiler()
        for camera_id in camera_ids:
            context.assert_camera_calibration_compatible(artifacts[camera_id], camera_id)
            _positive_directions(artifacts[camera_id], self.compiler.axes)
            _artifact_sha256(artifacts[camera_id])
        self.context = context
        self.camera_ids = camera_ids
        self.artifacts = dict(artifacts)
        self.intent_provider = intent_provider
        self.policy = policy
        self.compiler.compile(
            MultiViewVisualIntent({camera_id: ScreenDirection(0, 0) for camera_id in camera_ids}),
            self.artifacts,
        )
        self._stopped = False

    def step(self, instruction: str) -> PipelineStepResult:
        if self._stopped:
            raise WorkflowPreflightError("the previous waypoint failed or the episode ended")
        views = self.context.observe_views(self.camera_ids)
        intent = self.intent_provider.infer(views, instruction)
        if set(intent.directions) != set(self.camera_ids):
            raise WorkflowPreflightError("visual intent omitted or invented a camera view")
        prior = self.compiler.compile(intent, self.artifacts)
        policy_views = {camera_id: views[camera_id] for camera_id in self.policy_camera_ids}
        waypoint = self.policy.plan(policy_views, instruction, intent, prior)
        trace, status = self.context.move_waypoint(waypoint)
        reached = trace.waypoint_reached is True
        may_infer_again = reached and not (status.terminated or status.truncated)
        if not may_infer_again:
            self._stopped = True
        return PipelineStepResult(intent, prior, waypoint, trace, status,
                                  views[self.camera_ids[0]].state_id, may_infer_again)
