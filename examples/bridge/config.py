"""Scenario registration for the SimplerEnv WidowX Bridge example.

The primary and side views are two configured poses of Bridge's one native
``3rd_view_camera``. They are captured sequentially without advancing state.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

from robot_workflow.contracts import (
    CalibrationSpec, CameraKind, CameraSpec, Capability, EnvironmentProfile,
    MotionFrame, MotionSpec, WaypointSpec,
)


@dataclass(frozen=True, slots=True)
class BridgeView:
    id: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.id or len(self.eye) != 3 or len(self.target) != 3:
            raise ValueError("Bridge view requires an id and 3-D eye/target")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (*self.eye, *self.target)):
            raise ValueError("Bridge camera coordinates must be finite numbers")


@dataclass(frozen=True, slots=True)
class BridgeConfig:
    task: str = "widowx_carrot_on_plate"
    calibration_seed: int = 0
    native_camera_id: str = "3rd_view_camera"
    image_width: int = 640
    image_height: int = 480
    views: tuple[BridgeView, ...] = (
        BridgeView("primary", (-0.625, 0.05, 1.4), (-0.1, 0.05, 0.87)),
        BridgeView("side", (-0.18, 0.45, 1.0), (-0.18, 0.05, 0.95)),
    )
    probe_displacement_m: float = 0.02
    probe_repetitions: int = 3
    waypoint_position_tolerance_m: float = 0.012
    waypoint_rotation_tolerance_rad: float = 0.1
    waypoint_max_native_steps: int = 12
    max_translation_step_m: float = 0.01
    max_rotation_step_rad: float = 0.08
    max_waypoint_translation_m: float = 0.05
    gripper_open_command: int = 1
    gripper_close_command: int = -1

    def __post_init__(self) -> None:
        ids = tuple(view.id for view in self.views)
        if len(ids) < 2 or len(set(ids)) != len(ids) or ids[0] != "primary":
            raise ValueError("Bridge requires distinct views with 'primary' first")
        if not self.task or not self.native_camera_id:
            raise ValueError("Bridge task and native camera id must be nonempty")
        if isinstance(self.calibration_seed, bool) or not isinstance(self.calibration_seed, int) or self.calibration_seed < 0:
            raise ValueError("Bridge calibration_seed must be a nonnegative integer")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Bridge image dimensions must be positive")
        positive = (
            self.probe_displacement_m, self.waypoint_position_tolerance_m,
            self.waypoint_rotation_tolerance_rad, self.max_translation_step_m,
            self.max_rotation_step_rad, self.max_waypoint_translation_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Bridge movement, calibration and tolerance values must be positive and finite")
        if isinstance(self.probe_repetitions, bool) or self.probe_repetitions < 1:
            raise ValueError("Bridge probe_repetitions must be positive")
        if isinstance(self.waypoint_max_native_steps, bool) or self.waypoint_max_native_steps < 1:
            raise ValueError("Bridge waypoint_max_native_steps must be positive")
        if self.gripper_open_command == self.gripper_close_command:
            raise ValueError("Bridge gripper open and close commands must differ")


def _view_fingerprint(config: BridgeConfig, view: BridgeView) -> str:
    payload = {
        "native_camera_id": config.native_camera_id,
        "width": config.image_width,
        "height": config.image_height,
        "eye_world": view.eye,
        "target_world": view.target,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def make_bridge_profile(config: BridgeConfig = BridgeConfig()) -> EnvironmentProfile:
    """Register Bridge's robot, virtual camera views, and waypoint contract."""
    cameras = tuple(
        CameraSpec(view.id, CameraKind.NATIVE, config.image_width, config.image_height,
                   _view_fingerprint(config, view),
                   description=(f"{config.native_camera_id} rendered from configured {view.id} pose"))
        for view in config.views
    )
    task_slug = "".join(character.lower() if character.isalnum() else "-" for character in config.task).strip("-")
    return EnvironmentProfile(
        id=f"simpler.bridge.widowx.{task_slug}.seed-{config.calibration_seed}.v1",
        suite="SimplerEnv/Bridge",
        task_id=config.task,
        robot_profile_id="simpler.widowx.bridge.v1",
        cameras=cameras,
        motion=MotionSpec(
            id="simpler.widowx.ee-align.delta7.v1",
            supported_frames=frozenset({MotionFrame.ROBOT_BASE}),
            max_translation_m=config.max_waypoint_translation_m,
            max_rotation_rad=0.5,
            native_action_shape=(7,),
            reference_pose_contract=(
                f"tcp-in-robot-base-after-bridge-reset-seed-{config.calibration_seed}.v1"
            ),
            description="SimplerEnv ee_align 7-D delta [dx,dy,dz,rx,ry,rz,gripper]",
        ),
        waypoint_profiles=(WaypointSpec(
            "bridge.absolute_tcp.v1", MotionFrame.ROBOT_BASE,
            config.waypoint_position_tolerance_m,
            config.waypoint_rotation_tolerance_rad,
            config.waypoint_max_native_steps,
        ),),
        calibration=CalibrationSpec(
            id="bridge.cotracker.xyz-directions.v1",
            protocol_version="bridge-cotracker-xyz.v1",
            probe_frame=MotionFrame.ROBOT_BASE,
            supported_axes=frozenset({"x", "y", "z"}),
            max_displacement_m=config.probe_displacement_m,
            min_repetitions=config.probe_repetitions,
            artifact_schema_version="cotracker-directions.v1",
        ),
        capabilities=frozenset({
            Capability.TCP_FEEDBACK, Capability.WAYPOINT_EXECUTION,
            Capability.CALIBRATION_PROBES, Capability.TASK_EVALUATION,
            Capability.MULTI_VIEW_CAPTURE,
        }),
        task_phases=frozenset({"approach", "grasp", "transport", "release"}),
        metadata={
            "environment": "SimplerEnv Bridge / WidowX",
            "camera_capture": "one native camera repositioned for each view without env.step",
            "calibration_seed": str(config.calibration_seed),
            "gripper_commands": f"open={config.gripper_open_command},close={config.gripper_close_command}",
        },
    )
