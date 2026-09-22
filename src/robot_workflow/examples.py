"""Declarative profiles illustrating Google and Bridge separation.

These are not live adapters.  The fingerprints are stable profile identifiers
for this scaffold; a production integration must derive them from the actual
camera configuration and version them when the configuration changes.
"""

from .contracts import (
    CalibrationSpec,
    CameraKind,
    CameraSpec,
    Capability,
    EnvironmentProfile,
    MotionFrame,
    MotionSpec,
    WaypointSpec,
)


GOOGLE_COKE_PROFILE = EnvironmentProfile(
    id="simpler.google-coke.left-table-downward.v1",
    suite="simpler_env",
    task_id="google_robot_pick_coke_can",
    robot_profile_id="simpler.google.ee-align.v1",
    cameras=(
        CameraSpec(
            id="google.left_table_downward.v1",
            kind=CameraKind.INJECTED,
            width=640,
            height=480,
            fingerprint="simpler-render-camera:left-table-downward:v1",
            description="Registered presentation camera for Coke calibration.",
        ),
    ),
    motion=MotionSpec(
        id="simpler.google.delta7d.v1",
        supported_frames=frozenset({MotionFrame.ROBOT_BASE, MotionFrame.EE_ALIGNED}),
        max_translation_m=0.02,
        max_rotation_rad=0.10,
        native_action_shape=(7,),
        reference_pose_contract="simpler.google.tcp.robot_base.v1",
    ),
    waypoint_profiles=(
        WaypointSpec("google.reach_pose.v1", MotionFrame.ROBOT_BASE, 0.012, 0.12, 12),
    ),
    calibration=CalibrationSpec("google.cotracker-probes.v1", "cotracker-probes.v1", MotionFrame.ROBOT_BASE, frozenset({"x", "y", "z"}), 0.02, 3),
    capabilities=frozenset({
        Capability.TCP_FEEDBACK,
        Capability.WAYPOINT_EXECUTION,
        Capability.CALIBRATION_PROBES,
        Capability.TASK_EVALUATION,
    }),
    task_phases=frozenset({"approach", "align", "descend", "close", "lift", "hold"}),
)


BRIDGE_CARROT_PROFILE = EnvironmentProfile(
    id="simpler.widowx-bridge.carrot-third-view.v1",
    suite="simpler_env",
    task_id="widowx_carrot_on_plate",
    robot_profile_id="simpler.widowx.ee-gripper-link.v1",
    cameras=(
        CameraSpec(
            id="bridge.third_view.v1",
            kind=CameraKind.NATIVE,
            width=640,
            height=480,
            fingerprint="simpler-widowx:3rd-view-camera:bridge-real-eval:v1",
            description="Bridge native third-person evaluation camera.",
        ),
    ),
    motion=MotionSpec(
        id="simpler.widowx.delta7d.v1",
        supported_frames=frozenset({MotionFrame.ROBOT_BASE}),
        max_translation_m=0.015,
        max_rotation_rad=0.15,
        native_action_shape=(7,),
        reference_pose_contract="simpler.widowx.tcp.robot_base.v1",
    ),
    waypoint_profiles=(
        WaypointSpec("widowx.reach_pose.v1", MotionFrame.ROBOT_BASE, 0.012, 0.15, 18),
    ),
    calibration=CalibrationSpec("widowx.cotracker-probes.v1", "cotracker-probes.v1", MotionFrame.ROBOT_BASE, frozenset({"x", "y", "z"}), 0.015, 3),
    capabilities=frozenset({
        Capability.TCP_FEEDBACK,
        Capability.WAYPOINT_EXECUTION,
        Capability.CALIBRATION_PROBES,
        Capability.TASK_EVALUATION,
    }),
    task_phases=frozenset({
        "approach", "align", "descend", "grasp", "lift", "transport", "place", "release", "verify",
    }),
)
