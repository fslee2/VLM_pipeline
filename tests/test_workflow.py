from __future__ import annotations

import unittest

from robot_workflow import (
    CalibrationArtifact,
    CalibrationRequest,
    Capability,
    GripperIntent,
    MotionCommand,
    MotionFrame,
    MotionSpec,
    Pose,
    Waypoint,
    WorkflowEngine,
    WorkflowPreflightError,
)
from robot_workflow.examples import BRIDGE_CARROT_PROFILE, GOOGLE_COKE_PROFILE
from robot_workflow.ports import ExecutionTrace, RGBObservation, Transition


class FakeAdapter:
    """Pure test adapter; its encoder models the only intentional robot difference."""

    def __init__(self, profile, native_gripper):
        self._profile = profile
        self.native_gripper = native_gripper
        self.closed = False

    @property
    def profile(self):
        return self._profile

    def reset(self, *, seed=None, options=None):
        return self.capture_rgb(self.profile.cameras[0].id)

    def capture_rgb(self, camera_id):
        camera = self.profile.camera(camera_id)
        return RGBObservation(camera.id, camera.fingerprint, rgb=b"fake")

    def current_tcp_pose(self):
        return Pose(0, 0, 0)

    def execute_motion(self, command):
        native = self.native_gripper[command.gripper]
        transition = Transition(self.capture_rgb(self.profile.cameras[0].id), 0.0, False, False, 1, {"success": False})
        return ExecutionTrace("motion", ((command.frame.value, native),), transition)

    def execute_waypoint(self, waypoint):
        transition = Transition(self.capture_rgb(self.profile.cameras[0].id), 0.0, False, False, 2, {"success": False})
        return ExecutionTrace("waypoint", ((waypoint.profile_id,),), transition)

    def task_status(self, transition):
        from robot_workflow import TaskStatus
        return TaskStatus(success=bool(transition.info.get("success")), terminated=transition.terminated, truncated=transition.truncated)

    def close(self):
        self.closed = True


GOOGLE_GRIPPER = {GripperIntent.OPEN: -1.0, GripperIntent.HOLD: 0.0, GripperIntent.CLOSE: 1.0}
BRIDGE_GRIPPER = {GripperIntent.OPEN: 1.0, GripperIntent.HOLD: 0.0, GripperIntent.CLOSE: -1.0}


class WorkflowTests(unittest.TestCase):
    def test_semantic_open_encodes_per_robot(self):
        command = MotionCommand(MotionFrame.ROBOT_BASE, GripperIntent.OPEN, delta=Pose(0.001, 0, 0))
        google = WorkflowEngine(FakeAdapter(GOOGLE_COKE_PROFILE, GOOGLE_GRIPPER)).prepare(seed=0, camera_id="google.left_table_downward.v1")
        bridge = WorkflowEngine(FakeAdapter(BRIDGE_CARROT_PROFILE, BRIDGE_GRIPPER)).prepare(seed=0, camera_id="bridge.third_view.v1")
        self.assertEqual(google.move(command)[0].native_actions[0][1], -1.0)
        self.assertEqual(bridge.move(command)[0].native_actions[0][1], 1.0)

    def test_artifact_rejects_wrong_robot_camera_profile(self):
        request = CalibrationRequest("cotracker-probes.v1", "google.left_table_downward.v1", ("x", "y"), 0.02, 3)
        artifact = CalibrationArtifact.for_profile(GOOGLE_COKE_PROFILE, request, {"quality": "pass"})
        context = WorkflowEngine(FakeAdapter(BRIDGE_CARROT_PROFILE, BRIDGE_GRIPPER)).prepare(seed=0, camera_id="bridge.third_view.v1")
        with self.assertRaises(ValueError):
            context.assert_calibration_compatible(artifact)

    def test_waypoint_uses_task_specific_phase_registry(self):
        context = WorkflowEngine(FakeAdapter(BRIDGE_CARROT_PROFILE, BRIDGE_GRIPPER)).prepare(
            seed=0,
            camera_id="bridge.third_view.v1",
            required_capabilities=(Capability.WAYPOINT_EXECUTION,),
        )
        trace, _ = context.move_waypoint(Waypoint("widowx.reach_pose.v1", Pose(0.2, 0.1, 0.2), phase="transport"))
        self.assertEqual(trace.command_kind, "waypoint")
        with self.assertRaises(WorkflowPreflightError):
            context.move_waypoint(Waypoint("widowx.reach_pose.v1", Pose(0.2, 0.1, 0.2), phase="lift_can"))

    def test_missing_capability_fails_before_reset(self):
        limited = FakeAdapter(GOOGLE_COKE_PROFILE, GOOGLE_GRIPPER)
        limited._profile = type(GOOGLE_COKE_PROFILE)(
            id="limited.v1", suite="fake", task_id="fake", robot_profile_id="fake.robot.v1",
            cameras=GOOGLE_COKE_PROFILE.cameras, motion=GOOGLE_COKE_PROFILE.motion,
        )
        with self.assertRaises(WorkflowPreflightError):
            WorkflowEngine(limited).prepare(seed=0, camera_id="google.left_table_downward.v1", required_capabilities=(Capability.CALIBRATION_PROBES,))

    def test_late_camera_change_is_rejected(self):
        adapter = FakeAdapter(GOOGLE_COKE_PROFILE, GOOGLE_GRIPPER)
        context = WorkflowEngine(adapter).prepare(seed=0, camera_id="google.left_table_downward.v1")
        adapter.capture_rgb = lambda camera_id: RGBObservation(camera_id, "wrong-fingerprint", rgb=b"fake")
        with self.assertRaises(WorkflowPreflightError):
            context.observe()

    def test_non_finite_motion_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            MotionSpec(
                id="bad", supported_frames=frozenset({MotionFrame.ROBOT_BASE}),
                max_translation_m=float("nan"), max_rotation_rad=0.1,
                native_action_shape=(7,), reference_pose_contract="fake.v1",
            )

    def test_stale_artifact_schema_is_rejected(self):
        artifact = CalibrationArtifact(
            schema_version="old.v0",
            profile_id=GOOGLE_COKE_PROFILE.id,
            robot_profile_id=GOOGLE_COKE_PROFILE.robot_profile_id,
            camera_id="google.left_table_downward.v1",
            camera_fingerprint=GOOGLE_COKE_PROFILE.cameras[0].fingerprint,
            motion_profile_id=GOOGLE_COKE_PROFILE.motion.id,
            reference_pose_contract=GOOGLE_COKE_PROFILE.motion.reference_pose_contract,
            protocol_version="cotracker-probes.v1",
            payload={},
        )
        context = WorkflowEngine(FakeAdapter(GOOGLE_COKE_PROFILE, GOOGLE_GRIPPER)).prepare(seed=0, camera_id="google.left_table_downward.v1")
        with self.assertRaises(ValueError):
            context.assert_calibration_compatible(artifact)


if __name__ == "__main__":
    unittest.main()
