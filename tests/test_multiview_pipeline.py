from __future__ import annotations

import unittest
from dataclasses import replace
from itertools import product
from math import hypot

from robot_workflow import (
    CalibrationRequest, CalibrationSpec, CameraKind, CameraSpec, Capability,
    CoTrackerDirectionCompiler, EnvironmentProfile, MotionFrame, MotionSpec,
    MultiViewCoTrackerPipeline, MultiViewVisualIntent, Pose, ScreenDirection,
    TaskStatus, Waypoint, WaypointSpec, WorkflowEngine, WorkflowPreflightError,
    build_cotracker_direction_artifact,
)
from robot_workflow.ports import ExecutionTrace, RGBObservation, Transition


CAMERAS = (
    CameraSpec("front", CameraKind.NATIVE, 640, 480, "front-pose-1"),
    CameraSpec("side", CameraKind.INJECTED, 640, 480, "side-pose-1"),
)
PROFILE = EnvironmentProfile(
    id="fake.pick.v1", suite="fake", task_id="pick", robot_profile_id="robot.v1",
    cameras=CAMERAS,
    motion=MotionSpec("base-delta.v1", frozenset({MotionFrame.ROBOT_BASE}), 0.05, 0.1, (7,), "tcp-start.v1"),
    waypoint_profiles=(WaypointSpec("reach", MotionFrame.ROBOT_BASE, 0.012, 0.1, 12),),
    calibration=CalibrationSpec("cotracker", "cotracker-xyz.v1", MotionFrame.ROBOT_BASE,
                                frozenset({"x", "y", "z"}), 0.02, 3,
                                artifact_schema_version="cotracker-directions.v1"),
    capabilities=frozenset({Capability.MULTI_VIEW_CAPTURE, Capability.CALIBRATION_PROBES,
                            Capability.WAYPOINT_EXECUTION, Capability.TASK_EVALUATION}),
    task_phases=frozenset({"approach"}),
)
VECTORS = {
    "front": {"x": [1, 0], "y": [0, 1], "z": [1, 1]},
    "side": {"x": [1, 0], "y": [0, 1], "z": [-1, 1]},
}


def artifacts(profile=PROFILE, vectors=VECTORS):
    return {
        camera_id: build_cotracker_direction_artifact(
            profile, CalibrationRequest("cotracker-xyz.v1", camera_id, ("x", "y", "z"), 0.02, 3),
            vectors[camera_id], source_sha256="a" * 64,
        )
        for camera_id in vectors
    }


class FakeMultiViewAdapter:
    def __init__(self, profile=PROFILE):
        self._profile = profile
        self.state = 0
        self.capture_calls = 0
        self.mixed_state = False
        self.wrong_fingerprint = False
        self.reached = True
        self.execution_calls = 0

    @property
    def profile(self):
        return self._profile

    def reset(self, *, seed=None, options=None):
        self.state = 0
        return self.capture_rgb(self.profile.cameras[0].id)

    def capture_rgb(self, camera_id):
        camera = self.profile.camera(camera_id)
        return RGBObservation(camera.id, camera.fingerprint, rgb=camera_id.encode(), state_id=self.state)

    def capture_views(self, camera_ids):
        self.capture_calls += 1
        frames = {camera_id: self.capture_rgb(camera_id) for camera_id in camera_ids}
        if self.mixed_state:
            side = frames[camera_ids[-1]]
            frames[camera_ids[-1]] = RGBObservation(side.camera_id, side.camera_fingerprint,
                                                      side.rgb, state_id=self.state + 1)
        if self.wrong_fingerprint:
            side = frames[camera_ids[-1]]
            frames[camera_ids[-1]] = RGBObservation(side.camera_id, "wrong", side.rgb,
                                                      state_id=self.state)
        return frames

    def current_tcp_pose(self):
        return Pose(0, 0, 0)

    def execute_motion(self, command):
        raise AssertionError("this test uses waypoint execution")

    def execute_waypoint(self, waypoint):
        self.execution_calls += 1
        self.state += 1
        transition = Transition(self.capture_rgb(self.profile.cameras[0].id), 0.0, False, False, 2)
        return ExecutionTrace("waypoint", ((waypoint.profile_id,),), transition,
                              stop_reason="target_reached" if self.reached else "horizon",
                              waypoint_reached=self.reached)

    def task_status(self, transition):
        return TaskStatus(False, transition.terminated, transition.truncated)

    def close(self):
        pass


class FakeIntentProvider:
    def __init__(self, directions=None):
        self.calls = 0
        self.saw = None
        self.directions = directions or {
            "front": ScreenDirection(1, 1), "side": ScreenDirection(1, 1),
        }

    def infer(self, views, instruction):
        self.calls += 1
        self.saw = (tuple(views), instruction, {key: value.state_id for key, value in views.items()})
        return MultiViewVisualIntent(self.directions, phase="approach")


class FakePolicy:
    def __init__(self):
        self.calls = 0
        self.seen_views = None
        self.seen_prior = None

    def plan(self, views, instruction, intent, prior):
        self.calls += 1
        self.seen_views = tuple(views)
        self.seen_prior = prior
        return Waypoint("reach", Pose(0.1, 0.2, 0.3), phase=intent.phase)


class FakeCalibrationProvider:
    def __init__(self):
        self.calls = []

    def load_or_calibrate(self, context, camera_ids, axes):
        self.calls.append((context.profile_id, camera_ids, axes))
        return artifacts()


class MultiViewPipelineTests(unittest.TestCase):
    def prepare(self, adapter=None, **kwargs):
        adapter = adapter or FakeMultiViewAdapter()
        context = WorkflowEngine(adapter).prepare(seed=0, camera_id="front")
        intent = FakeIntentProvider()
        policy = FakePolicy()
        pipeline = MultiViewCoTrackerPipeline(
            context, camera_ids=("front", "side"), artifacts=artifacts(),
            intent_provider=intent, policy=policy, **kwargs,
        )
        return adapter, intent, policy, pipeline

    def test_one_reusable_step_uses_two_views_but_primary_only_policy(self):
        adapter, intent, policy, pipeline = self.prepare()
        result = pipeline.step("pick the object")
        self.assertEqual(adapter.capture_calls, 1)
        self.assertEqual(intent.saw, (("front", "side"), "pick the object", {"front": 0, "side": 0}))
        self.assertEqual(policy.seen_views, ("front",))
        self.assertEqual(dict(result.prior.signs), {"x": 1, "y": 1, "z": 0})
        self.assertEqual(len(result.prior.calibration_sha256), 2)
        self.assertFalse(hasattr(result.prior, "positive_robot_axis_image_directions"))
        self.assertTrue(result.may_infer_again)
        self.assertEqual(adapter.execution_calls, 1)
        pipeline.step("pick the object")
        self.assertEqual(intent.calls, 2)

    def test_calibration_provider_is_called_once_and_artifacts_reused_across_steps(self):
        adapter = FakeMultiViewAdapter()
        context = WorkflowEngine(adapter).prepare(seed=0, camera_id="front")
        calibration = FakeCalibrationProvider()
        intent = FakeIntentProvider()
        policy = FakePolicy()
        pipeline = MultiViewCoTrackerPipeline.prepare(
            context, camera_ids=("front", "side"), calibration_provider=calibration,
            intent_provider=intent, policy=policy,
        )

        self.assertEqual(calibration.calls, [(PROFILE.id, ("front", "side"), ("x", "y", "z"))])
        pipeline.step("pick")
        pipeline.step("pick")
        self.assertEqual(len(calibration.calls), 1)
        self.assertEqual(intent.calls, 2)
        self.assertEqual(adapter.capture_calls, 2)

    def test_policy_view_set_is_configurable_without_copying_workflow(self):
        _, _, policy, pipeline = self.prepare(policy_camera_ids=("front", "side"))
        pipeline.step("pick the object")
        self.assertEqual(policy.seen_views, ("front", "side"))

    def test_same_pipeline_runs_with_a_different_registered_robot_and_camera_names(self):
        other = replace(
            PROFILE, id="other.robot-task.v1", robot_profile_id="other.robot.v1",
            cameras=(CameraSpec("overhead", CameraKind.NATIVE, 640, 480, "overhead-pose"),
                     CameraSpec("wrist", CameraKind.NATIVE, 640, 480, "wrist-pose")),
            motion=replace(PROFILE.motion, id="other.delta.v1"),
        )
        adapter = FakeMultiViewAdapter(other)
        context = WorkflowEngine(adapter).prepare(seed=0, camera_id="overhead")
        measured = {"overhead": VECTORS["front"], "wrist": VECTORS["side"]}
        saved = artifacts(profile=other, vectors=measured)
        provider = FakeIntentProvider({"overhead": ScreenDirection(1, 1),
                                       "wrist": ScreenDirection(1, 1)})
        policy = FakePolicy()
        pipeline = MultiViewCoTrackerPipeline(
            context, camera_ids=("overhead", "wrist"), artifacts=saved,
            intent_provider=provider, policy=policy,
        )
        self.assertEqual(dict(pipeline.step("pick").prior.signs), {"x": 1, "y": 1, "z": 0})
        self.assertEqual(policy.seen_views, ("overhead",))

    def test_camera_set_may_include_more_than_two_views(self):
        third = CameraSpec("wrist", CameraKind.INJECTED, 640, 480, "wrist-pose-2")
        profile = replace(PROFILE, cameras=CAMERAS + (third,))
        adapter = FakeMultiViewAdapter(profile)
        context = WorkflowEngine(adapter).prepare(seed=0, camera_id="front")
        measured = {**VECTORS, "wrist": {"x": [0, 1], "y": [1, 0], "z": [-1, -1]}}
        provider = FakeIntentProvider({"front": ScreenDirection(1, 1),
                                       "side": ScreenDirection(1, 1),
                                       "wrist": ScreenDirection(1, 1)})
        pipeline = MultiViewCoTrackerPipeline(
            context, camera_ids=("front", "side", "wrist"),
            artifacts=artifacts(profile=profile, vectors=measured),
            intent_provider=provider, policy=FakePolicy(),
        )
        self.assertEqual(adapter.capture_calls, 0)
        result = pipeline.step("pick")
        self.assertEqual(provider.saw[0], ("front", "side", "wrist"))
        self.assertEqual(len(result.prior.calibration_sha256), 3)

    def test_view_specific_cotracker_compiles_z_without_vlm_axis_labels(self):
        visual = MultiViewVisualIntent({"front": ScreenDirection(1, 1),
                                        "side": ScreenDirection(-1, 1)})
        prior = CoTrackerDirectionCompiler().compile(visual, artifacts())
        self.assertEqual(dict(prior.signs), {"x": 0, "y": 0, "z": 1})

    def test_continuous_visual_deltas_preserve_all_synthetic_xyz_sign_combinations(self):
        calibrated = {
            camera: {axis: tuple(value / hypot(*vector) for value in vector)
                     for axis, vector in VECTORS[camera].items()}
            for camera in VECTORS
        }
        compiler = CoTrackerDirectionCompiler()
        for combination in product((-1, 0, 1), repeat=3):
            visual = MultiViewVisualIntent({
                camera: ScreenDirection(*(
                    sum(combination[k] * calibrated[camera]["xyz"[k]][coordinate]
                        for k in range(3)) for coordinate in (0, 1)
                )) for camera in calibrated
            })
            prior = compiler.compile(visual, artifacts())
            with self.subTest(combination=combination):
                self.assertEqual(tuple(prior.signs[axis] for axis in "xyz"), combination)

    def test_coarse_tri_state_can_lose_mixed_axis_information(self):
        calibrated = {
            camera: {axis: tuple(value / hypot(*vector) for value in vector)
                     for axis, vector in VECTORS[camera].items()}
            for camera in VECTORS
        }
        outputs = {}
        for combination in product((-1, 0, 1), repeat=3):
            quantized = tuple(
                tuple(0 if abs(sum(combination[k] * calibrated[camera]["xyz"[k]][coordinate]
                                   for k in range(3))) < 0.25
                      else 1 if sum(combination[k] * calibrated[camera]["xyz"[k]][coordinate]
                                    for k in range(3)) > 0 else -1
                      for coordinate in (0, 1))
                for camera in ("front", "side")
            )
            outputs.setdefault(quantized, []).append(combination)
        self.assertTrue(any(len(combinations) > 1 for combinations in outputs.values()))

    def test_frozen_measured_two_view_cotracker_vectors_compile_signed_single_axes(self):
        # Frozen values from the separate Simpler Bridge CoTracker primary/side
        # direction summaries, 2026-09-24. No live simulator or model is used.
        measured = {
            "front": {"x": [0.0375141818, 0.9992961263],
                      "y": [0.9999695168, 0.0078098111],
                      "z": [0.0337400613, -0.9994306500]},
            "side": {"x": [0.9896583800, -0.1434442766],
                     "y": [0.7153148345, 0.6988023833],
                     "z": [-0.0060049180, -0.9999819839]},
        }
        saved = artifacts(vectors=measured)
        compiler = CoTrackerDirectionCompiler()
        for axis in "xyz":
            for sign in (-1, 1):
                visual = MultiViewVisualIntent({
                    camera: ScreenDirection(*(
                        0 if abs(sign * measured[camera][axis][coordinate]) < 0.25
                        else 1 if sign * measured[camera][axis][coordinate] > 0 else -1
                        for coordinate in (0, 1)
                    )) for camera in measured
                })
                with self.subTest(axis=axis, sign=sign):
                    self.assertEqual(dict(compiler.compile(visual, saved).signs),
                                     {name: sign if name == axis else 0 for name in "xyz"})

    def test_camera_state_mismatch_fails_before_vlm_or_policy(self):
        adapter, intent, policy, pipeline = self.prepare()
        adapter.mixed_state = True
        with self.assertRaisesRegex(WorkflowPreflightError, "different environment states"):
            pipeline.step("pick")
        self.assertEqual(intent.calls, 0)
        self.assertEqual(policy.calls, 0)

    def test_camera_fingerprint_change_fails_before_vlm(self):
        adapter, intent, _, pipeline = self.prepare()
        adapter.wrong_fingerprint = True
        with self.assertRaisesRegex(WorkflowPreflightError, "invalid frame identity"):
            pipeline.step("pick")
        self.assertEqual(intent.calls, 0)

    def test_missing_or_wrong_camera_calibration_rejected_preflight(self):
        context = WorkflowEngine(FakeMultiViewAdapter()).prepare(seed=0, camera_id="front")
        with self.assertRaisesRegex(WorkflowPreflightError, "one CoTracker artifact"):
            MultiViewCoTrackerPipeline(context, camera_ids=("front", "side"),
                                       artifacts={"front": artifacts()["front"]},
                                       intent_provider=FakeIntentProvider(), policy=FakePolicy())
        switched = dict(artifacts())
        switched["side"] = switched["front"]
        with self.assertRaises(ValueError):
            MultiViewCoTrackerPipeline(context, camera_ids=("front", "side"), artifacts=switched,
                                       intent_provider=FakeIntentProvider(), policy=FakePolicy())

    def test_rank_deficient_view_geometry_rejected_before_vlm(self):
        bad_vectors = {camera: {"x": [1, 0], "y": [1, 0], "z": [1, 0]} for camera in ("front", "side")}
        context = WorkflowEngine(FakeMultiViewAdapter()).prepare(seed=0, camera_id="front")
        with self.assertRaisesRegex(WorkflowPreflightError, "cannot determine"):
            MultiViewCoTrackerPipeline(context, camera_ids=("front", "side"),
                                       artifacts=artifacts(vectors=bad_vectors),
                                       intent_provider=FakeIntentProvider(), policy=FakePolicy())

    def test_failed_waypoint_blocks_next_inference(self):
        adapter, intent, _, pipeline = self.prepare()
        adapter.reached = False
        result = pipeline.step("pick")
        self.assertFalse(result.may_infer_again)
        with self.assertRaisesRegex(WorkflowPreflightError, "previous waypoint failed"):
            pipeline.step("pick")
        self.assertEqual(intent.calls, 1)

    def test_cotracker_artifact_builder_rejects_missing_axis_and_nonfinite(self):
        request = CalibrationRequest("cotracker-xyz.v1", "front", ("x", "y", "z"), 0.02, 3)
        with self.assertRaisesRegex(ValueError, "exactly the requested axes"):
            build_cotracker_direction_artifact(PROFILE, request, {"x": [1, 0]}, source_sha256="a" * 64)
        bad = {**VECTORS["front"], "z": [float("nan"), 1]}
        with self.assertRaisesRegex(ValueError, "invalid numbers"):
            build_cotracker_direction_artifact(PROFILE, request, bad, source_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
