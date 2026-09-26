from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from robot_workflow.contracts import GripperIntent, Pose, Waypoint
from robot_workflow.pipeline import ScreenDirection
from robot_workflow.workflow import WorkflowEngine

from examples.bridge import (
    BridgeCoTrackerCalibrationProvider, BridgeConfig, BridgeSimplerAdapter,
    BridgeVisualIntentProvider, make_bridge_profile, prepare_bridge_pipeline,
)

try:
    import numpy as np
except ImportError:  # Bridge adapter's runtime dependency is optional for core users.
    np = None


class FakeSpace:
    dtype = "float32"
    shape = (7,)


class FakeBridgeEnv:
    action_space = FakeSpace()

    def __init__(self, config: BridgeConfig):
        self.config = config
        self.unwrapped = self
        self.tcp_x = 0.0
        self.active_view = "primary"
        self.render_calls = []
        self.step_calls = []
        self.closed = False

    def reset(self, *, seed=None, options=None):
        self.tcp_x = 0.0
        return self.get_obs(), {"seed": seed}

    def get_obs(self):
        self.render_calls.append(self.active_view)
        value = 10 if self.active_view == "primary" else 20
        rgb = np.full((480, 640, 3), value, dtype=np.uint8) if np is not None else None
        return {"image": {"3rd_view_camera": {"rgb": rgb}}}

    def step(self, action):
        self.step_calls.append(action)
        self.tcp_x += float(action[0])
        return self.get_obs(), 0.0, False, False, {"success": False}

    def close(self):
        self.closed = True


@unittest.skipIf(np is None, "NumPy is an optional Bridge runtime dependency")
class BridgeAdapterTests(unittest.TestCase):
    def make_adapter(self):
        config = BridgeConfig()
        env = FakeBridgeEnv(config)

        def configure(fake_env, view):
            fake_env.active_view = view.id

        def pose_reader(fake_env):
            return Pose(fake_env.tcp_x, 0.0, 0.0)

        def action_expander(target, current, gripper, action_space):
            dx = target.x - current.x
            dy = target.y - current.y
            dz = target.z - current.z
            norm = (dx * dx + dy * dy + dz * dz) ** 0.5
            scale = min(1.0, 0.01 / norm) if norm else 1.0
            return [[dx * scale, dy * scale, dz * scale, 0.0, 0.0, 0.0, gripper]]

        return (config, env, BridgeSimplerAdapter(
            config, env=env, camera_configurator=configure,
            pose_reader=pose_reader, action_expander=action_expander,
        ))

    def write_summaries(self, directory: Path, config: BridgeConfig) -> dict[str, Path]:
        paths = {}
        vectors = {
            "primary": {"x": [1.0, 0.0], "y": [0.0, 1.0], "z": [1.0, 1.0]},
            "side": {"x": [1.0, 0.0], "y": [0.0, 1.0], "z": [-1.0, 1.0]},
        }
        for view in config.views:
            payload = {
                "config": {
                    "reference_pose": "deterministic Bridge reset; TCP telemetry is audit-only",
                    "camera": {
                        "source": config.native_camera_id,
                        "runtime_override": {"eye_world": list(view.eye), "target_world": list(view.target)},
                    },
                    "cotracker": {"model": "CoTracker3 scaled offline", "checkpoint_sha256": "b" * 64},
                },
                "directions": {
                    f"plus_{axis}": {"normalized_direction": direction,
                                     "replicate_median_displacements_px": [[0.0, 1.0]] * 3,
                                     "valid_tracks": 25, "repeat_consistency_mean_cosine": 0.99}
                    for axis, direction in vectors[view.id].items()
                },
            }
            path = directory / f"{view.id}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths[view.id] = path
        return paths

    def test_profile_registers_bridge_task_and_two_view_channel(self):
        profile = make_bridge_profile()
        self.assertEqual(profile.task_id, "widowx_carrot_on_plate")
        self.assertEqual(tuple(camera.id for camera in profile.cameras), ("primary", "side"))
        self.assertEqual(profile.motion.native_action_shape, (7,))
        self.assertIn("open=1,close=-1", profile.metadata["gripper_commands"])

    def test_multiview_capture_uses_same_state_and_restores_primary(self):
        _, env, adapter = self.make_adapter()
        primary = adapter.reset(seed=17)
        views = adapter.capture_views(("primary", "side"))
        self.assertEqual(primary.state_id, 0)
        self.assertEqual(views["primary"].state_id, views["side"].state_id)
        self.assertEqual(views["primary"].rgb[0, 0, 0], 10)
        self.assertEqual(views["side"].rgb[0, 0, 0], 20)
        self.assertEqual(env.active_view, "primary")
        self.assertEqual(len(env.step_calls), 0)

    def test_waypoint_reports_reached_and_maps_semantic_close_to_minus_one(self):
        _, env, adapter = self.make_adapter()
        adapter.reset(seed=0)
        trace = adapter.execute_waypoint(
            Waypoint("bridge.absolute_tcp.v1", Pose(0.02, 0, 0), GripperIntent.CLOSE)
        )
        self.assertTrue(trace.waypoint_reached)
        self.assertEqual(trace.stop_reason, "target_reached")
        self.assertEqual(env.step_calls[0][6], -1.0)
        self.assertAlmostEqual(trace.tcp_after.x, 0.01)

    def test_policy_delta_is_capped_and_bridge_gripper_is_normalized(self):
        _, _, adapter = self.make_adapter()
        adapter.reset(seed=0)
        waypoint = adapter.waypoint_from_delta((0.2, 0, 0, 0, 0, 0, -1), phase="grasp")
        self.assertAlmostEqual(waypoint.target.x, 0.05)
        self.assertEqual(waypoint.gripper, GripperIntent.CLOSE)
        self.assertEqual(waypoint.phase, "grasp")

    def test_saved_summaries_become_camera_bound_cotracker_artifacts(self):
        config, _, adapter = self.make_adapter()
        context = WorkflowEngine(adapter).prepare(seed=0, camera_id="primary")
        with tempfile.TemporaryDirectory() as temp:
            paths = self.write_summaries(Path(temp), config)
            provider = BridgeCoTrackerCalibrationProvider(config, paths)
            artifacts = provider.load_or_calibrate(context, ("primary", "side"), ("x", "y", "z"))
        self.assertEqual(set(artifacts), {"primary", "side"})
        self.assertEqual(artifacts["side"].payload["estimator"], "CoTracker3")
        self.assertEqual(artifacts["side"].payload["positive_robot_axis_image_directions"]["x"], [1.0, 0.0])
        adapter.close()

    def test_calibration_summary_with_different_camera_pose_is_rejected(self):
        config, _, adapter = self.make_adapter()
        context = WorkflowEngine(adapter).prepare(seed=0, camera_id="primary")
        with tempfile.TemporaryDirectory() as temp:
            paths = self.write_summaries(Path(temp), config)
            side = json.loads(paths["side"].read_text(encoding="utf-8"))
            side["config"]["camera"]["runtime_override"]["eye_world"] = [99, 99, 99]
            paths["side"].write_text(json.dumps(side), encoding="utf-8")
            provider = BridgeCoTrackerCalibrationProvider(config, paths)
            with self.assertRaisesRegex(ValueError, "pose differs"):
                provider.load_or_calibrate(context, ("primary", "side"), ("x", "y", "z"))
        adapter.close()

    def test_full_bridge_example_runs_through_generic_pipeline_ports(self):
        config, env, adapter = self.make_adapter()
        calls = {"intent": 0, "policy": 0}

        class IntentClient:
            def infer_per_view(self, views, instruction):
                calls["intent"] += 1
                self.assert_synchronized = len({view.state_id for view in views.values()}) == 1
                return {
                    "directions": {
                        "primary": {"horizontal": 1.0, "vertical": 0.0},
                        "side": {"horizontal": 1.0, "vertical": 0.0},
                    },
                    "phase": "approach", "holding": "not_holding",
                }

        class ActionClient:
            def predict_delta(self, views, instruction, intent, prior):
                calls["policy"] += 1
                return (0.02, 0, 0, 0, 0, 0, 0)

        with tempfile.TemporaryDirectory() as temp:
            summaries = self.write_summaries(Path(temp), config)
            bridge, pipeline = prepare_bridge_pipeline(
                summary_paths=summaries,
                intent_provider=BridgeVisualIntentProvider(IntentClient()),
                action_client=ActionClient(), adapter=adapter, config=config,
                policy_camera_ids=("primary", "side"),
            )
            results = [pipeline.step("pick carrot") for _ in range(2)]
            self.assertTrue(all(result.may_infer_again for result in results))
            self.assertEqual(calls, {"intent": 2, "policy": 2})
            self.assertEqual(len(env.step_calls), 2)
            bridge.close()

    def test_pipeline_builder_refuses_nonmatching_calibration_reset_seed(self):
        config, _, adapter = self.make_adapter()
        with self.assertRaisesRegex(ValueError, "bound to the configured deterministic reset seed"):
            prepare_bridge_pipeline(
                summary_paths={}, intent_provider=object(), action_client=object(),
                seed=1, config=config, adapter=adapter,
            )


if __name__ == "__main__":
    unittest.main()
