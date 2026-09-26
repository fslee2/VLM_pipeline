"""SimplerEnv Bridge implementation of the generic environment ports.

Imports for SimplerEnv, NumPy, SAPIEN and SciPy are delayed until runtime so
the workflow package and its fake-adapter tests remain lightweight.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
import time
from typing import Any

from robot_workflow.contracts import (
    GripperIntent, MotionCommand, MotionFrame, Pose, TaskStatus, Waypoint,
)
from robot_workflow.ports import ExecutionTrace, RGBObservation, Transition
from robot_workflow.workflow import WorkflowPreflightError

from .config import BridgeConfig, BridgeView, make_bridge_profile


class BridgeSimplerAdapter:
    """Bridge/WidowX adapter; no Bridge assumptions leak into core workflow.

    ``env_factory``, ``camera_configurator``, ``pose_reader`` and
    ``action_expander`` are injectable seams for tests and alternate Simpler
    installations. Their defaults bind to SimplerEnv's Bridge APIs.
    """

    def __init__(self, config: BridgeConfig = BridgeConfig(), *, env: Any = None,
                 env_factory: Callable[..., Any] | None = None,
                 camera_configurator: Callable[[Any, BridgeView], None] | None = None,
                 pose_reader: Callable[[Any], Pose] | None = None,
                 action_expander: Callable[[Pose, Pose, float, Any], list[Any]] | None = None) -> None:
        self.config = config
        self._profile = make_bridge_profile(config)
        self._env = env
        self._env_factory = env_factory
        self._camera_configurator = camera_configurator or self._configure_camera
        self._pose_reader = pose_reader or self._read_tcp_pose
        self._action_expander = action_expander or self._expand_action
        self._state_id = -1
        self._terminated = False
        self._truncated = False
        self._last_info: Mapping[str, Any] = {}
        self._last_observation: RGBObservation | None = None
        self._motion_adapter: Any = None

    @property
    def profile(self):
        return self._profile

    @property
    def config(self) -> BridgeConfig:
        return self._config

    @config.setter
    def config(self, value: BridgeConfig) -> None:
        self._config = value

    def reset(self, *, seed: int | None = None,
              options: Mapping[str, Any] | None = None) -> RGBObservation:
        if self._env is None:
            factory = self._env_factory
            if factory is None:
                try:
                    import simpler_env
                except ImportError as exc:
                    raise RuntimeError(
                        "Bridge adapter needs SimplerEnv available in this Python environment"
                    ) from exc
                factory = simpler_env.make
            self._env = factory(self.config.task, render_mode="rgb_array")
        result = (self._env.reset(seed=seed) if options is None
                  else self._env.reset(seed=seed, options=options))
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("expected Gymnasium reset to return (observation, info)")
        observation, self._last_info = result
        shape = tuple(getattr(self._env.action_space, "shape", ()))
        if shape != (7,):
            raise WorkflowPreflightError(f"Bridge adapter expects native 7-D actions, got {shape}")
        low = getattr(self._env.action_space, "low", None)
        high = getattr(self._env.action_space, "high", None)
        if low is not None and high is not None:
            if not (low[6] <= self.config.gripper_open_command <= high[6]
                    and low[6] <= self.config.gripper_close_command <= high[6]):
                raise WorkflowPreflightError("Bridge semantic gripper commands exceed the native action bounds")
        self._state_id += 1
        self._terminated = self._truncated = False
        self._motion_adapter = None
        primary = self._view(self.profile.cameras[0].id)
        self._camera_configurator(self._env, primary)
        observation = self._env.unwrapped.get_obs()
        self._last_observation = self._capture(primary, observation)
        return self._last_observation

    def capture_rgb(self, camera_id: str) -> RGBObservation:
        if self._env is None:
            raise RuntimeError("reset Bridge adapter before capturing RGB")
        view = self._view(camera_id)
        self._camera_configurator(self._env, view)
        observation = self._env.unwrapped.get_obs()
        return self._capture(view, observation)

    def capture_views(self, camera_ids: tuple[str, ...]) -> Mapping[str, RGBObservation]:
        if self._env is None:
            raise RuntimeError("reset Bridge adapter before capturing views")
        if len(camera_ids) < 2 or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("capture_views needs distinct registered camera ids")
        captured = {}
        for camera_id in camera_ids:
            captured[camera_id] = self.capture_rgb(camera_id)
        # Leave the simulator in its canonical primary-camera pose. No env.step
        # occurred, so every RGBObservation carries the same state id.
        self._camera_configurator(self._env, self._view(self.profile.cameras[0].id))
        self._env.unwrapped.get_obs()
        return captured

    def current_tcp_pose(self) -> Pose:
        if self._env is None:
            raise RuntimeError("reset Bridge adapter before reading TCP pose")
        return self._pose_reader(self._env)

    def execute_motion(self, command: MotionCommand) -> ExecutionTrace:
        if command.frame is not MotionFrame.ROBOT_BASE:
            raise WorkflowPreflightError("Bridge adapter accepts robot-base motions only")
        current = self.current_tcp_pose()
        target = command.target if command.target is not None else self._add_delta(current, command.delta)
        waypoint = Waypoint("bridge.absolute_tcp.v1", target, command.gripper)
        return self.execute_waypoint(waypoint)

    def waypoint_from_delta(self, action: Any, *, phase: str = "") -> Waypoint:
        """Anchor the Bridge policy's local 7-D delta at current TCP pose.

        Rotation components follow the existing Simpler runner contract:
        ``rx/ry/rz`` are a rotation vector, right-composed with current ZYX
        yaw/pitch/roll. Translation is capped before conversion to an absolute
        target; semantic gripper polarity is translated only in this adapter.
        """
        values = tuple(float(value) for value in action)
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise ValueError("Bridge policy action must contain seven finite values")
        dx, dy, dz, rx, ry, rz, native_gripper = values
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance > self.config.max_waypoint_translation_m:
            ratio = self.config.max_waypoint_translation_m / distance
            dx, dy, dz = dx * ratio, dy * ratio, dz * ratio
        current = self.current_tcp_pose()
        if max(abs(rx), abs(ry), abs(rz)) <= 1e-12:
            yaw, pitch, roll = current.yaw, current.pitch, current.roll
        else:
            try:
                from scipy.spatial.transform import Rotation
            except ImportError as exc:
                raise RuntimeError("Bridge action conversion requires Simpler's SciPy runtime") from exc
            current_rotation = Rotation.from_euler("ZYX", (current.yaw, current.pitch, current.roll))
            target_rotation = current_rotation * Rotation.from_rotvec((rx, ry, rz))
            yaw, pitch, roll = target_rotation.as_euler("ZYX")
        target = Pose(current.x + dx, current.y + dy, current.z + dz,
                      float(yaw), float(pitch), float(roll))
        if native_gripper >= 0.5:
            gripper = GripperIntent.OPEN
        elif native_gripper <= -0.5:
            gripper = GripperIntent.CLOSE
        else:
            gripper = GripperIntent.HOLD
        return Waypoint("bridge.absolute_tcp.v1", target, gripper, phase)

    def execute_waypoint(self, waypoint: Waypoint) -> ExecutionTrace:
        if self._env is None:
            raise RuntimeError("reset Bridge adapter before executing a waypoint")
        if waypoint.profile_id != "bridge.absolute_tcp.v1":
            raise WorkflowPreflightError(f"unsupported Bridge waypoint profile: {waypoint.profile_id}")
        spec = self.profile.waypoint(waypoint.profile_id)
        start = self.current_tcp_pose()
        initial_position_error, initial_rotation_error = self._pose_errors(waypoint.target, start)
        if initial_position_error > self.profile.motion.max_translation_m:
            raise WorkflowPreflightError("Bridge waypoint exceeds the registered translation limit")
        if initial_rotation_error > self.profile.motion.max_rotation_rad:
            raise WorkflowPreflightError("Bridge waypoint exceeds the registered rotation limit")
        native_actions = []
        observation = self._last_observation
        terminated = truncated = False
        info: Mapping[str, Any] = self._last_info
        stop_reason = "horizon"
        reached = False
        for _ in range(spec.max_native_steps):
            current = self.current_tcp_pose()
            position_error, rotation_error = self._pose_errors(waypoint.target, current)
            # Match the established Bridge runner: always let the controller
            # issue at least one action for a decision, then test tolerance.
            if native_actions and position_error <= spec.position_tolerance_m and rotation_error <= spec.rotation_tolerance_rad:
                reached, stop_reason = True, "target_reached"
                break
            gripper = self._gripper_value(waypoint.gripper)
            actions = self._action_expander(waypoint.target, current, gripper, self._env.action_space)
            if not actions:
                raise WorkflowPreflightError("Bridge action expander returned no native action")
            action = actions[0]
            observation, reward, terminated, truncated, info = self._env.step(action)
            self._state_id += 1
            self._terminated, self._truncated = bool(terminated), bool(truncated)
            self._last_info = info if isinstance(info, Mapping) else {}
            self._last_observation = self._capture(self._view(self.profile.cameras[0].id), observation)
            native_actions.append(action)
            position_error, rotation_error = self._pose_errors(waypoint.target, self.current_tcp_pose())
            if position_error <= spec.position_tolerance_m and rotation_error <= spec.rotation_tolerance_rad:
                reached, stop_reason = True, "target_reached"
                break
            if terminated or truncated:
                stop_reason = "environment_terminal"
                break
        transition_observation = self._last_observation or observation
        transition = Transition(
            observation=transition_observation,
            reward=float(reward) if native_actions and reward is not None else None,
            terminated=bool(terminated), truncated=bool(truncated),
            native_steps=max(1, len(native_actions)), info=dict(info or {}),
        )
        return ExecutionTrace(
            command_kind="waypoint", native_actions=tuple(native_actions), transition=transition,
            tcp_before=start, tcp_after=self.current_tcp_pose(), stop_reason=stop_reason,
            waypoint_reached=reached,
        )

    def task_status(self, transition: Transition) -> TaskStatus:
        success = bool(transition.info.get("success", False))
        return TaskStatus(success, transition.terminated, transition.truncated,
                          diagnostics={key: value for key, value in transition.info.items()
                                       if key in {"success", "episode_stats", "is_success"}})

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        self._motion_adapter = None

    def _capture(self, view: BridgeView, observation: Mapping[str, Any]) -> RGBObservation:
        try:
            image = observation["image"][self.config.native_camera_id]
            image = image["rgb"] if "rgb" in image else image["Color"]
        except (KeyError, TypeError) as exc:
            raise WorkflowPreflightError(
                f"Bridge observation lacks image/{self.config.native_camera_id}/rgb|Color"
            ) from exc
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Bridge adapter requires NumPy at runtime") from exc
        rgb = np.asarray(image)
        if rgb.shape == (self.config.image_height, self.config.image_width, 4):
            rgb = rgb[..., :3]
        if rgb.shape != (self.config.image_height, self.config.image_width, 3):
            raise WorkflowPreflightError(
                f"expected {self.config.image_height}x{self.config.image_width} RGB, got {rgb.shape}"
            )
        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating) and float(np.nanmax(rgb)) <= 1.0:
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        camera = self.profile.camera(view.id)
        return RGBObservation(camera.id, camera.fingerprint, rgb.copy(), time.monotonic_ns(), self._state_id)

    def _configure_camera(self, env: Any, view: BridgeView) -> None:
        try:
            from mani_skill2_real2sim.utils.sapien_utils import look_at
        except ImportError as exc:
            raise RuntimeError("Bridge camera poses require SimplerEnv's SAPIEN runtime") from exc
        camera = env.unwrapped._cameras[self.config.native_camera_id].camera
        world_pose = look_at(view.eye, view.target)
        parent = getattr(camera, "parent", None)
        if parent is None:
            camera.set_pose(world_pose)
        else:
            camera.set_local_pose(parent.get_pose().inv() * world_pose)
        # The original Bridge background overlay is calibrated to the stock
        # view; moving the camera invalidates it, as in the established runner.
        env.unwrapped.rgb_overlay_img = None

    def _view(self, camera_id: str) -> BridgeView:
        self.profile.camera(camera_id)
        return next(view for view in self.config.views if view.id == camera_id)

    def _make_motion_adapter(self) -> Any:
        try:
            from envsim.adapters.absolute_pose import AbsolutePoseAdapter
        except ImportError as exc:
            raise RuntimeError(
                "Bridge waypoint execution needs Simpler's envsim AbsolutePoseAdapter and SciPy"
            ) from exc
        return AbsolutePoseAdapter(
            max_translation_step=self.config.max_translation_step_m,
            max_rotation_step=self.config.max_rotation_step_rad,
            action_space=self._env.action_space,
        )

    def _expand_action(self, target: Pose, current: Pose, gripper: float, action_space: Any) -> list[Any]:
        if self._motion_adapter is None:
            self._motion_adapter = self._make_motion_adapter()
        return self._motion_adapter.move_to(
            {"x": target.x, "y": target.y, "z": target.z,
             "yaw": target.yaw, "pitch": target.pitch, "roll": target.roll, "gripper": gripper},
            {"x": current.x, "y": current.y, "z": current.z,
             "yaw": current.yaw, "pitch": current.pitch, "roll": current.roll},
        )

    def _read_tcp_pose(self, env: Any) -> Pose:
        try:
            from scipy.spatial.transform import Rotation
        except ImportError as exc:
            raise RuntimeError("Bridge TCP pose conversion requires Simpler's SciPy runtime") from exc
        value = env.unwrapped.agent.robot.pose.inv() * env.unwrapped.tcp.pose
        q = value.q  # SAPIEN order is w, x, y, z; SciPy expects x, y, z, w.
        yaw, pitch, roll = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("ZYX")
        return Pose(*map(float, value.p), float(yaw), float(pitch), float(roll))

    @staticmethod
    def _pose_errors(target: Pose, current: Pose) -> tuple[float, float]:
        position = math.sqrt((target.x - current.x) ** 2 + (target.y - current.y) ** 2
                             + (target.z - current.z) ** 2)
        goal = _rotation_matrix_zyx(target.yaw, target.pitch, target.roll)
        actual = _rotation_matrix_zyx(current.yaw, current.pitch, current.roll)
        trace = sum(sum(goal[row][column] * actual[row][column]
                        for row in range(3)) for column in range(3))
        angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
        return position, angle

    def _gripper_value(self, intent: GripperIntent) -> float:
        if intent is GripperIntent.OPEN:
            return float(self.config.gripper_open_command)
        if intent is GripperIntent.CLOSE:
            return float(self.config.gripper_close_command)
        return 0.0

    @staticmethod
    def _add_delta(current: Pose, delta: Pose | None) -> Pose:
        if delta is None:
            raise ValueError("motion command lacks both target and delta")
        # Delta rotations are Euler components in robot-base ZYX order.
        try:
            from scipy.spatial.transform import Rotation
        except ImportError as exc:
            raise RuntimeError("Bridge motion composition requires Simpler's SciPy runtime") from exc
        actual = Rotation.from_euler("ZYX", (current.yaw, current.pitch, current.roll))
        increment = Rotation.from_euler("ZYX", (delta.yaw, delta.pitch, delta.roll))
        yaw, pitch, roll = (increment * actual).as_euler("ZYX")
        return Pose(current.x + delta.x, current.y + delta.y, current.z + delta.z,
                    float(yaw), float(pitch), float(roll))


def _rotation_matrix_zyx(yaw: float, pitch: float, roll: float) -> tuple[tuple[float, ...], ...]:
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cx, sx = math.cos(roll), math.sin(roll)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )
