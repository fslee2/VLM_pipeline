# Planned Simpler Integration Boundary

This repository now contains the generic synchronized-view, CoTracker-prior,
visual-intent, policy, and waypoint orchestration. It still does not contain a
live SimplerEnv adapter or run a SimplerEnv episode; passing portable tests must
not be represented as an H3 or Bridge task result.

## What moves behind a Simpler adapter

| Current responsibility | Future adapter responsibility |
|---|---|
| `simpler_env.make`, reset, `env.step`, close | `SimplerEnvironmentAdapter` lifecycle ownership |
| `env.unwrapped._render_cameras["render_camera"]` | Google injected-camera implementation |
| Registered Bridge primary/side capture without stepping between views | WidowX `capture_views` implementation with one shared `state_id` |
| `current_tcp_pose(env)` and gripper joint reads | Simpler telemetry implementation |
| `AbsolutePoseAdapter` interpolation | Registered Google/WidowX waypoint executor |
| Google/WidowX numeric gripper conversion | Per-robot semantic-motion encoder |
| `info["success"]` and task diagnostics | Per-task evaluation implementation |

The generic workflow must receive only the interfaces from
`robot_workflow.ports`; it must never receive a raw Simpler environment.

## Profile boundaries

The first two profiles are deliberately separate:

```text
simpler.google-coke.left-table-downward.v1
  camera: google.left_table_downward.v1
  task phases: approach, align, descend, close, lift, hold

simpler.widowx-bridge.carrot-third-view.v1
  camera: bridge.third_view.v1
  task phases: approach, align, descend, grasp, lift, transport, place, release, verify
```

The latter must not reuse the former's camera fingerprint, calibration
artifact, gripper encoding, task phase machine, or TCP reference contract.

## Integration gates

1. Implement Google adapter only; run existing non-VLM reset/capture/motion
   checks and prove its native action trace equals the legacy path.
2. Implement Bridge adapter with reset, a configured set of registered camera
   views, semantic gripper encoding, and post-action status extraction. Do not
   call privileged object poses from a policy-facing method.
3. Enable `CALIBRATION_PROBES` only after safe signed probe motions and camera
   fingerprint verification are recorded.
4. Convert each saved camera's measured CoTracker directions into a
   `CalibrationArtifact` bound to the registered camera/profile; instantiate
   the shared `MultiViewCoTrackerPipeline` with those artifacts and the live
   intent/policy clients. Do not copy the pipeline into a Bridge-specific runner.
5. Compare one episode's VLM inputs, compiled prior, waypoint and native
   action trace against the existing Simpler path before claiming behavioral
   equivalence. Rotation calibration is not part of the current translation
   direction compiler.

## Audit questions

- Does any workflow-level module import a simulator, use `unwrapped`, or
  encode a robot-specific gripper number?
- Does every calibration artifact reject a changed robot, camera fingerprint,
  motion profile, reference-pose contract, or protocol version?
- Can an unsupported feature fail during `prepare()` before a robot command is
  emitted?
- Does every adapter-owned waypoint trace retain its native actions and
  resulting terminal/truncation state?
