# Planned Simpler Integration Boundary

This repository is intentionally a clean contract scaffold. It does not yet
run a SimplerEnv episode and must not be represented as an H3 or Bridge result.

## What moves behind a Simpler adapter

| Current responsibility | Future adapter responsibility |
|---|---|
| `simpler_env.make`, reset, `env.step`, close | `SimplerEnvironmentAdapter` lifecycle ownership |
| `env.unwrapped._render_cameras["render_camera"]` | Google injected-camera implementation |
| Bridge `3rd_view_camera` capture | WidowX native-camera implementation |
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
2. Implement Bridge adapter with reset, `3rd_view_camera`, semantic gripper
   encoding, and post-action status extraction. Do not call privileged object
   poses from a policy-facing method.
3. Enable `CALIBRATION_PROBES` only after safe signed probe motions and camera
   fingerprint verification are recorded.
4. Move CoTracker, semantic planning, fusion, and H3 rollout onto
   `WorkflowContext` one stage at a time.

## Audit questions

- Does any workflow-level module import a simulator, use `unwrapped`, or
  encode a robot-specific gripper number?
- Does every calibration artifact reject a changed robot, camera fingerprint,
  motion profile, reference-pose contract, or protocol version?
- Can an unsupported feature fail during `prepare()` before a robot command is
  emitted?
- Does every adapter-owned waypoint trace retain its native actions and
  resulting terminal/truncation state?
