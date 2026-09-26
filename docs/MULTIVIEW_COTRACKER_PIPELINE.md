# Reusable multi-view + CoTracker workflow

This is a **package-level workflow**, not a Bridge camera script. A new
environment supplies registered cameras, a motion/waypoint adapter, one saved
CoTracker translation artifact per selected camera, a visual-intent provider,
and a waypoint policy. The same `MultiViewCoTrackerPipeline` then runs every
decision. A camera may be native, injected, wrist-mounted, or anything else
the adapter can register and capture without advancing the environment state.

```text
environment adapter (reset / registered camera capture / waypoint execution)
     │
     ├─ one-time CoTracker calibration per camera + reference-pose contract
     │                      ↓ saved compatible artifacts
     └─ synchronized named RGB views ──→ visual-intent provider
                                            │ image-space directions + phase
                                            ↓
                          deterministic CoTrackerDirectionCompiler
                                            │ compact robot-base XYZ signs
                                            ↓
                          waypoint policy (selected images, no raw tracks)
                                            ↓
                          adapter-owned execution + reached barrier
                                            └─ only then next observation/inference
```

The adapter must register every camera in `EnvironmentProfile.cameras`, declare
`MULTI_VIEW_CAPTURE`, and implement `capture_views(camera_ids)`. It can render
camera views sequentially, but cannot step the environment between captures.
Each `RGBObservation` must carry the same non-null environment `state_id`; the
workflow checks state IDs, camera IDs, and fingerprints before calling a VLM.
The adapter remains responsible for converting semantic gripper commands to
its native polarity and for deciding whether the waypoint met its configured
position/orientation tolerance. The pipeline refuses another inference after
an unreached waypoint or episode termination.

## Wiring the same pipeline in any environment

```python
from robot_workflow import (
    Capability, CoTrackerDirectionCompiler, MultiViewCoTrackerPipeline,
    WorkflowEngine,
)

# The integration provides these ports; no simulator or VLM SDK is imported
# by robot_workflow. primary_id and other_ids come from environment config.
context = WorkflowEngine(adapter).prepare(
    seed=0,
    camera_id=primary_id,
    required_capabilities=(Capability.MULTI_VIEW_CAPTURE,
                           Capability.CALIBRATION_PROBES,
                           Capability.WAYPOINT_EXECUTION),
)
camera_ids = (primary_id, *other_ids)
pipeline = MultiViewCoTrackerPipeline.prepare(
    context,
    camera_ids=camera_ids,
    calibration_provider=calibration_provider,
    intent_provider=visual_intent_client,
    policy=waypoint_policy,
    policy_camera_ids=(primary_id,),  # or camera_ids; policy view choice is config
    compiler=CoTrackerDirectionCompiler(axes=("x", "y", "z")),
)
result = pipeline.step("pick up the object and place it in the target")
```

`calibration_provider` implements `load_or_calibrate(context, camera_ids,
axes)`. On pipeline setup it first loads artifacts compatible with the
registered camera/robot/motion/reference-pose contract; if they are missing,
it may perform the setup probes, build artifacts, and persist them. The
pipeline retains those artifacts for every subsequent decision. Thus CoTracker
probe/tracking setup is not repeated per decision or episode when the saved
artifact still matches; changing the camera, robot/motion profile, or
reference-pose contract invalidates it and requires a new setup calibration.
`build_cotracker_direction_artifact` is the helper for wrapping one camera's
measured positive-axis image vectors, source SHA-256, and registered
`CalibrationRequest`; it stores normalized directions and profile/camera/
motion/reference-pose/protocol identity.

The visual-intent client receives only named RGB observations and instruction.
It returns `MultiViewVisualIntent` with per-view `ScreenDirection` (image right
and down positive), phase, and optional evidence. The deterministic compiler
alone reads CoTracker directions and returns `CompiledDirectionPrior`; the
policy receives only selected current images, intent, and this compact prior.
No simulator TCP/object pose, raw CoTracker points, or camera geometry is in
either VLM input through this API.

## What the tests prove, and what they do not

The standard-library suite checks synchronized-view rejection, wrong camera
and stale calibration rejection, rank-deficient geometry, signed XYZ compiler
behavior, configurable policy image count, waypoint barrier, and reuse with
different robot/camera registrations (including more than two views). It does
not run a live simulator or VLM.

The current artifact contains **unit image directions only**. An intent made
solely of left/right/up/down/hold labels can lose mixed-axis information:
different XYZ combinations may quantize to the same per-view labels. The
compiler accepts continuous image-space target deltas when a vision stage can
provide them; such deltas retain more information, but their scale across
views must be meaningful. The tests prove linear compiler behavior for
synthetic effects, not that a VLM can produce accurate continuous deltas.
An ill-conditioned or conflicting mapping is rejected instead of being called
a calibrated action. Translation magnitudes and **rotation** are not inferred
from this direction-only artifact; rotation needs its separate spatial
multi-point signature and a tested visual-intent contract.

## Change provenance

The pre-change source snapshot is commit
`77abb2cc9d59443e272a19934046da2dca7a25d1` on
`codex/dual-view-cotracker-workflow-20260925` in `fslee2/VLM_pipeline`.
It was pushed and its remote SHA checked before these workflow edits. This
document describes package code, not a claim that the Simpler adapter has
been migrated or that a Bridge episode succeeded.
