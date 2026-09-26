# Bridge example: SimplerEnv + WidowX

This folder is the first concrete environment integration for the generic
`robot_workflow` package. It registers the Bridge task and its cameras, adapts
the SimplerEnv lifecycle/observations/actions to the shared ports, loads the
existing CoTracker direction summaries as compatible calibration artifacts,
and builds `MultiViewCoTrackerPipeline` without editing or copying its core.

## Mapping onto the shared interfaces

| Shared workflow port | Bridge implementation |
|---|---|
| `EnvironmentProfile` | `make_bridge_profile(BridgeConfig(...))` |
| `EnvironmentAdapter` + `MultiViewCaptureAdapter` | `BridgeSimplerAdapter` |
| `CoTrackerCalibrationProvider` | `BridgeCoTrackerCalibrationProvider` |
| `VisualIntentProvider` | `BridgeVisualIntentProvider` around an injected client |
| `WaypointPolicy` | `BridgeWaypointPolicy` around an injected action client |
| Pipeline assembly | `prepare_bridge_pipeline(...)` |

The primary and side channels are two poses of the same native
`3rd_view_camera`, not two simultaneously mounted sensors. The adapter renders
them sequentially without calling `env.step`, verifies identity using
registered per-view fingerprints, and gives both observations the same local
environment `state_id`. It restores the primary camera pose before returning.
The reset/action flow uses SimplerEnv's `widowx_carrot_on_plate` task, reads TCP
pose in the WidowX robot-base frame, and routes target poses through the
existing Simpler `AbsolutePoseAdapter` (`ee_align`, 7-D action).

The Bridge CoTracker provider reads one frozen
`cotracker_direction_summary.json` for each view, verifies each summary's
camera eye/target against `BridgeConfig`, binds the `plus_x/plus_y/plus_z`
normalized directions to the profile, and hashes each source file. It does
not run CoTracker. These artifacts are loaded once when the pipeline is
prepared and then reused for all decisions.

The current Bridge summaries describe a deterministic reset calibration, so
`BridgeConfig.calibration_seed` defaults to `0` and the assembly helper refuses
to run another seed against that calibration. If a fresh calibration is
collected at another seed/reference pose, set that seed in the config and use
the matching summaries. This keeps the known camera + initial robot pose +
approximate wrist-orientation validity condition explicit. The historical
summary schema itself does not record a numeric reset TCP pose or seed, so the
configured calibration seed is a declared provenance assumption, not something
the summary loader can independently prove.

## Intent and action port contract

The core pipeline expects intent in each camera's image plane: horizontal and
vertical target-directed deltas, with image right/down positive. The compiler
then maps the per-view observations through each saved CoTracker artifact to
robot-base XYZ signs. Consequently, the Bridge intent client must return
`{"directions": {"primary": {"horizontal": ..., "vertical": ...},
"side": {...}}, "phase": ..., "holding": ..., "evidence": {...}}`.
It must **not** return robot-base XYZ signs directly; that is the behavior of
the older Bridge-specific dual-view runner, not this interface-aligned example.

The action client returns the Bridge policy's local 7-D delta
`[dx, dy, dz, rx, ry, rz, gripper]`. The adapter caps translation at 5 cm,
anchors that delta at the current TCP pose, interprets rotation-vector values
with the same right-composition used by the Simpler runner, and maps Bridge
gripper polarity (`+1=open`, `-1=close`, `0=hold`) to semantic intent. The
waypoint executor advances the simulator in bounded chunks and reports
`waypoint_reached`; the generic pipeline will not start another VLM decision
unless that barrier passes or the task terminates.

## Wiring example

Install no extra packages into this repository. Run in the Python environment
that already has the SimplerEnv Bridge runtime, NumPy, and SciPy available.
Place the workflow `src` directory and the Simpler checkout on `PYTHONPATH`.
The two client objects below are provider-specific API adapters supplied by
the deployment; they receive/return the port data described above.

```python
from pathlib import Path

from examples.bridge import (
    BridgeVisualIntentProvider,
    prepare_bridge_pipeline,
)

adapter, pipeline = prepare_bridge_pipeline(
    seed=0,
    summary_paths={
        "primary": Path("/data/bridge/primary_cotracker_direction_summary.json"),
        "side": Path("/data/bridge/side_cotracker_direction_summary.json"),
    },
    intent_provider=BridgeVisualIntentProvider(intent_api_client),
    action_client=bridge_policy_api_client,
    policy_camera_ids=("primary",),  # set to ("primary", "side") for dual-view policy input
)
try:
    while True:
        result = pipeline.step("Pick up the carrot and place it on the plate.")
        print(result.intent.phase, dict(result.prior.signs), result.trace.waypoint_reached)
        if not result.may_infer_again:
            break
finally:
    adapter.close()
```

## Verification boundary

`tests/test_bridge_adapter.py` uses a fake Simpler-like environment to test the
Bridge profile, view synchronization, calibration-summary conversion, gripper
polarity and waypoint barrier without launching a simulator or calling a VLM.
The adapter binds to real SimplerEnv APIs only when reset/action methods run.
Passing those tests demonstrates interface compatibility; it does not claim a
live Bridge episode or VLM success. Compare its intent semantics with the
existing `scripts/waypoint/h3_dual_view_intent_relay.py`: that runner gives
both calibration mappings to the VLM and asks it for robot-base XYZ directly,
while this example leaves calibration interpretation to the deterministic
compiler as required by the reusable workflow contract.
