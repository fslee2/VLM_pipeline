# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`robot-workflow-contracts` is a dependency-free library that puts a stable contract
above Gymnasium/simulator/robot SDK lifecycles so the same visual manipulation
workflow runs across different robot environments. It contains **no simulator,
VLM client, CoTracker model, or NumPy dependency** — those are replaceable ports
supplied by the integration project.

## Commands

The test suite uses only the Python standard library. No install, simulator, GPU,
or network is required.

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v          # full suite
python -m unittest tests.test_multiview_pipeline -v   # one module
python -m unittest tests.test_workflow.WorkflowTests.test_late_camera_change_is_rejected  # one test
```

`pyproject.toml` declares pytest config (`testpaths = ["tests"]`) but the suite is
plain `unittest` and must stay standard-library-only.

## Architecture

Strict layered dependency direction — each layer may import only the ones below it:

```
contracts.py   pure data types + artifact compatibility (no deps at all)
     ↑
ports.py       Protocol definitions: EnvironmentAdapter, MultiViewCaptureAdapter,
               RGBObservation, Transition, ExecutionTrace
     ↑
workflow.py    WorkflowEngine / WorkflowContext: preflight + capability gating
     ↑
pipeline.py    MultiViewCoTrackerPipeline: views -> intent -> compiler -> waypoint
```

`registry.py` maps scenario ids to `(EnvironmentProfile, adapter factory)` pairs
(explicit registration, no import-time auto-discovery). `examples.py` holds two
declarative profiles (Google Coke, Bridge Carrot) and **must not** import
SimplerEnv or encode robot poses.

### The boundary that matters

Everything environment-specific lives behind `ports.EnvironmentAdapter`. Code at
or above `workflow.py` must never touch a raw Gym env, call `unwrapped`, or
encode a robot-specific gripper number. Semantic `GripperIntent.OPEN` maps to
`-1.0` on Google and `+1.0` on WidowX; that polarity flip belongs solely to the
adapter.

`WorkflowContext.move_waypoint()` / multip-view `observe_views()` call
`_require(adapter, (Capability, ...))` before touching the backend, so an
unsupported feature fails during preflight rather than being silently
approximated. `Capability` is the single source of truth for optional features.

### Calibration artifacts

A `CalibrationArtifact` carries a `compatibility_key` (SHA-256 over profile id,
robot profile id, camera id, camera fingerprint, motion profile id,
reference-pose contract, protocol version, schema version). `assert_compatible`
rejects an artifact whose key does not match the runtime context — changing the
camera, robot/motion profile, reference pose, or protocol invalidates it. The
multi-view pipeline resolves calibration **once** in `prepare()`
(`calibration_provider.load_or_calibrate`) and reuses the artifacts every step;
do not re-run probe setup per decision or per episode.

### Pipeline data flow

`MultiViewCoTrackerPipeline.step()` enforces the ordering contract:

1. `context.observe_views(camera_ids)` — all frames must share one non-null
   `state_id`; camera ids and fingerprints are validated.
2. `intent_provider.infer(views, instruction)` → `MultiViewVisualIntent`
   (image-space `ScreenDirection`, right/down positive). The VLM sees images and
   the task only — never probe vectors, TCP pose, or camera geometry.
3. `CoTrackerDirectionCompiler.compile(intent, artifacts)` → `CompiledDirectionPrior`
   (robot-base x/y/z tri-state signs via least-squares; rank-deficient or
   conflicting geometry raises `WorkflowPreflightError`).
4. `policy.plan(policy_views, instruction, intent, prior)` → `Waypoint`.
5. `context.move_waypoint(waypoint)`; the adapter reports
   `ExecutionTrace.waypoint_reached`. Another inference is refused unless the
   waypoint was reached and the episode is still live.

Only the compiler reads CoTracker directions. Policies and the VLM must not.

### Known scope limits (do not overstate)

The artifact holds **unit image directions only** — coarse `left/right/up/down/hold`
labels can lose mixed-axis information. Rotation is *not* inferred; it needs a
separate multi-point-field contract. Tests prove portable wiring and compiler
math on synthetic effects, not a live Bridge episode or VLM/CoTracker accuracy.

## Docs

- `docs/MULTIVIEW_COTRACKER_PIPELINE.md` — pipeline wiring and data-flow contract.
- `docs/SIMPLER_INTEGRATION.md` — planned adapter migration boundary and
  integration gates (5 numbered gates before claiming behavioral equivalence).

Update these when the pipeline contract or adapter boundary changes; they encode
audit questions about simulator imports, unwrapped access, and artifact rejection.
