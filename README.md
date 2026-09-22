# Robot Workflow Contracts

A small, dependency-free foundation for running the same visual manipulation
workflow across different robot environments.

It deliberately does **not** wrap a whole robot workflow as a Gymnasium
wrapper.  Gymnasium remains the backend lifecycle (`reset`, `step`, `render`,
spaces); this project places a stable contract above it:

```text
Gymnasium / simulator / robot SDK
            |
     EnvironmentAdapter
            |
  canonical observation, motion, waypoint, calibration, evaluation ports
            |
       WorkflowEngine
 calibration -> semantic planner -> deterministic fusion -> policy -> execution
```

## Why this exists

Two environments can share a 7-D action shape but still be incompatible:
camera names and poses differ, TCP frames differ, gripper polarity can be
reversed, a waypoint can have a different convergence rule, and success may
live in different `info` fields.  These details belong to an environment
adapter and its profile, never in the VLM prompt or workflow core.

For example:

```text
Canonical intent:  GripperIntent.OPEN
Google encoder:    -1.0
WidowX encoder:    +1.0
```

## Design rules

1. The workflow never receives a raw Gym environment or uses `unwrapped`.
2. Policies emit `MotionCommand` and `GripperIntent`, not native numeric
   actions.
3. Every camera is named and fingerprinted. A calibration artifact is only
   valid for its exact robot, camera, motion profile, reference-pose contract,
   and calibration protocol.
4. Waypoint expansion belongs to the adapter: one policy action may execute
   several native steps, but all of them are captured in an `ExecutionTrace`.
5. Evaluation is post-action only. Privileged simulator state must not enter a
   policy observation.
6. Optional features are explicit capabilities. A workflow fails preflight
   rather than silently approximating an unsupported operation.

## Package map

```text
robot_workflow/
  contracts.py   Stable data types and artifact compatibility checks
  ports.py       Adapter protocols (the boundary for every environment)
  registry.py    Explicit profile/factory registration
  workflow.py    Backend-independent preflight and command routing
  examples.py    Google and Bridge profile declarations
```

`examples.py` is declarative only. It intentionally does not import SimplerEnv
or encode a robot pose. An integration repository should implement the actual
adapter in its own backend module.

## Minimal integration

```python
from robot_workflow import WorkflowEngine
from robot_workflow.examples import BRIDGE_CARROT_PROFILE

adapter = MyBridgeAdapter(BRIDGE_CARROT_PROFILE)
engine = WorkflowEngine(adapter)
context = engine.prepare(seed=0, camera_id="bridge.third_view.v1")

# A stored artifact must match this exact runtime context before use.
context.assert_calibration_compatible(artifact)
```

## Required adapter checklist

Every registered scenario must provide:

- lifecycle: create, `reset`, native-step ownership, close;
- at least one registered camera and RGB capture;
- a semantic-motion encoder and an action-bound validator;
- task-status extraction;
- a declared reference-pose contract.

It must additionally declare capabilities before it can be used for waypoint
execution, CoTracker probe calibration, or TCP feedback. A calibration-capable
profile also registers its protocol version, allowed probe axes, displacement
limit, and minimum repetition count.

## Verification

The test suite uses only the Python standard library:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

No simulator, model endpoint, package installation, or GPU is required.
