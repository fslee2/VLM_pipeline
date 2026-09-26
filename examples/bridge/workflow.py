"""Wire the Bridge adapter and ports into the reusable workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from robot_workflow.contracts import Capability
from robot_workflow.pipeline import MultiViewCoTrackerPipeline, VisualIntentProvider
from robot_workflow.workflow import WorkflowEngine

from .adapter import BridgeSimplerAdapter
from .calibration import BridgeCoTrackerCalibrationProvider
from .config import BridgeConfig
from .providers import BridgeActionClient, BridgeWaypointPolicy


def prepare_bridge_pipeline(
    *, summary_paths: Mapping[str, Path], intent_provider: VisualIntentProvider,
    action_client: BridgeActionClient, seed: int = 0,
    config: BridgeConfig = BridgeConfig(),
    adapter: BridgeSimplerAdapter | None = None,
    policy_camera_ids: tuple[str, ...] = ("primary",),
) -> tuple[BridgeSimplerAdapter, MultiViewCoTrackerPipeline]:
    """Create/reset Bridge, verify profile capabilities and cache calibration.

    Model clients are injected so API credentials and provider SDKs remain
    outside this environment adapter and the generic workflow package.
    """
    bridge = adapter or BridgeSimplerAdapter(config)
    if bridge.config != config:
        raise ValueError("adapter and workflow BridgeConfig must match")
    if seed != config.calibration_seed:
        raise ValueError(
            "Bridge CoTracker direction calibration is bound to the configured deterministic reset seed; "
            "use a matching saved calibration before selecting another seed"
        )
    camera_ids = tuple(view.id for view in config.views)
    context = WorkflowEngine(bridge).prepare(
        seed=seed, camera_id="primary",
        required_capabilities=(Capability.MULTI_VIEW_CAPTURE,
                               Capability.CALIBRATION_PROBES,
                               Capability.WAYPOINT_EXECUTION,
                               Capability.TASK_EVALUATION),
    )
    calibration_provider = BridgeCoTrackerCalibrationProvider(config, summary_paths)
    pipeline = MultiViewCoTrackerPipeline.prepare(
        context, camera_ids=camera_ids,
        calibration_provider=calibration_provider,
        intent_provider=intent_provider,
        policy=BridgeWaypointPolicy(action_client, bridge),
        policy_camera_ids=policy_camera_ids,
    )
    return bridge, pipeline
