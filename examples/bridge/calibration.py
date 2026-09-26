"""Load saved Bridge CoTracker direction summaries as workflow artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping

from robot_workflow.contracts import CalibrationArtifact, CalibrationRequest
from robot_workflow.pipeline import CoTrackerCalibrationProvider, build_cotracker_direction_artifact
from robot_workflow.workflow import WorkflowContext

from .config import BridgeConfig, make_bridge_profile


class BridgeCoTrackerCalibrationProvider(CoTrackerCalibrationProvider):
    """Adapt frozen ``cotracker_direction_summary.json`` files to core schema.

    This adapter never installs or runs CoTracker. The existing offline probe
    and tracker job creates summaries; this provider verifies their view poses,
    hashes the source file, wraps measured directions and lets the workflow
    retain the resulting artifacts for all decisions in the episode.
    """

    def __init__(self, config: BridgeConfig, summary_paths: Mapping[str, Path]) -> None:
        self.config = config
        self.summary_paths = {key: Path(value) for key, value in summary_paths.items()}
        self.load_count = 0

    def load_or_calibrate(self, context: WorkflowContext, camera_ids: tuple[str, ...],
                          axes: tuple[str, ...]) -> Mapping[str, CalibrationArtifact]:
        if set(camera_ids) != set(self.summary_paths):
            raise ValueError("provide exactly one CoTracker summary path for every configured Bridge view")
        self.load_count += 1
        artifacts = {}
        profile = make_bridge_profile(self.config)
        if profile.id != context.profile_id:
            raise ValueError("Bridge calibration config does not match the prepared workflow profile")
        for view_id in camera_ids:
            path = self.summary_paths[view_id]
            if not path.is_file():
                raise FileNotFoundError(f"Bridge CoTracker summary not found for {view_id}: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            self._check_camera_pose(summary, view_id)
            directions = summary.get("directions")
            if not isinstance(directions, dict):
                raise ValueError(f"{path}: missing CoTracker directions")
            positive = {}
            for axis in axes:
                row = directions.get(f"plus_{axis}")
                vector = row.get("normalized_direction") if isinstance(row, dict) else None
                if not isinstance(vector, list) or len(vector) != 2:
                    raise ValueError(f"{path}: missing measured plus_{axis}.normalized_direction")
                repeats = row.get("replicate_median_displacements_px", [])
                if not isinstance(repeats, list) or len(repeats) < self.config.probe_repetitions:
                    raise ValueError(
                        f"{path}: plus_{axis} has fewer than {self.config.probe_repetitions} repeats"
                    )
                positive[axis] = vector
            request = CalibrationRequest(
                protocol_version=profile.calibration.protocol_version,
                camera_id=view_id, axes=axes,
                displacement_m=self.config.probe_displacement_m,
                repetitions=self.config.probe_repetitions,
            )
            quality = {
                "source_summary": path.name,
                "cotracker_model": summary.get("config", {}).get("cotracker", {}).get("model"),
                "checkpoint_sha256": summary.get("config", {}).get("cotracker", {}).get("checkpoint_sha256"),
                "valid_tracks_by_axis": {
                    axis: directions[f"plus_{axis}"].get("valid_tracks") for axis in axes
                },
                "repeat_consistency_by_axis": {
                    axis: directions[f"plus_{axis}"].get("repeat_consistency_mean_cosine") for axis in axes
                },
            }
            artifacts[view_id] = build_cotracker_direction_artifact(
                profile, request, positive,
                source_sha256=sha256(path.read_bytes()).hexdigest(), quality=quality,
            )
        return artifacts

    def _check_camera_pose(self, summary: dict, view_id: str) -> None:
        view = next((item for item in self.config.views if item.id == view_id), None)
        if view is None:
            raise ValueError(f"{view_id!r} is not a configured Bridge view")
        try:
            camera = summary["config"]["camera"]
            override = camera["runtime_override"]
            if camera.get("source") != self.config.native_camera_id:
                raise ValueError("camera source does not match Bridge config")
            model = summary["config"].get("cotracker", {}).get("model", "")
            if "CoTracker3" not in model:
                raise ValueError("summary is not identified as a CoTracker3 result")
            reference_pose = summary["config"].get("reference_pose", "")
            if "deterministic Bridge reset" not in reference_pose:
                raise ValueError("summary does not declare the expected deterministic Bridge reset reference")
            eye = _parse_vector(override["eye_world"])
            target = _parse_vector(override["target_world"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{view_id}: summary lacks a verifiable Bridge camera pose") from exc
        if not _same_vector(eye, view.eye) or not _same_vector(target, view.target):
            raise ValueError(f"{view_id}: CoTracker summary camera pose differs from workflow config")


def _parse_vector(value: object) -> tuple[float, float, float]:
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("camera pose must be a 3-vector")
    vector = tuple(float(number) for number in value)
    if not all(math.isfinite(number) for number in vector):
        raise ValueError("camera pose must be finite")
    return vector


def _same_vector(left: tuple[float, ...], right: tuple[float, ...], tolerance: float = 1e-5) -> bool:
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))
