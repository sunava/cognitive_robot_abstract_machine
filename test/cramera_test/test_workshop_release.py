"""
Reject obsolete or incomplete workshop recordings before publication.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cramera.workshop_release import InvalidWorkshopRelease, WorkshopRelease


# %% complete recording fixture
@pytest.fixture
def recording(tmp_path: Path) -> Path:
    """
    Create one frame for every required successful native action.
    """
    steps = [f"Action {index + 1}" for index in range(10)]
    metrics = {
        "success": True,
        "action_statuses": ["SUCCEEDED"] * len(steps),
        "box_position_error": 0.001,
        "screw_depth": 0.11,
        "required_depth": 0.11,
        "driver_released": True,
        "support_released": True,
        "minimum_collision_distance": 0.0,
        "frames": len(steps),
        "contentsPhysics": {
            "model": "one-way-relative-contact-contents",
            "ticks": 1,
            "simulated_seconds": 0.6,
            "maximum_box_position_lag": 0.01,
            "maximum_box_angular_lag": 0.01,
            "maximum_relative_displacement": 0.001,
            "maximum_projection_correction": 0.01,
            "retained_contents": True,
            "escaped_contents": [],
            "contents_count": 5,
        },
    }
    scene = {
        "name": "aicor_workshop_pr2",
        "framesPerSecond": 50.0,
        "trajectory": "trajectory.json",
        "workshopDemo": {
            "kind": "native-recording",
            "steps": steps,
            "stepRanges": [
                {"name": name, "startFrame": index, "endFrame": index}
                for index, name in enumerate(steps)
            ],
            "result": metrics,
        },
    }
    objects = {
        key: [0, 0, 0, 0, 0, 0, 1]
        for key in [
            "screw_box",
            "driver",
            *[f"screw_{index + 1}" for index in range(5)],
        ]
    }
    trajectory = {
        "framesPerSecond": scene["framesPerSecond"],
        "frames": [{"robot_joint": 0.0} for _ in steps],
        "base": [[0, 0, 0, 0, 0, 0, 1] for _ in steps],
        "objects": [objects for _ in steps],
    }
    for name, data in (
        ("scene.json", scene),
        ("acceptance_metrics.json", metrics),
        ("trajectory.json", trajectory),
    ):
        (tmp_path / name).write_text(json.dumps(data))
    return tmp_path


def replace_metrics(directory: Path, field: str, value: object) -> None:
    """
    Keep the two acceptance documents consistent while testing one criterion.
    """
    metrics_path = directory / "acceptance_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics[field] = value
    metrics_path.write_text(json.dumps(metrics))
    scene_path = directory / "scene.json"
    scene = json.loads(scene_path.read_text())
    scene["workshopDemo"]["result"] = metrics
    scene_path.write_text(json.dumps(scene))


# %% acceptance and recording agreement
def test_complete_ten_action_recording_is_accepted(recording: Path) -> None:
    """
    Allow a complete finite recording with retained contents and matching stages.
    """
    WorkshopRelease(recording).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_statuses", ["SUCCEEDED", "SUCCEEDED"]),
        ("action_statuses", ["SUCCEEDED"] * 9 + ["FAILED"]),
        ("box_position_error", float("nan")),
        ("minimum_collision_distance", float("inf")),
        ("minimum_collision_distance", -0.002),
        ("screw_depth", 0.04),
        ("driver_released", False),
        ("support_released", False),
        ("frames", 11),
    ],
)
def test_success_flag_cannot_override_failed_acceptance(
    recording: Path, field: str, value: object
) -> None:
    """
    Recheck measured results instead of trusting a stale success flag.
    """
    replace_metrics(recording, field, value)
    with pytest.raises(InvalidWorkshopRelease):
        WorkshopRelease(recording).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retained_contents", False),
        ("escaped_contents", ["screw_5"]),
        ("contents_count", 4),
        ("maximum_box_position_lag", float("nan")),
    ],
)
def test_all_five_contents_must_have_valid_retention_evidence(
    recording: Path, field: str, value: object
) -> None:
    """
    Reject dropped, unmeasured or invalidly simulated contents.
    """
    metrics = json.loads((recording / "acceptance_metrics.json").read_text())
    contents = metrics["contentsPhysics"]
    contents[field] = value
    replace_metrics(recording, "contentsPhysics", contents)
    with pytest.raises(InvalidWorkshopRelease):
        WorkshopRelease(recording).validate()


@pytest.mark.parametrize(
    "mutation", ["kind", "range", "result", "rate", "missing_screw"]
)
def test_replay_metadata_and_frames_must_match_acceptance(
    recording: Path, mutation: str
) -> None:
    """
    Reject mixed exports, incorrect playback ranges and missing object states.
    """
    scene_path = recording / "scene.json"
    scene = json.loads(scene_path.read_text())
    trajectory_path = recording / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text())
    if mutation == "kind":
        scene["workshopDemo"]["kind"] = "keyframes"
    elif mutation == "range":
        scene["workshopDemo"]["stepRanges"][-1]["endFrame"] += 1
    elif mutation == "result":
        scene["workshopDemo"]["result"]["box_position_error"] = 0.0
    elif mutation == "rate":
        trajectory["framesPerSecond"] = 21.23
    else:
        del trajectory["objects"][-1]["screw_5"]
    scene_path.write_text(json.dumps(scene))
    trajectory_path.write_text(json.dumps(trajectory))
    with pytest.raises(InvalidWorkshopRelease):
        WorkshopRelease(recording).validate()
