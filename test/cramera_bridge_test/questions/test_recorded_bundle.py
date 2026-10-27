"""
Reading a recorded bundle: which of its models is the recorded robot.
"""

from __future__ import annotations

import json

from cramera_protocol.bundle.scene import SceneDescription
from cramera_bridge.questions.recorded.bundle import RecordedBundle

from .dataset.files import TWO_ROBOTS_SCENE


def test_the_robot_model_is_the_selected_robots():
    scene = SceneDescription.from_json(
        json.loads(TWO_ROBOTS_SCENE.read_text(encoding="utf-8"))
    )
    model = RecordedBundle.robot_model(scene)
    assert model.identifier == scene.robot.identifier
