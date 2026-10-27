"""
Shared fixtures of the question engine's tests: the recorded scenes of the dataset.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cramera.scenes import SceneCatalog, SceneDirectory, SceneSource

from .dataset.files import SCENES


@pytest.fixture
def scenes(tmp_path: Path) -> Path:
    """
    :return: A copy of the dataset's scenes that tests may change.
    """
    copy = tmp_path / "scenes"
    shutil.copytree(SCENES, copy)
    return copy


@pytest.fixture
def catalog(scenes: Path) -> SceneCatalog:
    """
    :return: A catalog of the dataset's scenes.
    """
    return SceneCatalog([SceneDirectory(scenes, SceneSource.LOCAL)])
