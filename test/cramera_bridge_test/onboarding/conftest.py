"""
Fixtures of the onboarding tests.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from cramera.locations import DataLocations

from ..dataset.onboarding.sliding_box_demo import SlidingBoxWorld

ONBOARDING_DATASET = Path(__file__).parent.parent / "dataset" / "onboarding"
"""
The demo files the onboarding tests run.
"""

SLIDING_BOX_DEMO = ONBOARDING_DATASET / "sliding_box_demo.py"
"""
A demo showing its world through coraplex's visualization.
"""

CORAPLEX_VISUALIZATION_MODULE = "coraplex.visualization"
"""
The module of coraplex's world visualization.
"""


@pytest.fixture
def locations(tmp_path: Path) -> DataLocations:
    """
    :return: Data locations in a temporary directory, with a shared scenes directory.
    """
    return DataLocations(data=tmp_path / "data", shared_scenes=tmp_path / "shared")


@pytest.fixture
def sliding_box() -> SlidingBoxWorld:
    """
    :return: A new world of a box above a table.
    """
    return SlidingBoxWorld.build()


@pytest.fixture
def coraplex_visualization() -> ModuleType:
    """
    Skip a test that runs coraplex's world visualization where coraplex cannot import
    it: it needs the Rerun SDK, which the CRAM workspace does not always install.

    :return: coraplex's visualization module.
    """
    return pytest.importorskip(CORAPLEX_VISUALIZATION_MODULE)
