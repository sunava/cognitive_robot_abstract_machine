"""
Fixtures of the scene tests.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ..dataset.bundles import ANNOTATED_BUNDLE


@pytest.fixture
def annotated_bundle(tmp_path) -> Path:
    """
    :return: A copy of the converted bundle that tests may change.
    """
    return Path(shutil.copytree(ANNOTATED_BUNDLE, tmp_path / ANNOTATED_BUNDLE.name))
