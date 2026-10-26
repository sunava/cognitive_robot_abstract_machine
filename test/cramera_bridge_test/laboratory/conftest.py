"""
The laboratory asset bundle used by the bridge's native integration tests.
"""

import os
from pathlib import Path

import pytest

BUNDLE_VARIABLE = "CRAMERA_LABORATORY_BUNDLE"
"""
The environment variable naming the laboratory bundle, which ships with the laboratory
plugin of the cramera repository and not with this workspace.
"""


@pytest.fixture(scope="module")
def laboratory_bundle() -> Path:
    """
    :return: The small laboratory bundle, independent of installing its plugin.
    """
    bundle = Path(os.environ.get(BUNDLE_VARIABLE, "")).expanduser()
    if not os.environ.get(BUNDLE_VARIABLE) or not bundle.is_dir():
        pytest.skip(f"{BUNDLE_VARIABLE} does not name the laboratory bundle")
    return bundle
