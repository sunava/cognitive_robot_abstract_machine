"""
Authored handle frames remain visible throughout browser articulation.
"""

from __future__ import annotations

from enum import StrEnum
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree

from typing_extensions import Any

import pytest

from .test_laboratory_bundle import laboratory_directory


# %% browser element adapter
class ElementField(StrEnum):
    """
    Element attributes consumed by the browser loader's DOM adapter.
    """

    NAME = "nodeName"
    """
    XML tag exposed as a DOM element name.
    """

    ATTRIBUTES = "attributes"
    """
    Parsed attribute values returned by getAttribute.
    """

    CHILDREN = "children"
    """
    Ordered child elements retained from the authored description.
    """


class BrowserFixture(StrEnum):
    """
    Location of parsed XML supplied to the Node test process.
    """

    DESCRIPTION = "CRAMERA_HANDLE_DESCRIPTION"
    """
    Environment variable carrying the parsed description file path.
    """

    FILENAME = "browser-elements.json"
    """
    Temporary DOM representation of the generated laboratory URDF.
    """


def browser_element(element: ElementTree.Element) -> dict[str, Any]:
    """
    Preserve parsed XML elements for the actual browser URDF loader.
    """
    return {
        ElementField.NAME: element.tag,
        ElementField.ATTRIBUTES: element.attrib,
        ElementField.CHILDREN: [browser_element(child) for child in element],
    }


# %% named handle frames
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_laboratory_handle_frames_follow_browser_joints(
    laboratory_directory: Path,
) -> None:
    """
    Load authored empty links and verify their rendered axes follow real joints.
    """
    description = ElementTree.parse(laboratory_directory / "environment.urdf")
    elements = laboratory_directory / BrowserFixture.FILENAME
    elements.write_text(json.dumps(browser_element(description.getroot())))
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_laboratory_handle_frames.js"),
        ],
        env={**os.environ, BrowserFixture.DESCRIPTION: str(elements)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
