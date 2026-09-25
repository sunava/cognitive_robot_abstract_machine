"""
Standalone Blender generation preserves portable workshop attribution.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from .test_laboratory_asset_rights import AssetDocument, rights_module


# %% isolated Blender authoring
@pytest.fixture
def authored_workshop(tmp_path: Path) -> Path:
    """
    Generate new meshes without touching packaged models or editable sources.
    """
    blender = shutil.which("blender")
    if blender is None:
        pytest.skip("Blender is unavailable")
    builder = (
        Path(__file__).resolve().parents[2] / "cramera/tools/workshop/build_blender.py"
    )
    output = tmp_path / "export"
    result = subprocess.run(
        [
            blender,
            "--background",
            "--python-exit-code",
            "1",
            "--python",
            str(builder),
            "--",
            "--output",
            str(output),
            "--source",
            str(tmp_path / "source"),
            "--no-render",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return output


# %% notices survive direct regeneration
def test_standalone_authoring_retains_workshop_asset_rights(
    authored_workshop: Path, rights_module: ModuleType
) -> None:
    """
    Fresh standalone exports carry the workshop notices and embedded attribution.
    """
    canonical = (
        Path(__file__).resolve().parents[2] / "cramera/tools/workshop/asset_rights"
    )
    rights = json.loads(
        (canonical / rights_module.RightsFile.METADATA).read_text(encoding="utf-8")
    )
    for filename in rights_module.RightsFile:
        assert (authored_workshop / filename).read_bytes() == (
            canonical / filename
        ).read_bytes()
    meshes = sorted(
        (authored_workshop / rights_module.PackagePath.ASSETS).glob(
            rights_module.PackagePath.GLBS
        )
    )
    assert meshes
    for path in meshes:
        asset = AssetDocument.read(path).document[rights_module.AssetField.ASSET]
        assert (
            asset[rights_module.AssetField.COPYRIGHT]
            == rights[rights_module.RightsField.COPYRIGHT]
        )
        embedded = asset[rights_module.AssetField.EXTRAS][
            rights_module.AssetField.RIGHTS
        ]
        for field in rights_module.RightsField:
            if field in (
                rights_module.RightsField.LICENSE_FILE,
                rights_module.RightsField.NOTICE_FILE,
            ):
                assert (path.parent / embedded[field]).resolve().read_bytes() == (
                    canonical / rights[field]
                ).read_bytes()
            else:
                assert embedded[field] == rights[field]
