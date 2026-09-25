"""
The authored medical laboratory uses its canonical portable resource name.
"""

from pathlib import Path

import pytest

from cramera import laboratory_world, paths
from cramera.laboratory_physics_session import LaboratoryPhysicsSession
from cramera.laboratory_world import LaboratoryAsset, LaboratoryWorld


# %% isolated packaged resource resolution
@pytest.fixture
def packaged_medical_laboratory() -> Path:
    """
    Locate the renamed portable bundle independently of user scene caches.
    """
    return (
        Path(laboratory_world.__file__).resolve().parent
        / LaboratoryAsset.RESOURCES
        / "aicor_medical_lab"
    )


def test_default_world_resolves_packaged_medical_laboratory(
    fixture_scene: Path, packaged_medical_laboratory: Path
) -> None:
    """
    Empty isolated scene roots select the installed medical laboratory bundle.
    """
    assert LaboratoryWorld.resolve_directory() == packaged_medical_laboratory
    assert LaboratoryWorld().bundle_directory == packaged_medical_laboratory


def test_contact_session_source_resolves_medical_laboratory(
    fixture_scene: Path,
    packaged_medical_laboratory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A source scene catalog exposes the renamed bundle to the contact session.
    """
    monkeypatch.setenv("CRAMERA_SCENES", str(packaged_medical_laboratory.parent))
    assert paths.resolve_scene_directory(LaboratoryPhysicsSession.SOURCE_SCENE) == (
        packaged_medical_laboratory
    )
    assert LaboratoryPhysicsSession.SOURCE_SCENE == LaboratoryAsset.SCENE_NAME
