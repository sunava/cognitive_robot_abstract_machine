"""
Explicit authored worlds retain the existing laboratory session contract.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from cramera import laboratory_physics_session
from cramera.laboratory_physics_session import LaboratoryPhysicsSession, PhysicsState

from .test_laboratory_physics_session import ControlledContactSimulation, physics_bundle


# %% configurable authored world
def test_explicit_source_is_used_instead_of_default_scene(
    physics_bundle: Path, tmp_path: Path, monkeypatch
) -> None:
    """
    A separately authored room can reuse the calibrated simulation session.
    """
    expanded = tmp_path / "expanded"
    shutil.copytree(physics_bundle, expanded)
    monkeypatch.setattr(
        laboratory_physics_session, "create_physics", ControlledContactSimulation
    )
    session = LaboratoryPhysicsSession(
        physics_bundle.parents[1], source_directory=expanded
    )
    try:
        assert session.start()["state"] == PhysicsState.RUNNING
        assert session.physics.bundle_directory == expanded
    finally:
        session.stop()


def test_missing_explicit_source_does_not_fall_back(
    physics_bundle: Path, tmp_path: Path, monkeypatch
) -> None:
    """
    A configured missing room reports failure without silently loading another.
    """
    monkeypatch.setattr(
        laboratory_physics_session, "create_physics", ControlledContactSimulation
    )
    session = LaboratoryPhysicsSession(
        physics_bundle.parents[1], source_directory=tmp_path / "absent"
    )
    try:
        assert session.start()["state"] == PhysicsState.FAILED
    finally:
        session.stop()
