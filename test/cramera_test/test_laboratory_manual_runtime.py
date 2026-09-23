"""
Manual revisions change physical start eligibility without moving samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from cramera.laboratory_centrifuge import InvalidCentrifugeOperation
from cramera.laboratory_centrifuge_command import CentrifugeAction, CentrifugeField
from cramera.laboratory_manual import CentrifugeManual, InvalidCentrifugeManual
from cramera.laboratory_physics import LaboratoryPhysics
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_centrifuge import load_pair, seat_tube
from .test_laboratory_physics import physics


# %% authoritative manual gate
def test_manual_revision_changes_start_for_the_same_balanced_pair(
    physics: LaboratoryPhysics,
) -> None:
    """
    The document changes eligibility while all physical state stays identical.
    """
    load_pair(physics)
    physics.centrifuge_command(CentrifugeAction.CLOSE)
    physics.step(5000)
    before = physics.data.qpos.copy()
    original = physics.centrifuge.snapshot()[CentrifugeField.SLOTS]
    physics.read_centrifuge_manual(CentrifugeManual.builtin("v2").document)
    np.testing.assert_array_equal(physics.data.qpos, before)
    assert physics.centrifuge.snapshot()[CentrifugeField.SLOTS] == original
    with pytest.raises(InvalidCentrifugeOperation):
        physics.centrifuge_command(CentrifugeAction.START)
    assert physics.centrifuge.target_rpm == 0
    physics.read_centrifuge_manual(CentrifugeManual.builtin("v1").document)
    physics.centrifuge_command(CentrifugeAction.START)
    assert physics.centrifuge.target_rpm == physics.centrifuge.parameters.default_rpm
    np.testing.assert_array_equal(physics.data.qpos, before)


def test_second_pair_satisfies_revised_manual(physics: LaboratoryPhysics) -> None:
    """
    Four physically secured balanced samples satisfy the stricter document.
    """
    physics.read_centrifuge_manual(CentrifugeManual.builtin("v2").document)
    load_pair(physics)
    for key, slot in (
        (LaboratoryBody.TEAL_TUBE, "C2"),
        (LaboratoryBody.RESERVE_TUBE, "C4"),
    ):
        physics.fill_liquid(key, 5.0)
        seat_tube(physics, key, slot)
        physics.centrifuge_command(CentrifugeAction.SECURE, slot=slot, key=key)
    physics.centrifuge_command(CentrifugeAction.CLOSE)
    physics.step(5000)
    physics.centrifuge_command(CentrifugeAction.START)
    assert physics.centrifuge.target_rpm == physics.centrifuge.parameters.default_rpm


def test_manual_cannot_change_during_rotor_cycle(physics: LaboratoryPhysics) -> None:
    """
    A running drive retains the validated manual governing its current cycle.
    """
    load_pair(physics)
    physics.centrifuge_command(CentrifugeAction.CLOSE)
    physics.step(5000)
    physics.centrifuge_command(CentrifugeAction.START)
    before = physics.centrifuge.knowledge.snapshot()
    with pytest.raises(InvalidCentrifugeOperation):
        physics.read_centrifuge_manual(CentrifugeManual.builtin("v2").document)
    assert physics.centrifuge.knowledge.snapshot() == before


def test_reset_preserves_read_manual(physics: LaboratoryPhysics) -> None:
    """
    Resetting physical objects does not erase acquired operating knowledge.
    """
    physics.read_centrifuge_manual(CentrifugeManual.builtin("v2").document)
    before = physics.centrifuge.knowledge.snapshot()
    physics.reset()
    assert physics.centrifuge.knowledge.snapshot() == before


def test_invalid_document_preserves_knowledge_and_physics(
    physics: LaboratoryPhysics,
) -> None:
    """
    A rejected document cannot partially replace valid operating knowledge.
    """
    physics.read_centrifuge_manual(CentrifugeManual.builtin("v2").document)
    before = physics.centrifuge.knowledge.snapshot()
    coordinates = physics.data.qpos.copy()
    with pytest.raises(InvalidCentrifugeManual):
        physics.read_centrifuge_manual("")
    assert physics.centrifuge.knowledge.snapshot() == before
    np.testing.assert_array_equal(physics.data.qpos, coordinates)
