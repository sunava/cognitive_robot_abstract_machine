"""
Instrument targets drive bounded physical joints without pose substitution.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import mujoco
import numpy as np
import pytest

from cramera.medical_lab_description import MedicalLabDescription
from cramera.medical_lab_dynamics import MedicalLabCollision, MedicalLabDrives


# %% independent physical hinge
@pytest.fixture()
def instrument_drives() -> MedicalLabDrives:
    """
    Compile a small authored hinge with the production instrument importer.
    """
    directory = Path(__file__).parent / "dataset" / "medical_lab"
    description = MedicalLabDescription.from_json(
        json.loads((directory / "description.json").read_text())
    )
    source = ElementTree.parse(directory / "hinge.urdf").getroot()
    root = ElementTree.Element("mujoco")
    ElementTree.SubElement(root, "compiler", angle="radian")
    ElementTree.SubElement(root, "option", timestep="0.001", integrator="implicitfast")
    world = ElementTree.SubElement(root, "worldbody")
    base = ElementTree.SubElement(world, "body", name="base")
    body = ElementTree.SubElement(base, "body", name="lid", pos="0 0 1")
    collision = MedicalLabCollision(description)
    collision.add_joint(body, source.find("link[@name='lid']"), source.find("joint"))
    collision.complete(root)
    model = mujoco.MjModel.from_xml_string(
        ElementTree.tostring(root, encoding="unicode")
    )
    data = mujoco.MjData(model)
    return MedicalLabDrives(model, data, description)


class TestMeasuredInstrumentMotion:
    """
    Setpoints change only actuator demand; MuJoCo determines the observed pose.
    """

    def test_target_does_not_assign_joint_position(self, instrument_drives) -> None:
        """
        An inspection command leaves physical state unchanged until stepped.
        """
        joint = instrument_drives.description.instruments[0].joints[0]
        before = instrument_drives.data.qpos.copy()
        instrument_drives.set_targets({joint.name: joint.inspection})
        np.testing.assert_array_equal(instrument_drives.data.qpos, before)

    def test_only_mechanical_mount_contact_is_excluded(self, instrument_drives) -> None:
        """
        The joint's mounting surfaces cannot eject the connected moving body.
        """
        assert instrument_drives.model.nexclude == 1

    def test_joint_reaches_measured_target_with_bounded_force(
        self, instrument_drives
    ) -> None:
        """
        A complete physical motion stays within its declared effort limit.
        """
        joint = instrument_drives.description.instruments[0].joints[0]
        instrument_drives.set_targets({joint.name: joint.inspection})
        peak = 0.0
        for _ in range(3000):
            instrument_drives.advance()
            mujoco.mj_step(instrument_drives.model, instrument_drives.data)
            peak = max(
                peak, float(np.max(np.abs(instrument_drives.data.actuator_force)))
            )
        assert instrument_drives.positions()[joint.name] == pytest.approx(
            joint.inspection, abs=0.008
        )
        assert peak <= joint.effort
        assert instrument_drives.reached({joint.name: joint.inspection})

    def test_invalid_batch_does_not_change_existing_targets(
        self, instrument_drives
    ) -> None:
        """
        A malformed multi-joint request cannot leave a partially applied target.
        """
        joint = instrument_drives.description.instruments[0].joints[0]
        before = instrument_drives.targets.copy()
        with pytest.raises(ValueError):
            instrument_drives.set_targets({joint.name: joint.inspection, "absent": 0.1})
        assert instrument_drives.targets == before

    def test_outside_limit_is_rejected(self, instrument_drives) -> None:
        """
        Commands preserve the authored physical joint range.
        """
        joint = instrument_drives.description.instruments[0].joints[0]
        with pytest.raises(ValueError):
            instrument_drives.set_targets({joint.name: joint.upper + 1.0})

    def test_hold_uses_measured_position(self, instrument_drives) -> None:
        """
        Pausing removes the remaining motion demand without teleporting a joint.
        """
        joint = instrument_drives.description.instruments[0].joints[0]
        instrument_drives.set_targets({joint.name: joint.inspection})
        for _ in range(200):
            instrument_drives.advance()
            mujoco.mj_step(instrument_drives.model, instrument_drives.data)
        before = instrument_drives.positions()
        instrument_drives.hold()
        assert instrument_drives.targets == before
        assert instrument_drives.positions() == before
