"""
Standard floating URDF joints preserve authored frames and passive motion.
"""

from pathlib import Path

import numpy as np
import pytest

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF


# %% passive floating joints
@pytest.mark.parametrize("prefix", ["", "laboratory"])
def test_floating_joint_preserves_authored_origin_and_name(prefix: str) -> None:
    """
    A floating joint needs no axis and retains its origin and fixed descendants.
    """
    parser = URDFParser.from_file(
        str(Path(__file__).parent / "dataset/floating_carrier.urdf"), prefix=prefix
    )
    world = parser.parse()
    world.validate()
    authored = next(joint for joint in parser.parsed.joints if joint.type == "floating")
    connection = world.get_connection_by_name(PrefixedName(authored.name, prefix))
    assert isinstance(connection, Connection6DoF)
    assert connection.name == PrefixedName(authored.name, prefix)
    assert connection.parent.name == PrefixedName(authored.parent, prefix)
    assert connection.child.name == PrefixedName(authored.child, prefix)
    assert len(connection.passive_dofs) == 7
    parent_T_carrier = HomogeneousTransformationMatrix.from_xyz_rpy(
        *authored.origin.xyz, *authored.origin.rpy
    )
    np.testing.assert_allclose(
        connection.origin.to_np(), parent_T_carrier.to_np(), atol=1e-12
    )
    fixed = next(
        joint for joint in parser.parsed.joints if joint.parent == authored.child
    )
    handle = world.get_body_by_name(PrefixedName(fixed.child, prefix))
    carrier_T_handle = HomogeneousTransformationMatrix.from_xyz_rpy(
        *fixed.origin.xyz, *fixed.origin.rpy
    )
    np.testing.assert_allclose(
        handle.global_pose.to_np(),
        connection.parent.global_pose.to_np()
        @ parent_T_carrier.to_np()
        @ carrier_T_handle.to_np(),
        atol=1e-12,
    )


def test_floating_joint_accepts_measured_pose_without_moving_its_parent() -> None:
    """
    A measured carrier transform updates its descendants through passive coordinates.
    """
    parser = URDFParser.from_file(
        str(Path(__file__).parent / "dataset/floating_carrier.urdf")
    )
    world = parser.parse()
    authored = next(joint for joint in parser.parsed.joints if joint.type == "floating")
    connection = world.get_connection_by_name(authored.name)
    before = connection.parent.global_pose.to_np().copy()
    parent_T_carrier = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.1, 0.3, 0.4, 0.2, -0.1, 0.6, reference_frame=connection.parent
    )
    connection.origin = parent_T_carrier
    np.testing.assert_allclose(
        connection.origin.to_np(), parent_T_carrier.to_np(), atol=1e-12
    )
    np.testing.assert_array_equal(connection.parent.global_pose.to_np(), before)
    np.testing.assert_allclose(
        connection.child.global_pose.to_np(),
        before @ parent_T_carrier.to_np(),
        atol=1e-12,
    )
