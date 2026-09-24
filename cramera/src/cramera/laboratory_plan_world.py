"""Apply detached laboratory measurements to a native plan's semantic world."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import mujoco
from typing_extensions import Protocol, Self

from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    WheeledDrive,
)


# %% detached measurement contract
class MeasuredLaboratory(Protocol):
    """Measured articulation available on the thread owning laboratory physics."""

    model: mujoco.MjModel
    """Compiled physical bodies and scalar joints."""
    data: mujoco.MjData
    """Current measured coordinates and world transforms."""

    def robot_state(self) -> dict[str, float]:
        """Return actual robot coordinates including fixed recorded articulation."""


@dataclass(frozen=True)
class LaboratoryJointMeasurement:
    """One named scalar joint observed at the capture instant."""

    joint_name: str
    """Description joint name shared by physical and semantic models."""
    position: float
    """Measured joint angle or displacement in SI units."""


@dataclass(frozen=True)
class LaboratoryPoseMeasurement:
    """An immutable body transform expressed in laboratory world coordinates."""

    body_name: str
    """Description body name shared by physical and semantic models."""
    position: tuple[float, float, float]
    """World translation in metres."""
    orientation: tuple[float, float, float, float]
    """World rotation in XYZW quaternion order."""


@dataclass(frozen=True)
class LaboratoryPlanWorld:
    """Detached physical state that can update a separately owned semantic world."""

    simulation_time: float
    """Physical simulation time at which all values were captured."""
    joints: tuple[LaboratoryJointMeasurement, ...]
    """Robot and instrument scalar coordinates copied from the authoritative state."""
    poses: tuple[LaboratoryPoseMeasurement, ...]
    """Body transforms copied without retaining references to physical arrays."""

    @classmethod
    def capture(cls, physics: MeasuredLaboratory) -> Self:
        """Copy measured state while the caller owns the physics stepping lock."""
        positions = physics.robot_state().copy()
        for index in range(physics.model.njnt):
            if physics.model.jnt_type[index] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            name = physics.model.joint(index).name
            if name is not None:
                positions[name] = float(
                    physics.data.qpos[physics.model.jnt_qposadr[index]]
                )
        return cls(
            simulation_time=float(physics.data.time),
            joints=tuple(
                LaboratoryJointMeasurement(name, float(position))
                for name, position in positions.items()
            ),
            poses=tuple(
                LaboratoryPoseMeasurement(
                    body_name=physics.model.body(index).name,
                    position=tuple(float(value) for value in physics.data.xpos[index]),
                    orientation=(
                        *(float(value) for value in physics.data.xquat[index, 1:]),
                        float(physics.data.xquat[index, 0]),
                    ),
                )
                for index in range(1, physics.model.nbody)
                if physics.model.body(index).name is not None
            ),
        )

    def apply(self, world: World) -> None:
        """Update matching semantic coordinates without modifying model geometry."""
        measured_joints = {joint.joint_name: joint.position for joint in self.joints}
        updated_degrees: set[UUID] = set()
        for connection in world.get_connections_by_type(ActiveConnection1DOF):
            name = str(connection.name)
            if name not in measured_joints or connection.raw_dof.id in updated_degrees:
                continue
            connection.position = measured_joints[name]
            updated_degrees.add(connection.raw_dof.id)

        measured_poses = {pose.body_name: pose for pose in self.poses}
        floating_connections = sorted(
            (
                connection
                for connection in world.connections
                if isinstance(connection, (Connection6DoF, WheeledDrive))
            ),
            key=lambda connection: len(
                world.compute_chain_of_connections(world.root, connection.child)
            ),
        )
        for connection in floating_connections:
            name = str(connection.child.name)
            if name not in measured_poses:
                continue
            measured = measured_poses[name]
            world_T_child = Pose.from_xyz_quaternion(
                *measured.position,
                *measured.orientation,
                reference_frame=world.root,
            ).to_homogeneous_matrix()
            connection.origin = world.transform(world_T_child, connection.parent)
