"""
Exercise native rigid cable construction and semantic pose synchronization.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from cramera.cable import (
    Cable,
    CableConfig,
    CableMujocoBuilder,
    CableMujocoSynchronizer,
)
from physics_simulators.mujoco_simulator import MujocoSimulator
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.geometry import Sphere


# %% native cable properties
def test_native_links_have_total_mass_and_ball_joints() -> None:
    """
    An unanchored cable has one free base and rigid ball-connected elements.
    """
    cable = Cable.create(World(), CableConfig(anchor_to_parent=False))
    specification = mujoco.MjSpec()
    model = specification.compile()
    compiled = cable.install(specification, model, mujoco.MjData(model))
    assert compiled.success
    assert compiled.model.nbody == cable.config.segment_count + 1
    assert compiled.model.neq == 0
    assert compiled.model.nflex == 0
    assert compiled.model.jnt_type.tolist() == [mujoco.mjtJoint.mjJNT_FREE] + [
        mujoco.mjtJoint.mjJNT_BALL
    ] * (cable.config.segment_count - 1)
    assert compiled.model.body_mass.sum() == pytest.approx(cable.config.mass)
    assert (
        tuple(
            compiled.model.body(index + 1).name
            for index in range(cable.config.segment_count)
        )
        == cable.physical_body_names
    )


def test_link_lengths_remain_fixed_under_gravity() -> None:
    """
    The native chain preserves adjacent link distances while it bends.
    """
    cable = Cable.create(World(), CableConfig(anchor_offset=(0, 0, 1)))
    specification = mujoco.MjSpec()
    model = specification.compile()
    compiled = cable.install(specification, model, mujoco.MjData(model))
    assert compiled.success
    for _ in range(400):
        mujoco.mj_step(compiled.model, compiled.data)
    positions = np.array(
        [compiled.data.body(name).xpos for name in cable.physical_body_names]
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(positions, axis=0), axis=1),
        cable.config.segment_length,
        atol=1e-7,
    )
    np.testing.assert_allclose(positions[0], cable.config.anchor_offset, atol=1e-10)
    assert positions[-1, 2] < cable.config.anchor_offset[2]


def test_parent_pose_and_local_offset_determine_spawn(
    world_with_two_bodies: tuple[World, Body, Body],
) -> None:
    """
    Local anchor offsets rotate with the parent instead of world axes.
    """
    world, parent, _ = world_with_two_bodies
    with world.modify_world():
        world.add_kinematic_structure_entity(parent)
    config = CableConfig(anchor_offset=(0.1, 0, 0.6), anchor_rpy=(0, 0, np.pi / 2))
    cable = Cable.create(world, config, parent)
    specification = mujoco.MjSpec()
    specification.worldbody.add_body(name=parent.name.name)
    model = specification.compile()
    compiled = cable.install(specification, model, mujoco.MjData(model))
    assert compiled.success
    mujoco.mj_forward(compiled.model, compiled.data)
    np.testing.assert_allclose(
        compiled.data.body(cable.physical_body_names[0]).xpos,
        config.anchor_offset,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        compiled.data.body(cable.physical_body_names[1]).xpos
        - compiled.data.body(cable.physical_body_names[0]).xpos,
        [0, config.segment_length, 0],
        atol=1e-7,
    )


# %% semantic placeholders and degraded builds
def test_new_semantic_points_publish_their_initial_pose() -> None:
    """
    Forward kinematics immediately exposes the configured spawn height.
    """
    cable = Cable.create(World(), CableConfig(anchor_offset=(0.2, 0, 0.35)))
    transform = cable.world.compute_forward_kinematics(
        cable.world.root, cable.placeholders[0]
    ).to_np()
    np.testing.assert_allclose(transform[:3, 3], cable.config.anchor_offset)


def test_added_coordinates_start_from_native_defaults() -> None:
    """
    New cable coordinates initialize correctly beside an already compiled rig.
    """
    cable = Cable.create(
        World(), CableConfig(anchor_to_parent=False, anchor_offset=(0.2, 0, 0.35))
    )
    specification = mujoco.MjSpec()
    body = specification.worldbody.add_body(name="rig")
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.02, 0, 0), mass=1)
    body.add_joint(name="rig_slide", type=mujoco.mjtJoint.mjJNT_SLIDE)
    model = specification.compile()
    data = mujoco.MjData(model)
    model, data = specification.recompile(model, data)
    compiled = cable.install(specification, model, data)
    assert compiled.success
    np.testing.assert_allclose(
        compiled.data.body(cable.physical_body_names[0]).xpos,
        cable.config.anchor_offset,
        atol=1e-7,
    )


def test_added_coordinates_initialize_with_existing_home_keyframe(
    tmp_path: Path,
) -> None:
    """
    Extending a keyed framework scene does not reuse unrelated joint coordinates.
    """
    cable = Cable.create(
        World(), CableConfig(anchor_to_parent=False, anchor_offset=(0.2, 0, 0.35))
    )
    output = tmp_path / "keyframe.xml"
    CableMujocoBuilder().build_world(cable.world, str(output))
    specification = mujoco.MjSpec.from_file(str(output))
    model = specification.compile()
    data = mujoco.MjData(model)
    body = specification.worldbody.add_body(name="rig")
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.02, 0, 0), mass=1)
    body.add_joint(name="rig_slide", type=mujoco.mjtJoint.mjJNT_SLIDE)
    model, data = specification.recompile(model, data)
    compiled = cable.install(specification, model, data)
    assert compiled.success
    np.testing.assert_allclose(
        compiled.data.body(cable.physical_body_names[0]).xpos,
        cable.config.anchor_offset,
        atol=1e-7,
    )


def test_cable_synchronizer_lifecycle(tmp_path: Path) -> None:
    """
    The callback owner can register, synchronize, and stop without leaking hooks.
    """
    cable = Cable.create(World(), CableConfig(anchor_offset=(0, 0, 1)))
    output = tmp_path / "synchronizer.xml"
    CableMujocoBuilder().build_world(cable.world, str(output))
    simulator = MujocoSimulator(file_path=str(output), _headless=True)
    synchronizer = CableMujocoSynchronizer(_world=cable.world, simulator=simulator)
    compiled = cable.install(
        simulator._mj_spec, simulator._mj_model, simulator._mj_data
    )
    simulator._mj_spec, simulator._mj_model, simulator._mj_data = (
        compiled.spec,
        compiled.model,
        compiled.data,
    )
    synchronizer._sim_to_world()
    synchronizer.stop()
    assert "read_data_from_simulator" not in simulator.__dict__


def test_rotated_parent_transforms_local_translation_and_direction(
    world_with_two_bodies: tuple[World, Body, Body],
) -> None:
    """
    A rotated parent rotates both the cable offset and its local x direction.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        world.add_kinematic_structure_entity(parent)
    with world.modify_world():
        world.add_kinematic_structure_entity(child)
        world.add_connection(
            FixedConnection(
                parent=parent,
                child=child,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=1, z=0.5, yaw=np.pi / 2
                ),
            )
        )
    config = CableConfig(anchor_offset=(0.2, 0, 0))
    cable = Cable.create(world, config, child)
    specification = mujoco.MjSpec()
    specification.worldbody.add_body(
        name=child.name.name, pos=(1, 0, 0.5), quat=(np.sqrt(0.5), 0, 0, np.sqrt(0.5))
    )
    model = specification.compile()
    compiled = cable.install(specification, model, mujoco.MjData(model))
    assert compiled.success
    np.testing.assert_allclose(
        compiled.data.body(cable.physical_body_names[0]).xpos, [1, 0.2, 0.5], atol=1e-8
    )
    np.testing.assert_allclose(
        compiled.data.body(cable.physical_body_names[1]).xpos,
        [1, 0.2 + config.segment_length, 0.5],
        atol=1e-7,
    )


def test_recompile_preserves_existing_scene_state_and_contact_parameters() -> None:
    """
    Adding a cable preserves moving objects and carries configured contact values.
    """
    cable = Cable.create(
        World(),
        CableConfig(anchor_to_parent=False, friction=(0.3, 0.004, 0.0002), condim=4),
    )
    specification = mujoco.MjSpec()
    body = specification.worldbody.add_body(name="existing_body", pos=(2, 0, 1))
    body.add_freejoint(name="existing_joint")
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.03, 0, 0), mass=1)
    model = specification.compile()
    data = mujoco.MjData(model)
    data.qpos[0] = 2.3
    data.qvel[0] = 0.7
    data.time = 0.1
    compiled = cable.install(specification, model, data)
    assert compiled.success
    assert compiled.data.qpos[0] == data.qpos[0]
    assert compiled.data.qvel[0] == data.qvel[0]
    assert compiled.data.time == data.time
    for name in cable.physical_body_names:
        geom = compiled.model.body(name).geomadr[0]
        np.testing.assert_allclose(
            compiled.model.geom_friction[geom], cable.config.friction
        )
        assert compiled.model.geom_condim[geom] == cable.config.condim


def test_placeholder_only_world_builds_without_duplicate_physical_links(
    tmp_path: Path,
) -> None:
    """
    Free semantic points are excluded from the MuJoCo mass and coordinates.
    """
    cable = Cable.create(World())
    output = tmp_path / "cable.xml"
    CableMujocoBuilder().build_world(cable.world, str(output))
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.nbody == 1
    assert model.nq == 0
    assert len(cable.placeholders) == cable.config.segment_count
    assert all(
        isinstance(body.parent_connection, Connection6DoF)
        for body in cable.placeholders
    )


def test_world_builder_preserves_existing_free_body_pose(
    tmp_path: Path, world_with_two_bodies: tuple[World, Body, Body]
) -> None:
    """
    Filtering points preserves native free-joint coordinate order and placement.
    """
    world, parent, child = world_with_two_bodies
    child.collision.shapes.append(Sphere(radius=0.03))
    with world.modify_world():
        world.add_kinematic_structure_entity(parent)
    with world.modify_world():
        world.add_kinematic_structure_entity(child)
        connection = Connection6DoF.create_with_dofs(world, parent, child)
        world.add_connection(connection)
    connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.2, z=0.4, yaw=0.3, reference_frame=parent
    )
    world.notify_state_change()
    expected = world.compute_forward_kinematics(parent, child).to_np()
    Cable.create(world)
    output = tmp_path / "existing-body.xml"
    CableMujocoBuilder().build_world(world, str(output))
    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body(child.name.name).xpos, expected[:3, 3])
    np.testing.assert_allclose(
        data.body(child.name.name).xmat.reshape(3, 3), expected[:3, :3]
    )


def test_builder_root_normalization_keeps_point_connections_current(
    tmp_path: Path, world_with_two_bodies: tuple[World, Body, Body]
) -> None:
    """
    Points remain writable after the framework normalizes a non-MuJoCo root.
    """
    world, parent, _ = world_with_two_bodies
    with world.modify_world():
        world.add_kinematic_structure_entity(parent)
    cable = Cable.create(world, CableConfig(anchor_offset=(0, 0, 0.5)))
    output = tmp_path / "normalized-root.xml"
    CableMujocoBuilder().build_world(world, str(output))
    assert cable.connections == tuple(
        point.parent_connection for point in cable.placeholders
    )


def test_recompile_preserves_existing_joint_forces_and_equality_switches() -> None:
    """
    World drive torques and disabled constraints survive cable installation.
    """
    cable = Cable.create(World(), CableConfig(anchor_to_parent=False))
    specification = mujoco.MjSpec()
    body = specification.worldbody.add_body(name="driven_body")
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.03, 0, 0), mass=1)
    body.add_joint(name="driven_joint", type=mujoco.mjtJoint.mjJNT_HINGE)
    specification.add_equality(
        name="existing_equality",
        type=mujoco.mjtEq.mjEQ_JOINT,
        objtype=mujoco.mjtObj.mjOBJ_JOINT,
        name1="driven_joint",
    )
    model = specification.compile()
    data = mujoco.MjData(model)
    data.qfrc_applied[0] = 0.4
    data.eq_active[0] = False
    compiled = cable.install(specification, model, data)
    assert compiled.success
    assert compiled.data.qfrc_applied[0] == data.qfrc_applied[0]
    assert compiled.data.eq_active[0] == data.eq_active[0]


def test_simulation_updates_semantic_points_without_changing_physics() -> None:
    """
    World poses mirror simulated cable links without writing back qpos.
    """
    cable = Cable.create(
        World(), CableConfig(anchor_to_parent=False, anchor_offset=(0, 0, 1))
    )
    specification = mujoco.MjSpec()
    model = specification.compile()
    compiled = cable.install(specification, model, mujoco.MjData(model))
    for _ in range(20):
        mujoco.mj_step(compiled.model, compiled.data)
    positions = compiled.data.qpos.copy()
    cable.sync(compiled.model, compiled.data)
    np.testing.assert_array_equal(compiled.data.qpos, positions)
    for placeholder, name in zip(cable.placeholders, cable.physical_body_names):
        actual = cable.world.compute_forward_kinematics(
            cable.world.root, placeholder
        ).to_np()
        np.testing.assert_allclose(actual[:3, 3], compiled.data.body(name).xpos)


def test_failed_recompile_keeps_existing_model_and_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A native build failure leaves the running scene usable and visibly degraded.
    """
    cable = Cable.create(World())
    specification = mujoco.MjSpec()
    model = specification.compile()
    data = mujoco.MjData(model)

    def fail_recompile(
        self: mujoco.MjSpec,
        previous_model: mujoco.MjModel,
        previous_data: mujoco.MjData,
    ) -> None:
        """
        Represent a native compiler rejecting the candidate cable.
        """
        raise ValueError("fixture compilation failure")

    monkeypatch.setattr(mujoco.MjSpec, "recompile", fail_recompile)
    compiled = cable.install(specification, model, data)
    assert not compiled.success
    assert compiled.spec is specification
    assert compiled.model is model
    assert compiled.data is data
    assert not cable.active
    assert len(cable.world.bodies) == cable.config.segment_count + 1


def test_missing_native_spec_api_retains_semantic_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Unavailable native construction APIs degrade without breaking the world.
    """
    cable = Cable.create(World())
    specification = mujoco.MjSpec()
    model = specification.compile()
    data = mujoco.MjData(model)
    monkeypatch.delattr(mujoco.MjSpec, "attach")
    compiled = cable.install(specification, model, data)
    assert not compiled.success
    assert compiled.model is model
    assert compiled.data is data
    assert not cable.active


@pytest.mark.parametrize(
    "arguments",
    [
        {"segment_count": True},
        {"segment_count": 1},
        {"mass": 0},
        {"radius": float("nan")},
        {"bend": -1},
        {"anchor_offset": (0, 0)},
        {"condim": 2},
        {"friction": (-1, 0, 0)},
    ],
)
def test_invalid_config_is_rejected(arguments: dict[str, object]) -> None:
    """
    Impossible geometry and contact parameters fail before editing the world.
    """
    with pytest.raises(ValueError):
        CableConfig(**arguments)
