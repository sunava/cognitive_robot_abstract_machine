"""
Measured rigid-body contacts for loose workshop contents and tools.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cramera.workshop_physics import (
    InvalidWorkshopCommand,
    WorkshopPhysics,
    WorkshopSimulationFailure,
)

from .test_workshop_bundle import workshop_directory

# %% independent bodies and compliant manipulation


@pytest.fixture
def physics(workshop_directory: Path) -> WorkshopPhysics:
    """
    Create the portable workshop without loading render meshes.
    """
    return WorkshopPhysics(workshop_directory)


def relative_position(
    physics: WorkshopPhysics, part: str, container: str
) -> np.ndarray:
    """
    Express a loose part's measured position in its container frame.
    """
    poses = physics.state()["objects"]
    container_pose = np.asarray(poses[container])
    return (
        Rotation.from_quat(container_pose[3:])
        .inv()
        .apply(np.asarray(poses[part][:3]) - container_pose[:3])
    )


def test_gravity_settles_independent_fasteners_inside_boxes(
    physics: WorkshopPhysics,
) -> None:
    physics.step(1.0)
    for key in physics.state()["objects"]:
        if key.startswith("screw_") and key != "screw_box":
            position = relative_position(physics, key, "screw_box")
            assert -0.14 < position[0] < 0.14
            assert -0.16 < position[1] < 0.16
            assert 0.005 < position[2] < 0.075
    contacts = physics.state()["contacts"]
    assert any("screw_box" in (item["bodyA"], item["bodyB"]) for item in contacts)
    assert physics.state()["objects"]["screw_box"][2] > 0.919
    assert np.isfinite(physics.data.qpos).all()


def test_grab_and_move_apply_force_without_teleporting(
    physics: WorkshopPhysics,
) -> None:
    physics.step(0.5)
    original = physics.state()["objects"]["screw_box"]
    target = list(original)
    target[2] += 0.20
    physics.grab("screw_box")
    physics.move(target)
    assert physics.state()["objects"]["screw_box"] == original
    physics.step(0.6)
    assert physics.state()["objects"]["screw_box"][2] > original[2] + 0.14
    assert physics.state()["target"] == target
    physics.release()
    physics.step(1.0)
    assert 0.918 < physics.state()["objects"]["screw_box"][2] < 0.925


def test_tilted_box_contents_slide_relative_to_box(physics: WorkshopPhysics) -> None:
    physics.step(0.6)
    physics.grab("screw_box")
    target = physics.state()["objects"]["screw_box"]
    target[2] += 0.23
    physics.move(target)
    physics.step(1.0)
    before = relative_position(physics, "screw_1", "screw_box")
    target[3:] = Rotation.from_euler("x", 30, degrees=True).as_quat().tolist()
    physics.move(target)
    physics.step(1.0)
    after = relative_position(physics, "screw_1", "screw_box")
    assert abs(after[1] - before[1]) > 0.025
    assert -0.17 < after[1] < 0.17
    assert np.isfinite(physics.data.qvel).all()


def test_worktop_resists_downward_drag(physics: WorkshopPhysics) -> None:
    physics.step(0.5)
    physics.grab("nail_box")
    target = physics.state()["objects"]["nail_box"]
    target[2] -= 0.20
    physics.move(target)
    physics.step(0.6)
    assert physics.state()["objects"]["nail_box"][2] > 0.915
    assert (
        np.linalg.norm(physics.data.xfrc_applied[physics.model.body("nail_box").id, :3])
        <= physics.parameters.maximum_force + 1e-8
    )


def test_drawer_target_drives_bounded_joint(physics: WorkshopPhysics) -> None:
    joint = "workshop_drawer_3_joint"
    physics.set_drawer(joint, 0.30)
    assert physics.state()["joints"][joint] == 0
    physics.step(1.0)
    assert 0.28 < physics.state()["joints"][joint] <= 0.301
    with pytest.raises(InvalidWorkshopCommand):
        physics.set_drawer(joint, 0.5)


def test_invalid_pose_and_unheld_motion_are_rejected(physics: WorkshopPhysics) -> None:
    with pytest.raises(InvalidWorkshopCommand):
        physics.move([0, 0, 1, 0, 0, 0, 1])
    physics.grab("screw_box")
    for pose in ([0] * 6, [0, 0, float("nan"), 0, 0, 0, 1], [0] * 7):
        with pytest.raises(InvalidWorkshopCommand):
            physics.move(pose)
    with pytest.raises(InvalidWorkshopCommand):
        physics.grab("workshop_environment")


def test_reset_restores_all_free_body_states(physics: WorkshopPhysics) -> None:
    original = physics.state()
    physics.step(0.2)
    physics.grab("screw_box")
    physics.reset()
    reset = physics.state()
    assert reset["objects"] == original["objects"]
    assert reset["time"] == 0
    assert reset["held"] is None


# %% physical obstructions and ideal thread engagement


def place_initial_body(physics: WorkshopPhysics, key: str, pose: list[float]) -> None:
    """
    Define an isolated regression's initial condition before integration.
    """
    address = physics.model.joint(f"{key}_free").qposadr[0]
    physics.data.qpos[address : address + 7] = [*pose[:3], pose[6], *pose[3:6]]
    mujoco.mj_forward(physics.model, physics.data)


def test_translated_box_does_not_rigidly_parent_contents(
    physics: WorkshopPhysics,
) -> None:
    physics.step(0.5)
    physics.grab("nail_box")
    target = physics.state()["objects"]["nail_box"]
    target[2] += 0.25
    physics.move(target)
    physics.step(0.8)
    before = relative_position(physics, "nail_1", "nail_box")
    target[1] -= 0.20
    physics.move(target)
    physics.step(0.15)
    after = relative_position(physics, "nail_1", "nail_box")
    assert np.linalg.norm(after - before) > 0.015


def test_drawer_motor_stalls_against_a_held_box(physics: WorkshopPhysics) -> None:
    place_initial_body(physics, "nail_box", [-0.78, -0.51, 0.68, 0, 0, 0, 1])
    physics.grab("nail_box")
    joint = "workshop_drawer_3_joint"
    physics.set_drawer(joint, 0.4)
    physics.step(1.0)
    assert physics.state()["joints"][joint] < 0.37
    assert any(
        "workshop_drawer_3" in (contact["bodyA"], contact["bodyB"])
        and "nail_box" in (contact["bodyA"], contact["bodyB"])
        for contact in physics.state()["contacts"]
    )


def seat_screw_and_driver(physics: WorkshopPhysics) -> None:
    """
    Start a contact test with a shaft in the pilot and blade in its slot.
    """
    physics.step(0.3)
    target = physics.state()["thread"]["screwTargetPose"]
    place_initial_body(physics, "screw_1", target)
    physics.step(0.15)
    goal = physics.state()["thread"]["driverTargetPose"]
    place_initial_body(physics, "driver", goal)
    physics.grab("driver")
    physics.move(goal)
    physics.step(0.20)


def test_driver_requires_aligned_contact_with_a_seated_screw(
    physics: WorkshopPhysics,
) -> None:
    physics.drive(0)
    physics.grab("driver")
    with pytest.raises(InvalidWorkshopCommand):
        physics.drive(1)
    seat_screw_and_driver(physics)
    assert physics.state()["thread"]["engaged"]
    assert physics.state()["thread"]["screw"] == "screw_1"


def test_measured_driver_rotation_advances_and_reverses_thread_pitch(
    physics: WorkshopPhysics,
) -> None:
    seat_screw_and_driver(physics)
    initial_depth = physics.state()["thread"]["depth"]
    physics.drive(1)
    physics.step(1.0)
    forward = physics.state()["thread"]
    assert forward["turns"] > 0.25
    assert forward["depth"] > initial_depth + 0.0015
    assert abs(forward["depth"] - forward["pitch"] * forward["turns"]) < 0.001
    physics.drive(0)
    physics.step(0.25)
    stopped = physics.state()["thread"]
    assert stopped["turns"] == forward["turns"]
    physics.drive(-1)
    physics.step(0.50)
    reverse = physics.state()["thread"]
    assert reverse["turns"] < forward["turns"] - 0.15
    assert reverse["depth"] < forward["depth"] - 0.0009


def test_thread_cannot_advance_without_measured_tool_rotation(
    physics: WorkshopPhysics,
) -> None:
    seat_screw_and_driver(physics)
    physics.parameters = replace(physics.parameters, maximum_torque=0.0)
    physics.drive(1)
    physics.step(0.5)
    assert abs(physics.state()["thread"]["turns"]) < 0.005


def test_thread_reactions_conserve_total_force_and_moment(
    physics: WorkshopPhysics,
) -> None:
    seat_screw_and_driver(physics)
    physics.data.xfrc_applied[:] = 0
    target = physics.pose("screw_1")
    target[0] += 0.003
    physics.apply_thread_constraint(target)
    forces = physics.data.xfrc_applied[:, :3]
    torques = physics.data.xfrc_applied[:, 3:]
    np.testing.assert_allclose(forces.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(
        (torques + np.cross(physics.data.xipos, forces)).sum(axis=0), 0, atol=1e-12
    )
    assert np.linalg.norm(forces[physics.objects["workshop_assembly"]]) > 1


def test_invalid_solver_state_is_reported_without_silent_reset(
    physics: WorkshopPhysics,
) -> None:
    physics.data.qpos[-1] = float("nan")
    with pytest.raises(WorkshopSimulationFailure):
        physics.step(0.01)


def test_automatic_solver_reset_is_reported_even_on_first_step(
    physics: WorkshopPhysics,
) -> None:
    physics.data.qvel[-1] = 1e20
    with pytest.raises(WorkshopSimulationFailure):
        physics.step(0.01)


def test_physical_pickup_position_release_and_driver_alignment(
    physics: WorkshopPhysics,
) -> None:
    physics.step(0.4)
    physics.grab("screw_1")
    target = physics.state()["objects"]["screw_1"]
    target[2] = 1.40
    physics.move(target)
    physics.step(0.7)
    goal = physics.state()["thread"]["screwTargetPose"]
    above = list(goal)
    above[2] = 1.40
    physics.move(above)
    physics.step(0.8)
    physics.move(goal)
    physics.step(0.8)
    physics.release()
    physics.step(0.2)
    assert physics.state()["thread"]["state"] == "seated"
    physics.grab("driver")
    target = physics.state()["objects"]["driver"]
    target[2] = 1.45
    physics.move(target)
    physics.step(0.7)
    goal = physics.state()["thread"]["driverTargetPose"]
    above = list(goal)
    above[2] = 1.45
    physics.move(above)
    physics.step(0.8)
    physics.move(goal)
    physics.step(0.8)
    assert physics.state()["thread"]["engaged"]
    physics.drive(1)
    physics.step(0.5)
    assert physics.state()["thread"]["turns"] > 0.15


def test_thread_stops_before_driving_the_head_into_wood(
    physics: WorkshopPhysics,
) -> None:
    seat_screw_and_driver(physics)
    maximum_depth = physics.scene["workshop"]["assembly"]["maximumDepth"]
    physics.drive(1)
    physics.step(maximum_depth / physics.geometry.pitch + 3.0)
    thread = physics.state()["thread"]
    assert thread["state"] == "limit"
    assert thread["depth"] <= maximum_depth + 0.001
    assert physics.direction == 0
