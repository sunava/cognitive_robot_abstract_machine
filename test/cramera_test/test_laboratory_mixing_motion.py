"""
Close pouring and circular held-container motion match laboratory manipulation.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cramera.laboratory_mixing_program import LaboratoryMixingProgram, MixingPhase
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics_robot import robot_physics


# %% pouring clearance
def test_pouring_lip_stays_close_to_receiver_opening(robot_physics) -> None:
    """
    Settled pouring targets leave two centimeters of clearance above the rim.
    """
    program = LaboratoryMixingProgram(robot_physics)
    stages = [
        stage
        for stage in program.stages
        if stage.phase == MixingPhase.POUR
        and Rotation.from_quat(stage.rotation).apply([0, 0, 1])[2] < 0.2
    ]
    assert stages
    for stage in stages:
        rotation = Rotation.from_quat(stage.rotation)
        interior = robot_physics.liquid.interior
        lip = np.asarray(stage.position) + rotation.apply(
            interior.lowest_lip(rotation.inv().apply([0, 0, 1]))
        )
        receiver_rim = program._slot(stage.receiver_slot) + [0, 0, interior.rim]
        assert lip[2] - receiver_rim[2] == pytest.approx(0.02)
        np.testing.assert_allclose(lip[:2], receiver_rim[:2], atol=1e-9)


# %% continuous swirling
def test_swirl_bottom_orbits_while_grasp_point_remains_stationary(
    robot_physics,
) -> None:
    """
    A lifted glass follows a circle through all quadrants around a fixed grip.
    """
    program = LaboratoryMixingProgram(robot_physics)
    program.stage_index = next(
        index
        for index, stage in enumerate(program.stages)
        if stage.phase == MixingPhase.SWIRL
    )
    assert program.stage.object_key == LaboratoryBody.CLEAR_TUBE
    times = np.linspace(0, program.stage.duration, 201)
    poses = [program._swirl_pose(time) for time in times]
    pivot = np.array([0, 0, program.parameters.tool_height])
    centers = np.array([pose[:3, 3] for pose in poses])
    grips = np.array([pose[:3, 3] + pose[:3, :3] @ pivot for pose in poses])
    np.testing.assert_allclose(
        grips, np.repeat(grips[:1], len(times), axis=0), atol=1e-12
    )
    assert np.ptp(centers[:, 0]) > 0.025
    assert np.ptp(centers[:, 1]) > 0.025
    np.testing.assert_allclose(poses[0], poses[-1], atol=1e-12)
