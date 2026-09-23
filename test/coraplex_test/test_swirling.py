"""
Continuous container motion and its Giskard action constraints.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from coraplex.datastructures.swirling import InvalidSwirl, SwirlProfile, SwirlTrajectory

# %% Fixed pivot geometry


@pytest.fixture
def swirl_trajectory():
    """
    A front-held upright container with an off-center grip pivot.
    """
    return SwirlTrajectory(
        root_T_container=np.eye(4),
        container_P_pivot=np.array([0.012, 0.0, 0.10]),
        profile=SwirlProfile(tilt_angle=0.22, cycles=3.0, duration=8.0),
    )


def test_swirl_keeps_the_pivot_fixed_in_world(swirl_trajectory):
    pivot = np.append(swirl_trajectory.container_P_pivot, 1.0)
    for elapsed in np.linspace(0, swirl_trajectory.profile.duration, 97):
        np.testing.assert_allclose(
            swirl_trajectory.pose_at(elapsed) @ pivot, pivot, atol=1e-12
        )


def test_lower_container_follows_a_circle_during_full_tilt(swirl_trajectory):
    profile = swirl_trajectory.profile
    bottom = np.append(swirl_trajectory.container_P_pivot - [0, 0, 0.10], 1.0)
    points = np.array(
        [
            (swirl_trajectory.pose_at(elapsed) @ bottom)[:3]
            for elapsed in np.linspace(
                profile.duration * profile.ramp_fraction,
                profile.duration * (1 - profile.ramp_fraction),
                81,
            )
        ]
    )
    offsets = points[:, :2] - swirl_trajectory.container_P_pivot[:2]
    np.testing.assert_allclose(
        np.linalg.norm(offsets, axis=1), 0.10 * np.sin(profile.tilt_angle), atol=1e-12
    )
    assert np.ptp(points[:, 0]) > 0.04
    assert np.ptp(points[:, 1]) > 0.04
    np.testing.assert_allclose(
        points[:, 2], bottom[2] + 0.10 * (1 - np.cos(profile.tilt_angle)), atol=1e-12
    )


def test_swirl_starts_and_finishes_at_rest(swirl_trajectory):
    duration = swirl_trajectory.profile.duration
    for elapsed in [-1, 0, duration, duration + 1]:
        np.testing.assert_allclose(
            swirl_trajectory.pose_at(elapsed), np.eye(4), atol=1e-12
        )
    step = 1e-4
    for elapsed in [step, duration - step]:
        delta = swirl_trajectory.pose_at(elapsed) - np.eye(4)
        assert np.linalg.norm(delta) / step < 1e-6


def test_swirl_plane_follows_initial_container_orientation(swirl_trajectory):
    root_T_container = np.eye(4)
    root_T_container[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.5]).as_matrix()
    root_T_container[:3, 3] = [0.4, 0.6, 0.8]
    rotated = replace(swirl_trajectory, root_T_container=root_T_container)
    for elapsed in np.linspace(0, rotated.profile.duration, 19):
        np.testing.assert_allclose(
            rotated.pose_at(elapsed),
            root_T_container @ swirl_trajectory.pose_at(elapsed),
            atol=1e-12,
        )


def test_swirl_radius_constructor_matches_requested_bottom_orbit():
    profile = SwirlProfile.from_radius(
        bottom_radius=0.015, pivot_to_bottom=0.10, cycles=2.0, duration=6.0
    )
    assert 0.10 * np.sin(profile.tilt_angle) == pytest.approx(0.015)


@pytest.mark.parametrize(
    "parameters",
    [
        dict(tilt_angle=-0.1),
        dict(tilt_angle=np.pi / 2),
        dict(cycles=0),
        dict(duration=0),
        dict(duration=np.nan),
        dict(ramp_fraction=0),
        dict(ramp_fraction=0.51),
    ],
)
def test_invalid_swirl_parameters_are_rejected(parameters):
    with pytest.raises(InvalidSwirl):
        SwirlProfile(**parameters)


def test_swirl_rejects_nonrigid_initial_pose(swirl_trajectory):
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    with pytest.raises(InvalidSwirl):
        replace(swirl_trajectory, root_T_container=invalid)
