"""
The authored balance measures supported load and retains a deliberate tare.
"""

import json

import mujoco
import numpy as np
import pytest

from cramera.laboratory_world import LaboratoryBody
from cramera.laboratory_physics import LaboratoryPhysics
from cramera.laboratory_scale import ScaleGeometry

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_physics import physics


# %% physical scale loading
def place_on_pan(
    physics: LaboratoryPhysics,
    key: str = LaboratoryBody.CLEAR_TUBE,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """
    Initialize a freely simulated object immediately above the authored pan.
    """
    pan = physics.model.geom(ScaleGeometry.PAN).id
    body = physics.objects[key].body_id
    joint = physics.model.body_jntadr[body]
    position = physics.model.jnt_qposadr[joint]
    velocity = physics.model.jnt_dofadr[joint]
    surface = physics.data.geom_xpos[pan].copy()
    surface[2] += physics.model.geom_size[pan, 1] + 0.003
    physics.data.qpos[position : position + 3] = surface + offset
    physics.data.qpos[position + 3 : position + 7] = [1, 0, 0, 0]
    physics.data.qvel[velocity : velocity + 6] = 0
    mujoco.mj_forward(physics.model, physics.data)


def test_empty_balance_is_stably_zero(physics: LaboratoryPhysics) -> None:
    """
    An unloaded pan settles at zero without attributing nearby rack objects.
    """
    physics.step(1500)
    scale = physics.snapshot()["scale"]
    assert scale["available"]
    assert scale["stable"]
    assert scale["grams"] == 0.0
    assert scale["objects"] == []


def test_supported_glass_and_liquid_have_actual_weight(
    physics: LaboratoryPhysics,
) -> None:
    """
    Pan reaction includes the dry vessel and its water-density liquid load.
    """
    physics.fill_liquid(LaboratoryBody.CLEAR_TUBE, 5.0)
    place_on_pan(physics)
    physics.step(2500)
    scale = physics.snapshot()["scale"]
    assert scale["stable"]
    assert scale["grams"] == pytest.approx(17.0, abs=0.015)
    assert scale["forceNewtons"] / np.linalg.norm(
        physics.model.opt.gravity
    ) == pytest.approx(0.017, abs=0.000015)
    assert scale["objects"] == [LaboratoryBody.CLEAR_TUBE]


def test_hovering_glass_does_not_register_as_load(physics: LaboratoryPhysics) -> None:
    """
    A supported cursor target above the pan applies no weighing load.
    """
    place_on_pan(physics, offset=(0, 0, 0.06))
    position = physics.data.xpos[
        physics.objects[LaboratoryBody.CLEAR_TUBE].body_id
    ].copy()
    physics.set_target(LaboratoryBody.CLEAR_TUBE, position.tolist())
    physics.step(1500)
    assert physics.snapshot()["scale"]["grams"] == 0.0


def test_housing_contact_is_not_weighing_pan_contact(
    physics: LaboratoryPhysics,
) -> None:
    """
    A glass resting on the instrument housing bypasses the balance pan.
    """
    place_on_pan(physics, offset=(0.12, 0, 0))
    physics.step(2000)
    assert physics.snapshot()["scale"]["grams"] == 0.0


def test_tare_subtracts_glass_and_tracks_added_liquid(
    physics: LaboratoryPhysics,
) -> None:
    """
    Taring an empty vessel permits reading subsequently added liquid alone.
    """
    physics.fill_liquid(LaboratoryBody.CLEAR_TUBE, 0.0)
    place_on_pan(physics)
    physics.step(2500)
    physics.tare_scale()
    assert physics.snapshot()["scale"]["grams"] == 0.0
    physics.fill_liquid(LaboratoryBody.CLEAR_TUBE, 5.0)
    physics.step(2500)
    scale = physics.snapshot()["scale"]
    assert scale["grams"] == pytest.approx(5.0, abs=0.015)
    assert scale["tareGrams"] == pytest.approx(12.0, abs=0.015)
    assert scale["grossGrams"] == pytest.approx(17.0, abs=0.015)


def test_removed_load_leaves_negative_tare_until_reset(
    physics: LaboratoryPhysics,
) -> None:
    """
    Removing a tared vessel preserves the offset until a deliberate reset.
    """
    place_on_pan(physics)
    physics.step(2500)
    physics.tare_scale()
    position = physics.data.xpos[
        physics.objects[LaboratoryBody.CLEAR_TUBE].body_id
    ].copy()
    physics.set_target(LaboratoryBody.CLEAR_TUBE, (position + [0, 0, 0.08]).tolist())
    physics.step(2500)
    scale = physics.snapshot()["scale"]
    assert scale["grams"] == pytest.approx(-12.0, abs=0.015)
    physics.reset()
    assert physics.snapshot()["scale"]["tareGrams"] == 0.0


def test_unsettled_balance_cannot_be_tared(physics: LaboratoryPhysics) -> None:
    """
    A still-falling load must settle before it defines the zero reference.
    """
    place_on_pan(physics, offset=(0, 0, 0.02))
    physics.step(40)
    with pytest.raises(ValueError):
        physics.tare_scale()


def test_scale_snapshot_can_be_published_as_json(physics: LaboratoryPhysics) -> None:
    """
    Actual solver readings cross the HTTP boundary without scalar coercion.
    """
    physics.step(600)
    assert json.loads(json.dumps(physics.snapshot()))["scale"]["available"] is True


def test_manual_rotation_damps_empty_glass_above_pan(
    physics: LaboratoryPhysics,
) -> None:
    """
    Low axial inertia must not turn cursor damping into sustained spin.
    """
    place_on_pan(physics, offset=(0.0, 0.0, 0.1))
    body = physics.objects[LaboratoryBody.CLEAR_TUBE].body_id
    joint = physics.model.body_jntadr[body]
    position = physics.model.jnt_qposadr[joint]
    physics.data.qpos[position + 3 : position + 7] = [np.cos(0.15), 0, 0, np.sin(0.15)]
    mujoco.mj_forward(physics.model, physics.data)
    physics.set_target(LaboratoryBody.CLEAR_TUBE, physics.data.xpos[body].tolist())
    physics.step(1500)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        physics.model, physics.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
    )
    assert np.linalg.norm(velocity[:3]) < 0.01
    assert abs(physics.data.xquat[body, 0]) > np.cos(0.001 / 2)


def test_first_stable_reading_has_finished_display_filtering(
    physics: LaboratoryPhysics,
) -> None:
    """
    Tare must not capture smoothing bias larger than a displayed digit.
    """
    physics.step(1000)
    place_on_pan(physics)
    for _ in range(150):
        physics.step(20)
        scale = physics.snapshot()["scale"]
        if scale["stable"] and scale["objects"]:
            break
    assert scale["stable"]
    assert scale["grams"] == 12.0
