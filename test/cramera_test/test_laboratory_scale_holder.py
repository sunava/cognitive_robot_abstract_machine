"""
Only the pan and its attached holder transmit load into the scale reading.
"""

from pathlib import Path

import mujoco
import pytest

from cramera.laboratory_scale import LaboratoryScale, ScaleField


# %% instrument load path
@pytest.mark.parametrize(
    "horizontal,counted", [(0.0, True), (-0.2, True), (0.2, False)]
)
def test_scale_counts_holder_load_but_excludes_housing(
    horizontal: float, counted: bool
):
    """
    A sample's entire weight registers through the holder or pan alone.
    """
    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).parent / "dataset/laboratory_scale_holder.xml")
    )
    data = mujoco.MjData(model)
    data.qpos[0] = horizontal
    scale = LaboratoryScale(model, data)
    for _ in range(2000):
        mujoco.mj_step(model, data)
        scale.observe()
    expected = model.body_mass[1] * 1000.0 if counted else 0.0
    reading = scale.snapshot()
    assert reading[ScaleField.STABLE]
    assert reading[ScaleField.GRAMS] == pytest.approx(expected, abs=0.015)
    assert reading[ScaleField.OBJECTS] == ([model.body(1).name] if counted else [])
