"""
Standalone instrument descriptions retain usable openings and moving frames.
"""

from __future__ import annotations

import importlib
import math
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

# %% standalone authoring modules


@pytest.fixture
def authoring(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    """
    Import the Blender-independent authoring modules from their tool directory.
    """
    directory = (
        Path(__file__).resolve().parents[2] / "cramera/tools/laboratory/instruments"
    )
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module("spec"), importlib.import_module("description")


def test_hollow_support_colliders_leave_the_insertion_axis_open(
    authoring: tuple[ModuleType, ModuleType], tmp_path: Path
) -> None:
    """
    A sleeve must collide at its walls, never through its central bore.
    """
    specification, descriptions = authoring
    model = specification.DeviceSpec("holder", "Holder", "An open insertion sleeve")
    part = model.part("base")
    part.ring("sleeve", (0, 0, 0.02), 0.012, 0.01, 0.04, specification.Finish.STEEL)
    descriptions.InstrumentDescription(model).write(tmp_path)
    root = ElementTree.parse(tmp_path / "model.urdf")
    collisions = root.findall(".//collision")
    assert len(collisions) >= 16
    for collision in collisions:
        position = [
            float(value) for value in collision.find("origin").get("xyz").split()
        ]
        size = [
            float(value) for value in collision.find("geometry/box").get("size").split()
        ]
        distance = (position[0] ** 2 + position[1] ** 2) ** 0.5
        assert distance - size[0] / 2 >= 0.01 - 1e-8


def test_grasp_frame_is_attached_to_the_moving_cover(
    authoring: tuple[ModuleType, ModuleType], tmp_path: Path
) -> None:
    """
    Handle frames must follow the articulated part instead of staying on the base.
    """
    specification, descriptions = authoring
    model = specification.DeviceSpec("cabinet", "Cabinet", "A covered chamber")
    model.part("base").box("floor", (0, 0, 0.01), (0.2, 0.2, 0.02))
    model.part(
        "cover",
        "base",
        origin=(0, 0.1, 0.2),
        joint=specification.JointKind.REVOLUTE,
        axis=(-1, 0, 0),
        lower=0,
        upper=1.5,
        open_value=1.2,
    ).box("lid", (0, -0.1, 0), (0.2, 0.2, 0.01))
    model.frame("handle_grasp", "cover", (0, -0.18, 0.03), "Open the cover")
    descriptions.InstrumentDescription(model).write(tmp_path)
    root = ElementTree.parse(tmp_path / "model.urdf")
    frame = root.find("joint[@name='cabinet_handle_grasp_joint']")
    assert frame.find("parent").get("link") == "cabinet_cover"
    joint = root.find("joint[@name='cabinet_cover_joint']")
    assert joint.get("type") == "revolute"
    assert float(joint.find("limit").get("upper")) == 1.5


# %% invalid portable descriptions
@pytest.mark.parametrize("name", ["../outside", "a/b", "a\\b", "", ".", "white space"])
def test_asset_part_names_cannot_escape_the_resource_folder(
    authoring: tuple[ModuleType, ModuleType], name: str
) -> None:
    """
    A body identifier must remain a safe local mesh filename and TF name.
    """
    specification, _ = authoring
    model = specification.DeviceSpec("holder", "Holder", "An independent resource")
    model.part(name).box("housing", (0, 0, 0), (0.1, 0.1, 0.1))
    with pytest.raises(ValueError):
        model.validate()


@pytest.mark.parametrize("axis", [(0, 0, 0), (0, 0, 2), (0, 0, math.nan)])
def test_joint_axes_must_be_finite_unit_vectors(
    authoring: tuple[ModuleType, ModuleType], axis: tuple[float, float, float]
) -> None:
    """
    URDF and Blender must receive the same unambiguous motion axis.
    """
    specification, _ = authoring
    model = specification.DeviceSpec("slider", "Slider", "A moving control")
    model.part("base")
    model.part(
        "control",
        "base",
        joint=specification.JointKind.PRISMATIC,
        axis=axis,
        lower=0,
        upper=0.1,
    )
    with pytest.raises(ValueError):
        model.validate()


def test_root_transform_cannot_be_silently_discarded(
    authoring: tuple[ModuleType, ModuleType],
) -> None:
    """
    A standalone root has no parent joint that could carry a placement offset.
    """
    specification, _ = authoring
    model = specification.DeviceSpec("holder", "Holder", "A portable reference frame")
    model.part("base", origin=(1, 0, 0))
    with pytest.raises(ValueError):
        model.validate()


@pytest.mark.parametrize(
    "dimensions", [(0.1, 0.1), (0.1, math.inf, 0.1), (0.1, -0.1, 0.1)]
)
def test_solid_dimensions_must_be_finite_and_complete(
    authoring: tuple[ModuleType, ModuleType], dimensions: tuple[float, ...]
) -> None:
    """
    Invalid solid extents are rejected before geometry or XML construction.
    """
    specification, _ = authoring
    model = specification.DeviceSpec("holder", "Holder", "A solid fixture")
    model.part("base").box("housing", (0, 0, 0), dimensions)
    with pytest.raises(ValueError):
        model.validate()


def test_cone_cannot_have_negative_end_radius(
    authoring: tuple[ModuleType, ModuleType],
) -> None:
    """
    A cone permits a pointed end but not an inverted negative-radius surface.
    """
    specification, _ = authoring
    model = specification.DeviceSpec("holder", "Holder", "A tapered fixture")
    model.part("base").cone(
        "tip", (0, 0, 0), -0.001, 0.01, 0.03, specification.Finish.STEEL
    )
    with pytest.raises(ValueError):
        model.validate()


# %% inertia and coordinate integrity
def test_noncolliding_solid_retains_mass_and_inertia(
    authoring: tuple[ModuleType, ModuleType], tmp_path: Path
) -> None:
    """
    Disabling contact response does not remove a moving solid's mass.
    """
    specification, descriptions = authoring
    model = specification.DeviceSpec("sensor", "Sensor", "A non-contact sensor")
    part = model.part("body", mass=0.25)
    part.box("housing", (0.02, 0.03, 0.04), (0.1, 0.2, 0.3), collision=False)
    descriptions.InstrumentDescription(model).write(tmp_path)
    link = ElementTree.parse(tmp_path / "model.urdf").find("link")
    assert link.findall("collision") == []
    inertial = link.find("inertial")
    assert inertial is not None
    assert float(inertial.find("mass").get("value")) == part.mass
    assert np.fromstring(inertial.find("origin").get("xyz"), sep=" ") == pytest.approx(
        part.shapes[0].center
    )
    dimensions = np.asarray(part.shapes[0].dimensions)
    expected = part.mass / 12 * (np.sum(dimensions**2) - dimensions**2)
    inertia = inertial.find("inertia")
    assert [float(inertia.get(key)) for key in ("ixx", "iyy", "izz")] == pytest.approx(
        expected
    )


def test_rotated_sleeve_contacts_preserve_the_bore(
    authoring: tuple[ModuleType, ModuleType],
) -> None:
    """
    Every point on a sleeve axis stays outside all rotated wall colliders.
    """
    specification, descriptions = authoring
    model = specification.DeviceSpec("holder", "Holder", "An angled sleeve")
    part = model.part("base")
    part.ring(
        "sleeve",
        (0.17, 0.21, 0.08),
        0.02,
        0.015,
        0.06,
        specification.Finish.STEEL,
        rotation=(0.6, 0.8, 0.4),
    )
    sleeve = part.shapes[0]
    primitives = descriptions.ContactPrimitive.from_shape(sleeve)
    orientation = descriptions.rotation_matrix(sleeve.rotation)
    for height in (-sleeve.dimensions[2] / 2, 0, sleeve.dimensions[2] / 2):
        for angle in np.linspace(0, math.tau, 17):
            radius = sleeve.dimensions[1] * 0.999
            point = np.asarray(sleeve.center) + orientation @ np.array(
                (radius * math.cos(angle), radius * math.sin(angle), height)
            )
            for primitive in primitives:
                local = descriptions.rotation_matrix(primitive.rotation).T @ (
                    point - primitive.center
                )
                assert not np.all(np.abs(local) < primitive.half_extents())
