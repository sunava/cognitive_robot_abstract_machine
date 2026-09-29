"""
Articulated sample instruments retain usable controls and open sockets.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


# %% definition loading
@pytest.fixture
def sample_devices(monkeypatch: pytest.MonkeyPatch):
    """
    Load model definitions through their standalone Blender tool import path.
    """
    directory = (
        Path(__file__).resolve().parents[2] / "cramera/tools/laboratory/instruments"
    )
    monkeypatch.syspath_prepend(str(directory))
    module = importlib.import_module("sample_devices")
    return {device.identifier: device for device in module.build_devices()}


# %% interaction geometry
def test_pipette_controls_have_independent_travel(sample_devices) -> None:
    """
    The plunger and ejector move independently and retain their control frames.
    """
    device = sample_devices["electronic_pipette"]
    parts = {part.name: part for part in device.parts}
    plunger = parts["plunger"]
    ejector = parts["tip_ejector"]
    assert plunger.parent == ejector.parent == parts["body"].name
    assert plunger.joint == ejector.joint == "prismatic"
    assert plunger.lower < plunger.upper
    assert ejector.lower < ejector.upper
    parents = {frame.name: frame.parent for frame in device.frames}
    assert parents["plunger_contact"] == plunger.name
    assert parents["ejector_contact"] == ejector.name


@pytest.mark.parametrize(
    "identifier,owner,role",
    [
        ("tip_rack", "socket_insert", "tip_socket"),
        ("thermomixer", "sample_block", "sample_well"),
    ],
)
def test_sample_insertion_frames_target_open_sleeves(
    sample_devices, identifier, owner, role
) -> None:
    """
    Each insertion frame is centered over a genuine open sleeve.
    """
    device = sample_devices[identifier]
    block = next(part for part in device.parts if part.name == owner)
    sleeves = [shape for shape in block.shapes if shape.kind == "ring"]
    targets = [frame for frame in device.frames if frame.role == role]
    assert len(sleeves) == len(targets) == 24
    for target in targets:
        sleeve = next(
            shape for shape in sleeves if shape.center[:2] == target.position[:2]
        )
        assert target.parent == block.name
        assert target.position[2] == pytest.approx(
            sleeve.center[2] + sleeve.dimensions[2] / 2
        )
        assert 0 < sleeve.dimensions[1] < sleeve.dimensions[0]
        for shape in block.shapes:
            if shape.kind != "box" or not shape.collision:
                continue
            inside = all(
                abs(target.position[axis] - shape.center[axis])
                < shape.dimensions[axis] / 2
                for axis in range(3)
            )
            assert not inside, (target.name, shape.name)


@pytest.mark.parametrize("identifier", ["tip_rack", "thermomixer"])
def test_lid_handle_follows_opening_cover(sample_devices, identifier) -> None:
    """
    The rear hinge lifts the front handle above the receptacles.
    """
    device = sample_devices[identifier]
    lid = next(part for part in device.parts if part.name == "lid")
    handle = next(frame for frame in device.frames if frame.name == "lid_handle")
    assert handle.parent == lid.name
    assert lid.joint == "revolute"
    assert handle.position[1] < 0
    assert lid.axis == (-1, 0, 0)
    assert 0 < lid.open_value <= lid.upper


def test_thermomixer_block_uses_bounded_orbital_travel(sample_devices) -> None:
    """
    Mixing is a small planar translation that retains the block orientation.
    """
    device = sample_devices["thermomixer"]
    parts = {part.name: part for part in device.parts}
    block = parts["sample_block"]
    longitudinal = parts[block.parent]
    lateral = parts[longitudinal.parent]
    assert longitudinal.joint == lateral.joint == "prismatic"
    assert lateral.axis == (1, 0, 0)
    assert longitudinal.axis == (0, 1, 0)
    assert lateral.lower < 0 < lateral.upper
    assert longitudinal.lower < 0 < longitudinal.upper


def test_sample_instruments_are_valid_standalone_devices(sample_devices) -> None:
    """
    Every definition passes the hierarchy and physical-dimension validation.
    """
    for device in sample_devices.values():
        device.validate()
