"""
New standalone instruments remain discoverable and portable as resource folders.
"""

from __future__ import annotations

import importlib
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from cramera.laboratory_devices import (
    DeviceResourceFile,
    LaboratoryDevice,
    LaboratoryDeviceResource,
    LaboratoryInstrument,
)
from semantic_digital_twin.adapters.urdf import URDFParser

# %% catalog integration


def test_instrument_catalog_matches_authored_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Every independently authored instrument has a public resource identifier.
    """
    directory = (
        Path(__file__).resolve().parents[2] / "cramera/tools/laboratory/instruments"
    )
    monkeypatch.syspath_prepend(str(directory))
    identifiers = {
        device.identifier
        for module in ("sample_devices", "chamber_devices", "analysis_devices")
        for device in importlib.import_module(module).build_devices()
    }
    assert {instrument.value for instrument in LaboratoryInstrument} == identifiers
    assert not set(LaboratoryInstrument) & set(LaboratoryDevice)


@pytest.mark.parametrize("instrument", list(LaboratoryInstrument))
def test_installed_instrument_references_only_its_local_meshes(
    instrument: LaboratoryInstrument,
) -> None:
    """
    A device can be moved without the room or another instrument resource.
    """
    resource = LaboratoryDeviceResource(instrument)
    description = ElementTree.parse(resource.urdf)
    meshes = description.findall(".//mesh")
    assert meshes
    for mesh in meshes:
        relative = Path(mesh.attrib["filename"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        assert (resource.directory / relative).is_file()
    assert (resource.directory / DeviceResourceFile.METADATA).is_file()
    assert (resource.directory / DeviceResourceFile.README).is_file()


def test_new_instrument_resource_can_be_copied_without_the_source_catalog(
    tmp_path: Path,
) -> None:
    """
    The same public copy helper accepts a newly authored instrument.
    """
    source = LaboratoryDeviceResource(LaboratoryInstrument.PH_PROBE)
    destination = source.copy_to(tmp_path)
    copied = LaboratoryDeviceResource(LaboratoryInstrument.PH_PROBE, tmp_path)
    assert copied.directory == destination
    assert copied.urdf.read_bytes() == source.urdf.read_bytes()
    for original in source.directory.rglob("*"):
        if original.is_file():
            assert (
                destination / original.relative_to(source.directory)
            ).read_bytes() == original.read_bytes()


@pytest.mark.parametrize("instrument", list(LaboratoryInstrument))
def test_independent_instrument_loads_as_a_cram_world(
    instrument: LaboratoryInstrument,
) -> None:
    """
    The production parser retains every device body and interaction frame.
    """
    resource = LaboratoryDeviceResource(instrument)
    description = ElementTree.parse(resource.urdf)
    world = URDFParser.from_file(str(resource.urdf), prefix="").parse()
    assert {body.name.name for body in world.bodies} == {
        link.attrib["name"] for link in description.findall("link")
    }
