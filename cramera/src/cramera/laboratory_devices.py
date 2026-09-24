"""Locate and copy independently reusable laboratory instrument resources."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# %% portable device catalog


class LaboratoryDevice(StrEnum):
    """Authored instruments available as separate material-bearing URDF models."""

    CENTRIFUGE = "centrifuge"
    """Housing, continuous rotor, and bounded protective lid."""
    DUAL_DISPENSER = "dual_dispenser"
    """Two-reservoir dispenser with an open receiving cradle."""
    SCALE = "scale"
    """Weighing instrument with its pan and open sample support."""
    TUBE_RACK = "tube_rack"
    """Six-position rack retaining open insertion apertures."""


class DeviceResourceFile(StrEnum):
    """Files and directories shared by installed and copied device resources."""

    DIRECTORY = "resources/laboratory_devices"
    """Catalog location relative to the CRAMERA Python package."""
    DESCRIPTION = "model.urdf"
    """Root URDF referring only to assets in the same device directory."""
    METADATA = "metadata.json"
    """Authored placement and frame information accompanying the model."""
    README = "README.md"
    """English description of model scope and local coordinates."""
    ASSETS = "assets"
    """Embedded-texture visual assets referenced by the URDF."""


@dataclass(frozen=True)
class LaboratoryDeviceResource:
    """A self-contained instrument folder suitable for CRAM's URDF parser."""

    device: LaboratoryDevice
    """Instrument selected from the installed catalog."""
    root_directory: Path | None = None
    """Optional catalog directory overriding the installed package resources."""

    @property
    def directory(self) -> Path:
        """Return the folder containing this device and its relative mesh assets."""
        root = (
            self.root_directory or Path(__file__).parent / DeviceResourceFile.DIRECTORY
        )
        return root / self.device.value

    @property
    def urdf(self) -> Path:
        """Return the independently loadable model description."""
        return self.directory / DeviceResourceFile.DESCRIPTION

    def copy_to(self, directory: Path) -> Path:
        """Copy the complete device folder without depending on the source package.

        :param directory: Parent directory receiving the named instrument folder.
        :return: Copied resource directory.
        """
        destination = directory / self.device.value
        shutil.copytree(self.directory, destination)
        return destination
