"""Locate and copy independently reusable laboratory instrument resources."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# %% portable device catalog


class LaboratoryDevice(StrEnum):
    """Instruments extracted from the assembled laboratory as independent models."""

    CENTRIFUGE = "centrifuge"
    """Housing, continuous rotor, and bounded protective lid."""
    DUAL_DISPENSER = "dual_dispenser"
    """Two-reservoir dispenser with an open receiving cradle."""
    SCALE = "scale"
    """Weighing instrument with its pan and open sample support."""
    TUBE_RACK = "tube_rack"
    """Six-position rack retaining open insertion apertures."""
    REFRIGERATOR = "refrigerator"
    """Compact sample refrigerator with a glass door and six storage positions."""


class LaboratoryInstrument(StrEnum):
    """Standalone instruments and accessories authored in editable Blender models."""

    VORTEX_MIXER = "vortex_mixer"
    """Contact cup with vertical travel and a speed control."""
    ELECTRONIC_PIPETTE = "electronic_pipette"
    """Handheld pipette with independently moving plunger and tip ejector."""
    TIP_RACK = "tip_rack"
    """Opening rack with twenty-four accessible tip sockets."""
    TUBE_CAPPER = "tube_capper"
    """Tube clamp, translating chuck and rotating closure head."""
    THERMOMIXER = "thermomixer"
    """Opening temperature-control instrument with an articulated sample block."""
    MICROPLATE_READER = "microplate_reader"
    """Optical plate instrument with an opening sample drawer."""
    MICROPLATE_96_WELL = "microplate_96_well"
    """Movable plate with ninety-six open wells."""
    INCUBATOR = "incubator"
    """Glazed chamber door and three separate shelf bodies."""
    DIGITAL_MICROSCOPE = "digital_microscope"
    """Focus carriage, two-axis specimen stage and objective selection."""
    PH_METER = "ph_meter"
    """Measurement console with an adjustable electrode support."""
    PH_PROBE = "ph_probe"
    """Separate glass electrode and guarded sensing bulb."""
    PCR_THERMOCYCLER = "pcr_thermocycler"
    """Thermal block with an opening heated cover."""
    BARCODE_SCANNER = "barcode_scanner"
    """Handheld scanner with an articulated trigger."""
    BARCODE_DOCK = "barcode_dock"
    """Independent charging cradle for the handheld scanner."""
    LABEL_PRINTER = "label_printer"
    """Opening media cover, rotating label roll and feed roller."""


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

    device: LaboratoryDevice | LaboratoryInstrument
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
