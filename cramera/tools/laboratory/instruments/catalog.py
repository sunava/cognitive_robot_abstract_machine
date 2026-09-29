"""
Complete collection of independently reusable laboratory instruments.
"""

from analysis_devices import build_devices as build_analysis_devices
from chamber_devices import build_devices as build_chamber_devices
from sample_devices import build_devices as build_sample_devices
from spec import DeviceSpec

# %% Collection


def build_devices() -> list[DeviceSpec]:
    """
    Return the original instruments and their separately movable accessories.
    """
    return build_sample_devices() + build_chamber_devices() + build_analysis_devices()
