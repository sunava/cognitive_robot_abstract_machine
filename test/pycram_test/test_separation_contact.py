from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path

import numpy as np


class SeparationModuleLoadError(RuntimeError):
    """
    Raised when the separation module cannot be loaded for tests.
    """


def load_separation_module():
    """
    Load the separation module used by the thesis demos.
    """
    module_path = (
        Path(__file__).resolve().parents[2]
        / "pycram"
        / "demos"
        / "thesis"
        / "primitives"
        / "seperation_devision.py"
    )
    module_spec = importlib_util.spec_from_file_location(
        "seperation_devision", module_path
    )
    if module_spec is None or module_spec.loader is None:
        raise SeparationModuleLoadError(
            "Could not create a module specification for seperation_devision."
        )
    module = importlib_util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class SeparationContactCase:
    start_point: np.ndarray
    normal: np.ndarray
    tangential_primary: np.ndarray
    tangential_secondary: np.ndarray
    length: float
    depth: float
    steps: int


def test_saw_contact_uses_full_length_along_tangent():
    separation_module = load_separation_module()
    case = SeparationContactCase(
        start_point=np.array([0.0, 0.0, 0.0], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
        tangential_primary=np.array([1.0, 0.0, 0.0], dtype=float),
        tangential_secondary=np.array([0.0, 1.0, 0.0], dtype=float),
        length=0.2,
        depth=0.1,
        steps=5,
    )
    separation_spec = separation_module.SeparationSpec(
        mode=separation_module.SepMode.SAW,
        length=case.length,
        depth=case.depth,
        n=case.steps,
        tangential_osc_amp=0.0,
    )

    poses = separation_module.compile_separation_contact(
        frame_id=None,
        p0=case.start_point,
        n=case.normal,
        spec=separation_spec,
        t1=case.tangential_primary,
        t2=case.tangential_secondary,
    )

    x_values = np.array([pose.pose.position.x for pose in poses], dtype=float)
    z_values = np.array([pose.pose.position.z for pose in poses], dtype=float)

    expected_x = np.linspace(-0.5 * case.length, 0.5 * case.length, case.steps)
    expected_z = np.linspace(0.0, -case.depth, case.steps)

    assert np.allclose(x_values, expected_x)
    assert np.allclose(z_values, expected_z)
