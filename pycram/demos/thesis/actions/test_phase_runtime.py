import numpy as np
import pytest
from dataclasses import dataclass

from pycram.demos.thesis.actions.phase_runtime import (
    AnchorKey,
    ParamKey,
    PhaseSpec,
    PrimitiveFamily,
)


@dataclass(frozen=True)
class CutPlaneAnchor:
    frame_id: str
    p0: np.ndarray
    n: np.ndarray


@dataclass(frozen=True)
class CutSpec:
    depth: float


def test_phase_spec_binds_anchor_and_params():
    spec = PhaseSpec(
        family=PrimitiveFamily.SEPARATION_CONTACT,
        anchor_key=AnchorKey.CUT_PLANE,
        param_key=ParamKey.CUT_SPEC,
        time_span=(0.0, 1.0),
    )

    anchors = {
        AnchorKey.CUT_PLANE: CutPlaneAnchor(
            frame_id="obj",
            p0=np.array([0.0, 0.0, 0.0]),
            n=np.array([0.0, 0.0, 1.0]),
        )
    }
    params = {ParamKey.CUT_SPEC: CutSpec(depth=0.02)}

    inst = spec.bind(anchors, params)

    assert inst.anchor.frame_id == "obj"
    assert float(inst.params.depth) == 0.02


def test_missing_anchor_raises():
    spec = PhaseSpec(
        family=PrimitiveFamily.SEPARATION_CONTACT,
        anchor_key=AnchorKey.CUT_PLANE,
        param_key=ParamKey.CUT_SPEC,
    )
    with pytest.raises(KeyError):
        spec.bind({}, {ParamKey.CUT_SPEC: CutSpec(depth=0.02)})


def test_missing_params_raises():
    spec = PhaseSpec(
        family=PrimitiveFamily.SEPARATION_CONTACT,
        anchor_key=AnchorKey.CUT_PLANE,
        param_key=ParamKey.CUT_SPEC,
    )
    with pytest.raises(KeyError):
        spec.bind(
            {
                AnchorKey.CUT_PLANE: CutPlaneAnchor(
                    "obj", np.zeros(3), np.array([0, 0, 1])
                )
            },
            {},
        )
