"""Unit tests for phase runtime binding and compilation."""

from dataclasses import dataclass

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped

from pycram.demos.thesis.actions.phase_runtime import (
    AnchorKey,
    ActionProgram,
    CompilerRegistry,
    MissingAnchorError,
    MissingCompilerError,
    MissingParameterError,
    ParamKey,
    PhaseInstance,
    PhaseSpec,
    PrimitiveFamily,
)


@dataclass(frozen=True)
class CutPlaneAnchor:
    """Test anchor used by phase runtime unit tests."""

    frame_id: str
    p0: np.ndarray
    n: np.ndarray


@dataclass(frozen=True)
class CutSpec:
    """Test spec used by phase runtime unit tests."""

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
    with pytest.raises(MissingAnchorError):
        spec.bind({}, {ParamKey.CUT_SPEC: CutSpec(depth=0.02)})


def test_missing_params_raises():
    spec = PhaseSpec(
        family=PrimitiveFamily.SEPARATION_CONTACT,
        anchor_key=AnchorKey.CUT_PLANE,
        param_key=ParamKey.CUT_SPEC,
    )
    with pytest.raises(MissingParameterError):
        spec.bind(
            {
                AnchorKey.CUT_PLANE: CutPlaneAnchor(
                    "obj", np.zeros(3), np.array([0, 0, 1])
                )
            },
            {},
        )


def test_missing_compiler_raises():
    phase = PhaseInstance(
        family=PrimitiveFamily.SEPARATION_CONTACT,
        anchor=CutPlaneAnchor("obj", np.zeros(3), np.array([0, 0, 1])),
        params=CutSpec(depth=0.02),
    )
    registry = CompilerRegistry(compilers={})
    with pytest.raises(MissingCompilerError):
        list(registry.compile(phase))


def test_action_program_compiles_phases():
    phase = PhaseInstance(
        family=PrimitiveFamily.SEPARATION_CONTACT,
        anchor=CutPlaneAnchor("obj", np.zeros(3), np.array([0, 0, 1])),
        params=CutSpec(depth=0.02),
    )

    def compiler(anchor, params):
        pose = PoseStamped()
        pose.header.frame_id = "obj"
        return [pose]

    registry = CompilerRegistry(
        compilers={PrimitiveFamily.SEPARATION_CONTACT: compiler}
    )
    program = ActionProgram(phases=(phase,))
    result = program.compile(registry)
    assert len(result) == 1
