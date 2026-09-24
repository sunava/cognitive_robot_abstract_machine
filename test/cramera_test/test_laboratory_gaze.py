"""
A persistent pointing constraint follows measured manipulation frames.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field

import mujoco
import numpy as np
import pytest

from cramera.laboratory_gaze import GazeField, GazeFrame, LaboratoryGaze
from cramera.laboratory_physics_robot import RobotHeadJoint, RobotJoint
from giskardpy.motion_statechart.tasks.pointing import Pointing


# %% independent measured articulation
@dataclass
class MeasuredHead:
    """
    A small contact simulator exposes the same measured head interface.
    """

    model: mujoco.MjModel = field(init=False)
    """
    Two limited hinges and movable root and manipulation frames.
    """

    data: mujoco.MjData = field(init=False)
    """
    Authoritative dynamics advanced independently of gaze commands.
    """

    robot_joints: dict[str, RobotJoint] = field(init=False, default_factory=dict)
    """
    Actuator metadata for the two independently controlled head axes.
    """

    commands: list[dict[str, float]] = field(init=False, default_factory=list)
    """
    Complete target commands submitted by the pointing controller.
    """

    def __post_init__(self) -> None:
        """
        Build finite-inertia articulation without optional mesh assets.
        """
        root = ElementTree.Element("mujoco")
        ElementTree.SubElement(root, "compiler", angle="radian")
        ElementTree.SubElement(root, "option", timestep="0.001", gravity="0 0 0")
        world = ElementTree.SubElement(root, "worldbody")
        support = ElementTree.SubElement(world, "body", name="support", mocap="true")
        pan = ElementTree.SubElement(support, "body", name="pan", pos="0 0 1.3")
        ElementTree.SubElement(
            pan, "joint", name=RobotHeadJoint.PAN, axis="0 0 1", range="-2.8 2.8"
        )
        tilt = ElementTree.SubElement(pan, "body", name="tilt", pos="0.068 0 0")
        ElementTree.SubElement(
            tilt, "joint", name=RobotHeadJoint.TILT, axis="0 1 0", range="-0.37 1.296"
        )
        for body in (pan, tilt):
            ElementTree.SubElement(
                body, "inertial", pos="0 0 0", mass="1", diaginertia="0.1 0.1 0.1"
            )
        ElementTree.SubElement(
            tilt,
            "body",
            name=GazeFrame.CAMERA,
            pos="0.05 0 0.1",
            quat=f"{math.sqrt(0.5)} 0 {math.sqrt(0.5)} 0",
        )
        ElementTree.SubElement(
            world, "body", name=GazeFrame.TOOL, mocap="true", pos="0.65 0.25 0.6"
        )
        actuators = ElementTree.SubElement(root, "actuator")
        for name in RobotHeadJoint:
            ElementTree.SubElement(
                actuators,
                "position",
                name=name,
                joint=name,
                kp="80",
                kv="8",
                forcerange="-2 2",
            )
        self.model = mujoco.MjModel.from_xml_string(
            ElementTree.tostring(root, encoding="unicode")
        )
        self.data = mujoco.MjData(self.model)
        for name in RobotHeadJoint:
            joint = self.model.joint(name)
            self.robot_joints[name] = RobotJoint(
                name,
                int(joint.qposadr[0]),
                self.model.actuator(name).id,
                float(joint.range[0]),
                float(joint.range[1]),
            )
        mujoco.mj_forward(self.model, self.data)

    def set_joint_targets(self, positions: dict[str, float]) -> None:
        """
        Command actuators while retaining the measured dynamic state.
        """
        self.commands.append(positions.copy())
        for name, value in positions.items():
            joint = self.robot_joints[name]
            assert joint.lower <= value <= joint.upper
            self.data.ctrl[joint.actuator_index] = value

    def run(self, gaze: LaboratoryGaze, seconds: float) -> None:
        """
        Interleave controller calls with genuine physical integration.
        """
        for _ in range(round(seconds / self.model.opt.timestep)):
            before = self.data.qpos.copy()
            gaze.advance()
            np.testing.assert_array_equal(self.data.qpos, before)
            mujoco.mj_step(self.model, self.data)


@pytest.fixture
def measured_head() -> MeasuredHead:
    """
    Create an independent moving target and articulated optical camera.
    """
    return MeasuredHead()


# %% continuing observation and bounded commands
def test_native_pointing_tracks_the_measured_tool(measured_head: MeasuredHead) -> None:
    """
    The native QP points the physical camera without setting joint poses.
    """
    gaze = LaboratoryGaze(measured_head)
    measured_head.run(gaze, 7.0)
    assert gaze.angle_error < gaze.parameters.threshold
    assert set(measured_head.commands[-1]) == set(RobotHeadJoint)
    assert len(gaze.executor.qp_controller.active_dofs) == len(RobotHeadJoint)


def test_tracking_continues_after_convergence_and_root_motion(
    measured_head: MeasuredHead,
) -> None:
    """
    A converged task follows changed tool and base measurements indefinitely.
    """
    gaze = LaboratoryGaze(measured_head)
    measured_head.run(gaze, 7.0)
    assert gaze.angle_error < gaze.parameters.threshold
    measured_head.data.mocap_pos[
        measured_head.model.body(GazeFrame.TOOL).mocapid[0]
    ] = (0.85, -0.4, 0.8)
    measured_head.data.mocap_pos[measured_head.model.body("support").mocapid[0]] = (
        -0.2,
        0.1,
        0.0,
    )
    mujoco.mj_forward(measured_head.model, measured_head.data)
    assert gaze.angle_error > gaze.parameters.threshold
    measured_head.run(gaze, 7.0)
    assert gaze.angle_error < gaze.parameters.threshold
    assert not gaze.executor.motion_statechart.is_end_motion()


def test_paused_interval_does_not_expand_the_control_step(
    measured_head: MeasuredHead,
) -> None:
    """
    Resuming after a long gap keeps a single bounded command interval.
    """
    gaze = LaboratoryGaze(measured_head)
    measured_head.run(gaze, 7.0)
    measured_head.data.time += 100.0
    measured_head.data.mocap_pos[
        measured_head.model.body(GazeFrame.TOOL).mocapid[0]
    ] = (0.65, -0.5, 0.4)
    mujoco.mj_forward(measured_head.model, measured_head.data)
    before = measured_head.data.qpos.copy()
    gaze.advance()
    for name, target in measured_head.commands[-1].items():
        actual = before[measured_head.robot_joints[name].position_index]
        assert (
            abs(target - actual)
            <= gaze.parameters.maximum_velocity / gaze.parameters.frequency
        )


def test_unreachable_target_respects_head_position_limits(
    measured_head: MeasuredHead,
) -> None:
    """
    A below-camera target saturates the head without moving another joint.
    """
    measured_head.data.mocap_pos[
        measured_head.model.body(GazeFrame.TOOL).mocapid[0]
    ] = (0.0, 0.0, -1.0)
    mujoco.mj_forward(measured_head.model, measured_head.data)
    gaze = LaboratoryGaze(measured_head)
    measured_head.run(gaze, 7.0)
    for name, target in measured_head.commands[-1].items():
        joint = measured_head.robot_joints[name]
        assert joint.lower <= target <= joint.upper
    assert gaze.angle_error > gaze.parameters.threshold


def test_camera_tracks_a_rotated_measured_support(measured_head: MeasuredHead) -> None:
    """
    The observed tool is transformed through the measured base orientation.
    """
    gaze = LaboratoryGaze(measured_head)
    measured_head.run(gaze, 7.0)
    support = measured_head.model.body("support").mocapid[0]
    angle = 0.6
    measured_head.data.mocap_quat[support] = (
        math.cos(angle / 2),
        0.0,
        0.0,
        math.sin(angle / 2),
    )
    mujoco.mj_forward(measured_head.model, measured_head.data)
    measured_head.run(gaze, 7.0)
    assert gaze.angle_error < gaze.parameters.threshold


def test_repeated_calls_at_one_physical_time_do_not_issue_extra_commands(
    measured_head: MeasuredHead,
) -> None:
    """
    Rendering or polling frequency cannot advance the physical control plan.
    """
    gaze = LaboratoryGaze(measured_head)
    gaze.advance()
    command_count = len(measured_head.commands)
    for _ in range(20):
        gaze.advance()
    assert len(measured_head.commands) == command_count


def test_snapshot_reports_current_measured_pointing_error(
    measured_head: MeasuredHead,
) -> None:
    """
    Live gaze evidence reflects new measurements before another QP prediction.
    """
    gaze = LaboratoryGaze(measured_head)
    before = gaze.snapshot()
    measured_head.data.mocap_pos[
        measured_head.model.body(GazeFrame.TOOL).mocapid[0]
    ] = (0.3, -0.6, 1.2)
    mujoco.mj_forward(measured_head.model, measured_head.data)
    after = gaze.snapshot()
    assert after == {
        GazeField.CONSTRAINT: Pointing.__name__,
        GazeField.TARGET_FRAME: GazeFrame.TOOL,
        GazeField.CAMERA_FRAME: GazeFrame.CAMERA,
        GazeField.ANGLE_ERROR: gaze.angle_error,
    }
    assert after[GazeField.ANGLE_ERROR] != before[GazeField.ANGLE_ERROR]
