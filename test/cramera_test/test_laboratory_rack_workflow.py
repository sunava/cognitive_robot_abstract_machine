"""
The native recipe retrieves its loaded rack before the first individual pickup.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from coraplex.datastructures.enums import TaskStatus
from cramera.laboratory_actions import (
    CloseInstrumentAction,
    LaboratoryContext,
    OpenInstrumentAction,
    RetrieveRackAction,
    TransferSampleAction,
)
from cramera.laboratory_handles import LaboratoryHandle
from cramera.laboratory_mixing_program import LaboratoryMixingProgram
from cramera.laboratory_physics_program import LaboratoryProgramPhysics, ProgramState
from cramera.laboratory_rack import RackGeometry
from cramera.laboratory_workflow import (
    LaboratoryWorkflowProgram,
    WorkflowLocation,
    WorkflowStep,
)
from cramera.laboratory_world import LaboratoryBody

from .test_laboratory_bundle import laboratory_directory
from .test_laboratory_mobile_robot import mobile_robot
from .test_laboratory_physics_robot import robot_physics
from .test_laboratory_rack_contacts import (
    RackSupportGeometry,
    assert_rack_transfer_contacts,
)
from .test_laboratory_workflow_adapter import workflow_world


# %% native startup and single-sample handoff
def test_native_workflow_retrieves_loaded_rack_before_first_sample_pickup(
    workflow_world: LaboratoryProgramPhysics,
) -> None:
    """
    Measured drawer and rack actions hand an empty sample to the usual dispenser action.
    """
    workflow_world.run_workflow(prepare=True)
    physics = workflow_world.physics
    program = workflow_world.program
    assert isinstance(program, LaboratoryWorkflowProgram)
    startup_actions = (
        OpenInstrumentAction,
        RetrieveRackAction,
        CloseInstrumentAction,
    )
    action_prefix = (*startup_actions, TransferSampleAction)
    startup_verified = False
    previous_step = None
    maximum_seconds = program.parameters.transfer_timeout * len(action_prefix)
    try:
        for _ in range(int(maximum_seconds / physics.timestep)):
            if program.step_index != previous_step:
                print(
                    f"{physics.data.time:.3f}s: {program.step.operation}",
                    flush=True,
                )
                previous_step = program.step_index
            before = physics.data.qpos.copy()
            program.advance()
            assert np.array_equal(physics.data.qpos, before)
            physics.step()
            assert_workflow_payload_contacts(program)
            before = physics.data.qpos.copy()
            program.observe()
            assert np.array_equal(physics.data.qpos, before)
            if program.step_index == len(startup_actions) and not startup_verified:
                assert_retrieved_rack(program)
                assert_native_measurements(program)
                assert program.execution.completed == len(startup_actions)
                startup_verified = True
            if program.step_index == len(action_prefix):
                break
            if program.state != ProgramState.RUNNING:
                break
        assert program.state == ProgramState.RUNNING, program.snapshot()
        assert startup_verified
        assert program.step_index == len(action_prefix), program.snapshot()
        assert program.execution.completed == len(action_prefix)
        assert isinstance(program.execution.plan.context, LaboratoryContext)
        performed = program.execution.plan.actions[: len(action_prefix)]
        assert tuple(type(item.action) for item in performed) == action_prefix
        assert all(item.status == TaskStatus.SUCCEEDED for item in performed)
        assert program.step.operation == WorkflowStep.DOSE_A
        sample = program.samples[0]
        assert sample.key == program.SAMPLE_KEYS[0]
        assert sample.location == WorkflowLocation.DISPENSER
        assert program.motion.state == ProgramState.SUCCEEDED
        assert program.motion.grasp_verified
        assert program.motion.holding
        np.testing.assert_allclose(
            physics.data.xpos[physics.objects[sample.key].body_id],
            physics.pipetting.target_position,
            atol=program.motion.parameters.position_tolerance,
        )
        assert physics.drawer.closed
        assert physics.rack.supported_by(LaboratoryBody.ROOM)
        assert physics.rack.occupants()[physics.rack.assignments[sample.key]] is None
        for remaining in program.SAMPLE_KEYS[1:]:
            assert physics.rack.occupants()[physics.rack.assignments[remaining]] == (
                remaining
            )
        assert all(
            physics.liquid.tubes[key].volume_ml == 0 for key in program.SAMPLE_KEYS
        )
        assert_native_measurements(program)
        assert not np.any(physics.data.xfrc_applied)
    finally:
        workflow_world.stop_program()


def assert_retrieved_rack(program: LaboratoryWorkflowProgram) -> None:
    """
    Require the closed drawer and four upright empty samples on the supported rack.
    """
    physics = program.physics
    assert physics.drawer.closed
    assert physics.rack.supported_by(LaboratoryBody.ROOM)
    np.testing.assert_allclose(
        physics.data.xpos[physics.rack.body],
        physics.model.body_pos[physics.rack.body],
        atol=program.parameters.position_tolerance,
    )
    occupants = physics.rack.occupants()
    for key in program.SAMPLE_KEYS:
        assert occupants[physics.rack.assignments[key]] == key
        assert physics.liquid.tubes[key].volume_ml == 0
        body = physics.objects[key].body_id
        axis = physics.data.xmat[body].reshape(3, 3)[:, 2]
        assert axis[2] >= math.cos(program.parameters.upright_tolerance)


def assert_native_measurements(program: LaboratoryWorkflowProgram) -> None:
    """
    Native rack, vessel and handle frames reflect the latest measured action boundary.
    """
    physics = program.physics
    world = program.execution.plan.world
    measured = {pose.body_name: pose for pose in program.execution.measurement.poses}
    for name in (RackGeometry.BODY, *program.SAMPLE_KEYS):
        np.testing.assert_allclose(
            world.get_body_by_name(name).global_pose.to_np()[:3, 3],
            measured[name].position,
            atol=1e-12,
        )
    for name in LaboratoryHandle:
        np.testing.assert_allclose(
            world.get_body_by_name(name).global_pose.to_np()[:3, 3],
            physics.data.site_xpos[physics.model.site(name).id],
            atol=1e-12,
        )


# %% intended contacts across native action boundaries
def assert_workflow_payload_contacts(program: LaboratoryWorkflowProgram) -> None:
    """
    Reject incidental robot or payload penetration during drawer and sample actions.
    """
    physics = program.physics
    if program.rack_motion is not None:
        assert_rack_transfer_contacts(physics, program.rack_motion)
        return
    payload = {RackGeometry.BODY, *program.SAMPLE_KEYS}
    for contact in physics.data.contact:
        if contact.dist >= -0.0002:
            continue
        geometries = (contact.geom1, contact.geom2)
        names = {physics.model.geom(geometry).name for geometry in geometries}
        bodies = [
            physics.model.body(physics.model.geom_bodyid[geometry]).name
            for geometry in geometries
        ]
        robots = [body.startswith("pr2/") for body in bodies]
        if all(robots):
            continue
        if not any(robots) and not payload.intersection(bodies):
            continue
        if any(robots) and any(
            physics.model.geom_type[geometry] == mujoco.mjtGeom.mjGEOM_PLANE
            for geometry in geometries
        ):
            continue
        fingertip = any(body in LaboratoryMixingProgram.FINGERTIPS for body in bodies)
        handle_grasp = (
            program.hinge is not None
            and program.hinge.handle in geometries
            and fingertip
        )
        sample_grasp = (
            program.motion is not None
            and program.motion.object_key in bodies
            and fingertip
        )
        sample_support = RackGeometry.BODY in bodies and any(
            body in program.SAMPLE_KEYS for body in bodies
        )
        drawer_support = (
            names
            == {
                RackSupportGeometry.BASE,
                RackGeometry.DRAWER_FLOOR,
            }
            and program.step.operation == WorkflowStep.OPEN_DRAWER
        )
        bench_support = names == {
            RackSupportGeometry.BASE,
            RackSupportGeometry.WORKTOP,
        } and program.step.operation in (
            WorkflowStep.CLOSE_DRAWER,
            WorkflowStep.DISPENSER,
        )
        assert (
            handle_grasp
            or sample_grasp
            or sample_support
            or drawer_support
            or bench_support
        ), (program.step.operation, bodies, names, contact.dist)
