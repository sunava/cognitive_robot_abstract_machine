"""
Native workshop plans retain independent pickup and placement actions.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np

from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from cramera.workshop_demo import WorkshopDemo
from cramera.workshop_world import WorkshopAsset, WorkshopWorld
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
)

from .test_workshop_world import workshop_bundle
from .test_workshop_bundle import workshop_directory

# %% modular native plan


def test_native_plan_has_pickup_then_pilot_hole_placement(workshop_bundle) -> None:
    """
    The demo composes ordinary native actions for the selected independent screw.
    """
    demo = WorkshopDemo(workshop=WorkshopWorld(bundle_directory=workshop_bundle))
    world = demo.workshop.build()
    context = demo.build_context(world)
    actions = demo.actions(context)
    assert len(actions) == 2
    assert isinstance(actions[0], PickUpAction)
    assert isinstance(actions[1], PlaceAction)
    assert actions[0].object_designator is world.get_body_by_name(WorkshopAsset.SCREW)
    assert actions[1].object_designator is actions[0].object_designator
    np.testing.assert_allclose(
        actions[1].target_location.to_np(), demo.workshop.target_pose.to_np()
    )
    assert demo.build_plan(context).root is not None


def test_execution_observer_tracks_every_unselected_loose_body(workshop_bundle) -> None:
    """
    Acceptance detects an unintended movement of any other fastener or box.
    """
    demo = WorkshopDemo(workshop=WorkshopWorld(bundle_directory=workshop_bundle))
    world = demo.workshop.build()
    context = demo.build_context(world)
    observer = demo.observe(context)
    assert {str(body.name) for body in observer.other_poses} == {
        item.name for item in demo.workshop.objects if item.name != WorkshopAsset.SCREW
    }


def test_stationary_bench_plan_excludes_drive_from_controller(workshop_bundle) -> None:
    """
    The arm controller cannot move the base while aligning a screw with its socket.
    """
    demo = WorkshopDemo(workshop=WorkshopWorld(bundle_directory=workshop_bundle))
    world = demo.workshop.build()
    context = demo.build_context(world)
    assert isinstance(context.robot.root.parent_connection, FixedConnection)
    np.testing.assert_allclose(
        context.robot.root.global_pose.to_np(),
        demo.workshop.base_pose.to_np(),
        atol=1e-12,
    )
    assert any(
        connection.is_controlled
        for connection in context.robot.left_arm.active_connections
    )


def test_arm_worktop_margin_keeps_real_collision_protection(workshop_bundle) -> None:
    """
    A bench-scale planning margin still forbids any arm penetration.
    """
    demo = WorkshopDemo(workshop=WorkshopWorld(bundle_directory=workshop_bundle))
    world = demo.workshop.build()
    context = demo.build_context(world)
    gripper = context.robot.left_arm.end_effector.bodies_with_collision
    arm = [
        body
        for body in context.robot.left_arm.bodies_with_collision
        if body not in gripper
    ]
    matching = [
        rule
        for rule in world.collision_manager.default_rules
        if isinstance(rule, AvoidCollisionBetweenGroups) and rule.body_group_a == arm
    ]
    assert len(matching) == 1
    assert matching[0].buffer_zone_distance == demo.arm_clearance
    assert matching[0].violated_distance == 0.0


def test_replay_preserves_relative_original_asset_notice_links(
    workshop_bundle, tmp_path
) -> None:
    """
    Bundled original GLBs keep working permission and attribution references.
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "cramera/src/cramera/resources/aicor_workshop/assets/bin.glb"
    )
    output = tmp_path / "replay"
    copies = [
        output / "meshes/objects/screw_box.glb",
        output / "meshes/recording/assets/drawer.glb",
    ]
    for destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    demo = WorkshopDemo(
        workshop=WorkshopWorld(bundle_directory=workshop_bundle),
        output_directory=output,
    )
    demo.copy_notices()
    for mesh in copies:
        data = mesh.read_bytes()
        document_length = struct.unpack_from("<I", data, 12)[0]
        rights = json.loads(data[20 : 20 + document_length])["asset"]["extras"][
            "assetRights"
        ]
        for field in ("licenseFile", "noticeFile"):
            notice = mesh.parent / rights[field]
            assert notice.read_bytes() == (workshop_bundle / notice.name).read_bytes()
    assert (output / "rights.json").read_bytes() == (
        workshop_bundle / "rights.json"
    ).read_bytes()
