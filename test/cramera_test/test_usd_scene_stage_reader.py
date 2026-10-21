"""
Reading composed USD stages into the neutral scene description.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import pytest

pxr = pytest.importorskip("pxr")

from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

from cramera.onboard.usd_scene.exceptions import (
    PrimNotFound,
    RobotExcluded,
    SelectedPrimLeftOut,
    StageNotFound,
)
from cramera.onboard.usd_scene.scene_description import (
    DetachedBody,
    EnvironmentArticulation,
    ExcludedPrim,
    ExclusionReason,
    GeneratedName,
    ImportedJoint,
    ImportedLink,
    JointKind,
    LightKind,
    MaterialSource,
    SceneDescription,
    SemanticAttributeName,
    SemanticNamespace,
    SkipReason,
)
from cramera.onboard.usd_scene.stage_reader import PrimExclusion, StageReader

DATASET = Path(__file__).parent / "dataset" / "usd_scene"
"""
Small stages, one per behaviour.
"""


def read(name: str, *prim_paths: str) -> SceneDescription:
    """
    The description of a fixture stage.

    :param name: The stage file name.
    :param prim_paths: The selected prims, none for the whole stage.
    """
    return StageReader(DATASET / name, list(prim_paths)).read()


def open_stage(name: str) -> Usd.Stage:
    """
    A fixture stage, to read authored values for comparison.

    :param name: The stage file name.
    """
    return Usd.Stage.Open(str(DATASET / name))


def world_points(link: ImportedLink, path: str) -> numpy.ndarray:
    """
    The world positions of a prim's mesh vertices.

    :param link: The link holding the mesh.
    :param path: The prim path.
    """
    mesh = next(mesh for mesh in link.meshes if mesh.path == path)
    return mesh.points @ link.pose.rotation.T + link.pose.translation


def link_holding(description: SceneDescription, path: str) -> ImportedLink:
    """
    The link a prim's mesh belongs to.

    :param description: The scene.
    :param path: The prim path.
    """
    return next(
        link
        for link in description.links
        if any(mesh.path == path for mesh in link.meshes)
    )


# %% geometry and materials


class TestTexturedQuad:
    """
    An OmniPBR quad with three textures and semantic attributes.
    """

    def test_texture_coordinates_are_kept_as_authored(self):
        description = read("textured_quad.usda")
        stage = open_stage("textured_quad.usda")
        authored = numpy.asarray(
            UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Decal"))
            .GetPrimvar("st")
            .Get()
        )
        mesh = link_holding(description, "/World/Decal").meshes[0]
        assert numpy.array_equal(
            mesh.texture_coordinates[mesh.triangles.ravel()],
            authored[[0, 1, 2, 0, 2, 3]],
        )

    def test_material_inputs_resolve_textures(self):
        description = read("textured_quad.usda")
        material = description.materials["/World/Looks/DecalPaint"]
        assert material.source is MaterialSource.OMNI_PBR
        assert (
            material.inputs["diffuse_texture"].path
            == DATASET / "textures" / "albedo.png"
        )
        assert material.inputs["diffuse_tint"] == (0.5, 1.0, 1.0)
        assert DATASET / "textures" / "roughness.png" in description.used_files

    def test_semantic_attributes_are_grouped_by_namespace(self):
        description = read("textured_quad.usda")
        semantics = {item.path: item for item in description.semantics}
        assert semantics["/World/Decal"].attributes == {
            "sdt": {"class": "AprilTag", "tagId": 7}
        }
        assert (
            semantics["/World/Decal"].link
            == link_holding(description, "/World/Decal").name
        )


def test_scaled_cube_keeps_its_authored_extent():
    description = read("glass_pane.usda")
    stage = open_stage("glass_pane.usda")
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(stage.GetPrimAtPath("/World/Pane"))
        .ComputeAlignedRange()
    )
    points = world_points(link_holding(description, "/World/Pane"), "/World/Pane")
    assert numpy.allclose(points.min(axis=0), bounds.GetMin())
    assert numpy.allclose(points.max(axis=0), bounds.GetMax())


def test_y_up_centimetre_stage_becomes_z_up_metres():
    description = read("y_up_centimetres.usda")
    stage = open_stage("y_up_centimetres.usda")
    authored = numpy.asarray(
        UsdGeom.Mesh(stage.GetPrimAtPath("/World/Marker")).GetPointsAttr().Get()
    )
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    points = world_points(link_holding(description, "/World/Marker"), "/World/Marker")
    assert numpy.allclose(points[:, 2], authored[:, 1] * scale)
    assert numpy.allclose(points[:, 0], authored[:, 0] * scale)
    assert numpy.allclose(points[:, 1], -authored[:, 2] * scale)


def test_unbound_geometry_is_shaded_by_its_display_colour():
    description = read("y_up_centimetres.usda")
    mesh = link_holding(description, "/World/Marker").meshes[0]
    material = description.materials[mesh.material]
    assert material.source is MaterialSource.DISPLAY_COLOR
    assert material.inputs["displayColor"] == pytest.approx((0.1, 0.8, 0.2))


def test_instance_proxies_are_converted_per_instance():
    description = read("instanced_bolts.usda")
    first = world_points(
        link_holding(description, "/World/BoltA/Head"), "/World/BoltA/Head"
    )
    second = world_points(
        link_holding(description, "/World/BoltB/Head"), "/World/BoltB/Head"
    )
    assert numpy.allclose(second - first, [1.0, 0.0, 0.0])
    assert {link.path for link in description.links} == {
        "/World",
        "/World/BoltA",
        "/World/BoltB",
    }


def test_guide_proxy_invisible_and_unsupported_prims_are_skipped():
    description = read("hidden_prims.usda")
    assert [mesh.path for link in description.links for mesh in link.meshes] == [
        "/World/Visible"
    ]
    assert dict(description.report.skipped) == {
        SkipReason.GUIDE_PURPOSE: 1,
        SkipReason.PROXY_PURPOSE: 1,
        SkipReason.INVISIBLE: 2,
        SkipReason.UNSUPPORTED_GEOMETRY: 1,
    }


def test_selected_prims_limit_the_conversion():
    description = read("glass_pane.usda", "/World/Frame")
    assert [mesh.path for link in description.links for mesh in link.meshes] == [
        "/World/Frame"
    ]


def test_nested_selections_convert_each_prim_once():
    description = read("glass_pane.usda", "/World", "/World/Frame")
    paths = [mesh.path for link in description.links for mesh in link.meshes]
    assert sorted(paths) == ["/World/Frame", "/World/Pane"]


def test_missing_stage_and_prim_are_refused():
    with pytest.raises(StageNotFound):
        StageReader(DATASET / "missing.usda").read()
    with pytest.raises(PrimNotFound):
        read("glass_pane.usda", "/World/Nothing")


# %% joints


class TestJointedCabinet:
    """
    A cabinet whose door, drawer and wheel are jointed to its body.
    """

    @pytest.fixture(scope="class")
    def description(self) -> SceneDescription:
        return read("jointed_cabinet.usda")

    def joint(self, description: SceneDescription, path: str):
        return next(joint for joint in description.joints if joint.path == path)

    def test_joint_kinds_follow_the_usd_joint_types(self, description):
        kinds = {
            joint.path: joint.kind
            for joint in description.joints
            if joint.path.startswith("/World/Cabinet/Joints")
        }
        assert kinds == {
            "/World/Cabinet/Joints/RootJoint": JointKind.FIXED,
            "/World/Cabinet/Joints/DoorHinge": JointKind.REVOLUTE,
            "/World/Cabinet/Joints/DrawerSlide": JointKind.PRISMATIC,
            "/World/Cabinet/Joints/WheelSpin": JointKind.CONTINUOUS,
            "/World/Cabinet/Joints/FlapHinge": JointKind.REVOLUTE,
        }

    def test_disabled_joint_yields_to_the_enabled_joint_of_its_body(self, description):
        assert description.report.skipped_examples[SkipReason.LOOP_CLOSING_JOINT] == [
            "/World/Cabinet/Joints/DrawerLock"
        ]

    def test_disabled_joint_still_moves_its_only_body(self, description):
        flap = self.joint(description, "/World/Cabinet/Joints/FlapHinge")
        assert not flap.enabled
        assert flap in description.movable_joints

    def test_initial_values_are_recovered_from_body_poses(self, description):
        stage = open_stage("jointed_cabinet.usda")
        door_angle = (
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/Cabinet/Door"))
            .GetOrderedXformOps()[1]
            .Get()
        )
        drawer = (
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/Cabinet/Drawer"))
            .GetOrderedXformOps()[0]
            .Get()
        )
        slide = UsdPhysics.PrismaticJoint(
            stage.GetPrimAtPath("/World/Cabinet/Joints/DrawerSlide")
        )
        assert self.joint(
            description, "/World/Cabinet/Joints/DoorHinge"
        ).initial_value == pytest.approx(numpy.radians(door_angle))
        assert self.joint(
            description, "/World/Cabinet/Joints/DrawerSlide"
        ).initial_value == pytest.approx(drawer[0] - slide.GetLocalPos0Attr().Get()[0])
        assert self.joint(
            description, "/World/Cabinet/Joints/WheelSpin"
        ).initial_value == pytest.approx(0.0)

    def test_limits_are_converted_to_radians_and_metres(self, description):
        stage = open_stage("jointed_cabinet.usda")
        hinge = UsdPhysics.RevoluteJoint(
            stage.GetPrimAtPath("/World/Cabinet/Joints/DoorHinge")
        )
        joint = self.joint(description, "/World/Cabinet/Joints/DoorHinge")
        assert (joint.lower, joint.upper) == pytest.approx(
            (
                numpy.radians(hinge.GetLowerLimitAttr().Get()),
                numpy.radians(hinge.GetUpperLimitAttr().Get()),
            )
        )
        spin = self.joint(description, "/World/Cabinet/Joints/WheelSpin")
        assert (spin.lower, spin.upper) == (None, None)

    def test_joint_tree_reproduces_every_link_frame(self, description):
        links = {link.name: link for link in description.links}
        for joint in description.joints:
            placed = (
                links[joint.parent].pose
                @ joint.origin
                @ joint.motion(joint.initial_value)
            )
            assert placed.is_close(links[joint.child].pose), joint.name

    def test_geometry_keeps_its_authored_world_position(self, description):
        stage = open_stage("jointed_cabinet.usda")
        bounds = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            .ComputeWorldBound(stage.GetPrimAtPath("/World/Cabinet/Door/Leaf"))
            .ComputeAlignedRange()
        )
        points = world_points(
            link_holding(description, "/World/Cabinet/Door/Leaf"),
            "/World/Cabinet/Door/Leaf",
        )
        assert numpy.allclose(points.min(axis=0), bounds.GetMin())
        assert numpy.allclose(points.max(axis=0), bounds.GetMax())

    def test_rigid_body_without_joint_is_free_and_attached_to_its_asset(
        self, description
    ):
        links = {link.path: link for link in description.links}
        assert links["/World/Cabinet/Bin"].free
        assert not links["/World/Cabinet/Body"].free
        attachment = next(
            joint
            for joint in description.joints
            if joint.child == links["/World/Cabinet/Bin"].name
        )
        assert (attachment.kind, attachment.parent) == (
            JointKind.FIXED,
            links["/World/Cabinet"].name,
        )

    def test_joint_to_the_world_hangs_from_the_enclosing_asset(self, description):
        links = {link.path: link for link in description.links}
        root_joint = self.joint(description, "/World/Cabinet/Joints/RootJoint")
        assert root_joint.parent == links["/World/Cabinet"].name
        cabinet = next(
            joint
            for joint in description.joints
            if joint.child == links["/World/Cabinet"].name
        )
        assert cabinet.parent == description.root_link

    def test_asset_semantics_belong_to_the_asset_link(self, description):
        semantics = {item.path: item for item in description.semantics}
        links = {link.path: link for link in description.links}
        assert semantics["/World/Cabinet"].link == links["/World/Cabinet"].name

    def test_inherited_material_binding_is_used(self, description):
        mesh = link_holding(description, "/World/Cabinet/Body/Shell").meshes[0]
        assert (
            description.materials[mesh.material].source
            is MaterialSource.PREVIEW_SURFACE
        )

    def test_names_are_unique_and_drop_the_default_prim(self, description):
        names = [link.name for link in description.links]
        assert len(names) == len(set(names))
        assert "Cabinet__Door" in names
        assert description.root_link == "World"


def joint_at(description: SceneDescription, path: str) -> ImportedJoint:
    """
    The joint authored at a path.

    :param description: The scene.
    :param path: The joint prim's path.
    """
    return next(joint for joint in description.joints if joint.path == path)


def reproduces_link_frames(description: SceneDescription) -> bool:
    """
    Whether every joint, moved to its initial value, places its child link where the
    stage put it.

    :param description: The scene.
    """
    links = {link.name: link for link in description.links}
    return all(
        (
            links[joint.parent].pose @ joint.origin @ joint.motion(joint.initial_value)
        ).is_close(links[joint.child].pose)
        for joint in description.joints
    )


class TestJointNamingOnlyItsFirstBody:
    """
    A door hinged to the world by a joint that names only ``physics:body0``.
    """

    @pytest.fixture(scope="class")
    def description(self) -> SceneDescription:
        return read("body0_hinge.usda")

    def test_value_keeps_the_sign_usd_measures_the_second_frame_with(self, description):
        stage = open_stage("body0_hinge.usda")
        opened = (
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/Door"))
            .GetOrderedXformOps()[1]
            .Get()
        )
        hinge = joint_at(description, "/World/Hinge")
        assert hinge.initial_value == pytest.approx(-numpy.radians(opened))
        assert hinge.lower <= hinge.initial_value <= hinge.upper

    def test_joint_tree_reproduces_the_door(self, description):
        assert reproduces_link_frames(description)


class TestHingeOpenedPastHalfATurn:
    """
    A lid opened 200 degrees on a hinge limited to 0 to 270 degrees.
    """

    @pytest.fixture(scope="class")
    def description(self) -> SceneDescription:
        return read("wide_hinge.usda")

    def test_value_lies_within_the_limits(self, description):
        stage = open_stage("wide_hinge.usda")
        opened = (
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/Lid"))
            .GetOrderedXformOps()[1]
            .Get()
        )
        assert joint_at(description, "/World/Hinge").initial_value == pytest.approx(
            numpy.radians(opened)
        )

    def test_joint_tree_reproduces_the_lid(self, description):
        assert reproduces_link_frames(description)


def test_stage_without_default_prim_roots_its_links_in_the_world():
    description = read("no_default_prim.usda")
    assert description.root_link == GeneratedName.WORLD
    assert [joint.parent for joint in description.joints] == [GeneratedName.WORLD]


# %% robots and excluded prims

ROBOT_MARKERS = "robot_markers.usda"
"""
A cell with one robot marked by Isaac Sim's robot schema, one marked by its entity type,
and environment prims jointed to both.
"""


def read_robot_markers(exclusion: PrimExclusion) -> SceneDescription:
    """
    The description of the robot marker stage under an exclusion.

    :param exclusion: The prims to leave out.
    """
    return StageReader(DATASET / ROBOT_MARKERS, exclusion=exclusion).read()


def mesh_paths(description: SceneDescription) -> set[str]:
    """
    The prims whose geometry a description holds.

    :param description: The scene.
    """
    return {mesh.path for link in description.links for mesh in link.meshes}


def authored_joint_paths(description: SceneDescription) -> set[str]:
    """
    The joint prims a description converted, without the generated attachments.

    :param description: The scene.
    """
    return {joint.path for joint in description.joints if joint.child_body is not None}


def lies_in(path: str, subtrees: list[str]) -> bool:
    """
    Whether a prim path lies in one of some subtrees.

    :param path: The prim path.
    :param subtrees: The subtree root paths.
    """
    return any(path == root or path.startswith(root + "/") for root in subtrees)


class TestMarkedRobotsAreLeftOut:
    """
    Without being asked to keep them, the robots a stage marks are not converted.
    """

    @pytest.fixture(scope="class")
    def description(self) -> SceneDescription:
        return read_robot_markers(PrimExclusion())

    def test_robots_are_reported_with_the_mark_found(self, description):
        assert description.report.excluded == [
            ExcludedPrim("/World/Robots/Rover", ExclusionReason.ISAAC_ROBOT_SCHEMA),
            ExcludedPrim("/World/Robots/Walker", ExclusionReason.ROBOT_ENTITY_TYPE),
        ]

    def test_only_the_environment_geometry_is_converted(self, description):
        assert mesh_paths(description) == {
            "/World/Floor",
            "/World/Robots/Pedestal/Column",
            "/World/Cell/Bench",
            "/World/Cart/Tray",
            "/World/Hook/Bar",
        }

    def test_no_link_stands_for_a_prim_of_a_robot(self, description):
        robots = [excluded.path for excluded in description.report.excluded]
        assert [
            link.path for link in description.links if lies_in(link.path, robots)
        ] == []

    def test_joints_to_robot_prims_are_skipped(self, description):
        assert description.report.skipped_examples[
            SkipReason.JOINT_TO_EXCLUDED_PRIM
        ] == ["/World/Joints/Tow", "/World/Joints/Dock", "/World/Joints/Grip"]
        assert authored_joint_paths(description) == set()

    def test_every_joint_connects_converted_links(self, description):
        names = {link.name for link in description.links}
        assert all(
            joint.parent in names and joint.child in names
            for joint in description.joints
        )

    def test_body_towed_by_a_robot_is_free_on_the_world(self, description):
        cart = next(link for link in description.links if link.path == "/World/Cart")
        joint = next(joint for joint in description.joints if joint.child == cart.name)
        assert (joint.parent, joint.kind, cart.free) == (
            description.root_link,
            JointKind.FIXED,
            True,
        )

    def test_prim_held_only_by_a_robot_is_static_geometry(self, description):
        hook = next(link for link in description.links if link.path == "/World/Hook")
        joint = next(joint for joint in description.joints if joint.child == hook.name)
        assert (joint.parent, joint.child_body, hook.free) == (
            description.root_link,
            None,
            False,
        )

    def test_cameras_and_semantics_of_robots_are_left_out(self, description):
        assert description.cameras == []
        assert [item.path for item in description.semantics] == ["/World/Cell"]

    def test_bodies_that_hung_from_a_robot_are_reported_detached(self, description):
        assert description.report.detached == [
            DetachedBody("/World/Hook", "/World/Robots/Walker", "/World/Joints/Grip"),
            DetachedBody(
                "/World/Cart", "/World/Robots/Rover/Chassis", "/World/Joints/Tow"
            ),
        ]


def test_entity_type_mark_is_the_attribute_the_stage_authors():
    stage = open_stage(ROBOT_MARKERS)
    walker = stage.GetPrimAtPath("/World/Robots/Walker")
    assert walker.HasAttribute(
        SemanticNamespace.SEMANTIC_DIGITAL_TWIN.attribute_name(
            SemanticAttributeName.ENTITY_TYPE
        )
    )


def test_excluded_prim_leaves_out_its_subtree():
    description = read_robot_markers(PrimExclusion(excluded_paths=["/World/Robots"]))
    assert description.report.excluded == [
        ExcludedPrim("/World/Robots", ExclusionReason.REQUESTED)
    ]
    assert mesh_paths(description) == {
        "/World/Floor",
        "/World/Cell/Bench",
        "/World/Cart/Tray",
        "/World/Hook/Bar",
    }


def test_kept_robots_stay_in_the_environment():
    kept = read_robot_markers(PrimExclusion(keep_robots=True))
    assert kept.report.excluded == []
    assert mesh_paths(kept) == {
        "/World/Floor",
        "/World/Robots/Rover/Chassis/Body",
        "/World/Robots/Rover/Chassis/Wheel/Tyre",
        "/World/Robots/Walker/Torso",
        "/World/Robots/Pedestal/Column",
        "/World/Cell/Bench",
        "/World/Cart/Tray",
        "/World/Hook/Bar",
    }
    assert authored_joint_paths(kept) == {
        "/World/Robots/Rover/Joints/WheelJoint",
        "/World/Joints/Tow",
        "/World/Joints/Dock",
        "/World/Joints/Grip",
    }


def test_robot_converted_as_its_own_model_is_kept():
    description = read_robot_markers(PrimExclusion(robot_path="/World/Robots/Rover"))
    assert description.report.excluded == [
        ExcludedPrim("/World/Robots/Walker", ExclusionReason.ROBOT_ENTITY_TYPE)
    ]
    assert mesh_paths(description) == {
        "/World/Floor",
        "/World/Robots/Rover/Chassis/Body",
        "/World/Robots/Rover/Chassis/Wheel/Tyre",
        "/World/Robots/Pedestal/Column",
        "/World/Cell/Bench",
        "/World/Cart/Tray",
        "/World/Hook/Bar",
    }


def test_robot_root_link_below_the_marked_prim_keeps_its_robot():
    description = read_robot_markers(
        PrimExclusion(robot_path="/World/Robots/Rover/Chassis")
    )
    assert [excluded.path for excluded in description.report.excluded] == [
        "/World/Robots/Walker"
    ]


def test_missing_excluded_prim_is_refused():
    with pytest.raises(PrimNotFound):
        read_robot_markers(PrimExclusion(excluded_paths=["/World/Nothing"]))


def test_robot_inside_an_excluded_prim_is_refused():
    with pytest.raises(RobotExcluded):
        read_robot_markers(
            PrimExclusion(
                excluded_paths=["/World/Robots"], robot_path="/World/Robots/Rover"
            )
        )


@pytest.mark.parametrize(
    "prim_path, left_out",
    [
        (
            "/World/Robots/Rover",
            ExcludedPrim("/World/Robots/Rover", ExclusionReason.ISAAC_ROBOT_SCHEMA),
        ),
        (
            "/World/Robots/Rover/Chassis",
            ExcludedPrim("/World/Robots/Rover", ExclusionReason.ISAAC_ROBOT_SCHEMA),
        ),
        (
            "/World/Robots/Walker/Torso",
            ExcludedPrim("/World/Robots/Walker", ExclusionReason.ROBOT_ENTITY_TYPE),
        ),
    ],
)
def test_selected_prim_in_a_marked_robot_is_refused(prim_path, left_out):
    with pytest.raises(SelectedPrimLeftOut) as refused:
        StageReader(DATASET / ROBOT_MARKERS, [prim_path]).read()
    assert (refused.value.prim_path, refused.value.left_out) == (prim_path, left_out)


def test_selected_prim_in_an_excluded_prim_is_refused():
    with pytest.raises(SelectedPrimLeftOut) as refused:
        StageReader(
            DATASET / ROBOT_MARKERS,
            ["/World/Robots/Pedestal"],
            exclusion=PrimExclusion(excluded_paths=["/World/Robots"]),
        ).read()
    assert refused.value.left_out == ExcludedPrim(
        "/World/Robots", ExclusionReason.REQUESTED
    )


@pytest.mark.parametrize(
    "exclusion",
    [PrimExclusion(keep_robots=True), PrimExclusion(robot_path="/World/Robots/Rover")],
)
def test_selected_prim_in_a_kept_robot_is_converted(exclusion):
    description = StageReader(
        DATASET / ROBOT_MARKERS, ["/World/Robots/Rover/Chassis"], exclusion=exclusion
    ).read()
    assert mesh_paths(description) == {
        "/World/Robots/Rover/Chassis/Body",
        "/World/Robots/Rover/Chassis/Wheel/Tyre",
    }


def test_joints_to_robots_beside_the_selected_prims_are_skipped():
    whole_stage = read_robot_markers(PrimExclusion())
    robots = [excluded.path for excluded in whole_stage.report.excluded]
    description = StageReader(
        DATASET / ROBOT_MARKERS, ["/World/Cart", "/World/Hook", "/World/Joints"]
    ).read()
    assert (
        description.report.skipped_examples[SkipReason.JOINT_TO_EXCLUDED_PRIM]
        == whole_stage.report.skipped_examples[SkipReason.JOINT_TO_EXCLUDED_PRIM]
    )
    assert authored_joint_paths(description) == set()
    assert [link.path for link in description.links if lies_in(link.path, robots)] == []
    assert description.report.detached == whole_stage.report.detached


# %% robot parts and articulations the stage leaves unmarked


def test_robot_parts_beside_a_marked_robot_are_reported_with_their_moving_joints():
    description = StageReader(DATASET / "detached_gripper.usda").read()
    assert description.report.detached == [
        DetachedBody(
            "/World/Tote",
            "/World/Robots/Rover/Chassis",
            "/World/Robots/Rover/Joints/Carry",
        ),
        DetachedBody(
            "/World/Robots/RoverGripper/Palm",
            "/World/Robots/Rover/Chassis",
            "/World/Robots/RoverGripper/Mount",
            ["/World/Robots/RoverGripper/FingerSlide"],
        ),
    ]


def test_kept_robot_detaches_nothing():
    description = StageReader(
        DATASET / "detached_gripper.usda",
        exclusion=PrimExclusion(keep_robots=True),
    ).read()
    assert description.report.detached == []


UNMARKED_ARTICULATION = "unmarked_articulation.usda"
"""
An unmarked robot, a marked one and environment articulations with and without moving
joints.
"""


@pytest.mark.parametrize(
    "exclusion",
    [
        PrimExclusion(),
        PrimExclusion(keep_robots=True),
        PrimExclusion(robot_path="/World/Robots/Arm"),
    ],
)
def test_articulations_with_moving_joints_outside_robots_are_reported(exclusion):
    description = StageReader(
        DATASET / UNMARKED_ARTICULATION, exclusion=exclusion
    ).read()
    assert description.report.articulations == [
        EnvironmentArticulation(
            "/World/Robots/Amr",
            ["/World/Robots/Amr/Joints/DriveX", "/World/Robots/Amr/Joints/LidSlide"],
        ),
        EnvironmentArticulation("/World/Cabinet", ["/World/Cabinet/Joints/Hinge"]),
        EnvironmentArticulation("/World/Lift/Mast", ["/World/Lift/Joints/Raise"]),
    ]


def test_excluded_articulation_is_not_reported():
    description = StageReader(
        DATASET / UNMARKED_ARTICULATION,
        exclusion=PrimExclusion(excluded_paths=["/World/Robots/Amr"]),
    ).read()
    assert [item.path for item in description.report.articulations] == [
        "/World/Cabinet",
        "/World/Lift/Mast",
    ]


# %% lights


def test_lights_carry_direction_size_and_exposure():
    description = read("lit_room.usda")
    stage = open_stage("lit_room.usda")
    lights = {light.kind: light for light in description.lights}
    panel = UsdLux.RectLight(stage.GetPrimAtPath("/World/Panel"))
    assert (
        lights[LightKind.RECTANGLE].intensity
        == panel.GetIntensityAttr().Get() * 2 ** panel.GetExposureAttr().Get()
    )
    assert (
        lights[LightKind.RECTANGLE].width,
        lights[LightKind.RECTANGLE].height,
    ) == pytest.approx((panel.GetWidthAttr().Get(), panel.GetHeightAttr().Get()))
    assert numpy.allclose(
        lights[LightKind.RECTANGLE].emission_direction, [0.0, 0.0, -1.0]
    )
    assert lights[LightKind.DISTANT].color_temperature == 5600.0
    tilt = numpy.radians(30.0)
    assert numpy.allclose(
        lights[LightKind.DISTANT].emission_direction,
        [0.0, numpy.sin(tilt), -numpy.cos(tilt)],
    )
    assert lights[LightKind.DOME].color == pytest.approx((0.8, 0.88, 1.0))
