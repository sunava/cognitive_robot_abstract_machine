"""
The laboratory as a CRAM world: the bench's environment with its joints, the vessels and
the stopper as free bodies with their collision shapes, the closed cabinet panels the
environment leaves without collision, and the handles annotated for grasping.

Loading builds a planning world; it starts no motion and no simulation. Robots are
added through the usual :class:`~semantic_digital_twin.specifications.robots.RobotSpecification`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    FileUriResolver,
    PackageUriResolver,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cramera_bridge.scenes.bundle_world import BundleEnvironment, PoseText
from cramera_protocol.laboratory.cabinet import CabinetStructure
from cramera_protocol.bundle.scene import ObjectDescription
from cramera_protocol.laboratory.layout import (
    BenchLayout,
    LaboratorySceneKey,
    LaboratorySemanticsKey,
    VesselDimensions,
)

# %% names


class LaboratoryHandle(StrEnum):
    """
    The frames on the laboratory's handle bars.
    """

    DRAWER = "laboratory_drawer_handle"
    """
    The bench drawer's handle.
    """

    RACK = "laboratory_rack_handle"
    """
    The carrying handle of the rack.
    """

    CABINET_LOWER = "laboratory_cabinet_lower_handle"
    """
    The handle of the lower cabinet door.
    """

    CABINET_MIDDLE = "laboratory_cabinet_middle_handle"
    """
    The handle of the middle cabinet door.
    """

    CABINET_RIGHT = "laboratory_cabinet_right_handle"
    """
    The handle of the right cabinet door.
    """

    CENTRIFUGE_LID = "laboratory_centrifuge_lid_handle"
    """
    The handle of the centrifuge's lid.
    """

    REFRIGERATOR_DOOR = "laboratory_refrigerator_door_handle"
    """
    The handle of the refrigerator's door.
    """


class LaboratoryLink(StrEnum):
    """
    The environment links the world refers to.
    """

    RACK = "laboratory_rack"
    """
    The rack, which the vessels rest in.
    """


# %% environment


@dataclass
class LaboratoryEnvironment(BundleEnvironment):
    """
    Builds CRAM worlds holding a laboratory bundle.
    """

    allow_support_contacts: bool = field(default=False, kw_only=True)
    """
    Whether collision checks leave out the contacts between the vessels and the rack
    they rest in; nothing else is left out.
    """

    @property
    def bench(self) -> BenchLayout:
        """
        The bench the bundle describes.
        """
        return BenchLayout.from_json(
            self.contents.scene.extensions[LaboratorySceneKey.LABORATORY]
        )

    @property
    def vessels(self) -> VesselDimensions:
        """
        The dimensions every vessel shares, from the bundle's semantics.
        """
        return VesselDimensions.from_json(
            self.contents.semantics.extensions[LaboratorySemanticsKey.VESSEL]
        )

    def populate(self, world: World) -> None:
        """
        Add the environment, the free objects with their collision shapes, the cabinet
        panels and the handle annotations.

        :param world: The world receiving the laboratory.
        """
        super().populate(world)
        self.annotate_handles(world)
        self.add_cabinet(world)
        if self.allow_support_contacts:
            self.allow_rack_contacts(world)

    def add_object(self, world: World, description: ObjectDescription) -> Body:
        """
        Add a free object from its own URDF, with its collision shapes and inertia, at
        its spawn pose; an object without a URDF is drawn from its mesh.

        :param world: The world receiving the object.
        :param description: The object's description.
        :return: The object's body.
        """
        if description.urdf is None:
            return super().add_object(world, description)
        path = self.directory / description.urdf
        model = URDFParser.from_file(
            str(path),
            prefix="",
            path_resolver=CompositePathResolver(
                [FileUriResolver(base_directory=str(path.parent)), PackageUriResolver()]
            ),
        ).parse()
        root = model.root.id
        world.merge_world_at_pose(model, PoseText(description.spawn).native(world.root))
        return world.get_kinematic_structure_entity_by_id(root)

    def annotate_handles(self, world: World) -> None:
        """
        :param world: The world holding the laboratory.
        """
        with world.modify_world():
            for body in world.bodies:
                if body.name.name in LaboratoryHandle:
                    world.add_semantic_annotation(Handle(root=body))

    def add_cabinet(self, world: World) -> None:
        """
        :param world: The world holding the laboratory.
        """
        cabinet = CabinetStructure()
        BodySpecification(
            name=cabinet.name,
            shapes=ShapeCollection(
                shapes=[
                    Box(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            *panel.center
                        ),
                        scale=Scale(*panel.size),
                    )
                    for panel in cabinet.panels()
                ]
            ),
            visual_shapes=ShapeCollection(),
        ).spawn(world)

    def allow_rack_contacts(self, world: World) -> None:
        """
        :param world: The world holding the laboratory.
        """
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    body_group_a=[
                        world.get_body_by_name(placement.key)
                        for placement in self.bench.tubes
                    ],
                    body_group_b=[world.get_body_by_name(LaboratoryLink.RACK)],
                )
            )

    def slot_pose(self, identifier: str, world: World) -> Pose:
        """
        :param identifier: A rack slot.
        :param world: A world holding the laboratory.
        :return: Where a vessel's bottom stands in the slot, in the world's root frame.
        """
        pose = self.bench.slot(identifier).pose
        return Pose.from_xyz_quaternion(
            pose.x,
            pose.y,
            pose.z,
            pose.qx,
            pose.qy,
            pose.qz,
            pose.qw,
            reference_frame=world.root,
        )
