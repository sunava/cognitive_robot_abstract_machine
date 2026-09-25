"""
Load the portable woodworking workshop into independent CRAM worlds.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from cramera import paths
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection

# %% scene vocabulary


class WorkshopAsset(StrEnum):
    """
    Portable workshop resources and stable native body identifiers.
    """

    SCENE_NAME = "aicor_workshop"
    SCENE = "scene.json"
    RESOURCES = "resources"
    SCREW = "screw_1"
    SOCKET = "workshop_socket"
    ENVIRONMENT = "workshop_environment"
    ASSEMBLY = "workshop_assembly"
    FLOOR = "workshop_floor"


@dataclass(frozen=True)
class WorkshopObject:
    """
    A separate loose body with its own world placement.
    """

    name: str
    """
    Native root body identifier.
    """

    description: Path
    """
    URDF path relative to the resource bundle.
    """

    pose: Pose
    """
    Initial object transform in the world frame.
    """

    @classmethod
    def from_json(cls, value: dict) -> WorkshopObject:
        """
        Read an independently movable object from the renderer manifest.
        """
        return cls(
            value["key"], Path(value["urdf"]), Pose.from_xyz_quaternion(*value["spawn"])
        )


@dataclass
class WorkshopWorld:
    """
    Reusable environment with optional robot specifications and semantic handles.
    """

    bundle_directory: Path = field(
        default_factory=lambda: WorkshopWorld.resolve_directory()
    )
    """
    Resource directory containing the complete scene description.
    """

    base_pose: Pose = field(
        default_factory=lambda: Pose.from_xyz_rpy(x=0.16, y=-0.76, yaw=math.pi / 2)
    )
    """
    PR2 base starting pose clear of the front workbench edge.
    """

    scene: dict = field(init=False)
    """
    Authored rendering and interaction metadata.
    """

    objects: tuple[WorkshopObject, ...] = field(init=False)
    """
    Independent loose-part URDF declarations.
    """

    def __post_init__(self) -> None:
        """
        Resolve typed object declarations from the portable scene manifest.
        """
        self.bundle_directory = Path(self.bundle_directory)
        self.scene = json.loads(
            (self.bundle_directory / WorkshopAsset.SCENE).read_text()
        )
        self.objects = tuple(
            WorkshopObject.from_json(item) for item in self.scene["objects"]
        )

    @staticmethod
    def resolve_directory() -> Path:
        """
        Prefer the local authored scene and otherwise load packaged resources.
        """
        local = paths.resolve_scene_directory(WorkshopAsset.SCENE_NAME)
        return (
            local
            if local is not None
            else Path(__file__).resolve().parent
            / WorkshopAsset.RESOURCES
            / WorkshopAsset.SCENE_NAME
        )

    @property
    def target_pose(self) -> Pose:
        """
        Return a fresh screw-tip pose inside the open pilot hole.
        """
        return Pose.from_xyz_quaternion(
            *self.scene["workshop"]["assembly"]["socketPose"]
        )

    def create_world(self, *, robots: Sequence[RobotSpecification] = ()) -> World:
        """
        Create independent object and drawer state for any supplied robots.
        """
        world = WorldSpecification(robots=list(robots)).to_domain_object()
        [environment_entry] = [
            model for model in self.scene["models"] if not model["robot"]
        ]
        environment = URDFParser.from_file(
            str(self.bundle_directory / environment_entry["urdf"]), prefix=""
        ).parse()
        with world.modify_world():
            connection = FixedConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=environment.root,
                parent_T_connection_expression=HomogeneousTransformationMatrix(),
            )
            world.merge_world(environment, root_connection=connection)
        with world.modify_world():
            for drawer in self.scene["workshop"]["drawers"]:
                world.add_semantic_annotation(
                    Handle(root=world.get_body_by_name(drawer["handle"]))
                )
        for item in self.objects:
            object_world = URDFParser.from_file(
                str(self.bundle_directory / item.description), prefix=""
            ).parse()
            world.merge_world_at_pose(object_world, item.pose.to_homogeneous_matrix())
        return world

    def build(self) -> World:
        """
        Create a PR2 planning world facing the woodworking bench.
        """
        return self.create_world(
            robots=[
                RobotSpecification(
                    semantic_annotation_type=PR2,
                    odom_T_robot_start=self.base_pose.to_homogeneous_matrix(),
                )
            ]
        )
