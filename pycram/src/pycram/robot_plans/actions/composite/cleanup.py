from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import Optional, Union

from krrood.entity_query_language.factories import entity, variable
from pycram.datastructures.enums import Arms
from pycram.plans.failures import NavigationGoalNotReachedError
from pycram.plans.factories import execute_single
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.composite.transporting import TransportAction
from semantic_digital_twin.reasoning.predicates import (
    movable_obstacle,
    near,
    on_supporting_surface,
)
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


def resolve_supporting_surface_annotation(
    world, support: Union[Body, HasSupportingSurface]
) -> HasSupportingSurface:
    if isinstance(support, HasSupportingSurface):
        return support

    for annotation in world.semantic_annotations:
        if isinstance(annotation, HasSupportingSurface) and annotation.root == support:
            return annotation

    raise ValueError(f"No HasSupportingSurface annotation found for body '{support}'.")


def query_bowls_on_support(
    world, support: Union[Body, HasSupportingSurface]
) -> list[Bowl]:
    support_annotation = resolve_supporting_surface_annotation(world, support)
    bowl = variable(Bowl, domain=world.get_semantic_annotations_by_type(Bowl))
    support_var = variable(support_annotation.__class__, domain=[support_annotation])
    result = list(
        entity(bowl).where(on_supporting_surface(bowl, support_var)).evaluate()
    )
    return sorted(result, key=lambda item: str(item.root.name))


def query_movable_obstacles_near(
    world,
    reference: Union[Body, HasSupportingSurface],
    radius: float = 1.0,
) -> list[Body]:
    reference_body = (
        reference.root if isinstance(reference, HasSupportingSurface) else reference
    )
    obstacle = variable(Body, domain=world.bodies)
    result = list(
        entity(obstacle)
        .where(
            movable_obstacle(obstacle),
            near(obstacle, reference_body, float(radius)),
        )
        .evaluate()
    )
    return sorted(result, key=lambda item: str(item.name))


@dataclass
class RelocateObstacleAction(ActionDescription):
    """
    Move a blocking obstacle away from a reference body by directly updating its world pose.
    """

    obstacle_designator: Body
    reference_body: Body
    relocation_distance: float = 0.75

    def execute(self) -> None:
        if self.obstacle_designator.parent_connection is None:
            raise ValueError(
                f"Obstacle '{self.obstacle_designator}' has no parent connection."
            )

        world_root = self.world.root
        obstacle_pose = self.world.transform(
            self.obstacle_designator.global_transform, world_root
        ).to_pose()
        reference_pose = self.world.transform(
            self.reference_body.global_transform, world_root
        ).to_pose()

        direction = np.array(
            [
                float(obstacle_pose.x - reference_pose.x),
                float(obstacle_pose.y - reference_pose.y),
            ],
            dtype=float,
        )
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([0.0, 1.0], dtype=float)
        else:
            direction /= norm

        obstacle_pose.x = float(obstacle_pose.x) + float(direction[0]) * float(
            self.relocation_distance
        )
        obstacle_pose.y = float(obstacle_pose.y) + float(direction[1]) * float(
            self.relocation_distance
        )

        parent = self.obstacle_designator.parent_connection.parent
        parent_pose = self.world.transform(
            obstacle_pose.to_homogeneous_matrix(), parent
        )

        with self.world.modify_world():
            self.obstacle_designator.parent_connection.origin = parent_pose
            self.world.update_forward_kinematics()


@dataclass
class CleanUpAction(ActionDescription):
    """
    Query bowls on a support surface and transport them to a target location.
    """

    source_support: Body
    target_location: Union[Pose, Body]
    arm: Optional[Arms] = None
    obstacle_search_radius: float = 1.0
    max_navigation_retries: int = 1

    def _resolved_target_location(self) -> Pose:
        if isinstance(self.target_location, Body):
            return self.target_location.global_pose.to_pose()
        return self.target_location

    def _recover_from_navigation_failure(self) -> bool:
        obstacles = query_movable_obstacles_near(
            self.world,
            self.source_support,
            radius=self.obstacle_search_radius,
        )
        if not obstacles:
            return False

        self.add_subplan(
            execute_single(
                RelocateObstacleAction(
                    obstacle_designator=obstacles[0],
                    reference_body=self.source_support,
                    relocation_distance=0.75,
                )
            )
        ).perform()
        return True

    def _transport_with_recovery(self, bowl_body: Body) -> None:
        attempts = 0
        while True:
            try:
                self.add_subplan(
                    execute_single(
                        TransportAction(
                            object_designator=bowl_body,
                            target_location=self._resolved_target_location(),
                            arm=self.arm,
                        )
                    )
                ).perform()
                return
            except NavigationGoalNotReachedError:
                if attempts >= int(self.max_navigation_retries):
                    raise
                if not self._recover_from_navigation_failure():
                    raise
                attempts += 1

    def execute(self) -> None:
        bowls = query_bowls_on_support(self.world, self.source_support)
        for bowl in bowls:
            self._transport_with_recovery(bowl.root)
