"""
Where an object can be put down on or in an annotated surface: free spots sampled from
the surface's supporting region when the step runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Iterator, List, Optional

from cramera_bridge.plans.workspace import SurfaceAnnotations
from cramera_bridge.exceptions import SurfaceNotInWorld
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

SAMPLES_PER_SURFACE = 50
"""
How many spots are sampled from each matching surface.
"""

# %% targets


@dataclass(frozen=True)
class SurfaceTargets:
    """
    The poses an object can be put down at on the surfaces of one annotation class, or
    on one named surface; iterated lazily, so the spots are sampled when the step runs
    against the world as it then is.
    """

    world: World
    """
    The world holding the surfaces.
    """

    body: Body
    """
    The object put down.
    """

    surface_type: str
    """
    The class name of the surface annotation.
    """

    surface_name: Optional[str] = None
    """
    The name of one surface annotation, or None for every surface of the class.
    """

    def surfaces(self) -> List[HasSupportingSurface]:
        """
        :return: The matching surfaces of the world.
        :raises SurfaceNotInWorld: When the world has none.
        """
        annotation = SurfaceAnnotations().named(self.surface_type)
        surfaces = [
            surface
            for surface in self.world.get_semantic_annotations_by_type(annotation)
            if self.surface_name is None or str(surface.name) == self.surface_name
        ]
        if not surfaces:
            raise SurfaceNotInWorld(self.surface_type, self.surface_name)
        return surfaces

    def __iter__(self) -> Iterator[Pose]:
        """
        :return: Object poses resting on the surfaces, in the order the surfaces rate
            their spots, keeping the object's current orientation.
        """
        lift = self.bottom_offset()
        orientation = self.body.global_pose.orientation.to_np()
        for surface in self.surfaces():
            for point in surface.sample_points_from_surface(amount=SAMPLES_PER_SURFACE):
                position = self.world.transform(point, self.world.root).to_np()
                yield Pose.from_xyz_quaternion(
                    position[0],
                    position[1],
                    position[2] + lift,
                    *orientation[:4],
                    reference_frame=self.world.root,
                )

    def first(self) -> Pose:
        """
        :return: The best rated pose.
        :raises SurfaceNotInWorld: When no surface offers a spot.
        """
        for pose in self:
            return pose
        raise SurfaceNotInWorld(self.surface_type, self.surface_name)

    def bottom_offset(self) -> float:
        """
        :return: How far the object's origin lies above its lowest point, so it rests on
            a spot rather than sinking into it.
        """
        box = self.body.collision.as_bounding_box_collection_in_frame(
            self.body
        ).bounding_box()
        return -float(box.min_z)
