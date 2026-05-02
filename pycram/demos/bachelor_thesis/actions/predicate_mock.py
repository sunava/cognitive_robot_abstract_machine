"""
put knowledge queries here, like:
empty(supportingSurface) = true
def misplaced(object):
    get objects location
    if location = place where belongs -> return false
    else -> return true
"""
from sqlalchemy.dialects.oracle.dictionary import all_objects

from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.reasoning.queries import semantic_annotations_on_surfaces
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Food, Fruit, Cuttlery, Plate, Cup, Bowl
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World


import math

from semantic_digital_twin.world_description.world_entity import SemanticAnnotation, Body

# TODO: Work in progress
def misplaced(obj: Body, world: World):
    """
    returns true if object is misplaced and false if it is already at the correct location
    """
    print(obj.name)
    print(world.get_semantic_annotation_by_name(obj.name.name.slice(0, -4)))

    sem_annotation = world.get_semantic_annotation_by_id(obj.id)

    if isinstance(sem_annotation, Food):
        correct_location = world.get_semantic_annotations_by_name("cooking_table")
    elif isinstance(sem_annotation, (Cuttlery, Plate, Bowl, Cup)):
        correct_location = world.get_semantic_annotations_by_name("counterTop")
    else:
        correct_location = world.get_semantic_annotations_by_name("table")


    if isinstance(correct_location, HasSupportingSurface):
        if semantic_annotations_on_surfaces([correct_location], world).__contains__(sem_annotation):
            return False, correct_location
        else:
            return True, correct_location
    else:
        TypeError("Correct location is not of type HasSupportingSurface")
        return True



    


def reachable(object_location: Pose, robot_location: Pose, world: World):

    # --- distance check ---
    dx = object_location.x - robot_location.x
    dy = object_location.y - robot_location.y
    dist = math.sqrt(dx**2 + dy**2)

    if dist > 0.8 or dist < 0.3:
        return False

    # --- height check ---
    if object_location.z > Scalar(1.2):
        return False

    # --- blocking check ---
    for obj in world.bodies:

        if obj.global_pose is None:
            continue

        if is_between(robot_location, object_location, obj.global_pose):
            if is_same_height(obj.global_pose, object_location):
                return False

    return True


def is_between(p1: Pose, p2: Pose, p: Pose):
    """
    Checks if point p lies approximately on the segment p1->p2
    """

    # vector projection
    dx1 = p2.x - p1.x
    dy1 = p2.y - p1.y

    dx2 = p.x - p1.x
    dy2 = p.y - p1.y

    dot = dx1*dx2 + dy1*dy2
    len_sq = dx1*dx1 + dy1*dy1

    if dot < 0 or dot > len_sq:
        return False

    # distance from line
    cross = abs(dx1*dy2 - dy1*dx2)
    dist = cross / math.sqrt(len_sq)

    return dist < 0.1


def is_same_height(p1: Pose, p2: Pose):
    return abs(p1.z - p2.z) < 0.1
