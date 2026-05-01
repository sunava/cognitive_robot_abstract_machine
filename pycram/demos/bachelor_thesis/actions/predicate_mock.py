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
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World


import math

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
