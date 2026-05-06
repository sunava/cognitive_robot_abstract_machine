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
from semantic_digital_twin.semantic_annotations.semantic_annotations import Food, Fruit, Cuttlery, Plate, Cup, Bowl, \
    Bottle, CounterTop, Table, ShelfLayer
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

from time import sleep


import math

from semantic_digital_twin.world_description.world_entity import SemanticAnnotation, Body

# TODO: Work in progress - How do I get from body to Semantic annotation
def misplaced(obj: Body, world: World):
    """
    returns true if object is misplaced and false if it is already at the correct location
    """

    sem_annotation = world.get_semantic_annotation_by_name(obj.name)

    if isinstance(sem_annotation, (Food, Bottle)):
        correct_location = world.get_semantic_annotations_by_name("cooking_table")[0]
    elif isinstance(sem_annotation, (Cuttlery, Plate, Bowl, Cup)):
        correct_location = world.get_semantic_annotations_by_name("counterTop")[0]
    else:
        correct_location = world.get_semantic_annotations_by_name("table")[0]


    if isinstance(correct_location, (CounterTop, Table, ShelfLayer)):
        # print("on correct surface: ", semantic_annotations_on_surfaces([correct_location], world))
        if semantic_annotations_on_surfaces([correct_location], world).__contains__(sem_annotation):
            return False, correct_location
        else:
            return True, correct_location
    else:
        TypeError("Correct location is not of type HasSupportingSurface")
        return True, correct_location


def is_empty(furniture: HasSupportingSurface, known_objects : list[SemanticAnnotation], world: World):
    """
    returns false, if the furniture has an already perceived object on it
    returns true, if it is empty (only acknowledged objects that are already perceived)
    """
    for ann in semantic_annotations_on_surfaces([furniture], world):
        if ann in known_objects:
            return False
    return True

def human_near():
    # only useful in real scenario, no human in simulation
    return False

    


def reachable(object : SemanticAnnotation):
    # debug, WIP for later
    return True


    surfaces = []
    for annotation in world.semantic_annotations:
        if isinstance(annotation, (Table, CounterTop)):#, ShelfLayer)):
            surfaces.append(annotation)


    object_surface = None
    for surface in surfaces:
        for anno in semantic_annotations_on_surfaces([surface], world):
            print(surface.name, anno.name)
            if anno == object_to_reach:
                object_surface = surface

    if object_surface is None:
        Exception(f"Object {object_to_reach} is not on a Table, CounterTop or ShelfLayer surface")

    surface_region = object_surface.supported_surface[0].get_region().combined_mesh()

    edges = surface_region.edges_boundary

    print("Region: ", surface_region, "\n Edges: ", edges)

    # TODO: calculate if the distance to the reachable edge is smaller then a minimum distance to estimate if the robot could reach the object

    # # --- distance check ---
    # dx = object_location.x - robot_location.x
    # dy = object_location.y - robot_location.y
    # dist = math.sqrt(dx**2 + dy**2)
    #
    # if dist > 0.8 or dist < 0.3:
    #     return False
    #
    # # --- height check ---
    # if object_location.z > Scalar(1.2):
    #     return False
    #
    # # --- blocking check ---
    # for obj in world.bodies:
    #
    #     if obj.global_pose is None:
    #         continue
    #
    #     if is_between(robot_location, object_location, obj.global_pose):
    #         if is_same_height(obj.global_pose, object_location):
    #             return False

    # return True


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

