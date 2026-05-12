from abc import ABC, abstractmethod
import os
from typing import Any
from contextlib import contextmanager

from hypothesis.stateful import precondition
from scipy.constants import precision

from demos.bachelor_thesis.actions.predicate_mock import (
    reachable,
    is_empty,
    semantic_annotations_on_surface_cached,
    is_supported_by_surface_cached,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import SemanticAnnotationNotInWorldError
from semantic_digital_twin.reasoning.queries import semantic_annotations_on_surfaces
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Plate, Spoon, Knife, Cup, Milk, \
    Banana, Bread, Cuttlery, Table
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation, Body


def perf_print(message: str):
    pass


@contextmanager
def perf_step(label: str):
    yield


class Task(ABC):
    name : str
    reward: float
    duration: float
    world: World
    """
    expected time needed for the task in seconds
    """

    @abstractmethod
    def precondition(self):
        """
        each precondition is safed as bool. E.g. preconditions: precond1(), precond2(), precond3(), ... is safed as
        list [bool of precond1, bool of precond2, bool of precond3, ...]
        """
        pass

    @abstractmethod
    def effect(self):

        pass

    @abstractmethod
    def constraints(self):
        pass

    @abstractmethod
    def calculate_feasibility_custom(self) -> float:
        pass

    def calculate_feasibility(self, weight_list_preconditions: list, weight_list_constraints) -> float:
        weight = []
        weight.extend(weight_list_preconditions)
        weight.extend(weight_list_constraints)

        all_elem = []
        all_elem.extend(self.precondition())
        all_elem.extend(self.constraints())

        weight_max = 0
        weight_sum = 0
        index = 0
        for elem in all_elem:
            weight_max += weight[index]
            if elem:
                weight_sum += weight[index]
            else:
                pass
            index += 1

        if weight_max == 0:
            res = 0
        else:
            res = weight_sum / weight_max
        return res

    def calculate_current_score(self):
        return self.reward * self.calculate_feasibility_custom()

    def calculate_current_score_normalized(self):
        return self.calculate_current_score()/self.duration

    @abstractmethod
    def update_to_current_world_state(self, world: World, perceived_objects : list[SemanticAnnotation]):
        pass




class PutAwayObjectTask(Task):
    required_objects: list[SemanticAnnotation]

    def __init__(self, name : str, required_objects : list[SemanticAnnotation], world: World, perceived_objects : list[SemanticAnnotation]):
        ## different for all instances of this task ##
        self.name = name
        self.required_objects = required_objects

        ## set for all instances of this task ##
        self.reward = 200
        self.duration = 30

        ## world stuff ##
        self.world = world
        self.perceived_objects = perceived_objects

    def precondition(self):
        preconditions = []
        for obj in self.required_objects:
            found = False
            if self.perceived_objects.__contains__(obj):
                found = True
                preconditions.append(True)
            if not found:
                preconditions.append(False)

        return preconditions

    def constraints(self):
        constraints = []

        for obj in self.required_objects:
            if reachable(obj):
                constraints.append(True)
            else:
                constraints.append(False)

        return constraints

    def effect(self):
        # TODO
        pass

    def calculate_feasibility_custom(self):
        """
        weight: 1 for each precondition
        weight: 0.5 for object constraints
        """
        with perf_step(f"{self.name}.precondition for feasibility"):
            preconditions = self.precondition()
        with perf_step(f"{self.name}.constraints for feasibility"):
            constraints = self.constraints()

        weight_preconditions = []
        weight_constraints = []
        for i in range(len(preconditions)):
            weight_preconditions.append(float(1))
        for j in range(len(constraints)):
            weight_constraints.append(float(0.5))

        return self.calculate_feasibility(weight_preconditions, weight_constraints)


    def update_to_current_world_state(self, world: World, perceived_objects : list[SemanticAnnotation]):
        self.world = world
        self.perceived_objects = perceived_objects


class SetTableTask(Task):
    required_objects: list[Any]
    preconditions : list[Any]
    required_instances : list[Any]
    reward: float
    duration: float
    world: World
    table : Table
    def __init__(
        self,
        name: str,
        table: Table,
        world: World,
        perceived_objects: list[SemanticAnnotation],
        surface_cache: dict | None = None,
    ):
        with perf_step(f"{name}.__init__"):
            ## different for all instances of this task ##
            self.name = name
            self.table = table

            ## set for all instances of this task ##
            self.required_objects = [Bowl, Plate, Spoon, Knife, Cup, Milk, Banana, Bread]
            self.reward = 200 * len(self.required_objects) + 50 + 50 + 100 + 15 # 50 for each thing of cutlery and 100 for plate
            self.duration = 30 * len(self.required_objects)

            ## world stuff ##
            self.world = world
            self.perceived_objects = perceived_objects
            self.surface_cache = surface_cache
            with perf_step(f"{name} initial precondition"):
                self.preconditions = self.precondition()

    def precondition(self):
        preconditions = []

        # TODO: check if empty works
        with perf_step(f"{self.name}.precondition table empty check"):
            preconditions.append(
                is_empty(
                    self.table,
                    self.perceived_objects,
                    self.world,
                    self.surface_cache,
                )
            )

        with perf_step(f"{self.name}.precondition match required objects"):
            for obj in self.required_objects:
                found = False
                for ob in self.perceived_objects:
                    if isinstance(ob, obj):
                        found = True
                        preconditions.append(True)
                        break
                if not found:
                    preconditions.append(False)

        return preconditions

    def constraints(self):
        constraints = []

        with perf_step(f"{self.name}.constraints match required objects"):
            for obj in self.required_objects:
                found = False
                for ob in self.perceived_objects:
                    if isinstance(ob, obj):
                        found = True
                        constraints.append(reachable(ob))
                        break
                if not found:
                    constraints.append(False)

        return constraints

    def effect(self):
        # TODO
        pass

    def calculate_feasibility_custom(self):
        """
        weight: 3 for each precondition - table empty
        weight: 1 for each precondition - object exists
        weight: 0.5 for object constraints
        """
        with perf_step(f"{self.name}.constraints for feasibility"):
            constraints = self.constraints()

        weight_preconditions = []
        weight_constraints = []
        for i in range(len(self.precondition())):
            if i == 0:
                weight_preconditions.append(3)
            else:
                weight_preconditions.append(float(1))
        for j in range(len(constraints)):
            weight_constraints.append(float(0.5))

        return self.calculate_feasibility(weight_preconditions, weight_constraints)

    def update_to_current_world_state(
        self,
        world: World,
        perceived_objects: list[SemanticAnnotation],
        surface_cache: dict | None = None,
    ):
        with perf_step(f"{self.name}.update_to_current_world_state"):
            self.world = world
            self.perceived_objects = perceived_objects
            self.surface_cache = surface_cache
            self.preconditions = self.precondition()

class CleanTableTask(Task):
    required_objects: list[SemanticAnnotation]
    table: Table

    def __init__(
        self,
        name: str,
        table: Table,
        world: World,
        perceived_objects: list[SemanticAnnotation],
        surface_cache: dict | None = None,
    ):
        ## different for all instances of this task ##
        self.name = name

        ## world stuff ##
        self.world = world
        self.perceived_objects = perceived_objects
        self.required_objects = []
        self.table = table
        self.surface_cache = surface_cache

        ## set for all instances of this task ##
        objects_on_table = semantic_annotations_on_surface_cached(table, world, self.surface_cache)
        for obj in objects_on_table:
            if self.perceived_objects.__contains__(obj):
                self.required_objects.append(obj)

        cutlery = []
        plates = []
        for obj in self.required_objects:
            if isinstance(obj, Cuttlery):
                cutlery.append(obj)
            if isinstance(obj, Plate):
                plates.append(obj)
        self.reward = 200 * len(self.required_objects) + len(cutlery) * 50 + len(plates) * 100 + 15  # 50 for each thing of cutlery and 100 for plate
        self.duration = 30 * len(self.required_objects)


    def precondition(self):
        # because required objects are the perceived objects
        preconditions = []
        for obj in self.required_objects:
            preconditions.append(True)
        return preconditions

    def constraints(self):
        constraints = []
        for obj in self.required_objects:
            if obj is None:
                constraints.append(False)
            elif reachable(obj):
                constraints.append(True)
            else:
                constraints.append(False)
        return constraints

    def effect(self):
        # TODO
        pass

    def calculate_feasibility_custom(self):
        """
        weight: 1 for each precondition
        weight: 0.5 for object constraints
        """
        weight_preconditions = []
        weight_constraints = []

        for i in range(len(self.precondition())):
            weight_preconditions.append(float(1))
        for j in range(len(self.constraints())):
            weight_constraints.append(float(0.5))

        return self.calculate_feasibility(weight_preconditions, weight_constraints)


    def update_to_current_world_state(
        self,
        world: World,
        perceived_objects: list[SemanticAnnotation],
        surface_cache: dict | None = None,
    ):
        self.world = world
        self.perceived_objects = perceived_objects
        self.surface_cache = surface_cache
        self.required_objects = []

        ## set for all instances of this task ##
        objects_on_table = semantic_annotations_on_surface_cached(self.table, world, self.surface_cache)
        for obj in objects_on_table:
            if self.perceived_objects.__contains__(obj):
                if not self.required_objects.__contains__(obj):
                    self.required_objects.append(obj)

        cutlery = []
        plates = []
        for obj in self.required_objects:
            if isinstance(obj, Cuttlery):
                cutlery.append(obj)
            if isinstance(obj, Plate):
                plates.append(obj)
        self.reward = 200 * len(self.required_objects) + len(cutlery) * 50 + len(
            plates) * 100 + 15  # 50 for each thing of cutlery and 100 for plate
        self.duration = 30 * len(self.required_objects)


class LoadDishwasherTask(Task):
    required_objects: list[SemanticAnnotation]
    world : World
    # TODO: continue

    def __init__(
        self,
        name: str,
        perceived_objects: list[SemanticAnnotation],
        location_dishes: HasSupportingSurface,
        world: World,
        surface_cache: dict | None = None,
        support_cache: dict | None = None,
        required_objects: list[SemanticAnnotation] | None = None,
    ):
        self.location_dishes = location_dishes      # where the dishes are that are supposed to be put away (e.g. kitchen counter)
        self.name = name
        self.perceived_objects = perceived_objects
        self.required_objects = [] if required_objects is None else list(required_objects)
        self.world = world
        self.surface_cache = surface_cache
        self.support_cache = support_cache

        if required_objects is None:

            objects_on_counter = semantic_annotations_on_surface_cached(
                location_dishes,
                world,
                self.surface_cache,
            )

            for obj in objects_on_counter:
                if self.perceived_objects.__contains__(obj) and isinstance(obj, (Cuttlery, Plate, Cup, Bowl)):
                    self.required_objects.append(obj)

        self.reward = len(self.required_objects) * 200 + len(self.required_objects) * 70 + 100 + 160 + 100 + 200
        self.duration = 30 * len(self.required_objects) + 60 # extra time for opening/closing dishwasher



    def precondition(self):
        # because required objects are the perceived objects
        preconditions = []

        # Dishwasher rack has to be empty with respect to already perceived objects.
        dishwasher_rack = self.world.get_semantic_annotation_by_name("dishwasher_rack")
        dishwasher_rack_empty = True
        for annotation in self.perceived_objects:
            if is_supported_by_surface_cached(
                annotation,
                dishwasher_rack,
                self.support_cache,
            ):
                dishwasher_rack_empty = False
                break
        preconditions.append(dishwasher_rack_empty)

        for obj in self.required_objects:
            preconditions.append(True)
        return preconditions

    def constraints(self):
        constraints = []
        for obj in self.required_objects:
            if reachable(obj):
                constraints.append(True)
            else:
                constraints.append(False)
        return constraints

    def effect(self):
        # TODO
        pass

    def calculate_feasibility_custom(self):
        """
        weight: 3 for precondition dishwasher empty
        weight: 1 for each precondition object exists
        weight: 0.5 for object constraints
        """
        weight_preconditions = []
        weight_constraints = []

        for i in range(len(self.precondition())):
            if i == 0:
                weight_preconditions.append(float(3))
            else:
                weight_preconditions.append(float(1))
        for j in range(len(self.constraints())):
            weight_constraints.append(float(0.5))

        return self.calculate_feasibility(weight_preconditions, weight_constraints)


    def update_to_current_world_state(
        self,
        world: World,
        perceived_objects: list[SemanticAnnotation],
        surface_cache: dict | None = None,
        support_cache: dict | None = None,
        required_objects: list[SemanticAnnotation] | None = None,
    ):
        self.perceived_objects = perceived_objects
        self.required_objects = []
        self.world = world
        self.surface_cache = surface_cache
        self.support_cache = support_cache

        if required_objects is not None:
            self.required_objects = list(required_objects)
            self.reward = len(self.required_objects) * 200 + len(self.required_objects) * 70 + 100 + 160 + 100 + 200
            self.duration = 30 * len(self.required_objects) + 60 # extra time for opening/closing dishwasher
            return

        objects_on_counter = semantic_annotations_on_surface_cached(
            self.location_dishes,
            world,
            self.surface_cache,
        )
        self.required_objects = []

        for obj in objects_on_counter:
            if self.perceived_objects.__contains__(obj) and isinstance(obj, (Cuttlery, Plate, Cup, Bowl)):
                self.required_objects.append(obj)

        self.reward = len(self.required_objects) * 200 + len(self.required_objects) * 70 + 100 + 160 + 100 + 200
        self.duration = 30 * len(self.required_objects) + 60 # extra time for opening/closing dishwasher




class UnloadDishwasherTask(Task):
    required_objects: list[SemanticAnnotation]
    world : World

    # TODO: continue

    def __init__(
        self,
        name: str,
        perceived_objects: list[SemanticAnnotation],
        world: World,
        surface_cache: dict | None = None,
        required_objects: list[SemanticAnnotation] | None = None,
    ):
        self.name = name
        self.perceived_objects = perceived_objects
        self.required_objects = [] if required_objects is None else list(required_objects)
        self.world = world
        self.surface_cache = surface_cache
        self.dishwasher_rack = None

        try:
            self.dishwasher_rack = world.get_semantic_annotation_by_name("dishwasher_rack")
            if not isinstance(self.dishwasher_rack, HasSupportingSurface):
                Exception("dishwasher_rack needs to be of type HasSupportingSurface")
        except SemanticAnnotationNotInWorldError:
            Exception("no semantic annotation named dishwasher_rack")
        finally:
            Exception("no semantic annotation named dishwasher_rack, please add annotation")

        if required_objects is None:
            objects_in_dishwasher = semantic_annotations_on_surface_cached(
                self.dishwasher_rack,
                world,
                self.surface_cache,
            )

            for obj in objects_in_dishwasher:
                self.required_objects.append(obj)

        self.reward = len(self.required_objects) * 200 + len(self.required_objects) * 70 + 100 + 200
        self.duration = 30 * len(self.required_objects) + 60 # extra time for opening/closing dishwasher


    def precondition(self):
        # because required objects are the perceived objects
        preconditions = []

        for obj in self.required_objects:
            preconditions.append(True)
        return preconditions

    def constraints(self):
        constraints = []
        for obj in self.required_objects:
            if reachable(obj):
                constraints.append(True)
            else:
                constraints.append(False)
        return constraints

    def effect(self):
        # TODO
        pass

    def calculate_feasibility_custom(self):
        """
        weight: 1 for each precondition object exists
        weight: 0.5 for object constraints
        """
        weight_preconditions = []
        weight_constraints = []

        for i in range(len(self.precondition())):
            weight_preconditions.append(float(1))
        for j in range(len(self.constraints())):
            weight_constraints.append(float(0.5))

        return self.calculate_feasibility(weight_preconditions, weight_constraints)

    def update_to_current_world_state(
        self,
        world: World,
        perceived_objects: list[SemanticAnnotation],
        surface_cache: dict | None = None,
        required_objects: list[SemanticAnnotation] | None = None,
    ):
        self.perceived_objects = perceived_objects
        self.required_objects = []
        self.world = world
        self.surface_cache = surface_cache

        if required_objects is not None:
            self.required_objects = list(required_objects)
            self.reward = len(self.required_objects) * 200 + len(self.required_objects) * 70 + 100 + 200
            self.duration = 30 * len(self.required_objects) + 60 # extra time for opening/closing dishwasher
            return

        objects_on_counter = semantic_annotations_on_surface_cached(
            self.dishwasher_rack,
            world,
            self.surface_cache,
        )
        self.required_objects = []

        for obj in objects_on_counter:
            self.required_objects.append(obj)

        self.reward = len(self.required_objects) * 200 + len(self.required_objects) * 70 + 100 + 200
        self.duration = 30 * len(self.required_objects) + 60 # extra time for opening/closing dishwasher



        # constraints[0] = reachable(HomogeneousTransformationMatrix.from_xyz_rpy(x = obj.global_pose.x, y=obj.global_pose.y, z=obj.global_pose.z, reference_frame=world), self.robot.left_arm.root,
        # self.robot.left_arm.manipulator.tool_frame)
