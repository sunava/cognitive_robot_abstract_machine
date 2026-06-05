from abc import ABC, abstractmethod
from typing_extensions import Any
from contextlib import contextmanager


from demos.bachelor_thesis.actions.predicate_mock import (
    reachable,
    is_empty,
    semantic_annotations_on_surface_cached,
    is_supported_by_surface_cached,
)
from semantic_digital_twin.exceptions import SemanticAnnotationNotInWorldError
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Plate, Spoon, Knife, Cup, Milk, \
    Banana, Bread, Cuttlery, Table, DishwasherTab
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

### Rewards #########################################
REWARD_PER_OBJECT = 200
REWARD_CUTTLERY = 50
REWARD_PLATE = 100
REWARD_NAVIGATE_TO_TABLE = 15
REWARD_PICK_DISHWASHER_TAB = 100
REWARD_DISHWASHER_TAB_INTO_SLOT = 160
REWARD_PULL_PUSH_DISHWASHER_RACK = 100
REWARD_OPEN_CLOSE_DISHWASHER = 200

BONUS_REWARD_PER_OBJECT_OUT_OF_OR_IN_DISHWASHER = 70

### Duration ########################################
DURATION_PER_OBJECT = 30
DURATION_HANDLE_DISHWASHER = 60


#####################################################

class Task(ABC):
    name: str
    required_objects: list
    perceived_objects: list[SemanticAnnotation]
    reward: float
    duration: float
    world: World
    """
    expected time needed for the task in seconds
    """

    @abstractmethod
    def precondition(self) -> list[bool]:
        """
        each precondition is stored as bool. E.g. preconditions: precond1(), precond2(), precond3(), ... is safed as
        list [bool of precond1, bool of precond2, bool of precond3, ...]
        """
        pass

    def preconditions_helper(self):
        preconditions = []
        for obj in self.required_objects:
            if obj in self.perceived_objects:
                preconditions.append(True)
            else:
                preconditions.append(False)

        return preconditions

    @abstractmethod
    def effect(self):

        pass

    @abstractmethod
    def constraints(self) -> list[bool]:
        pass

    def constraints_helper(self) -> list[bool]:
        constraints = []

        for obj in self.required_objects:
            if reachable(obj):
                constraints.append(True)
            else:
                constraints.append(False)

        return constraints

    @abstractmethod
    def calculate_feasibility(self) -> float:
        pass

    def calculate_feasibility_helper(self, weight_list_preconditions: list, weight_list_constraints) -> float:
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

    def calculate_current_score(self) -> float:
        return self.reward * self.calculate_feasibility()

    def calculate_current_score_normalized(self) -> float:
        return self.calculate_current_score()/self.duration

    @abstractmethod
    def update_to_current_world_state(self, **kwargs) -> None:
        pass




class PutAwayObjectTask(Task):

    def __init__(self, name : str, required_objects : list[SemanticAnnotation], world: World, perceived_objects : list[SemanticAnnotation]):
        ## different for all instances of this task ##
        self.name = name
        self.required_objects = required_objects

        ## set for all instances of this task ##
        self.reward = REWARD_PER_OBJECT
        self.duration = DURATION_PER_OBJECT

        ## world stuff ##
        self.world = world
        self.perceived_objects = perceived_objects

    def precondition(self):
        return self.preconditions_helper()

    def constraints(self):
        return self.constraints_helper()

    def effect(self):
        # TODO
        pass

    def calculate_feasibility(self):
        """
        weight: 1 for each precondition
        weight: 0.5 for object constraints
        """
        preconditions = self.precondition()
        constraints = self.constraints()

        weight_preconditions = []
        weight_constraints = []
        for i in range(len(preconditions)):
            weight_preconditions.append(float(1))
        for j in range(len(constraints)):
            weight_constraints.append(float(0.5))

        return self.calculate_feasibility_helper(weight_preconditions, weight_constraints)


    def update_to_current_world_state(self, world: World, perceived_objects : list[SemanticAnnotation]):
        self.world = world
        self.perceived_objects = perceived_objects


class SetTableTask(Task):
    preconditions : list[Any]
    table : Table
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
        self.table = table

        ## set for all instances of this task ##
        self.required_objects = [Bowl, Plate, Spoon, Knife, Cup, Milk, Banana, Bread]

        self.reward = REWARD_PER_OBJECT * len(self.required_objects) + REWARD_CUTTLERY + REWARD_CUTTLERY + REWARD_PLATE + REWARD_NAVIGATE_TO_TABLE # 50 for each thing of cutlery and 100 for plate
        self.duration = DURATION_PER_OBJECT * len(self.required_objects)

        ## world stuff ##
        self.world = world
        self.perceived_objects = perceived_objects
        self.surface_cache = surface_cache

    def precondition(self):
        preconditions = []

        preconditions.append(
            is_empty(
                self.table,
                self.perceived_objects,
                self.world,
                self.surface_cache,
            )
        )

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

    def calculate_feasibility(self):
        """
        weight: 3 for each precondition - table empty
        weight: 1 for each precondition - object exists
        weight: 0.5 for object constraints
        """
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

        return self.calculate_feasibility_helper(weight_preconditions, weight_constraints)

    def update_to_current_world_state(
        self,
        world: World,
        perceived_objects: list[SemanticAnnotation],
        surface_cache: dict | None = None,
    ):
        self.world = world
        self.perceived_objects = perceived_objects
        self.surface_cache = surface_cache

class CleanTableTask(Task):
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
            if obj in self.perceived_objects:
                self.required_objects.append(obj)

        self._calculate_reward_and_duration()

    def precondition(self):
        return self.preconditions_helper()

    def constraints(self):
        return self.constraints_helper()

    def effect(self):
        # TODO
        pass

    def calculate_feasibility(self):
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

        return self.calculate_feasibility_helper(weight_preconditions, weight_constraints)


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
            if obj in self.perceived_objects:
                if not (obj in self.required_objects):
                    self.required_objects.append(obj)

        self._calculate_reward_and_duration()

    def _calculate_reward_and_duration(self):
        cutlery = []
        plates = []
        for obj in self.required_objects:
            if isinstance(obj, Cuttlery):
                cutlery.append(obj)
            if isinstance(obj, Plate):
                plates.append(obj)

        self.reward = REWARD_PER_OBJECT * len(self.required_objects) + len(cutlery) * REWARD_CUTTLERY + len(plates) * REWARD_PLATE + REWARD_NAVIGATE_TO_TABLE
        self.duration = DURATION_PER_OBJECT * len(self.required_objects)


class LoadDishwasherTask(Task):

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

        self.dishwasher_tab = None
        for ob in perceived_objects:
            if isinstance(ob, DishwasherTab):
                self.dishwasher_tab = ob

        if required_objects is None:

            objects_on_counter = semantic_annotations_on_surface_cached(
                location_dishes,
                world,
                self.surface_cache,
            )

            for obj in objects_on_counter:
                if (obj in self.perceived_objects) and isinstance(obj, (Cuttlery, Plate, Cup, Bowl)):
                    self.required_objects.append(obj)

        self._calculate_reward_and_duration()



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

        if self.dishwasher_tab is not None:
            preconditions.append(True)
        else:
            preconditions.append(False)

        for obj in self.required_objects:
            preconditions.append(True)
        return preconditions

    def constraints(self):
        # last constraint for dishwasher tab:
        # True, if dishwasher tab is reachable
        # False, if dishwasher tab unreachable or not perceived
        if self.dishwasher_tab is not None:
            dishwasher_tab_bool = reachable(self.dishwasher_tab)
        else:
            dishwasher_tab_bool = False

        list_const = self.constraints_helper()
        list_const.append(dishwasher_tab_bool)
        return list_const

    def effect(self):
        # TODO
        pass

    def calculate_feasibility(self):
        """
        weight: 3 for precondition dishwasher empty
        weight: 2 for precondition dishwasher tab is there
        weight: 1 for each precondition object exists
        weight: 0.5 for object constraints
        """
        weight_preconditions = []
        weight_constraints = []

        for i in range(len(self.precondition())):
            if i == 0:
                weight_preconditions.append(float(3))
            elif i == 1:
                weight_preconditions.append(float(2))
            else:
                weight_preconditions.append(float(1))

        for j in range(len(self.constraints())):
            weight_constraints.append(float(0.5))

        return self.calculate_feasibility_helper(weight_preconditions, weight_constraints)


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

        self.dishwasher_tab = None
        for ob in perceived_objects:
            if isinstance(ob, DishwasherTab):
                self.dishwasher_tab = ob

        if required_objects is not None:
            self.required_objects = list(required_objects)
            self._calculate_reward_and_duration()
            return

        objects_on_counter = semantic_annotations_on_surface_cached(
            self.location_dishes,
            world,
            self.surface_cache,
        )
        self.required_objects = []

        for obj in objects_on_counter:
            if (obj in self.perceived_objects) and isinstance(obj, (Cuttlery, Plate, Cup, Bowl)):
                self.required_objects.append(obj)

        self._calculate_reward_and_duration()

    def _calculate_reward_and_duration(self):
        self.reward = (len(self.required_objects) * REWARD_PER_OBJECT
                       + len(self.required_objects) * BONUS_REWARD_PER_OBJECT_OUT_OF_OR_IN_DISHWASHER
                       + REWARD_PICK_DISHWASHER_TAB + REWARD_DISHWASHER_TAB_INTO_SLOT + REWARD_PULL_PUSH_DISHWASHER_RACK
                       + REWARD_OPEN_CLOSE_DISHWASHER)
        self.duration = DURATION_PER_OBJECT * len(self.required_objects) + DURATION_HANDLE_DISHWASHER


class UnloadDishwasherTask(Task):
    world : World

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
                raise Exception("dishwasher_rack needs to be of type HasSupportingSurface")
        except SemanticAnnotationNotInWorldError:
            raise Exception("no semantic annotation named dishwasher_rack")

        if required_objects is None:
            objects_in_dishwasher = semantic_annotations_on_surface_cached(
                self.dishwasher_rack,
                world,
                self.surface_cache,
            )

            for obj in objects_in_dishwasher:
                self.required_objects.append(obj)

        self._calculate_reward_and_duration()

    def precondition(self):
        return self.preconditions_helper()

    def constraints(self):
        return self.constraints_helper()

    def effect(self):
        # TODO
        pass

    def calculate_feasibility(self):
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

        return self.calculate_feasibility_helper(weight_preconditions, weight_constraints)

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
            self._calculate_reward_and_duration()
            return

        objects_on_counter = semantic_annotations_on_surface_cached(
            self.dishwasher_rack,
            world,
            self.surface_cache,
        )
        self.required_objects = []

        for obj in objects_on_counter:
            self.required_objects.append(obj)

        self._calculate_reward_and_duration()


    def _calculate_reward_and_duration(self):
        self.reward = (len(self.required_objects) * REWARD_PER_OBJECT
                       + len(self.required_objects) * BONUS_REWARD_PER_OBJECT_OUT_OF_OR_IN_DISHWASHER
                       + REWARD_PULL_PUSH_DISHWASHER_RACK + REWARD_OPEN_CLOSE_DISHWASHER)
        self.duration = DURATION_PER_OBJECT * len(self.required_objects) + DURATION_HANDLE_DISHWASHER


