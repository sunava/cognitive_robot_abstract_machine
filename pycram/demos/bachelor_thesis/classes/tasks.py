from abc import ABC, abstractmethod
from typing import Any

from hypothesis.stateful import precondition
from scipy.constants import precision

from demos.bachelor_thesis.actions.predicate_mock import reachable, is_empty
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.reasoning.queries import semantic_annotations_on_surfaces
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Plate, Spoon, Knife, Cup, Milk, \
    Banana, Bread, Cuttlery, Table
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation, Body


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
    def calculate_feasibility(self) -> float:
        pass

    def calculate_current_score(self):
        return self.reward * self.calculate_feasibility()

    def calculate_current_score_normalized(self):
        return self.calculate_current_score()/self.duration

    @abstractmethod
    def update_to_current_world_state(self, world: World, perceived_objects : list[Body]):
        pass




class PutAwayObjectTask(Task):
    required_objects: list[PrefixedName]

    def __init__(self, name : str, required_objects : list[PrefixedName], world: World, perceived_objects : list[Body]):
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
            for ob in self.perceived_objects:
                if ob.name == obj:
                    found = True
                    preconditions.append(True)
            if not found:
                preconditions.append(False)

        return preconditions

    def constraints(self):
        constraints = []

        for obj in self.required_objects:
            if reachable(self.world.get_semantic_annotation_by_name(obj)):
                constraints.append(True)
            else:
                constraints.append(False)

        return constraints

    def effect(self):
        # TODO
        pass

    def calculate_feasibility(self):
        """
        weight: 1 for each precondition
        weight: 0.5 for object constraints
        """
        weight = []
        for i in range(len(self.precondition())):
            weight.append(float(1))
        for j in range(len(self.constraints())):
            weight.append(float(0.5))

        all_elem = []
        all_elem.extend(self.precondition())
        all_elem.extend(self.constraints())

        weight_max = 0
        weight_sum = 0
        index = 0
        for elem in all_elem:
            weight_max += weight[index]
            if elem == True:
                weight_sum += weight[index]
            else:
                pass
            index += 1

        return weight_sum/weight_max

    def update_to_current_world_state(self, world: World, perceived_objects : list[Body]):
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
    def __init__(self, name : str, table : Table, world: World, perceived_objects : list[Body]):
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
        out = self.precondition()
        self.preconditions = out[0]
        self.required_instances = out[1]

    def precondition(self):
        preconditions = []
        required_instances = []

        perceived_objects_as_annotations = []

        for obj in self.perceived_objects:
            perceived_objects_as_annotations.append(self.world.get_semantic_annotation_by_name(obj.name))
        # TODO: check if empty works
        preconditions.append(is_empty(self.table, perceived_objects_as_annotations, self.world))

        for obj in self.required_objects:
            found = False
            for ob in self.perceived_objects:
                obi = self.world.get_semantic_annotation_by_name(ob.name)
                if isinstance(obi, obj):
                    found = True
                    preconditions.append(True)
                    required_instances.append(ob)
            if not found:
                required_instances.append(None)
                preconditions.append(False)

        return preconditions, required_instances

    def constraints(self):
        constraints = []
        required_objects = self.required_instances

        for obj in required_objects:
            if obj is None:
                constraints.append(False)
            elif reachable(self.world.get_semantic_annotation_by_name(obj.name)):
                constraints.append(True)
            else:
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
        weight = []
        for i in range(len(self.required_instances) + 1):
            if i == 0:
                weight.append(3)
            else:
                weight.append(float(1))
        for j in range(len(self.constraints())):
            weight.append(float(0.5))

        all_elem = []
        all_elem.extend(self.preconditions)
        all_elem.extend(self.constraints())

        weight_max = 0
        weight_sum = 0
        index = 0
        for elem in all_elem:
            weight_max += weight[index]
            if elem == True:
                weight_sum += weight[index]
            else:
                pass
            index += 1

        return weight_sum / weight_max

    def update_to_current_world_state(self, world: World, perceived_objects : list[Body]):
        self.world = world
        self.perceived_objects = perceived_objects
        out = self.precondition()
        self.preconditions = out[0]
        self.required_instances = out[1]

class CleanTableTask(Task):
    required_objects: list[Body]
    table: Table

    def __init__(self, name: str, table : Table, world: World, perceived_objects : list[Body]):
        ## different for all instances of this task ##
        self.name = name

        ## world stuff ##
        self.world = world
        self.perceived_objects = perceived_objects
        self.required_objects = []
        self.table = table

        ## set for all instances of this task ##
        objects_on_table = semantic_annotations_on_surfaces([table], world)
        for obj in objects_on_table:
            if self.perceived_objects.__contains__(world.get_body_by_name(obj.name)):
                self.required_objects.append(world.get_body_by_name(obj.name))

        cutlery = []
        plates = []
        for obj in self.required_objects:
            obi = world.get_semantic_annotation_by_name(obj.name)
            if isinstance(obi, Cuttlery):
                cutlery.append(obi)
            if isinstance(obi, Plate):
                plates.append(obi)
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
            elif reachable(self.world.get_semantic_annotation_by_name(obj.name)):
                constraints.append(True)
            else:
                constraints.append(False)
        return constraints

    def effect(self):
        # TODO
        pass

    def calculate_feasibility(self):
        """
        weight: 1 for each precondition
        weight: 0.5 for object constraints
        """
        weight = []
        for i in range(len(self.precondition())):
            weight.append(float(1))
        for j in range(len(self.constraints())):
            weight.append(float(0.5))

        all_elem = []
        all_elem.extend(self.precondition())
        all_elem.extend(self.constraints())

        weight_max = 0
        weight_sum = 0
        index = 0
        for elem in all_elem:
            weight_max += weight[index]
            if elem == True:
                weight_sum += weight[index]
            else:
                pass
            index += 1

        result = 0
        if weight_max == 0:
            result = 0
        else:
            result = weight_sum / weight_max
        return result

    def update_to_current_world_state(self, world: World, perceived_objects : list[Body]):
        self.world = world
        self.perceived_objects = perceived_objects
        self.required_objects = []

        ## set for all instances of this task ##
        objects_on_table = semantic_annotations_on_surfaces([self.table], world)
        for obj in objects_on_table:
            if self.perceived_objects.__contains__(world.get_body_by_name(obj.name)):
                if not self.required_objects.__contains__(world.get_body_by_name(obj.name)):
                    self.required_objects.append(world.get_body_by_name(obj.name))

        cutlery = []
        plates = []
        for obj in self.required_objects:
            obi = world.get_semantic_annotation_by_name(obj.name)
            if isinstance(obi, Cuttlery):
                cutlery.append(obi)
            if isinstance(obi, Plate):
                plates.append(obi)
        self.reward = 200 * len(self.required_objects) + len(cutlery) * 50 + len(
            plates) * 100 + 15  # 50 for each thing of cutlery and 100 for plate
        self.duration = 30 * len(self.required_objects)



        # constraints[0] = reachable(HomogeneousTransformationMatrix.from_xyz_rpy(x = obj.global_pose.x, y=obj.global_pose.y, z=obj.global_pose.z, reference_frame=world), self.robot.left_arm.root,
        # self.robot.left_arm.manipulator.tool_frame)




