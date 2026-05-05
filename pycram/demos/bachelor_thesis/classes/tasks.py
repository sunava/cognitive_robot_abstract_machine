from abc import ABC, abstractmethod
from typing import Any

from demos.bachelor_thesis.events.event_handler import EventDispatcher
from demos.bachelor_thesis.actions.predicate_mock import reachable
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World




class Task(ABC):
    name : str
    reward: float
    duration: float
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
    def calculate_feasibility(self):
        pass



class PutAwayObjectTask(Task):
    required_objects: list[PrefixedName]
    world: World
    handler: EventDispatcher

    def __init__(self, name : str, required_objects : list[PrefixedName], world: World, handler: EventDispatcher):
        self.name = name
        self.reward = 200
        self.duration = 30
        self.required_objects = required_objects
        self.world = world
        self.handler = handler

    def precondition(self):
        preconditions = []
        for obj in self.required_objects:
            found = False
            for ob in self.handler.perceived_objects:
                print(ob)
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




        # constraints[0] = reachable(HomogeneousTransformationMatrix.from_xyz_rpy(x = obj.global_pose.x, y=obj.global_pose.y, z=obj.global_pose.z, reference_frame=world), self.robot.left_arm.root,
        # self.robot.left_arm.manipulator.tool_frame)




