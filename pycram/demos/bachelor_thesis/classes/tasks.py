from abc import ABC, abstractmethod

from demos.bachelor_thesis.events.event_handler import EventDispatcher
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body, SemanticAnnotation
from semantic_digital_twin.reasoning.predicates import reachable



class Task(ABC):
    name : str

    @abstractmethod
    def precondition(self):
        pass

    @abstractmethod
    def postcondition(self):
        """
        each precondition is safed as bool. E.g. preconditions: precond1(), precond2(), precond3(), ... is safed as
        list [bool of precond1, bool of precond2, bool of precond3, ...]
        """
        pass

    @abstractmethod
    def constraints(self):
        pass

    @abstractmethod
    def calculate_feasibility(self):
        pass



class TransportTask(Task):
    required_objects: list[SemanticAnnotation]
    world: World
    handler: EventDispatcher

    def __init__(self, name : str, required_objects : list[SemanticAnnotation], world: World, handler: EventDispatcher):
        self.name = name
        self.required_objects = required_objects
        self.world = world
        self.handler = handler

    def precondition(self):
        precontitions = list[bool]

    def constraints(self):
        constraints = list[bool]
        obj = self.required_objects[0]

        constraints[0] = reachable(HomogeneousTransformationMatrix.from_xyz_rpy(x = obj.global_pose.x, y=obj.global_pose.y, z=obj.global_pose.z, reference_frame=world), self.robot.left_arm.root,
        self.robot.left_arm.manipulator.tool_frame)



