from abc import abstractmethod, ABC
from dataclasses import dataclass

from typing_extensions import Self

from pycram.datastructures.enums import AxisIdentifier
from semantic_digital_twin.semantic_annotations.mixins import HasHandle
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import RootedSemanticAnnotation, Body


@dataclass
class Tool(RootedSemanticAnnotation, ABC):
    @abstractmethod
    def tool_alignment(self) -> AxisIdentifier:
        pass

@dataclass
class ToolWithHandle(Tool, HasHandle, ABC):

    @property
    def tip(self)  -> Body:
        return self.root


@dataclass
class Whisk(ToolWithHandle):

    def tool_alignment(self) -> AxisIdentifier:
        return AxisIdentifier.X

@dataclass
class Knife(ToolWithHandle):
    def tool_alignment(self) -> AxisIdentifier:
        return AxisIdentifier.Z

    @classmethod
    def from_world(cls, world: World) -> Self:
        obj = cls(root=world.get_body_by_name())
        handle = world.get_body_by_name()
        obj.add_handle(Handle(root=handle))
        return obj


@dataclass
class Sponge(Tool):
    def tool_alignment(self) -> AxisIdentifier:
        return AxisIdentifier.Y

