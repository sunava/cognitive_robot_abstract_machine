from abc import abstractmethod, ABC
from dataclasses import dataclass

from typing_extensions import Self

from pycram.datastructures.enums import AxisIdentifier
from semantic_digital_twin.semantic_annotations.mixins import HasHandle, HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body



