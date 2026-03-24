from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Self, Type

from typing_extensions import List, Any, Optional

from krrood.ormatic.dao import AlternativeMapping, T


@dataclass
class AssociationWithMappedClass:
    """
    Context of a pycram plan
    """

    reference_to_mapped_class: MappedClass


@dataclass
class BuildFirst:
    """
    Spatial type grrrrrrrrr
    """

    value: str
    backreference_to_ordinary_class: Optional[OrdinaryClass] = None


@dataclass
class BuildFirstAssociation:
    """
    Spatial type association like in Shape
    """

    build_first: BuildFirst


@dataclass
class MappedClass:
    """
    Something like a World in SemDT
    """

    ordinary_instances: List[OrdinaryClass]
    relations_between_ordinary_instances: List[OrdinaryClassRelation]


@dataclass
class OrdinaryClassRelation:
    """
    Something like a connection in SemDT
    """

    parent: OrdinaryClass
    """
    Something like the parent of a connection in SemDT
    """

    child: OrdinaryClass
    """
    Something like the child of a connection in SemDT
    """

    build_first: BuildFirst
    """
    Something like the spatial type of a connection in SemDT ( e. g. 1DOFConnection )
    """


@dataclass
class OrdinaryClass:
    """
    Something like Kinematic Structure Entity
    """

    association: BuildFirstAssociation
    """
    A reference to another mapped class that has a reference to something that must be built first (like Shape.origin)
    """


@dataclass
class Entrypoint:
    """
    Something like a Plan
    """

    reference_to_mapped_class: MappedClass
    """
    Something like the initial_world of a pycram plan
    """

    association_with_mapped_class: AssociationWithMappedClass
    """
    Something like the context of a pycram plan
    """


@dataclass
class BuildFirstMapping(AlternativeMapping[BuildFirst]):

    value: str
    backreference_to_ordinary_class: OrdinaryClass

    @classmethod
    def from_domain_object(cls, obj: T) -> Self:
        return cls(obj.value, obj.backreference_to_ordinary_class)

    def to_domain_object(self) -> T:
        return BuildFirst(self.value, self.backreference_to_ordinary_class)


@dataclass
class MappedClassMapping(AlternativeMapping[MappedClass]):

    ordinary_instances: List[OrdinaryClass]
    relations_between_ordinary_instances: List[OrdinaryClassRelation]

    @classmethod
    def from_domain_object(cls, obj: T) -> Self:
        return cls(
            obj.ordinary_instances,
            obj.relations_between_ordinary_instances,
        )

    def to_domain_object(self) -> T:
        for ordinary_instance in self.ordinary_instances:
            assert isinstance(ordinary_instance, OrdinaryClass)
            assert isinstance(
                ordinary_instance.association.build_first,
                BuildFirst,
            )

        for relation in self.relations_between_ordinary_instances:
            assert isinstance(relation.build_first, BuildFirst)

        return MappedClass(
            self.ordinary_instances, self.relations_between_ordinary_instances
        )


@dataclass
class EntryPointMapping(AlternativeMapping[Entrypoint]):
    reference_to_mapped_class: MappedClass
    association_with_mapped_class: AssociationWithMappedClass

    @classmethod
    def from_domain_object(cls, obj: T) -> Self:
        return cls(
            reference_to_mapped_class=obj.reference_to_mapped_class,
            association_with_mapped_class=obj.association_with_mapped_class,
        )

    def to_domain_object(self) -> T:
        return Entrypoint(
            self.reference_to_mapped_class, self.association_with_mapped_class
        )
