"""
The semantic attributes a converted USD stage authored on its prims, ``semantics.json``,
as semantic annotations of the bodies holding the prims' geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any, Dict, List, Optional, Type

from cramera_protocol.document import JsonDocument, document_field, extension_field
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Desk,
    Fridge,
    Houseplant,
    Shelf,
    Sink,
    Table,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% file


class SemanticNamespace(StrEnum):
    """
    The attribute namespaces of a prim entry.
    """

    SEMANTIC_DIGITAL_TWIN = "sdt"
    """
    Attributes describing the prim for the semantic digital twin.
    """

    ONTOLOGY = "ont"
    """
    Ontology types of the prim.
    """


class PrimEntryKey(StrEnum):
    """
    The keys of one prim entry.
    """

    LINK = "link"
    """
    The link holding the prim's geometry.
    """


class SemanticAttributeName(StrEnum):
    """
    The ``sdt`` attributes with a field of their own.
    """

    CLASS = "class"
    """
    The prim's class.
    """

    ENTITY_TYPE = "entityType"
    """
    The kind of entity.
    """

    CANONICAL_NAME = "canonicalName"
    """
    The prim's display name.
    """

    COMPONENT_ID = "componentId"
    """
    The identifier of the device the prim stands for.
    """

    CAPABILITIES = "capabilities"
    """
    What the prim's device can do.
    """


class OntologyAttributeName(StrEnum):
    """
    The ``ont`` attributes.
    """

    TYPES = "types"
    """
    The asserted ontology types.
    """

    INFERRED_TYPES = "inferredTypes"
    """
    The inferred ontology types.
    """


@dataclass(kw_only=True)
class PrimEntry(JsonDocument):
    """
    One prim of ``semantics.json``.
    """

    link: str = document_field(PrimEntryKey.LINK)
    """
    The link holding the prim's geometry.
    """

    semantic_attributes: Dict[str, Any] = document_field(
        SemanticNamespace.SEMANTIC_DIGITAL_TWIN, default_factory=dict
    )
    """
    The prim's ``sdt`` attributes by name.
    """

    ontology_attributes: Dict[str, Any] = document_field(
        SemanticNamespace.ONTOLOGY, default_factory=dict
    )
    """
    The prim's ``ont`` attributes by name.
    """


class SemanticsKey(StrEnum):
    """
    The keys of ``semantics.json``.
    """

    PRIMS = "prims"
    """
    The annotated prims.
    """


@dataclass(kw_only=True)
class Semantics(JsonDocument):
    """
    The content of ``semantics.json``.
    """

    prims: Dict[str, PrimEntry] = document_field(
        SemanticsKey.PRIMS, default_factory=dict
    )
    """
    The annotated prims by their USD path.
    """

    extensions: Dict[str, Any] = extension_field()
    """
    The stage's units, up axis and source, which building a world does not read.
    """


# %% annotations


@dataclass(frozen=True)
class PrimAttribute:
    """
    One ``sdt`` attribute as the stage authored it.
    """

    name: str
    """
    The attribute's name without its namespace.
    """

    value: Any
    """
    The authored value: a text, number, truth value or list.
    """


@dataclass(eq=False)
class UsdPrimAnnotation(HasRootBody):
    """
    The semantic attributes of one USD prim, on the body holding its geometry; prims
    merged into one body give one annotation each.
    """

    prim_path: str = field(kw_only=True)
    """
    The USD path of the prim.
    """

    semantic_class: Optional[str] = field(default=None, kw_only=True)
    """
    The authored ``sdt:class``.
    """

    entity_type: Optional[str] = field(default=None, kw_only=True)
    """
    The authored ``sdt:entityType``.
    """

    canonical_name: Optional[str] = field(default=None, kw_only=True)
    """
    The authored ``sdt:canonicalName``.
    """

    component_id: Optional[str] = field(default=None, kw_only=True)
    """
    The authored ``sdt:componentId``.
    """

    capabilities: List[str] = field(default_factory=list, kw_only=True)
    """
    The authored ``sdt:capabilities``.
    """

    ontology_types: List[str] = field(default_factory=list, kw_only=True)
    """
    The authored ``ont:types``.
    """

    inferred_ontology_types: List[str] = field(default_factory=list, kw_only=True)
    """
    The authored ``ont:inferredTypes``.
    """

    attributes: List[PrimAttribute] = field(default_factory=list, kw_only=True)
    """
    Every ``sdt`` attribute of the prim, including those with a field of their own.
    """

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = PrefixedName(self.prim_path)
        super().__post_init__()

    def __hash__(self) -> int:
        return hash((self.__class__, self.root, self.prim_path))

    @classmethod
    def of_prim(cls, path: str, entry: PrimEntry, body: Body) -> UsdPrimAnnotation:
        """
        :param path: The USD path of the prim.
        :param entry: The prim's entry of ``semantics.json``.
        :param body: The body of the prim's link.
        :return: The prim's annotation on the body.
        """
        semantic = entry.semantic_attributes
        ontology = entry.ontology_attributes
        return cls(
            root=body,
            prim_path=path,
            semantic_class=semantic.get(SemanticAttributeName.CLASS),
            entity_type=semantic.get(SemanticAttributeName.ENTITY_TYPE),
            canonical_name=semantic.get(SemanticAttributeName.CANONICAL_NAME),
            component_id=semantic.get(SemanticAttributeName.COMPONENT_ID),
            capabilities=list(semantic.get(SemanticAttributeName.CAPABILITIES, [])),
            ontology_types=list(ontology.get(OntologyAttributeName.TYPES, [])),
            inferred_ontology_types=list(
                ontology.get(OntologyAttributeName.INFERRED_TYPES, [])
            ),
            attributes=[
                PrimAttribute(name=name, value=value)
                for name, value in semantic.items()
            ],
        )


# %% class mapping


class AuthoredClass(StrEnum):
    """
    The ``sdt:class`` values that stand for a semantic digital twin annotation.
    """

    LAB_REFRIGERATOR = "lab_refrigerator"
    """
    A laboratory refrigerator.
    """

    STORAGE_SHELVING = "storage_shelving"
    """
    A free-standing shelving rack.
    """

    LAB_SINK_UNIT = "lab_sink_unit"
    """
    A sink unit with its tap.
    """

    POTTED_PLANT = "potted_plant"
    """
    A plant in a planter.
    """

    OPERATOR_DESK = "operator_desk"
    """
    A desk for an operator.
    """

    GAS_CYLINDER_CABINET = "gas_cylinder_cabinet"
    """
    A cabinet holding gas cylinders.
    """

    LAB_BENCH_ISLAND = "lab_bench_island"
    """
    A free-standing laboratory bench.
    """

    LAB_BENCH_WALL = "lab_bench_wall"
    """
    A laboratory bench along a wall.
    """


@dataclass
class SemanticClassMapping:
    """
    Which semantic digital twin annotation an authored ``sdt:class`` stands for.
    """

    annotation_types: Dict[str, Type[HasRootBody]] = field(
        default_factory=lambda: {
            AuthoredClass.LAB_REFRIGERATOR: Fridge,
            AuthoredClass.STORAGE_SHELVING: Shelf,
            AuthoredClass.LAB_SINK_UNIT: Sink,
            AuthoredClass.POTTED_PLANT: Houseplant,
            AuthoredClass.OPERATOR_DESK: Desk,
            AuthoredClass.GAS_CYLINDER_CABINET: Cabinet,
            AuthoredClass.LAB_BENCH_ISLAND: Table,
            AuthoredClass.LAB_BENCH_WALL: Table,
        }
    )
    """
    The annotation type per ``sdt:class`` value.
    """

    def annotation_for(self, entry: PrimEntry, body: Body) -> Optional[HasRootBody]:
        """
        :param entry: A prim's entry of ``semantics.json``.
        :param body: The body of the prim's link.
        :return: The annotation the prim's class stands for, named after the body, or
            None for a class without a counterpart.
        """
        annotation_type = self.annotation_types.get(
            entry.semantic_attributes.get(SemanticAttributeName.CLASS)
        )
        if annotation_type is None:
            return None
        return annotation_type(name=body.name, root=body)
