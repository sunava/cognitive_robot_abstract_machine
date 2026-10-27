"""
The names a query box offers.
"""

from __future__ import annotations

import pytest

from cramera_protocol.questions import VocabularyKind
from cramera_bridge.questions.entity import NamedEntity
from cramera_bridge.questions.exceptions import UnknownVocabularyName
from cramera_bridge.questions.knowledge import QueryDomain
from cramera_bridge.questions.vocabulary import QueryVocabulary
from cramera_bridge.questions.workspace_classes import (
    ClassLocation,
    WorkspaceClassIndex,
    WorkspacePackage,
)
from .dataset.entities import Coordinates, CoordinatesFormat, ShelfItem

VOCABULARY = QueryVocabulary(
    domains=[QueryDomain("item", ShelfItem, [])],
    extra_names={
        "origin": Coordinates(0.0),
        "Coordinates": Coordinates,
        "writer": CoordinatesFormat(),
    },
    class_index=WorkspaceClassIndex.of_classes(
        [
            ClassLocation(
                NamedEntity.__name__, NamedEntity.__module__, WorkspacePackage.KRROOD
            ),
            ClassLocation(NamedEntity.__name__, "elsewhere", WorkspacePackage.ROBOKUDO),
        ]
    ),
)

# %% entries


def test_variables_come_first_then_their_types():
    entries = VOCABULARY.entries()
    assert (entries[0].name, entries[0].kind) == ("item", VocabularyKind.VARIABLE)
    assert (entries[1].name, entries[1].kind) == (
        ShelfItem.__name__,
        VocabularyKind.ENTITY_TYPE,
    )


def test_further_names_keywords_and_classes_are_offered_once():
    kinds = {entry.name: entry for entry in VOCABULARY.entries()}
    assert kinds["origin"].kind is VocabularyKind.VALUE
    assert kinds["Coordinates"].kind is VocabularyKind.ENTITY_TYPE
    assert kinds["an"].kind is VocabularyKind.FACTORY
    assert kinds[NamedEntity.__name__].kind is VocabularyKind.CLASS
    assert kinds[NamedEntity.__name__].further_modules == 1


# %% members


def test_members_list_fields_then_properties_then_methods():
    members = VOCABULARY.members_of("writer")
    assert [(member.name, member.kind) for member in members][:3] == [
        ("prefix", VocabularyKind.FIELD),
        ("accepts", VocabularyKind.METHOD),
        ("text", VocabularyKind.METHOD),
    ]
    assert "size" in [member.name for member in VOCABULARY.members_of("item")]


def test_a_value_and_a_class_name_have_members():
    assert [member.name for member in VOCABULARY.members_of("origin")] == ["x"]
    assert "name" in [
        member.name for member in VOCABULARY.members_of(NamedEntity.__name__)
    ]


def test_an_unknown_name_has_no_members():
    with pytest.raises(UnknownVocabularyName):
        VOCABULARY.members_of("nothing")


def test_properties_are_described_without_running_them():
    assert QueryVocabulary.kind_of(property(lambda self: 1)) is VocabularyKind.PROPERTY
    assert QueryVocabulary.kind_of(1) is None
