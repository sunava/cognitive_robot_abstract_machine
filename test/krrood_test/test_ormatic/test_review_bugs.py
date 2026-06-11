"""
Reproduction tests for the bugs found during the ORMatic package review.

Each test is labelled with the bug id from the review
(see /home/tom/.claude/plans/look-at-the-ormatic-wild-frog.md):

- B1: ``mark_initialized`` is never called -> repeated ``from_dao`` on a shared
  state re-runs all phases including ``__post_init__``.
- B2: state reuse corrupts the alternative-mapping dependency graph and
  reference registry -> ``to_domain_object`` is called repeatedly.
- B3: a root DAO that is alternatively mapped and part of a reference cycle is
  converted twice -> identity split between the returned object and the object
  the cycle points to.
- B5: ``ORMatic.from_package(ignore_krrood_test_classes=False)`` includes no
  alternative mappings at all due to inverted condition logic.
- B6: user-supplied ``type_mappings`` are clobbered by the interface-dependency
  loop in ``ORMatic.from_package``.
- B7: empty collections alias the DAO's instrumented collection into the
  domain object.
- B8: ``TypeType`` / ``JSONDataType`` crash on None and ``TypeType`` is missing
  ``cache_ok``.
- B9: ``DataAccessObject.to_dao`` / ``AlternativeMapping.to_dao`` crash when
  ``state=None`` is passed although the parameter is annotated Optional.
- B10: DAO lookup caches go permanently stale when a DAO class is defined
  after the first (failed) lookup.
- B14: an exception during conversion leaves ``is_processing`` /
  ``discovery_mode`` poisoned on the state.

B11 (tuple-typed collection fields) is intentionally not reproduced: tuples
cannot serve as SQLAlchemy relationship collection classes at all, so such
fields were never supported end-to-end.
"""

from dataclasses import dataclass

import pytest
from sqlalchemy import select

from krrood.ormatic.custom_types import TypeType, JSONDataType
from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
from krrood.ormatic.data_access_objects.helper import to_dao, get_dao_class
from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState
from krrood.ormatic.ormatic import ORMatic
from ..dataset import ormatic_interface as ormatic_interface_module
from ..dataset.example_classes import *
from ..dataset.ormatic_interface import *


# %% B1


def test_b1_shared_state_does_not_rerun_post_init(session, database, monkeypatch):
    """
    B1: Converting an already converted root DAO again with a shared state
    must not re-run population and ``__post_init__``.
    """
    container = ContainerGeneration([ItemWithBackreference(10), ItemWithBackreference(20)])
    session.add(to_dao(container))
    session.commit()
    session.expunge_all()

    container_dao = session.scalars(select(ContainerGenerationDAO)).one()

    calls = []
    original_post_init = ContainerGeneration.__post_init__

    def counting_post_init(self):
        calls.append(self)
        original_post_init(self)

    monkeypatch.setattr(ContainerGeneration, "__post_init__", counting_post_init)

    state = FromDataAccessObjectState()
    container_1 = container_dao.from_dao(state)
    container_2 = container_dao.from_dao(state)

    # both conversions return the same instance
    assert container_1 is container_2
    # the container's __post_init__ must have run exactly once
    assert len(calls) == 1


# %% B2


def test_b2_shared_state_converts_alternative_mappings_once(
    session, database, monkeypatch
):
    """
    B2: Converting two root DAOs with a shared state must not duplicate the
    class dependency graph nodes nor call ``to_domain_object`` repeatedly for
    the same alternative mapping instance.
    """
    entity = Entity("shared")
    to_state = ToDataAccessObjectState()
    dao_1 = to_dao(AlternativeMappingAggregator([entity], []), to_state)
    dao_2 = to_dao(AlternativeMappingAggregator([entity], []), to_state)
    session.add_all([dao_1, dao_2])
    session.commit()
    session.expunge_all()

    aggregator_daos = session.scalars(select(AlternativeMappingAggregatorDAO)).all()
    assert len(aggregator_daos) == 2

    calls = []
    original_to_domain_object = EntityMapping.to_domain_object

    def counting_to_domain_object(self):
        calls.append(self)
        return original_to_domain_object(self)

    monkeypatch.setattr(EntityMapping, "to_domain_object", counting_to_domain_object)

    state = FromDataAccessObjectState()
    aggregator_1 = aggregator_daos[0].from_dao(state)
    aggregator_2 = aggregator_daos[1].from_dao(state)

    # the shared entity is converted exactly once
    assert len(calls) == 1
    # no duplicated nodes in the class dependency graph
    node_types = list(state._class_dependencies.nodes())
    assert len(node_types) == len(set(node_types))
    # identity of the shared entity is preserved across both conversions
    assert aggregator_1.entities1[0] is aggregator_2.entities1[0]


# %% B3


def test_b3_alternatively_mapped_root_in_cycle_keeps_identity(session, database):
    """
    B3: If the root DAO is alternatively mapped and the object graph cycles
    back to it, the returned domain object must be the same instance the
    cycle points to.
    """
    backreference = Backreference({1: 1})
    reference = Reference(0, backreference)
    backreference.reference = reference

    session.add(to_dao(backreference))
    session.commit()
    session.expunge_all()

    queried = session.scalars(select(BackreferenceMappingDAO)).one()
    reconstructed = queried.from_dao()

    assert isinstance(reconstructed, Backreference)
    assert reconstructed.reference.backreference is reconstructed


def test_b3_repeated_from_dao_with_shared_state_returns_same_object(
    session, database
):
    """
    B3 (and B1): two ``from_dao`` calls on the same alternatively mapped DAO
    with a shared state must return the same domain object.
    """
    backreference = Backreference({1: 1})
    reference = Reference(0, backreference)
    backreference.reference = reference

    session.add(to_dao(backreference))
    session.commit()
    session.expunge_all()

    queried = session.scalars(select(BackreferenceMappingDAO)).one()
    state = FromDataAccessObjectState()
    first = queried.from_dao(state)
    second = queried.from_dao(state)
    assert first is second


# %% B5


def test_b5_from_package_includes_alternative_mappings_when_not_ignoring():
    """
    B5: ``ignore_krrood_test_classes=False`` must include alternative
    mappings instead of dropping all of them.
    """
    ormatic = ORMatic.from_package(
        packages=[],
        ormatic_interface_dependencies=[],
        ignored_classes=set(),
        type_mappings={},
        ignore_krrood_test_classes=False,
    )
    assert len(ormatic.alternative_mappings) > 0


# %% B6


@dataclass
class _UserMappedClass:
    """Throwaway domain class for the user type mapping in the B6 test."""

    x: int = 0


def test_b6_from_package_keeps_user_type_mappings_with_interface_dependencies():
    """
    B6: user-supplied ``type_mappings`` must survive when
    ``ormatic_interface_dependencies`` is non-empty.
    """
    ormatic = ORMatic.from_package(
        packages=[],
        ormatic_interface_dependencies=[ormatic_interface_module],
        ignored_classes=set(),
        type_mappings={_UserMappedClass: ConceptType},
    )
    assert _UserMappedClass in set(ormatic.type_mappings.keys())


# %% B7


def test_b7_empty_collection_is_not_aliased(session, database):
    """
    B7: an empty collection on the domain object must be a fresh container,
    not the DAO's live instrumented collection.
    """
    positions = KRROODPositions([], ["a"])
    session.add(to_dao(positions))
    session.commit()
    session.expunge_all()

    queried = session.scalars(select(KRROODPositionsDAO)).one()
    reconstructed = queried.from_dao()

    assert reconstructed.positions == []
    assert reconstructed.positions is not queried.positions

    # mutating the domain object must not touch the DAO
    reconstructed.positions.append(KRROODPosition(1, 2, 3))
    assert len(queried.positions) == 0


# %% B8


def test_b8_type_type_handles_none():
    """B8: ``TypeType`` must map None to NULL instead of crashing."""
    type_type = TypeType()
    assert type_type.process_bind_param(None, None) is None


def test_b8_type_type_has_cache_ok():
    """B8: ``TypeType`` must set ``cache_ok`` to keep statement caching enabled."""
    assert TypeType.cache_ok is True


def test_b8_json_data_type_handles_none():
    """B8: ``JSONDataType`` must map NULL to None instead of crashing."""
    json_data_type = JSONDataType()
    assert json_data_type.process_result_value(None, None) is None
    assert json_data_type.process_bind_param(None, None) is None


# %% B9


def test_b9_to_dao_classmethod_accepts_none_state():
    """B9: the ``state`` parameter is Optional, passing None must work."""
    dao = KRROODPositionDAO.to_dao(KRROODPosition(1, 2, 3), None)
    assert dao.x == 1


def test_b9_alternative_mapping_to_dao_accepts_none_state():
    """B9: ``AlternativeMapping.to_dao`` with ``state=None`` must work."""
    mapping = EntityMapping.to_dao(Entity("x"), None)
    assert mapping.overwritten_name == "x"


# %% B10


class _LateDomainClass:
    """Throwaway domain class whose DAO is defined after the first lookup."""


def test_b10_dao_lookup_recovers_after_late_dao_definition():
    """
    B10: a failed DAO lookup must not be cached forever; defining the DAO
    class afterwards has to make the lookup succeed.
    """
    assert get_dao_class(_LateDomainClass) is None

    class _LateDomainClassDAO(DataAccessObject[_LateDomainClass]):
        pass

    assert get_dao_class(_LateDomainClass) is _LateDomainClassDAO


# %% B14


def test_b14_state_flags_are_reset_on_error(session, database, monkeypatch):
    """
    B14: an exception during conversion must not leave the state flags
    poisoned (``is_processing`` / ``discovery_mode``).
    """
    entity = Entity("x")
    session.add(to_dao(AlternativeMappingAggregator([entity], [])))
    session.commit()
    session.expunge_all()

    queried = session.scalars(select(AlternativeMappingAggregatorDAO)).one()

    def raising_to_domain_object(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(EntityMapping, "to_domain_object", raising_to_domain_object)

    state = FromDataAccessObjectState()
    with pytest.raises(RuntimeError):
        queried.from_dao(state)

    assert state.is_processing is False
    assert state.discovery_mode is False
