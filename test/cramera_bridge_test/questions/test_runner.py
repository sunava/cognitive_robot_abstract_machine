"""
Building, evaluating and rendering queries.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cramera_protocol.questions import AnswerColumn, AnswerKind
from cramera_bridge.questions.exceptions import EmptyQuery
from cramera_bridge.questions.highlights import EntityNameKeys
from cramera_bridge.questions.knowledge import QueryDomain
from cramera_bridge.questions.replay import REPLAY_LEAD_SECONDS, REPLAY_TAIL_SECONDS
from cramera_bridge.questions.runner import (
    VALUE_PRECISION,
    AnswerRenderer,
    QueryRunner,
)
from .dataset.entities import (
    Coordinates,
    CoordinatesFormat,
    Delivery,
    GrowingShelf,
    ShelfItem,
)
from .dataset.files import QueryFile, QuerySource

LIGHT = ShelfItem(name="feather", size=0.1, shelf="top")
HEAVY = ShelfItem(name="anvil", size=9.0, shelf="bottom")
MOMENT = datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc)


def runner_over(shelf, **names) -> QueryRunner:
    """
    :param shelf: What the variable ``item`` ranges over.
    :param names: Further names a query may use.
    :return: A runner over the items.
    """
    return QueryRunner(
        domains=[QueryDomain("item", ShelfItem, shelf)], extra_names=names
    )


def answer(runner: QueryRunner, code: str, **renderer_settings):
    """
    :param runner: The runner to ask.
    :param code: Query source.
    :param renderer_settings: Further settings of the renderer.
    :return: The answer, rendered over the runner's entity types.
    """
    return runner.answer(
        code, AnswerRenderer(entity_types=runner.entity_types(), **renderer_settings)
    )


# %% building


def test_the_last_expression_is_the_query_and_statements_set_it_up():
    runner = runner_over([LIGHT, HEAVY])
    result = answer(runner, QuerySource(QueryFile.HEAVY_ITEMS).text())
    assert [row[AnswerColumn.ENTITY] for row in result.rows] == [HEAVY.name]
    assert result.kind is AnswerKind.ENTITIES


def test_a_source_ending_in_a_statement_answers_with_its_result():
    runner = runner_over([LIGHT], items=[LIGHT, HEAVY])
    result = answer(runner, QuerySource(QueryFile.BOUND_RESULT).text())
    assert [row[AnswerColumn.VALUE] for row in result.rows] == [LIGHT.name, HEAVY.name]
    assert result.verbalization is None


def test_an_empty_query_is_refused():
    with pytest.raises(EmptyQuery):
        runner_over([]).build("")


def test_an_unknown_name_and_broken_syntax_raise():
    runner = runner_over([])
    with pytest.raises(NameError):
        runner.build(QuerySource(QueryFile.UNKNOWN_NAME).text())
    with pytest.raises(SyntaxError):
        runner.build(QuerySource(QueryFile.BROKEN).text())


def test_every_query_gets_a_fresh_variable_and_reads_the_current_items():
    shelf = GrowingShelf([LIGHT])
    runner = runner_over(shelf)
    first = runner.namespace()["item"]
    shelf.items.append(HEAVY)
    assert runner.namespace()["item"] is not first
    result = answer(runner, "an(entity(item))")
    assert result.count == len(shelf.items)


def test_the_entity_type_is_in_reach_under_its_class_name():
    assert runner_over([]).namespace()[ShelfItem.__name__] is ShelfItem


def test_a_query_is_read_back_as_english():
    verbalization = runner_over([LIGHT]).verbalize("an(entity(item))")
    assert ShelfItem.__name__ in verbalization.text
    assert runner_over([LIGHT]).verbalize("1 + 1") is None


# %% rendering


def test_an_entity_row_holds_its_fields_but_not_its_name_or_internal_ones():
    row = answer(runner_over([LIGHT]), "an(entity(item))").rows[0]
    assert row == {
        AnswerColumn.ENTITY: LIGHT.name,
        AnswerColumn.TYPE: ShelfItem.__name__,
        "size": LIGHT.size,
        "shelf": LIGHT.shelf,
        "stocked": ShelfItem.stocked,
    }


def test_selected_columns_drop_their_owner():
    result = answer(runner_over([HEAVY]), QuerySource(QueryFile.SELECTION).text())
    assert result.kind is AnswerKind.ROWS
    assert result.rows == [{"name": HEAVY.name, "size": HEAVY.size}]


def test_columns_keep_full_names_when_shortening_would_merge_them():
    assert AnswerRenderer.column_names(["Item.name", "Shelf.name"]) == [
        "Item.name",
        "Shelf.name",
    ]


def test_rows_stop_at_the_limit_and_say_so():
    result = answer(runner_over([LIGHT, HEAVY]), "an(entity(item))", limit=1)
    assert (result.count, result.more) == (1, True)


def test_numbers_are_rounded_and_unknown_values_written_by_their_format():
    renderer = AnswerRenderer(value_formats=[CoordinatesFormat()])
    assert renderer.text_of(1 / 3) == round(1 / 3, VALUE_PRECISION)
    assert renderer.text_of(Coordinates(2.0)) == CoordinatesFormat().text(
        Coordinates(2.0)
    )
    assert renderer.text_of(object) == repr(object)


def test_a_single_value_is_one_row():
    rows, more = AnswerRenderer().rows_of(42)
    assert [row.values for row in rows] == [{AnswerColumn.VALUE: 42}]
    assert AnswerRenderer().rows_of(None) == ([], False)


# %% highlights


def test_entities_and_texts_naming_known_keys_light_up():
    keys = EntityNameKeys(
        keys_by_name={HEAVY.name: "anvil-key"}, known_keys=frozenset({"top"})
    )
    result = answer(
        runner_over([LIGHT, HEAVY]), "an(entity(item))", highlight_keys=keys
    )
    assert result.highlight == sorted(["anvil-key", LIGHT.name, "top"])


def test_a_text_naming_nothing_shown_is_left_alone():
    assert EntityNameKeys().keys_of("nowhere") == []
    assert EntityNameKeys().keys_of(3) == []


# %% replay


def test_a_row_with_a_moment_carries_its_replay_window():
    delivery = Delivery(name="parcel", timestamp=MOMENT)
    runner = QueryRunner(domains=[QueryDomain("delivery", Delivery, [delivery])])
    entity_answer = answer(runner, "an(entity(delivery))")
    selection_answer = answer(runner, QuerySource(QueryFile.DELIVERY_TIMES).text())
    for result in (entity_answer, selection_answer):
        window = result.replay[0]
        assert window.start == MOMENT.timestamp() - REPLAY_LEAD_SECONDS
        assert window.end == MOMENT.timestamp() + REPLAY_TAIL_SECONDS
    assert selection_answer.rows[0]["timestamp"] == MOMENT.isoformat(
        sep=" ", timespec="seconds"
    )


def test_a_row_without_a_moment_offers_no_replay():
    assert answer(runner_over([LIGHT]), "an(entity(item))").replay == [None]
