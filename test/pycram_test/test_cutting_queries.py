from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "pycram/src/pycram/external_interfaces/sparql_queries/cutting.py"
)


class FakeSPARQLClient:
    def __init__(self, url=None, responses=None, error=None):
        self.url = url
        self.responses = list(responses or [])
        self.error = error
        self.queries = []
        self.return_format = None

    def setReturnFormat(self, value):
        self.return_format = value

    def setQuery(self, query):
        self.queries.append(query)

    def queryAndConvert(self):
        if self.error is not None:
            raise self.error
        if not self.responses:
            raise AssertionError("No fake SPARQL response configured")
        return self.responses.pop(0)


@pytest.fixture
def cutting_module(monkeypatch):
    owlready_stub = types.ModuleType("owlready2")

    sparql_wrapper_stub = types.ModuleType("SPARQLWrapper")
    sparql_wrapper_stub.JSON = object()
    sparql_wrapper_stub.SPARQLWrapper = FakeSPARQLClient

    monkeypatch.setitem(sys.modules, "owlready2", owlready_stub)
    monkeypatch.setitem(sys.modules, "SPARQLWrapper", sparql_wrapper_stub)

    spec = spec_from_file_location("test_cutting_module", MODULE_PATH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_check_food_part_returns_boolean_and_builds_expected_query(cutting_module):
    fake_sparql = FakeSPARQLClient(responses=[{"boolean": True}])
    cutting_module.sparql = fake_sparql

    result = cutting_module.check_food_part("FOODON_00001234", "Peel")

    assert result is True
    assert len(fake_sparql.queries) == 1
    assert "ASK {" in fake_sparql.queries[0]
    assert "foodon:FOODON_00001234" in fake_sparql.queries[0]
    assert "cut:Peel" in fake_sparql.queries[0]


@pytest.mark.parametrize(
    ("function_name", "response", "expected", "expected_fragment"),
    [
        (
            "get_prior_task",
            {"results": {"bindings": [{"res": {"value": "washing"}}]}},
            "washing",
            "cut:Slicing",
        ),
        (
            "get_cutting_tool",
            {"results": {"bindings": [{"res": {"value": "ChefKnife"}}]}},
            "ChefKnife",
            "foodon:FOODON_00003523",
        ),
        (
            "get_cutting_position",
            {"results": {"bindings": [{"res": {"value": "top"}}]}},
            "top",
            "cut:Dicing",
        ),
        (
            "get_repetition",
            {"results": {"bindings": [{"res": {"value": "at least 1"}}]}},
            "at least 1",
            "cut:Julienne",
        ),
        (
            "get_peel_tool",
            {"results": {"bindings": [{"res": {"value": "YPeeler"}}]}},
            "YPeeler",
            "foodon:FOODON_00002403",
        ),
    ],
)
def test_query_helpers_return_first_binding(
    cutting_module, function_name, response, expected, expected_fragment
):
    fake_sparql = FakeSPARQLClient(responses=[response])
    cutting_module.sparql = fake_sparql

    if "foodon:" in expected_fragment:
        result = getattr(cutting_module, function_name)(
            expected_fragment.removeprefix("foodon:")
        )
    else:
        result = getattr(cutting_module, function_name)(expected_fragment)

    assert result == expected
    assert expected_fragment in fake_sparql.queries[0]


@pytest.mark.parametrize(
    ("function_name", "argument", "fallback"),
    [
        ("get_prior_task", "cut:Slicing", None),
        ("get_cutting_tool", "FOODON_00003523", "Knife"),
        ("get_cutting_position", "cut:Slicing", "middle"),
        ("get_repetition", "cut:Slicing", "1"),
        ("get_peel_tool", "FOODON_00003523", "Peeler"),
    ],
)
def test_query_helpers_use_fallbacks_when_bindings_are_empty(
    cutting_module, function_name, argument, fallback
):
    fake_sparql = FakeSPARQLClient(responses=[{"results": {"bindings": []}}])
    cutting_module.sparql = fake_sparql

    result = getattr(cutting_module, function_name)(argument)

    assert result == fallback


def test_get_cutting_knowledge_aggregates_prerequisites_and_peeling_tool(cutting_module):
    fake_sparql = FakeSPARQLClient(
        responses=[
            {"boolean": True},
            {"boolean": False},
            {"boolean": True},
            {"boolean": False},
            {"results": {"bindings": [{"res": {"value": "washing"}}]}},
            {"results": {"bindings": [{"res": {"value": "ChefKnife"}}]}},
            {"results": {"bindings": [{"res": {"value": "middle"}}]}},
            {"results": {"bindings": [{"res": {"value": "exactly 1"}}]}},
            {"results": {"bindings": [{"res": {"value": "YPeeler"}}]}},
        ]
    )
    cutting_module.sparql = fake_sparql

    result = cutting_module.get_cutting_knowledge("cut:Slicing", "FOODON_00003523")

    assert result == {
        "query_success": True,
        "verb": "cut:Slicing",
        "foodobject": "FOODON_00003523",
        "prior_task": "washing",
        "cutting_tool": "ChefKnife",
        "cutting_position": "middle",
        "repetition": "exactly 1",
        "remove_peel": True,
        "remove_core": False,
        "remove_stem": True,
        "remove_shell": False,
        "required_prerequisites": ["peeling", "stem_removal"],
        "peeling_tool": "YPeeler",
    }


def test_get_cutting_knowledge_skips_peel_tool_when_not_needed(cutting_module):
    fake_sparql = FakeSPARQLClient(
        responses=[
            {"boolean": False},
            {"boolean": False},
            {"boolean": False},
            {"boolean": False},
            {"results": {"bindings": []}},
            {"results": {"bindings": []}},
            {"results": {"bindings": []}},
            {"results": {"bindings": []}},
        ]
    )
    cutting_module.sparql = fake_sparql

    result = cutting_module.get_cutting_knowledge("cut:Slicing", "FOODON_00003523")

    assert result["peeling_tool"] is None
    assert result["required_prerequisites"] == []
    assert len(fake_sparql.responses) == 0


def test_safe_get_cutting_knowledge_returns_structured_error(cutting_module):
    cutting_module.sparql = FakeSPARQLClient(error=RuntimeError("network down"))

    result = cutting_module.safe_get_cutting_knowledge(
        "cut:Slicing", "FOODON_00003523"
    )

    assert result["query_success"] is False
    assert result["query_error"] == "RuntimeError: network down"
    assert result["required_prerequisites"] == []
    assert result["peeling_tool"] is None
