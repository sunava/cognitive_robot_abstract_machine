from dataclasses import dataclass

from krrood.entity_query_language.factories import query


@dataclass
class SingleArgumentAction:
    arm: str


@dataclass
class MixedArgumentAction:
    arm: str
    keep_joint_states: bool


def test_query_accepts_constructor_style_positional_arguments():
    action_query = query(SingleArgumentAction)("both")

    assert action_query.kwargs == {"arm": "both"}
    assert action_query.construct_instance() == SingleArgumentAction("both")


def test_query_accepts_mixed_positional_and_keyword_arguments():
    action_query = query(MixedArgumentAction)("left", keep_joint_states=True)

    assert action_query.kwargs == {"arm": "left", "keep_joint_states": True}
    assert action_query.construct_instance() == MixedArgumentAction("left", True)
