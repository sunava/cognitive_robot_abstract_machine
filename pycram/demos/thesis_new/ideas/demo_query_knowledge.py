from pprint import pformat

from pycram.external_interfaces.sparql_queries.cutting import safe_get_cutting_knowledge
from pycram.external_interfaces.sparql_queries.mixing import safe_get_mixing_knowledge
from pycram.external_interfaces.sparql_queries.pouring import safe_get_pouring_knowledge


DEFAULT_CUTTING_VERB = "cut:Slicing"
DEFAULT_CUTTING_FOODON = "FOODON_00003523"
DEFAULT_MIXING_TASK = "Whisking"
DEFAULT_POURING_VERB = "pour:Draining"
DEFAULT_POURING_FOODON = "obo:FOODON_03301304"


def _print_block(title, payload):
    print(f"[{title}]")
    print(pformat(payload, sort_dicts=False))
    print()


def main_query_knowledge(seed=None, robot_name=None, environment_name=None):
    _print_block(
        "cutting_query",
        safe_get_cutting_knowledge(DEFAULT_CUTTING_VERB, DEFAULT_CUTTING_FOODON),
    )
    _print_block(
        "mixing_query",
        safe_get_mixing_knowledge(DEFAULT_MIXING_TASK),
    )
    _print_block(
        "pouring_query",
        safe_get_pouring_knowledge(DEFAULT_POURING_VERB, DEFAULT_POURING_FOODON),
    )


if __name__ == "__main__":
    main_query_knowledge()
