import os
import sys
import threading

import pytest

from krrood.entity_query_language.factories import entity, variable, an
from krrood.symbol_graph.symbol_graph import SymbolGraph
from ..dataset.example_classes import KRROODPosition

try:
    import pydot
    import pygraphviz
except ImportError:
    pydot = None
    pygraphviz = None


@pytest.mark.skipif(
    not (pydot and pygraphviz), reason="pydot and graphviz not installed"
)
def test_visualize_symbol_graph():
    SymbolGraph().clear()
    symbol_graph = SymbolGraph()
    symbol_graph.to_dot("symbol_graph.svg", format_="svg", graph_type="type")
    assert len(symbol_graph._class_diagram.wrapped_classes) >= 59
    if os.path.exists("symbol_graph.svg"):
        os.remove("symbol_graph.svg")


def test_memory_leak():
    """
    Test if the SymbolGraph does not artificially keep objects alive that would be
    garbage collected.
    """

    def create_data():
        point = KRROODPosition(1, 2, 3)
        return point

    create_data()

    q = an(entity(variable(KRROODPosition, domain=None)))
    result = list(q.evaluate())

    assert result == []

    assert len(SymbolGraph().wrapped_instances) == 0


def test_concurrent_mutations_are_serialized():
    """
    The symbol graph is a process wide singleton that several threads mutate at once.

    Creating a `Symbol` adds a node from whatever thread happens to construct it, and
    query evaluation sweeps the garbage collected ones away with
    `remove_dead_instances`. Without a lock, two sweeps running at the same time work
    on the same snapshot of the graph and both try to remove the same dead node, so the
    slower one dies with ``ValueError: list.remove(x): x not in list``. This hammers
    both operations from several threads and fails if any of them raises.
    """
    symbol_graph = SymbolGraph()
    number_of_threads = 8
    rounds_per_thread = 20
    dead_instances_per_round = 200
    start = threading.Barrier(number_of_threads)
    errors = []

    def hammer(thread_index: int) -> None:
        try:
            start.wait()
            for round_index in range(rounds_per_thread):
                # The positions are dead as soon as the list is dropped, which turns
                # their nodes into ones that the next sweep has to remove.
                [
                    KRROODPosition(float(thread_index), float(round_index), float(i))
                    for i in range(dead_instances_per_round)
                ]
                symbol_graph.remove_dead_instances()
        except BaseException as e:  # noqa: BLE001 - reported to the main thread
            errors.append(e)

    threads = [
        threading.Thread(target=hammer, args=(thread_index,), daemon=True)
        for thread_index in range(number_of_threads)
    ]
    # Switch threads as often as possible so that the sweeps really do interleave.
    previous_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(previous_switch_interval)

    assert errors == []
    symbol_graph.remove_dead_instances()
    assert all(node.instance is not None for node in symbol_graph.wrapped_instances)
