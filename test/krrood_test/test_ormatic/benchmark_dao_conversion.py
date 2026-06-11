"""
Benchmark for the ORMatic DAO conversion routines (``to_dao`` / ``from_dao``).

Builds a sizable mixed object graph from the krrood test dataset classes
(scalars, single relationships, association collections, alternative mappings,
``__post_init__`` back-references), then times:

- ``to_dao``: converting all domain roots with a shared conversion state
- ``insert``: persisting all DAOs (add_all + commit) into in-memory SQLite
- ``query``: loading all root DAOs back (selectin-loaded relationships)
- ``from_dao``: reconstructing all domain objects with a shared state

Run standalone from the repository root (the generated
``test/krrood_test/dataset/ormatic_interface.py`` must exist, e.g. after a
pytest run of the krrood test suite)::

    python test/krrood_test/test_ormatic/benchmark_dao_conversion.py \
        --groups 100 --positions 10 --repetitions 5 [--json results.json]

Results are also written to ``PERFORMANCE_RESULTS.md`` (manually curated)
to compare the state before and after the performance upgrades.
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

# make the repository root importable so `test.krrood_test...` resolves
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState
from krrood.ormatic.utils import create_engine
from krrood.symbol_graph.symbol_graph import SymbolGraph

from test.krrood_test.dataset.example_classes import (
    AlternativeMappingAggregator,
    ContainerGeneration,
    Entity,
    ItemWithBackreference,
    KRROODOrientation,
    KRROODPose,
    KRROODPosition,
    KRROODPositions,
)
from test.krrood_test.dataset.ormatic_interface import (
    AlternativeMappingAggregatorDAO,
    Base,
    ContainerGenerationDAO,
    KRROODPoseDAO,
    KRROODPositionsDAO,
)

ROOT_DAO_CLASSES = [
    KRROODPositionsDAO,
    KRROODPoseDAO,
    AlternativeMappingAggregatorDAO,
    ContainerGenerationDAO,
]


def build_object_graph(groups: int, positions_per_group: int):
    """
    Build a list of domain root objects with a mix of relationship kinds.

    :param groups: Number of object groups; each group contributes four roots.
    :param positions_per_group: Number of positions per KRROODPositions root.
    :return: The list of domain roots and the total number of domain objects.
    """
    roots = []
    total_objects = 0
    for group_index in range(groups):
        positions = [
            KRROODPosition(group_index, j, j + 1) for j in range(positions_per_group)
        ]
        roots.append(KRROODPositions(positions, ["a", "b", "c"]))

        roots.append(
            KRROODPose(
                KRROODPosition(group_index, 1, 2),
                KRROODOrientation(1.0, 2.0, 3.0, None),
            )
        )

        entities = [Entity(f"entity_{group_index}_{k}") for k in range(3)]
        roots.append(AlternativeMappingAggregator(entities, entities[:2]))

        roots.append(
            ContainerGeneration([ItemWithBackreference(j) for j in range(3)])
        )

        total_objects += (
            1 + positions_per_group  # positions root + positions
            + 3  # pose + position + orientation
            + 1 + 3  # aggregator + entities
            + 1 + 3  # container + items
        )
    return roots, total_objects


def run_once(groups: int, positions_per_group: int):
    """
    Run one full benchmark repetition and return the phase timings in seconds.
    """
    SymbolGraph().clear()
    SymbolGraph()
    roots, total_objects = build_object_graph(groups, positions_per_group)
    timings = {"objects": total_objects, "roots": len(roots)}

    start = time.perf_counter()
    to_state = ToDataAccessObjectState()
    daos = [to_dao(root, to_state) for root in roots]
    timings["to_dao"] = time.perf_counter() - start

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(engine)()
    try:
        start = time.perf_counter()
        session.add_all(daos)
        session.commit()
        timings["insert"] = time.perf_counter() - start

        session.expunge_all()

        start = time.perf_counter()
        queried = []
        for dao_class in ROOT_DAO_CLASSES:
            queried.extend(session.scalars(select(dao_class)).all())
        timings["query"] = time.perf_counter() - start

        start = time.perf_counter()
        from_state = FromDataAccessObjectState()
        reconstructed = [dao.from_dao(from_state) for dao in queried]
        timings["from_dao"] = time.perf_counter() - start

        assert len(reconstructed) == len(roots)
    finally:
        session.close()
        engine.dispose()
        SymbolGraph().clear()

    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--positions", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--json", type=str, default=None, help="dump results to file")
    args = parser.parse_args()

    # warmup: prime SQLAlchemy mappers and all lookup caches
    run_once(5, 3)

    results = [run_once(args.groups, args.positions) for _ in range(args.repetitions)]

    phases = ["to_dao", "insert", "query", "from_dao"]
    summary = {
        "groups": args.groups,
        "positions_per_group": args.positions,
        "repetitions": args.repetitions,
        "objects": results[0]["objects"],
        "roots": results[0]["roots"],
        "phases": {},
    }
    print(
        f"\n{summary['objects']} domain objects, {summary['roots']} roots, "
        f"{args.repetitions} repetitions"
    )
    print(f"{'phase':<10} {'mean [ms]':>12} {'min [ms]':>12} {'max [ms]':>12}")
    for phase in phases:
        values = [r[phase] for r in results]
        mean, minimum, maximum = statistics.mean(values), min(values), max(values)
        summary["phases"][phase] = {"mean": mean, "min": minimum, "max": maximum}
        print(
            f"{phase:<10} {mean * 1000:>12.1f} {minimum * 1000:>12.1f} "
            f"{maximum * 1000:>12.1f}"
        )

    if args.json:
        with open(args.json, "w") as file:
            json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()
