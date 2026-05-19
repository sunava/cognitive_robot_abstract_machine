#!/usr/bin/env python3
"""
PC-style causal discovery for the bread-cutting CSV.

The script learns a graph skeleton and partial orientations from observational
data using conditional independence tests based on Gaussian partial correlation
and Fisher's Z transform.

Default setup:
    - input: cut_all_breads_results.csv
    - rows: PR2 and HSRB only, so robot identity is binary
    - variables: robot_is_pr2, final_success_numeric, and scene geometry

Assumptions are strong: no hidden confounders, no feedback cycles, and roughly
linear/Gaussian conditional relationships. Treat the output as a hypothesis
generator, not proof.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = (
    PROJECT_ROOT
    / "pycram"
    / "demos"
    / "thesis_new"
    / "records"
    / "cut_all_breads_results.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "pycram"
    / "demos"
    / "thesis_new"
    / "records"
    / "causality"
    / "pc_discovery"
)

ALPHA = 0.01
MAX_CONDITIONING_SET_SIZE = 3
MIN_ABS_PARTIAL_CORR_TO_REPORT = 0.03

VARIABLES = [
    "robot_is_pr2",
    "final_success_numeric",
    "object_world_x",
    "object_world_y",
    "object_world_z",
    "target_world_x",
    "target_world_y",
    "target_world_z",
    "object_size_x",
    "object_size_y",
    "object_size_z",
    "object_volume_aabb",
    "object_yaw_rad",
    "cut_normal_world_x",
    "cut_normal_world_y",
    "cut_normal_world_z",
    "cut_normal_world_yaw_rad",
    "slice_thickness_m",
    "num_cuts_x",
    "pointer_stride",
]


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT_CSV)
    data = data[data["robot_name"].isin(["pr2", "hsrb"])].copy()
    data["robot_is_pr2"] = (data["robot_name"] == "pr2").astype(float)
    data["final_success_numeric"] = data["final_success"].astype(bool).astype(float)

    for column in VARIABLES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data[VARIABLES].dropna().reset_index(drop=True)
    non_constant_columns = [
        column for column in data.columns if data[column].nunique(dropna=True) > 1
    ]
    data = data[non_constant_columns]

    standardized = (data - data.mean()) / data.std(ddof=0)
    standardized = (
        standardized.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    )
    print(f"Loaded {len(standardized):,} complete PR2/HSRB rows.")
    print(f"Variables used: {len(standardized.columns)}")
    return standardized


def partial_correlation(
    data: pd.DataFrame, x: str, y: str, conditioning_set: tuple[str, ...]
) -> float:
    columns = [x, y] + list(conditioning_set)
    matrix = data[columns].to_numpy(dtype=float)
    correlation = np.corrcoef(matrix, rowvar=False)
    precision = np.linalg.pinv(correlation)
    denominator = math.sqrt(max(precision[0, 0] * precision[1, 1], 1e-12))
    value = -precision[0, 1] / denominator
    return float(np.clip(value, -0.999999, 0.999999))


def fisher_z_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: tuple[str, ...],
) -> tuple[float, float]:
    r = partial_correlation(data, x, y, conditioning_set)
    dof = len(data) - len(conditioning_set) - 3
    if dof <= 0:
        return r, 1.0
    z_value = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
    p_value = 2 * (1 - norm.cdf(abs(z_value)))
    return r, float(p_value)


def ordered_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def run_pc_skeleton(
    data: pd.DataFrame,
) -> tuple[
    set[tuple[str, str]],
    dict[tuple[str, str], tuple[str, ...]],
    dict[tuple[str, str], dict],
]:
    variables = list(data.columns)
    edges = {ordered_pair(a, b) for a, b in itertools.combinations(variables, 2)}
    separating_sets: dict[tuple[str, str], tuple[str, ...]] = {}
    edge_tests: dict[tuple[str, str], dict] = {}

    for conditioning_size in range(MAX_CONDITIONING_SET_SIZE + 1):
        removed_this_round = 0
        for x, y in list(edges):
            neighbors = {
                b if a == x else a for a, b in edges if x in (a, b) and y not in (a, b)
            }
            if len(neighbors) < conditioning_size:
                continue

            for conditioning_set in itertools.combinations(
                sorted(neighbors), conditioning_size
            ):
                r, p_value = fisher_z_test(data, x, y, conditioning_set)
                edge_tests[(x, y)] = {
                    "conditioning_set": conditioning_set,
                    "partial_corr": r,
                    "p_value": p_value,
                }
                if p_value > ALPHA:
                    edges.remove((x, y))
                    separating_sets[(x, y)] = conditioning_set
                    removed_this_round += 1
                    break
        print(
            f"conditioning size {conditioning_size}: "
            f"removed {removed_this_round}, remaining edges {len(edges)}"
        )

    for edge in edges:
        if edge not in edge_tests:
            r, p_value = fisher_z_test(data, edge[0], edge[1], tuple())
            edge_tests[edge] = {
                "conditioning_set": tuple(),
                "partial_corr": r,
                "p_value": p_value,
            }
    return edges, separating_sets, edge_tests


def orient_edges(
    variables: list[str],
    skeleton_edges: set[tuple[str, str]],
    separating_sets: dict[tuple[str, str], tuple[str, ...]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    undirected = set(skeleton_edges)
    directed: set[tuple[str, str]] = set()

    def adjacent(a: str, b: str) -> bool:
        return (
            ordered_pair(a, b) in undirected or (a, b) in directed or (b, a) in directed
        )

    def orient(a: str, b: str) -> bool:
        edge = ordered_pair(a, b)
        if edge not in undirected:
            return False
        undirected.remove(edge)
        directed.add((a, b))
        return True

    # V-structures: X - Z - Y and X, Y non-adjacent, Z not in sep(X,Y): X -> Z <- Y
    for x, z, y in itertools.permutations(variables, 3):
        if x >= y:
            continue
        if (
            ordered_pair(x, z) in undirected
            and ordered_pair(y, z) in undirected
            and not adjacent(x, y)
        ):
            sep_set = separating_sets.get(ordered_pair(x, y), tuple())
            if z not in sep_set:
                orient(x, z)
                orient(y, z)

    changed = True
    while changed:
        changed = False
        # Meek R1: X -> Y - Z and X,Z non-adjacent implies Y -> Z
        for x, y in list(directed):
            for z in variables:
                if z in (x, y):
                    continue
                if ordered_pair(y, z) in undirected and not adjacent(x, z):
                    changed = orient(y, z) or changed

    return directed, undirected


def write_outputs(
    data: pd.DataFrame,
    directed: set[tuple[str, str]],
    undirected: set[tuple[str, str]],
    edge_tests: dict[tuple[str, str], dict],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for source, target in sorted(directed):
        test = edge_tests.get(ordered_pair(source, target), {})
        if abs(test.get("partial_corr", 0.0)) < MIN_ABS_PARTIAL_CORR_TO_REPORT:
            continue
        rows.append(
            {
                "source": source,
                "target": target,
                "edge_type": "directed",
                "partial_corr": test.get("partial_corr"),
                "abs_partial_corr": abs(test.get("partial_corr", 0.0)),
                "p_value": test.get("p_value"),
                "conditioning_set": ",".join(test.get("conditioning_set", tuple())),
            }
        )

    for a, b in sorted(undirected):
        test = edge_tests.get((a, b), {})
        if abs(test.get("partial_corr", 0.0)) < MIN_ABS_PARTIAL_CORR_TO_REPORT:
            continue
        rows.append(
            {
                "source": a,
                "target": b,
                "edge_type": "undirected",
                "partial_corr": test.get("partial_corr"),
                "abs_partial_corr": abs(test.get("partial_corr", 0.0)),
                "p_value": test.get("p_value"),
                "conditioning_set": ",".join(test.get("conditioning_set", tuple())),
            }
        )

    edges = pd.DataFrame(rows).sort_values("abs_partial_corr", ascending=False)
    edges.to_csv(OUTPUT_DIR / "pc_edges.csv", index=False)

    background_rows = []
    for a, b in sorted(set(directed) | {tuple(edge) for edge in undirected}):
        if a == "final_success_numeric" and b != "final_success_numeric":
            source, target, edge_type = b, a, "background_oriented"
        elif b == "final_success_numeric" and a != "final_success_numeric":
            source, target, edge_type = a, b, "background_oriented"
        elif a == "robot_is_pr2" and b == "final_success_numeric":
            source, target, edge_type = a, b, "background_oriented"
        elif b == "robot_is_pr2" and a == "final_success_numeric":
            source, target, edge_type = b, a, "background_oriented"
        elif (a, b) in directed:
            source, target, edge_type = a, b, "pc_directed"
        elif (b, a) in directed:
            source, target, edge_type = b, a, "pc_directed"
        else:
            source, target, edge_type = a, b, "undirected"

        test = edge_tests.get(
            ordered_pair(source, target), edge_tests.get(ordered_pair(a, b), {})
        )
        background_rows.append(
            {
                "source": source,
                "target": target,
                "edge_type": edge_type,
                "partial_corr": test.get("partial_corr"),
                "abs_partial_corr": abs(test.get("partial_corr", 0.0)),
                "p_value": test.get("p_value"),
                "conditioning_set": ",".join(test.get("conditioning_set", tuple())),
            }
        )

    background_edges = pd.DataFrame(background_rows).sort_values(
        "abs_partial_corr", ascending=False
    )
    background_edges.to_csv(
        OUTPUT_DIR / "pc_edges_background_oriented.csv", index=False
    )

    adjacency = pd.DataFrame(0, index=data.columns, columns=data.columns)
    for source, target in directed:
        adjacency.loc[source, target] = 1
    for a, b in undirected:
        adjacency.loc[a, b] = 1
        adjacency.loc[b, a] = 1
    adjacency.to_csv(OUTPUT_DIR / "pc_adjacency.csv")

    dot_lines = ["digraph PCDiscovery {"]
    for column in data.columns:
        shape = "doublecircle" if column == "final_success_numeric" else "ellipse"
        dot_lines.append(f'  "{column}" [shape={shape}];')
    for source, target in directed:
        dot_lines.append(f'  "{source}" -> "{target}";')
    for a, b in undirected:
        dot_lines.append(f'  "{a}" -> "{b}" [dir=none];')
    dot_lines.append("}")
    (OUTPUT_DIR / "pc_graph.dot").write_text(
        "\n".join(dot_lines) + "\n", encoding="utf-8"
    )

    report_lines = [
        "PC Causal Discovery Report",
        "==========================",
        "",
        f"Input CSV: {INPUT_CSV}",
        f"Rows used: {len(data)}",
        f"Variables used: {len(data.columns)}",
        f"Alpha: {ALPHA}",
        f"Max conditioning-set size: {MAX_CONDITIONING_SET_SIZE}",
        "",
        f"Directed edges: {len(directed)}",
        f"Undirected edges: {len(undirected)}",
        "",
        "Top reported edges:",
        (
            edges.head(30).to_string(index=False)
            if not edges.empty
            else "No edges above reporting threshold."
        ),
        "",
        "Top edges with minimal background orientation:",
        (
            background_edges.head(30).to_string(index=False)
            if not background_edges.empty
            else "No edges."
        ),
        "",
        "Caveat:",
        "This is observational causal discovery under strong assumptions: no hidden confounders,",
        "acyclic graph, and approximately linear/Gaussian conditional-independence behavior.",
        "Binary variables like robot_is_pr2 and final_success_numeric are included as numeric",
        "proxies, so directions involving them should be treated as hypotheses.",
    ]
    (OUTPUT_DIR / "pc_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(edges.head(25).to_string(index=False))
    print(f"Wrote {OUTPUT_DIR / 'pc_edges.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'pc_edges_background_oriented.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'pc_adjacency.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'pc_graph.dot'}")
    print(f"Wrote {OUTPUT_DIR / 'pc_report.txt'}")


def main() -> None:
    data = load_data()
    skeleton, separating_sets, edge_tests = run_pc_skeleton(data)
    directed, undirected = orient_edges(list(data.columns), skeleton, separating_sets)
    write_outputs(data, directed, undirected, edge_tests)


if __name__ == "__main__":
    main()
