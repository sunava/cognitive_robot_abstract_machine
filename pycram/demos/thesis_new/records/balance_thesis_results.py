#!/usr/bin/env python3
"""Create balanced thesis result CSVs without modifying the raw records."""

from __future__ import annotations

import csv
import random
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "balanced"
RANDOM_SEED = 20260518

INPUTS = [
    ("cut", "cut_all_breads_results.csv"),
    ("mix", "mix_all_bowls_results.csv"),
    ("wipe", "wipe_all_spaces_results.csv"),
]


def infer_wipe_environment(row: dict[str, str]) -> str:
    support_surface = row.get("support_surface_name", "")
    if "apartment" in support_surface:
        return "apartment"
    if "__" in support_surface:
        return "isr"
    return "kitchen"


def normalized_environment(action: str, row: dict[str, str]) -> str:
    world_name = row.get("world_name", "")
    if action == "wipe" and world_name == "map":
        return infer_wipe_environment(row)
    return world_name


def read_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for action, filename in INPUTS:
        with (SCRIPT_DIR / filename).open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["action_name"] = action
                row["normalized_world_name"] = normalized_environment(action, row)
                rows.append(row)
    return rows


def union_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    preferred = ["action_name", "normalized_world_name"]
    seen = set(preferred)
    fields = list(preferred)
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def balance_key(row: dict[str, str], mode: str) -> tuple[str, ...]:
    if mode == "global":
        return ("all",)
    if mode == "action":
        return (row["action_name"],)
    if mode == "action-environment":
        return (row["action_name"], row["normalized_world_name"])
    raise ValueError(f"Unsupported balance mode: {mode}")


def parse_args() -> str:
    parser = ArgumentParser(
        description="Balance thesis result CSVs without modifying raw records."
    )
    parser.add_argument(
        "--mode",
        choices=["global", "action", "action-environment"],
        default="action-environment",
        help=(
            "global: one count for every action/robot/environment cell; "
            "action: one count per action; "
            "action-environment: one count per action/environment across robots."
        ),
    )
    return parser.parse_args().mode


def main() -> None:
    mode = parse_args()
    output_dirs = [OUTPUT_DIR / mode]
    if mode == "action-environment":
        output_dirs.append(OUTPUT_DIR)
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows()

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["action_name"],
            row.get("robot_name", ""),
            row["normalized_world_name"],
        )
        groups[key].append(row)

    target_counts: dict[tuple[str, ...], int] = {}
    for key, group_rows in groups.items():
        action, _robot, environment = key
        mode_key = balance_key(
            {"action_name": action, "normalized_world_name": environment}, mode
        )
        target_counts[mode_key] = min(
            target_counts.get(mode_key, len(group_rows)), len(group_rows)
        )
    rng = random.Random(RANDOM_SEED)

    balanced_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    for key in sorted(groups):
        group_rows = groups[key]
        action, robot, environment = key
        mode_key = balance_key(
            {"action_name": action, "normalized_world_name": environment}, mode
        )
        target_count = target_counts[mode_key]
        sampled_rows = sorted(
            rng.sample(group_rows, target_count),
            key=lambda row: (
                row.get("run_id", ""),
                row.get("task_instance_id", ""),
                row.get("seed", ""),
            ),
        )
        balanced_rows.extend(sampled_rows)
        summary_rows.append(
            {
                "balance_mode": mode,
                "action_name": action,
                "robot_name": robot,
                "environment": environment,
                "raw_count": str(len(group_rows)),
                "balanced_count": str(target_count),
                "dropped_count": str(len(group_rows) - target_count),
            }
        )

    fields = union_fieldnames(balanced_rows)
    for output_dir in output_dirs:
        write_csv(output_dir / "combined_balanced_observed.csv", balanced_rows, fields)
        write_csv(
            output_dir / "balance_summary_observed.csv",
            summary_rows,
            [
                "balance_mode",
                "action_name",
                "robot_name",
                "environment",
                "raw_count",
                "balanced_count",
                "dropped_count",
            ],
        )

        for action, _filename in INPUTS:
            action_rows = [row for row in balanced_rows if row["action_name"] == action]
            write_csv(
                output_dir / f"{action}_balanced_observed.csv", action_rows, fields
            )

    counts = Counter(
        (
            row["action_name"],
            row.get("robot_name", ""),
            row["normalized_world_name"],
        )
        for row in balanced_rows
    )
    print(f"balance_mode={mode}")
    print(f"balanced_rows={len(balanced_rows)}")
    for key in sorted(counts):
        print(*key, counts[key], sep=",")


if __name__ == "__main__":
    main()
