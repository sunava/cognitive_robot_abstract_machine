#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    records_dir = Path(__file__).resolve().parent / "records"
    parser = argparse.ArgumentParser(
        description=(
            "Create thesis plots for object-articulated cutting bindings from the "
            "cut_all_breads_results.csv log."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=records_dir / "cut_all_breads_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=records_dir / "object_articulation_plots",
    )
    return parser.parse_args()


REQUIRED_FIELDS = [
    "anchor_local_x",
    "anchor_local_y",
    "anchor_local_z",
    "anchor_norm_x",
    "anchor_norm_y",
    "anchor_norm_z",
    "cut_normal_world_x",
    "cut_normal_world_y",
    "cut_normal_world_z",
    "cut_normal_world_yaw_rad",
    "object_size_x",
    "object_size_y",
    "object_size_z",
    "object_volume_aabb",
    "object_yaw_rad",
    "robot_decision",
    "outcome",
    "bread_name",
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ensure_required_fields(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("Input CSV has no rows.")
    fieldnames = set(rows[0].keys())
    missing = [field for field in REQUIRED_FIELDS if field not in fieldnames]
    if missing:
        raise RuntimeError(
            "CSV is missing object-articulation fields. "
            "Rerun the updated cutting experiment first. Missing: " + ", ".join(missing)
        )


def to_float_array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        text = row.get(key, "")
        values.append(float(text) if text not in ("", None) else np.nan)
    return np.asarray(values, dtype=float)


def wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def decision_color_map(
    decisions: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    unique = sorted(set(decisions))
    cmap = plt.get_cmap("tab10")
    return {decision: cmap(i % 10) for i, decision in enumerate(unique)}


def plot_anchor_distribution(rows: list[dict[str, str]], output_dir: Path) -> None:
    decisions = [row.get("robot_decision", "") for row in rows]
    colors = decision_color_map(decisions)
    anchor_x = to_float_array(rows, "anchor_norm_x")
    anchor_y = to_float_array(rows, "anchor_norm_y")
    anchor_z = to_float_array(rows, "anchor_norm_z")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    for decision in sorted(colors):
        mask = np.array(
            [row.get("robot_decision", "") == decision for row in rows], dtype=bool
        )
        axes[0].scatter(
            anchor_x[mask],
            anchor_z[mask],
            s=28,
            alpha=0.8,
            label=decision,
            color=colors[decision],
        )
        axes[1].scatter(
            anchor_x[mask],
            anchor_y[mask],
            s=28,
            alpha=0.8,
            label=decision,
            color=colors[decision],
        )

    axes[0].set_title("Anchor distribution in normalized object frame (x-z)")
    axes[0].set_xlabel("normalized local x")
    axes[0].set_ylabel("normalized local z")
    axes[1].set_title("Anchor distribution in normalized object frame (x-y)")
    axes[1].set_xlabel("normalized local x")
    axes[1].set_ylabel("normalized local y")
    for axis in axes:
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "anchor_position_distribution.png", dpi=180)
    plt.close(fig)


def plot_orientation_relation(rows: list[dict[str, str]], output_dir: Path) -> None:
    object_yaw = to_float_array(rows, "object_yaw_rad")
    cut_yaw = to_float_array(rows, "cut_normal_world_yaw_rad")
    yaw_error = wrap_to_pi(cut_yaw - object_yaw)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(object_yaw, cut_yaw, s=26, alpha=0.85, color="tab:blue")
    low = float(np.nanmin([object_yaw.min(), cut_yaw.min()]))
    high = float(np.nanmax([object_yaw.max(), cut_yaw.max()]))
    axes[0].plot([low, high], [low, high], linestyle="--", color="0.4")
    axes[0].set_xlabel("object yaw [rad]")
    axes[0].set_ylabel("cut normal yaw [rad]")
    axes[0].set_title("Chosen cut orientation vs. object orientation")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(
        yaw_error[np.isfinite(yaw_error)], bins=24, color="tab:orange", alpha=0.85
    )
    axes[1].set_xlabel("yaw difference [rad]")
    axes[1].set_ylabel("count")
    axes[1].set_title("Orientation residual")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "cut_orientation_vs_object_orientation.png", dpi=180)
    plt.close(fig)


def plot_geometry_strategy_relation(
    rows: list[dict[str, str]], output_dir: Path
) -> None:
    decisions = [row.get("robot_decision", "") for row in rows]
    colors = decision_color_map(decisions)
    size_x = to_float_array(rows, "object_size_x")
    size_y = to_float_array(rows, "object_size_y")
    size_z = to_float_array(rows, "object_size_z")
    volume = to_float_array(rows, "object_volume_aabb")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for decision in sorted(colors):
        mask = np.array(
            [row.get("robot_decision", "") == decision for row in rows], dtype=bool
        )
        axes[0].scatter(
            size_x[mask],
            size_y[mask],
            s=30 + 120 * np.clip(size_z[mask], 0.0, None),
            alpha=0.8,
            color=colors[decision],
            label=decision,
        )
    axes[0].set_xlabel("object size x [m]")
    axes[0].set_ylabel("object size y [m]")
    axes[0].set_title("Object geometry vs. selected strategy")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    unique_decisions = sorted(colors)
    grouped = [
        volume[
            np.array(
                [row.get("robot_decision", "") == decision for row in rows], dtype=bool
            )
        ]
        for decision in unique_decisions
    ]
    axes[1].boxplot(grouped, labels=unique_decisions, vert=True)
    axes[1].set_ylabel("AABB volume [m$^3$]")
    axes[1].set_title("Geometry-strategy relation via object volume")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "geometry_vs_selected_strategy.png", dpi=180)
    plt.close(fig)


def plot_anchor_heatmap(rows: list[dict[str, str]], output_dir: Path) -> None:
    anchor_x = to_float_array(rows, "anchor_norm_x")
    anchor_z = to_float_array(rows, "anchor_norm_z")
    fig, axis = plt.subplots(figsize=(6, 5))
    hist = axis.hist2d(
        anchor_x, anchor_z, bins=20, range=[[0.0, 1.0], [0.0, 1.0]], cmap="YlOrRd"
    )
    axis.set_xlabel("normalized local x")
    axis.set_ylabel("normalized local z")
    axis.set_title("Heatmap of chosen anchor $p^*$ across objects")
    axis.grid(True, alpha=0.15)
    fig.colorbar(hist[3], ax=axis, label="count")
    fig.tight_layout()
    fig.savefig(output_dir / "anchor_heatmap.png", dpi=180)
    plt.close(fig)


def plot_single_binding_concept(rows: list[dict[str, str]], output_dir: Path) -> None:
    row = rows[0]
    mins = np.array(
        [
            float(row["object_aabb_min_x"]),
            float(row["object_aabb_min_z"]),
        ]
    )
    maxs = np.array(
        [
            float(row["object_aabb_max_x"]),
            float(row["object_aabb_max_z"]),
        ]
    )
    anchor = np.array(
        [
            float(row["anchor_local_x"]),
            float(row["anchor_local_z"]),
        ]
    )
    normal = np.array(
        [
            float(row["cut_normal_local_x"]),
            float(row["cut_normal_local_z"]),
        ]
    )

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.add_patch(
        plt.Rectangle(
            mins,
            maxs[0] - mins[0],
            maxs[1] - mins[1],
            fill=False,
            linewidth=2.0,
            color="black",
        )
    )
    axis.arrow(
        0.0, 0.0, 0.04, 0.0, width=0.0006, color="tab:blue", length_includes_head=True
    )
    axis.arrow(
        0.0, 0.0, 0.0, 0.04, width=0.0006, color="tab:green", length_includes_head=True
    )
    axis.text(0.042, 0.0, "x", color="tab:blue", va="center")
    axis.text(0.0, 0.042, "z", color="tab:green", ha="center")
    axis.scatter([anchor[0]], [anchor[1]], color="tab:red", s=70, zorder=3)
    axis.arrow(
        anchor[0],
        anchor[1],
        0.03 * normal[0],
        0.03 * normal[1],
        width=0.0006,
        color="tab:red",
        length_includes_head=True,
    )
    axis.text(anchor[0], anchor[1], "  $p^*$", color="tab:red", va="bottom")
    axis.set_title("Single-object cut binding in local object frame")
    axis.set_xlabel("local x [m]")
    axis.set_ylabel("local z [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "single_object_binding_concept.png", dpi=180)
    plt.close(fig)


def write_binding_summary(rows: list[dict[str, str]], output_dir: Path) -> None:
    decisions = [row.get("robot_decision", "") for row in rows]
    unique, counts = np.unique(np.asarray(decisions, dtype=object), return_counts=True)
    lines = [
        "Object-articulated cutting summary",
        "",
        "Logged quantities",
        "- selected local anchor p* in the object frame",
        "- selected contact normal n in local and world coordinates",
        "- object dimensions from the local axis-aligned bounding box",
        "- object world pose and orientation",
        "- chosen cutting technique parameters",
        "",
        "Decision counts",
    ]
    for decision, count in zip(unique.tolist(), counts.tolist()):
        lines.append(f"- {decision}: {count}")
    (output_dir / "binding_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    ensure_required_fields(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_anchor_distribution(rows, args.output_dir)
    plot_orientation_relation(rows, args.output_dir)
    plot_geometry_strategy_relation(rows, args.output_dir)
    plot_anchor_heatmap(rows, args.output_dir)
    plot_single_binding_concept(rows, args.output_dir)
    write_binding_summary(rows, args.output_dir)
    print(f"Wrote object-articulation plots to {args.output_dir}")


if __name__ == "__main__":
    main()
