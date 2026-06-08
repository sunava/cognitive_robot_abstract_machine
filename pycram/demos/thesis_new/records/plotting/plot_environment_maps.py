#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

THESIS_NEW_ROOT = Path(__file__).resolve().parents[2]
PYCRAM_ROOT = THESIS_NEW_ROOT.parents[1]
WORKSPACE_ROOT = PYCRAM_ROOT.parent
for source_root in [
    THESIS_NEW_ROOT,
    THESIS_NEW_ROOT / "src",
    PYCRAM_ROOT / "src",
    WORKSPACE_ROOT / "krrood" / "src",
    WORKSPACE_ROOT / "probabilistic_model" / "src",
    WORKSPACE_ROOT / "random_events" / "src",
    WORKSPACE_ROOT / "semantic_digital_twin" / "src",
]:
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt


def parse_args(
    *,
    default_input_name: str = "cut_all_breads_results.csv",
    default_output_dir_name: str = "environment_maps",
    default_environment_name: str = "",
    task_label: str = "Cutting",
) -> argparse.Namespace:
    records_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            f"Create top-down environment maps for {task_label.lower()} experiment "
            "success/failure."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=records_dir / default_input_name,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=records_dir / default_output_dir_name,
    )
    parser.add_argument(
        "--environment-name",
        default=default_environment_name,
        help="Optional thesis environment name. Defaults to inference from CSV/world_name.",
    )
    parser.add_argument(
        "--robot-name",
        default="",
        help="Optional thesis robot name. Defaults to first robot_name in CSV.",
    )
    parser.add_argument(
        "--split-by-world-name",
        action="store_true",
        help=(
            "Write one environment-map set per distinct world_name into separate "
            "subdirectories. Enabled automatically when the CSV contains multiple "
            "world_name values."
        ),
    )
    parser.add_argument(
        "--task-label",
        default=task_label,
        help="Task label used in the generated text summary.",
    )
    return parser.parse_args()


def parse_all_tasks_args() -> argparse.Namespace:
    records_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Create top-down environment maps for one or all thesis experiment tasks."
        )
    )
    parser.add_argument(
        "--task",
        choices=[*TASK_CONFIGS.keys(), "all"],
        default="all",
        help="Task to plot. Defaults to all tasks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=records_dir / "environment_maps",
        help="Root output directory. Task/world/robot subdirectories are created below it.",
    )
    parser.add_argument(
        "--environment-name",
        default="",
        help="Optional thesis environment override for all selected tasks.",
    )
    parser.add_argument(
        "--robot-name",
        default="",
        help="Optional thesis robot override for all selected tasks.",
    )
    parser.add_argument(
        "--split-by-world-name",
        action="store_true",
        help="Force one map set per distinct world_name/robot_name group.",
    )
    return parser.parse_args()


REQUIRED_FIELDS = [
    "target_world_x",
    "target_world_y",
    "outcome",
    "robot_decision",
    "recovery_used",
    "support_surface_name",
    "support_world_x",
    "support_world_y",
    "support_yaw_rad",
    "support_size_x",
    "support_size_y",
]
INPUT_TEMPLATE_FIELDS = ["world_name", "robot_name", *REQUIRED_FIELDS]
TASK_CONFIGS = {
    "cutting": {
        "input_name": "csv/cutting_result.csv",
        "output_dir_name": "environment_maps/cutting",
        "environment_name": "",
        "task_label": "Cutting",
    },
    "mixing": {
        "input_name": "csv/mixing_reuslt.csv",
        "output_dir_name": "environment_maps/mixing",
        "environment_name": "",
        "task_label": "Mixing",
    },
    "pouring": {
        "input_name": "csv/pouring_results.csv",
        "output_dir_name": "environment_maps/pouring",
        "environment_name": "",
        "task_label": "Pouring",
    },
    "wiping": {
        "input_name": "csv/raw_wiping_merged.csv",
        "output_dir_name": "environment_maps/wiping",
        "environment_name": "",
        "task_label": "Wiping",
    },
}

FIXED_PLOT_BOUNDS_BY_ENVIRONMENT = {
    "isr": (-6.0, 5.0, -5.0, 6.0),
    "isr-testbed": (-6.0, 5.0, -5.0, 6.0),
}


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(",".join(INPUT_TEMPLATE_FIELDS) + "\n", encoding="utf-8")
        print(f"Created missing input CSV template: {path}")
        return []

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def slugify(text: str) -> str:
    cleaned = "".join(
        ch.lower() if ch.isalnum() else "_" for ch in (text or "").strip()
    )
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed or "unknown_world"


def infer_wiping_environment_from_support(row: dict[str, str]) -> str:
    support_surface = row.get("support_surface_name", "")
    if "apartment" in support_surface:
        return "apartment"
    if "__" in support_surface:
        return "isr"
    return "kitchen"


def infer_row_environment_key(row: dict[str, str]) -> str:
    for field in ["normalized_world_name", "environment_name"]:
        value = (row.get(field) or "").strip()
        if value:
            return value

    world_name = (row.get("world_name") or "").strip()
    if world_name == "map" and row.get("task_name", "").lower() == "wiping":
        return infer_wiping_environment_from_support(row)
    if world_name == "map" and row.get("support_surface_name"):
        return infer_wiping_environment_from_support(row)
    return world_name or "unknown_world"


def group_rows_by_world_and_robot(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        world_name = infer_row_environment_key(row)
        robot_name = (row.get("robot_name") or "").strip() or "unknown_robot"
        grouped.setdefault((world_name, robot_name), []).append(row)
    return dict(sorted(grouped.items()))


def ensure_required_fields(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("Input CSV has no rows.")
    fields = set(rows[0].keys())
    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    if missing:
        raise RuntimeError(
            "CSV is missing environment-map fields. "
            "Rerun the updated experiment first. Missing: " + ", ".join(missing)
        )


def to_float(value: str) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def unique_surface_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    keyed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row.get("support_surface_name", ""),
            row.get("support_world_x", ""),
            row.get("support_world_y", ""),
        )
        if key[0]:
            keyed[key] = row
    return list(keyed.values())


def infer_environment_name(rows: list[dict[str, str]], override: str) -> str:
    if override:
        return override
    environment_key = infer_row_environment_key(rows[0])
    if environment_key and environment_key != "unknown_world":
        return environment_key
    world_name = rows[0].get("world_name", "") or ""
    if world_name.endswith("_root"):
        return world_name[: -len("_root")]
    if world_name:
        return world_name
    return "apartment"


def infer_robot_name(rows: list[dict[str, str]], override: str) -> str:
    if override:
        return override
    return rows[0].get("robot_name", "") or "pr2"


def rotated_rectangle(
    center_x: float, center_y: float, size_x: float, size_y: float, yaw: float
) -> np.ndarray:
    half = np.array(
        [
            [-0.5 * size_x, -0.5 * size_y],
            [0.5 * size_x, -0.5 * size_y],
            [0.5 * size_x, 0.5 * size_y],
            [-0.5 * size_x, 0.5 * size_y],
            [-0.5 * size_x, -0.5 * size_y],
        ],
        dtype=float,
    )
    c = np.cos(yaw)
    s = np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return half @ rotation.T + np.array([center_x, center_y], dtype=float)


def body_topdown_polygons(body, reference_frame) -> list[np.ndarray]:
    try:
        bbc = body.collision.as_bounding_box_collection_in_frame(reference_frame)
    except Exception:
        return []

    polygons: list[np.ndarray] = []
    for bb in getattr(bbc, "bounding_boxes", []):
        min_x = float(bb.min_x)
        min_y = float(bb.min_y)
        max_x = float(bb.max_x)
        max_y = float(bb.max_y)
        if not np.all(np.isfinite([min_x, min_y, max_x, max_y])):
            continue
        if (max_x - min_x) <= 1e-4 or (max_y - min_y) <= 1e-4:
            continue
        polygons.append(
            np.array(
                [
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                    [min_x, min_y],
                ],
                dtype=float,
            )
        )
    return polygons


def background_body_skip_reason(name: str, robot_name: str = "") -> str | None:
    lowered = (name or "").lower()
    if not lowered:
        return "empty_name"
    if lowered.startswith(("bread_", "knife_", "whisk_", "bowl_")):
        return "dynamic_task_object"
    # Preserve key ISR environment furniture/structure even when names contain
    # generic tokens like "room" or "base_link".
    if any(token in lowered for token in ["wall", "table", "chair", "bed"]):
        return None
    if any(
        token in lowered
        for token in [
            "pr2",
            "hsrb",
            "stretch",
            "tiago",
            "armar",
            "justin",
            "g1",
            "robot",
        ]
    ):
        return "robot_body_name"
    if robot_name and robot_name.lower() in lowered:
        return "selected_robot_name"
    if any(
        token in lowered
        for token in [
            "floor",
            "ceiling",
            "ground",
            "room",
            "apartment_root",
            "apartment",
        ]
    ):
        return "world_shell_name"
    if any(
        token in lowered
        for token in [
            "base_footprint",
            "base_bellow",
            "caster",
            "torso",
            "head_pan",
            "head_tilt",
            "stereo",
            "sensor_mount",
            "laser_tilt",
            "shoulder",
            "upper_arm",
            "forearm",
            "elbow",
            "wrist",
            "gripper",
            "finger",
            "palm_link",
            "force_torque",
            "imu_link",
            "optical_frame",
            "camera_frame",
            "tool_frame",
        ]
    ):
        return "robot_link_name"
    if lowered.endswith(("_link", "_frame")):
        return "link_or_frame_suffix"
    return None


def is_world_background_body(name: str, robot_name: str = "") -> bool:
    return background_body_skip_reason(name, robot_name=robot_name) is None


def inspect_world_background(
    rows: list[dict[str, str]], environment_name: str, robot_name: str
) -> tuple[list[tuple[str, np.ndarray]], list[dict[str, str]]]:
    try:
        from world_setup import setup_thesis_world
    except ModuleNotFoundError as exc:
        return [], [
            {
                "name": "world_setup",
                "status": "skipped",
                "reason": f"missing_dependency:{exc.name}",
            }
        ]

    try:
        world = setup_thesis_world(
            robot_name=robot_name, environment_name=environment_name
        )
    except ModuleNotFoundError as exc:
        return [], [
            {
                "name": f"{environment_name}/{robot_name}",
                "status": "skipped",
                "reason": f"missing_dependency:{exc.name}",
            }
        ]
    except Exception as exc:
        return [], [
            {
                "name": f"{environment_name}/{robot_name}",
                "status": "skipped",
                "reason": f"world_setup_error:{type(exc).__name__}",
            }
        ]
    bodies = list(getattr(world, "bodies", []))
    reference_frame = getattr(world, "root", None)
    polygons: list[tuple[str, np.ndarray]] = []
    report: list[dict[str, str]] = []

    for body in bodies:
        name = getattr(getattr(body, "name", None), "name", None) or getattr(
            body, "name", ""
        )
        if not isinstance(name, str):
            report.append(
                {"name": repr(name), "status": "skipped", "reason": "non_string_name"}
            )
            continue

        skip_reason = background_body_skip_reason(name, robot_name=robot_name)
        if skip_reason is not None:
            report.append({"name": name, "status": "skipped", "reason": skip_reason})
            continue

        try:
            body_polygons = body_topdown_polygons(body, reference_frame)
        except Exception as exc:
            report.append(
                {
                    "name": name,
                    "status": "skipped",
                    "reason": f"polygon_error:{type(exc).__name__}",
                }
            )
            continue

        if not body_polygons:
            report.append(
                {"name": name, "status": "skipped", "reason": "no_xy_polygons"}
            )
            continue

        kept_count = 0
        skipped_large = 0
        for polygon in body_polygons:
            extent_x = float(np.max(polygon[:, 0]) - np.min(polygon[:, 0]))
            extent_y = float(np.max(polygon[:, 1]) - np.min(polygon[:, 1]))
            if extent_x > 20.0 or extent_y > 20.0:
                skipped_large += 1
                continue
            polygons.append((name, polygon))
            kept_count += 1

        if kept_count > 0:
            report.append(
                {
                    "name": name,
                    "status": "kept",
                    "reason": f"polygons={kept_count}",
                }
            )
        else:
            report.append(
                {
                    "name": name,
                    "status": "skipped",
                    "reason": f"oversized_polygons={skipped_large}",
                }
            )

    return polygons, report


def load_world_background(
    rows: list[dict[str, str]], environment_name: str, robot_name: str
) -> list[tuple[str, np.ndarray]]:
    polygons, _ = inspect_world_background(rows, environment_name, robot_name)
    return polygons


def relevant_bounds(
    rows: list[dict[str, str]],
) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = to_float(row.get("target_world_x", ""))
        y = to_float(row.get("target_world_y", ""))
        if np.isfinite(x) and np.isfinite(y):
            xs.append(float(x))
            ys.append(float(y))
        sx = to_float(row.get("support_world_x", ""))
        sy = to_float(row.get("support_world_y", ""))
        ssx = to_float(row.get("support_size_x", ""))
        ssy = to_float(row.get("support_size_y", ""))
        if np.all(np.isfinite([sx, sy, ssx, ssy])):
            xs.extend([float(sx - 0.6 * ssx), float(sx + 0.6 * ssx)])
            ys.extend([float(sy - 0.6 * ssy), float(sy + 0.6 * ssy)])
    if not xs or not ys:
        return None
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    pad_x = max(0.2, 0.15 * (max_x - min_x if max_x > min_x else 1.0))
    pad_y = max(0.2, 0.15 * (max_y - min_y if max_y > min_y else 1.0))
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def combined_plot_bounds(
    environment_name: str,
    rows: list[dict[str, str]],
    world_polygons: list[tuple[str, np.ndarray]],
) -> tuple[float, float, float, float] | None:
    fixed_bounds = FIXED_PLOT_BOUNDS_BY_ENVIRONMENT.get(environment_name)
    if fixed_bounds is not None:
        return fixed_bounds

    xs: list[float] = []
    ys: list[float] = []

    row_bounds = relevant_bounds(rows)
    if row_bounds is not None:
        min_x, max_x, min_y, max_y = row_bounds
        xs.extend([min_x, max_x])
        ys.extend([min_y, max_y])

    for _, polygon in world_polygons:
        if polygon.size == 0:
            continue
        xs.extend(polygon[:, 0].tolist())
        ys.extend(polygon[:, 1].tolist())

    if not xs or not ys:
        return None

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    pad_x = max(0.2, 0.05 * (max_x - min_x if max_x > min_x else 1.0))
    pad_y = max(0.2, 0.05 * (max_y - min_y if max_y > min_y else 1.0))
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def outcome_style(row: dict[str, str]) -> tuple[str, str, str]:
    success = row.get("outcome", "") == "success"
    recovery = to_bool(row.get("recovery_used", ""))
    if success and not recovery:
        return ("tab:green", "o", "direct success")
    if success and recovery:
        return ("tab:orange", "^", "success after recovery")
    return ("tab:red", "x", "failure")


def annotate_outcome_counts(axis, rows: list[dict[str, str]]) -> None:
    direct_success = sum(
        1
        for row in rows
        if row.get("outcome", "") == "success"
        and not to_bool(row.get("recovery_used", ""))
    )
    recovered_success = sum(
        1
        for row in rows
        if row.get("outcome", "") == "success" and to_bool(row.get("recovery_used", ""))
    )
    failures = sum(1 for row in rows if row.get("outcome", "") != "success")
    summary = (
        f"direct success: {direct_success}\n"
        f"success after recovery: {recovered_success}\n"
        f"failure: {failures}"
    )
    return summary


def legend_labels(rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    direct_success = sum(
        1
        for row in rows
        if row.get("outcome", "") == "success"
        and not to_bool(row.get("recovery_used", ""))
    )
    recovered_success = sum(
        1
        for row in rows
        if row.get("outcome", "") == "success" and to_bool(row.get("recovery_used", ""))
    )
    failures = sum(1 for row in rows if row.get("outcome", "") != "success")
    return [
        ("tab:green", "o", f"direct success ({direct_success})"),
        ("tab:orange", "^", f"success after recovery ({recovered_success})"),
        ("tab:red", "x", f"failure ({failures})"),
    ]


def plot_environment_success_map(
    environment_name: str,
    rows: list[dict[str, str]],
    output_dir: Path,
    world_polygons: list[tuple[str, np.ndarray]],
) -> None:
    fig, axis = plt.subplots(figsize=(12, 8))
    bounds = combined_plot_bounds(environment_name, rows, world_polygons)

    for name, polygon in world_polygons:
        axis.fill(
            polygon[:, 0],
            polygon[:, 1],
            color="#dddddd",
            alpha=0.55,
            edgecolor="#aaaaaa",
            linewidth=0.7,
            zorder=0,
        )

    for surface in unique_surface_rows(rows):
        x = to_float(surface.get("support_world_x", ""))
        y = to_float(surface.get("support_world_y", ""))
        sx = to_float(surface.get("support_size_x", ""))
        sy = to_float(surface.get("support_size_y", ""))
        yaw = to_float(surface.get("support_yaw_rad", ""))
        if not np.all(np.isfinite([x, y, sx, sy])):
            continue
        polygon = rotated_rectangle(x, y, sx, sy, 0.0 if not np.isfinite(yaw) else yaw)
        axis.fill(
            polygon[:, 0],
            polygon[:, 1],
            color="#e6d3a7",
            alpha=0.6,
            edgecolor="#8f7a54",
            linewidth=1.2,
            zorder=1,
        )
        axis.text(
            x,
            y,
            surface.get("support_surface_name", ""),
            ha="center",
            va="center",
            fontsize=8,
            alpha=0.8,
        )

    for row in rows:
        x = to_float(row.get("target_world_x", ""))
        y = to_float(row.get("target_world_y", ""))
        if not np.all(np.isfinite([x, y])):
            continue
        color, marker, _ = outcome_style(row)
        axis.scatter([x], [y], s=52, color=color, marker=marker, alpha=0.9, zorder=3)

    axis.set_title("Task achievability over world geometry")
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_aspect("equal", adjustable="box")
    if bounds is not None:
        min_x, max_x, min_y, max_y = bounds
        axis.set_xlim(min_x, max_x)
        axis.set_ylim(min_y, max_y)
    axis.grid(True, alpha=0.25)
    handles = []
    labels = []
    for color, marker, label in legend_labels(rows):
        handle = plt.Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="None",
            markerfacecolor=color if marker != "x" else "none",
            markeredgecolor=color,
            markeredgewidth=1.6,
            color=color,
            markersize=9,
        )
        handles.append(handle)
        labels.append(label)
    axis.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        borderaxespad=0.0,
        ncol=3,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="0.5",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.27)
    fig.savefig(
        output_dir / "environment_success_map.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def write_environment_summary(
    rows: list[dict[str, str]], output_dir: Path, task_label: str
) -> None:
    total = len(rows)
    success_count = sum(1 for row in rows if row.get("outcome", "") == "success")
    recovery_success_count = sum(
        1
        for row in rows
        if row.get("outcome", "") == "success" and to_bool(row.get("recovery_used", ""))
    )
    failure_count = total - success_count
    world_names = sorted(
        {(row.get("world_name") or "").strip() or "unknown_world" for row in rows}
    )
    lines = [
        f"{task_label} environment-map summary",
        "",
        f"worlds: {', '.join(world_names)}",
        f"total trials: {total}",
        f"successes: {success_count}",
        f"successes after recovery: {recovery_success_count}",
        f"failures: {failure_count}",
        "",
        "Interpretation",
        "- Green points indicate directly achievable task placements.",
        "- Orange points indicate placements that required a recovery strategy.",
        "- Red points indicate placements where the generalized action plan was not successfully realized.",
    ]
    (output_dir / "environment_map_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_background_debug_report(
    output_dir: Path,
    environment_name: str,
    robot_name: str,
    report_rows: list[dict[str, str]],
) -> None:
    kept = [row for row in report_rows if row["status"] == "kept"]
    skipped = [row for row in report_rows if row["status"] != "kept"]
    lines = [
        "2D background debug report",
        "",
        f"environment: {environment_name}",
        f"robot: {robot_name}",
        f"kept bodies: {len(kept)}",
        f"skipped bodies: {len(skipped)}",
        "",
        "Entries",
    ]
    for row in sorted(report_rows, key=lambda item: (item["status"], item["name"])):
        lines.append(f"{row['status']}\t{row['reason']}\t{row['name']}")
    (output_dir / "environment_background_debug.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def render_environment_map_set(
    rows: list[dict[str, str]],
    output_dir: Path,
    environment_name: str,
    robot_name: str,
    task_label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    world_polygons, background_report = inspect_world_background(
        rows, environment_name, robot_name
    )
    plot_environment_success_map(environment_name, rows, output_dir, world_polygons)
    write_environment_summary(rows, output_dir, task_label)
    write_background_debug_report(
        output_dir, environment_name, robot_name, background_report
    )


def render_input_file(
    input_path: Path,
    output_dir: Path,
    environment_name_override: str,
    robot_name_override: str,
    split_by_world_name: bool,
    task_label: str,
) -> None:
    rows = load_rows(input_path)
    if not rows:
        print(f"No rows to plot yet: {input_path}")
        return

    ensure_required_fields(rows)
    grouped_rows = group_rows_by_world_and_robot(rows)
    split_by_world = split_by_world_name or len(grouped_rows) > 1

    if split_by_world:
        written_dirs: list[Path] = []
        for (world_name, robot_name_key), world_rows in grouped_rows.items():
            group_output_dir = output_dir / slugify(world_name) / slugify(robot_name_key)
            environment_name = infer_environment_name(
                world_rows, environment_name_override
            )
            robot_name = infer_robot_name(world_rows, robot_name_override)
            render_environment_map_set(
                world_rows,
                group_output_dir,
                environment_name,
                robot_name,
                task_label,
            )
            written_dirs.append(group_output_dir)
        written = ", ".join(str(path) for path in written_dirs)
        print(f"Wrote per-world/per-robot environment maps to: {written}")
        return

    environment_name = infer_environment_name(rows, environment_name_override)
    robot_name = infer_robot_name(rows, robot_name_override)
    render_environment_map_set(
        rows,
        output_dir,
        environment_name,
        robot_name,
        task_label,
    )
    print(f"Wrote environment maps to {output_dir}")


def render_task(
    task_key: str,
    output_root: Path | None = None,
    environment_name_override: str = "",
    robot_name_override: str = "",
    split_by_world_name: bool = False,
) -> None:
    records_dir = Path(__file__).resolve().parents[1]
    config = TASK_CONFIGS[task_key]
    output_dir = (
        output_root / task_key
        if output_root is not None
        else records_dir / str(config["output_dir_name"])
    )
    render_input_file(
        input_path=records_dir / str(config["input_name"]),
        output_dir=output_dir,
        environment_name_override=(
            environment_name_override or str(config["environment_name"])
        ),
        robot_name_override=robot_name_override,
        split_by_world_name=split_by_world_name,
        task_label=str(config["task_label"]),
    )


def main(
    *,
    default_input_name: str = "cut_all_breads_results.csv",
    default_output_dir_name: str = "environment_maps/cutting",
    default_environment_name: str = "",
    task_label: str = "Cutting",
) -> None:
    args = parse_args(
        default_input_name=default_input_name,
        default_output_dir_name=default_output_dir_name,
        default_environment_name=default_environment_name,
        task_label=task_label,
    )
    render_input_file(
        input_path=args.input,
        output_dir=args.output_dir,
        environment_name_override=args.environment_name,
        robot_name_override=args.robot_name,
        split_by_world_name=args.split_by_world_name,
        task_label=args.task_label,
    )


def main_all_tasks() -> None:
    args = parse_all_tasks_args()
    task_keys = TASK_CONFIGS.keys() if args.task == "all" else [args.task]
    for task_key in task_keys:
        render_task(
            task_key,
            output_root=args.output_dir,
            environment_name_override=args.environment_name,
            robot_name_override=args.robot_name,
            split_by_world_name=args.split_by_world_name,
        )


if __name__ == "__main__":
    main_all_tasks()
