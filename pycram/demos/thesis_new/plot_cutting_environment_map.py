#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PYCRAM_ROOT = Path(__file__).resolve().parents[2]
if str(PYCRAM_ROOT) not in sys.path:
    sys.path.insert(0, str(PYCRAM_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from demos.thesis_new.thesis_math.world_utils import body_local_aabb
from demos.thesis_new.world_setup import setup_thesis_world
from pycram.tf_transformations import quaternion_matrix


def parse_args() -> argparse.Namespace:
    records_dir = Path(__file__).resolve().parent / "records"
    parser = argparse.ArgumentParser(
        description=(
            "Create top-down environment maps for cutting experiment success/failure."
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
        default=records_dir / "environment_maps",
    )
    parser.add_argument(
        "--environment-name",
        default="",
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


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def slugify(text: str) -> str:
    cleaned = "".join(
        ch.lower() if ch.isalnum() else "_" for ch in (text or "").strip()
    )
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed or "unknown_world"


def group_rows_by_world_and_robot(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        world_name = (row.get("world_name") or "").strip() or "unknown_world"
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
            "Rerun the updated cutting experiment first. Missing: " + ", ".join(missing)
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


def body_topdown_polygon(body) -> np.ndarray | None:
    try:
        mins, maxs = body_local_aabb(body, use_visual=False, apply_shape_scale=True)
    except Exception:
        return None
    size_x = float(maxs[0] - mins[0])
    size_y = float(maxs[1] - mins[1])
    if (
        not np.isfinite(size_x)
        or not np.isfinite(size_y)
        or size_x <= 1e-4
        or size_y <= 1e-4
    ):
        return None

    pose = body.global_pose
    pos = np.asarray(pose.to_position().to_np(), dtype=float).reshape(-1)[:3]
    quat = np.asarray(pose.to_quaternion().to_np(), dtype=float).reshape(-1)[:4]
    rotation = quaternion_matrix(quat)[:3, :3]
    corners_local = np.array(
        [
            [mins[0], mins[1], 0.0],
            [maxs[0], mins[1], 0.0],
            [maxs[0], maxs[1], 0.0],
            [mins[0], maxs[1], 0.0],
            [mins[0], mins[1], 0.0],
        ],
        dtype=float,
    )
    corners_world = corners_local @ rotation.T + pos
    return corners_world[:, :2]


def is_world_background_body(name: str, robot_name: str = "") -> bool:
    lowered = (name or "").lower()
    if not lowered:
        return False
    if lowered.startswith(("bread_", "knife_", "whisk_", "bowl_")):
        return False
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
        return False
    if robot_name and robot_name.lower() in lowered:
        return False
    if any(
        token in lowered
        for token in [
            "floor",
            "wall",
            "ceiling",
            "ground",
            "room",
            "apartment_root",
            "apartment",
        ]
    ):
        return False
    if any(
        token in lowered
        for token in [
            "base_footprint",
            "base_link",
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
        return False
    if lowered.endswith(("_link", "_frame")):
        return False
    return True


def load_world_background(
    rows: list[dict[str, str]], environment_name: str, robot_name: str
) -> list[tuple[str, np.ndarray]]:
    world = setup_thesis_world(robot_name=robot_name, environment_name=environment_name)
    bodies = list(getattr(world, "bodies", []))
    polygons: list[tuple[str, np.ndarray]] = []
    for body in bodies:
        name = getattr(getattr(body, "name", None), "name", None) or getattr(
            body, "name", ""
        )
        if not isinstance(name, str):
            continue
        if not is_world_background_body(name, robot_name=robot_name):
            continue
        polygon = body_topdown_polygon(body)
        if polygon is None:
            continue
        extent_x = float(np.max(polygon[:, 0]) - np.min(polygon[:, 0]))
        extent_y = float(np.max(polygon[:, 1]) - np.min(polygon[:, 1]))
        if extent_x > 20.0 or extent_y > 20.0:
            continue
        polygons.append((name, polygon))
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
    rows: list[dict[str, str]],
    world_polygons: list[tuple[str, np.ndarray]],
) -> tuple[float, float, float, float] | None:
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
    rows: list[dict[str, str]],
    output_dir: Path,
    world_polygons: list[tuple[str, np.ndarray]],
) -> None:
    fig, axis = plt.subplots(figsize=(12, 8))
    bounds = combined_plot_bounds(rows, world_polygons)

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
        bbox_to_anchor=(0.5, -0.30),
        borderaxespad=0.0,
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="0.5",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.34)
    fig.savefig(output_dir / "environment_success_map.png", dpi=180)
    plt.close(fig)


def write_environment_summary(rows: list[dict[str, str]], output_dir: Path) -> None:
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
        "Cutting environment-map summary",
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


def render_environment_map_set(
    rows: list[dict[str, str]],
    output_dir: Path,
    environment_name: str,
    robot_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    world_polygons = load_world_background(rows, environment_name, robot_name)
    plot_environment_success_map(rows, output_dir, world_polygons)
    write_environment_summary(rows, output_dir)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    ensure_required_fields(rows)
    grouped_rows = group_rows_by_world_and_robot(rows)
    split_by_world = args.split_by_world_name or len(grouped_rows) > 1

    if split_by_world:
        written_dirs: list[Path] = []
        for (world_name, robot_name_key), world_rows in grouped_rows.items():
            output_dir = args.output_dir / slugify(world_name) / slugify(robot_name_key)
            environment_name = infer_environment_name(world_rows, args.environment_name)
            robot_name = infer_robot_name(world_rows, args.robot_name)
            render_environment_map_set(
                world_rows, output_dir, environment_name, robot_name
            )
            written_dirs.append(output_dir)
        written = ", ".join(str(path) for path in written_dirs)
        print(f"Wrote per-world/per-robot environment maps to: {written}")
        return

    environment_name = infer_environment_name(rows, args.environment_name)
    robot_name = infer_robot_name(rows, args.robot_name)
    render_environment_map_set(rows, args.output_dir, environment_name, robot_name)
    print(f"Wrote environment maps to {args.output_dir}")


if __name__ == "__main__":
    main()
