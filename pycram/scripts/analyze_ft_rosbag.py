#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


@dataclass(frozen=True)
class TopicInfo:
    topic_id: int
    name: str
    type_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze force/torque topics from a ROS 2 sqlite rosbag (.db3) and "
            "export plots, summaries, and CSV files."
        )
    )
    parser.add_argument("bag", type=Path, help="Path to the rosbag .db3 file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ft_analysis_output"),
        help="Directory for plots, CSVs, and summaries",
    )
    parser.add_argument(
        "--topic-prefix",
        default="/ft/",
        help="Only analyze topics beginning with this prefix",
    )
    parser.add_argument(
        "--event-topic",
        default="/events/read_split",
        help="Optional event topic to overlay on plots",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.75,
        help="Half-window around each event for event-aligned summaries",
    )
    parser.add_argument(
        "--max-event-lines",
        type=int,
        default=40,
        help="Maximum number of event markers drawn per plot",
    )
    parser.add_argument(
        "--plot-limit",
        type=int,
        default=12,
        help="Maximum number of numeric fields plotted per topic",
    )
    parser.add_argument(
        "--video-topic",
        default="",
        help="Optional image topic to render as side-by-side video with FT plots",
    )
    parser.add_argument(
        "--video-ft-topic",
        default="/ft/l_gripper_motor",
        help="FT topic to show in the side-by-side video plot",
    )
    parser.add_argument(
        "--video-ft-field",
        default="wrench.force.norm",
        help="FT field to plot in the side-by-side video",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=10.0,
        help="Output fps for the rendered side-by-side video",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=1400,
        help="Output video width in pixels",
    )
    return parser.parse_args()


def sanitize_topic_name(topic_name: str) -> str:
    return topic_name.strip("/").replace("/", "__") or "root_topic"


def open_bag_connection(bag_path: Path) -> sqlite3.Connection:
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    connection = sqlite3.connect(str(bag_path))
    connection.row_factory = sqlite3.Row
    return connection


def load_topics(connection: sqlite3.Connection) -> list[TopicInfo]:
    rows = connection.execute(
        "SELECT id, name, type FROM topics ORDER BY name"
    ).fetchall()
    return [
        TopicInfo(topic_id=int(row["id"]), name=row["name"], type_name=row["type"])
        for row in rows
    ]


def flatten_numeric_message(message: Any, prefix: str = "") -> dict[str, float]:
    values: dict[str, float] = {}

    if isinstance(message, (bool, int, float)):
        values[prefix or "value"] = float(message)
        return values

    if isinstance(message, (list, tuple)):
        for index, item in enumerate(message):
            child_prefix = f"{prefix}[{index}]" if prefix else f"value[{index}]"
            values.update(flatten_numeric_message(item, child_prefix))
        return values

    if hasattr(message, "get_fields_and_field_types"):
        for field_name in message.get_fields_and_field_types():
            if field_name == "header":
                continue
            field_value = getattr(message, field_name)
            child_prefix = f"{prefix}.{field_name}" if prefix else field_name
            values.update(flatten_numeric_message(field_value, child_prefix))
        return values

    return values


def message_to_label(message: Any) -> str:
    if hasattr(message, "data"):
        return str(getattr(message, "data"))
    if hasattr(message, "get_fields_and_field_types"):
        parts = []
        for field_name in message.get_fields_and_field_types():
            parts.append(f"{field_name}={getattr(message, field_name)!r}")
        return ", ".join(parts)
    return repr(message)


def extract_topic_samples(
    connection: sqlite3.Connection, topic: TopicInfo
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    msg_type = get_message(topic.type_name)
    rows = connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic.topic_id,),
    ).fetchall()

    timestamps_ns: list[int] = []
    per_field: dict[str, list[float]] = {}
    for row in rows:
        message = deserialize_message(row["data"], msg_type)
        numeric_values = flatten_numeric_message(message)
        if not numeric_values:
            continue
        timestamps_ns.append(int(row["timestamp"]))
        for key in numeric_values:
            per_field.setdefault(key, [])
        for key in per_field:
            per_field[key].append(float(numeric_values.get(key, np.nan)))

    if not timestamps_ns:
        return np.array([], dtype=np.int64), {}

    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    arrays = {key: np.asarray(values, dtype=float) for key, values in per_field.items()}
    add_wrench_magnitudes(arrays)
    return timestamps, arrays


def add_wrench_magnitudes(per_field: dict[str, np.ndarray]) -> None:
    force_keys = ["wrench.force.x", "wrench.force.y", "wrench.force.z"]
    torque_keys = ["wrench.torque.x", "wrench.torque.y", "wrench.torque.z"]
    if all(key in per_field for key in force_keys):
        per_field["wrench.force.norm"] = np.sqrt(
            sum(np.square(per_field[key]) for key in force_keys)
        )
    if all(key in per_field for key in torque_keys):
        per_field["wrench.torque.norm"] = np.sqrt(
            sum(np.square(per_field[key]) for key in torque_keys)
        )


def extract_event_series(
    connection: sqlite3.Connection, topics: list[TopicInfo], event_topic_name: str
) -> list[tuple[int, str]]:
    event_topic = next(
        (topic for topic in topics if topic.name == event_topic_name), None
    )
    if event_topic is None:
        return []

    msg_type = get_message(event_topic.type_name)
    rows = connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (event_topic.topic_id,),
    ).fetchall()
    events: list[tuple[int, str]] = []
    for row in rows:
        message = deserialize_message(row["data"], msg_type)
        events.append((int(row["timestamp"]), message_to_label(message)))
    return events


def extract_image_samples(
    connection: sqlite3.Connection, topics: list[TopicInfo], image_topic_name: str
) -> list[tuple[int, np.ndarray]]:
    image_topic = next(
        (topic for topic in topics if topic.name == image_topic_name), None
    )
    if image_topic is None:
        return []

    msg_type = get_message(image_topic.type_name)
    rows = connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (image_topic.topic_id,),
    ).fetchall()

    samples: list[tuple[int, np.ndarray]] = []
    for row in rows:
        message = deserialize_message(row["data"], msg_type)
        image = ros_image_to_bgr(message)
        if image is not None:
            samples.append((int(row["timestamp"]), image))
    return samples


def ros_image_to_bgr(message: Any) -> np.ndarray | None:
    height = int(getattr(message, "height", 0))
    width = int(getattr(message, "width", 0))
    encoding = str(getattr(message, "encoding", "")).lower()
    data = bytes(getattr(message, "data", b""))
    if height <= 0 or width <= 0 or not data:
        return None

    if encoding in {"rgb8", "bgr8"}:
        channels = 3
        array = np.frombuffer(data, dtype=np.uint8)
        expected = height * width * channels
        if array.size < expected:
            return None
        image = array[:expected].reshape((height, width, channels))
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()

    if encoding in {"rgba8", "bgra8"}:
        channels = 4
        array = np.frombuffer(data, dtype=np.uint8)
        expected = height * width * channels
        if array.size < expected:
            return None
        image = array[:expected].reshape((height, width, channels))
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if encoding == "mono8":
        array = np.frombuffer(data, dtype=np.uint8)
        expected = height * width
        if array.size < expected:
            return None
        image = array[:expected].reshape((height, width))
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return None


def write_csv(
    path: Path, relative_time_s: np.ndarray, series: dict[str, np.ndarray]
) -> None:
    field_names = list(series)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(["time_s", *field_names]) + "\n")
        for index in range(relative_time_s.shape[0]):
            row = [f"{relative_time_s[index]:.9f}"]
            for field_name in field_names:
                value = series[field_name][index]
                row.append("" if math.isnan(value) else f"{value:.12g}")
            handle.write(",".join(row) + "\n")


def compute_topic_summary(
    topic_name: str,
    relative_time_s: np.ndarray,
    series: dict[str, np.ndarray],
) -> dict[str, Any]:
    field_summary: dict[str, Any] = {}
    duration_s = float(relative_time_s[-1]) if relative_time_s.size else 0.0
    sample_count = int(relative_time_s.size)
    sample_rate_hz = sample_count / duration_s if duration_s > 0.0 else None

    for field_name, values in series.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        field_summary[field_name] = {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "abs_max": float(np.max(np.abs(finite))),
            "rms": float(np.sqrt(np.mean(np.square(finite)))),
        }

    return {
        "topic": topic_name,
        "sample_count": sample_count,
        "duration_s": duration_s,
        "approx_sample_rate_hz": sample_rate_hz,
        "fields": field_summary,
    }


def compute_event_windows(
    relative_time_s: np.ndarray,
    series: dict[str, np.ndarray],
    events: list[tuple[float, str]],
    window_seconds: float,
) -> list[dict[str, Any]]:
    if not events:
        return []

    window_rows: list[dict[str, Any]] = []
    for event_time_s, event_label in events:
        start = event_time_s - window_seconds
        stop = event_time_s + window_seconds
        mask = (relative_time_s >= start) & (relative_time_s <= stop)
        if not np.any(mask):
            continue

        row: dict[str, Any] = {
            "event_time_s": float(event_time_s),
            "event_label": event_label,
            "window_start_s": float(start),
            "window_stop_s": float(stop),
        }
        for field_name, values in series.items():
            segment = values[mask]
            finite = segment[np.isfinite(segment)]
            if finite.size == 0:
                continue
            row[f"{field_name}__mean"] = float(np.mean(finite))
            row[f"{field_name}__abs_max"] = float(np.max(np.abs(finite)))
            row[f"{field_name}__rms"] = float(np.sqrt(np.mean(np.square(finite))))
        window_rows.append(row)

    return window_rows


def write_event_windows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            values = [
                (
                    json.dumps(row.get(column, ""))
                    if isinstance(row.get(column, ""), str)
                    else str(row.get(column, ""))
                )
                for column in columns
            ]
            handle.write(",".join(values) + "\n")


def write_events_csv(path: Path, rows: list[tuple[float, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8") as handle:
        handle.write("event_time_s,event_label\n")
        for event_time_s, event_label in rows:
            handle.write(f"{event_time_s:.9f},{json.dumps(event_label)}\n")


def plot_event_labels(
    axis: Any, event_rows: list[tuple[float, str]], max_event_lines: int
) -> None:
    ymin, ymax = axis.get_ylim()
    y_text = ymax - 0.06 * (ymax - ymin if ymax != ymin else 1.0)
    for event_time_s, event_label in event_rows[:max_event_lines]:
        axis.axvline(event_time_s, color="tab:red", alpha=0.18, linewidth=0.8)
        axis.text(
            event_time_s,
            y_text,
            event_label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color="tab:red",
            alpha=0.8,
        )


def plot_topic(
    path: Path,
    topic_name: str,
    relative_time_s: np.ndarray,
    series: dict[str, np.ndarray],
    event_rows: list[tuple[float, str]],
    plot_limit: int,
    max_event_lines: int,
) -> None:
    field_names = list(series)[:plot_limit]
    if not field_names:
        return

    figure, axes = plt.subplots(
        len(field_names),
        1,
        figsize=(14, max(3, 2.3 * len(field_names))),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for axis, field_name in zip(axes, field_names):
        axis.plot(relative_time_s, series[field_name], linewidth=1.0)
        axis.set_ylabel(field_name, fontsize=8)
        axis.grid(True, alpha=0.25)
        plot_event_labels(axis, event_rows, max_event_lines)

    axes[-1].set_xlabel("time [s]")
    figure.suptitle(topic_name)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_overview(
    path: Path,
    topic_summaries: list[dict[str, Any]],
    topic_samples: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    common_fields: list[str],
) -> None:
    if not common_fields:
        return

    field_names = common_fields[:6]
    figure, axes = plt.subplots(
        len(field_names),
        1,
        figsize=(14, max(3, 2.4 * len(field_names))),
        sharex=False,
    )
    axes = np.atleast_1d(axes)

    for axis, field_name in zip(axes, field_names):
        for summary in topic_summaries:
            topic_name = summary["topic"]
            relative_time_s, series = topic_samples[topic_name]
            axis.plot(
                relative_time_s, series[field_name], linewidth=1.0, label=topic_name
            )
        axis.set_ylabel(field_name, fontsize=8)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)

    axes[-1].set_xlabel("time [s]")
    figure.suptitle("FT topic comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def topic_variant_key(topic_name: str) -> str:
    suffixes = sorted(
        [
            "_clean_derivative_avg",
            "_clean_derivative",
            "_clean_avg",
            "_clean",
            "_zeroed_derivative_avg",
            "_zeroed_derivative",
            "_zeroed_avg",
            "_zeroed",
            "_derivative_avg",
            "_derivative",
            "_avg",
        ],
        key=len,
        reverse=True,
    )
    base = topic_name.split("/")[-1]
    for suffix in suffixes:
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def topic_variant_label(topic_name: str) -> str:
    base = topic_name.split("/")[-1]
    root = topic_variant_key(topic_name)
    label = base[len(root) :].lstrip("_")
    return label or "raw"


def plot_variant_comparison(
    path: Path,
    topic_samples: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]],
    event_rows: list[tuple[float, str]],
    max_event_lines: int,
) -> None:
    groups: dict[str, list[str]] = {}
    for topic_name in topic_samples:
        groups.setdefault(topic_variant_key(topic_name), []).append(topic_name)

    groups = {key: value for key, value in groups.items() if len(value) >= 2}
    if not groups:
        return

    preferred_fields = [
        "wrench.force.x",
        "wrench.force.y",
        "wrench.force.z",
        "wrench.force.norm",
        "wrench.torque.x",
        "wrench.torque.norm",
    ]
    figure, axes = plt.subplots(
        len(groups),
        1,
        figsize=(15, max(4, 3.2 * len(groups))),
        sharex=False,
    )
    axes = np.atleast_1d(axes)

    for axis, (group_name, topic_names) in zip(axes, sorted(groups.items())):
        plotted = False
        for topic_name in sorted(topic_names):
            relative_time_s, series = topic_samples[topic_name]
            field_name = next(
                (field for field in preferred_fields if field in series), None
            )
            if field_name is None:
                continue
            axis.plot(
                relative_time_s,
                series[field_name],
                linewidth=1.1,
                label=f"{topic_variant_label(topic_name)}: {field_name}",
            )
            plotted = True
        if plotted:
            axis.legend(fontsize=7)
        axis.set_title(group_name)
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("time [s]")
        plot_event_labels(axis, event_rows, max_event_lines)

    figure.suptitle("FT variant comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def compute_event_metrics(
    topic_name: str,
    relative_time_s: np.ndarray,
    series: dict[str, np.ndarray],
    events: list[tuple[float, str]],
    window_seconds: float,
) -> list[dict[str, Any]]:
    preferred_fields = [
        "wrench.force.norm",
        "wrench.force.x",
        "wrench.torque.norm",
        "wrench.torque.x",
    ]
    metric_fields = [field for field in preferred_fields if field in series]
    if not metric_fields:
        metric_fields = list(series)[:2]

    rows: list[dict[str, Any]] = []
    for event_time_s, event_label in events:
        mask = (relative_time_s >= event_time_s - window_seconds) & (
            relative_time_s <= event_time_s + window_seconds
        )
        if not np.any(mask):
            continue
        row: dict[str, Any] = {
            "topic": topic_name,
            "variant": topic_variant_label(topic_name),
            "event_time_s": float(event_time_s),
            "event_label": event_label,
        }
        time_segment = relative_time_s[mask]
        for field_name in metric_fields:
            values = series[field_name][mask]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            peak_index = int(np.nanargmax(np.abs(values)))
            row[f"{field_name}__mean"] = float(np.nanmean(values))
            row[f"{field_name}__rms"] = float(np.sqrt(np.nanmean(np.square(values))))
            row[f"{field_name}__abs_peak"] = float(np.nanmax(np.abs(values)))
            row[f"{field_name}__time_to_peak_s"] = float(
                time_segment[peak_index] - event_time_s
            )
        rows.append(row)
    return rows


def write_event_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            values = [
                (
                    json.dumps(row.get(column, ""))
                    if isinstance(row.get(column, ""), str)
                    else str(row.get(column, ""))
                )
                for column in columns
            ]
            handle.write(",".join(values) + "\n")


def write_interpretation_report(
    path: Path,
    topic_summaries: list[dict[str, Any]],
    event_metric_rows: list[dict[str, Any]],
) -> None:
    lines: list[str] = ["FT analysis interpretation", ""]
    if not topic_summaries:
        lines.append("No decodable FT topics were found.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    candidates: list[tuple[float, str, str]] = []
    for summary in topic_summaries:
        for field_name, stats in summary["fields"].items():
            if "force.norm" in field_name or field_name.endswith("force.x"):
                candidates.append(
                    (float(stats["abs_max"]), summary["topic"], field_name)
                )
    if candidates:
        best_abs_max, best_topic, best_field = max(candidates, key=lambda row: row[0])
        lines.append(
            f"Strongest overall contact signature: {best_topic} via {best_field} "
            f"(abs peak {best_abs_max:.4g})."
        )

    if event_metric_rows:
        grouped: dict[str, list[float]] = {}
        for row in event_metric_rows:
            for key, value in row.items():
                if key.endswith("__abs_peak") and isinstance(value, (int, float)):
                    grouped.setdefault(row["topic"], []).append(float(value))
        if grouped:
            topic_name, values = max(grouped.items(), key=lambda item: np.mean(item[1]))
            lines.append(
                f"Largest event-aligned peaks: {topic_name} "
                f"(mean abs peak {float(np.mean(values)):.4g} across events)."
            )

    lines.append("")
    lines.append("Interpretation hints")
    lines.append(
        "- Rising force or torque peaks near an event suggest contact onset or split transition."
    )
    lines.append(
        "- Sustained force levels are more useful for cutting engagement than single-sample spikes."
    )
    lines.append(
        "- Derivative variants are usually better for timing transitions; clean/zeroed variants are better for stable phase magnitude."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def nearest_index(sorted_times_s: np.ndarray, target_time_s: float) -> int:
    if sorted_times_s.size == 0:
        return 0
    index = int(np.searchsorted(sorted_times_s, target_time_s, side="left"))
    if index <= 0:
        return 0
    if index >= sorted_times_s.size:
        return int(sorted_times_s.size - 1)
    before = sorted_times_s[index - 1]
    after = sorted_times_s[index]
    return int(
        index - 1
        if abs(target_time_s - before) <= abs(after - target_time_s)
        else index
    )


def render_ft_video_frame(
    image_bgr: np.ndarray,
    current_time_s: float,
    ft_time_s: np.ndarray,
    ft_values: np.ndarray,
    event_rows: list[tuple[float, str]],
    ft_topic_name: str,
    ft_field_name: str,
    video_width: int,
) -> np.ndarray:
    figure = plt.figure(figsize=(video_width / 100.0, 6.0), dpi=100)
    grid = figure.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    axis_image = figure.add_subplot(grid[0, 0])
    axis_plot = figure.add_subplot(grid[0, 1])

    axis_image.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axis_image.set_title(f"RGB @ {current_time_s:.2f}s")
    axis_image.axis("off")

    past_mask = ft_time_s <= current_time_s
    future_mask = ft_time_s > current_time_s
    if np.any(future_mask):
        axis_plot.plot(
            ft_time_s[future_mask],
            ft_values[future_mask],
            color="0.75",
            linewidth=1.0,
            alpha=0.6,
        )
    if np.any(past_mask):
        axis_plot.plot(
            ft_time_s[past_mask],
            ft_values[past_mask],
            color="tab:blue",
            linewidth=1.8,
        )
    axis_plot.axvline(current_time_s, color="black", linewidth=1.4, alpha=0.8)
    ymin = float(np.nanmin(ft_values)) if ft_values.size else -1.0
    ymax = float(np.nanmax(ft_values)) if ft_values.size else 1.0
    ypad = max(1e-6, 0.08 * (ymax - ymin if ymax != ymin else 1.0))
    axis_plot.set_ylim(ymin - ypad, ymax + ypad)
    axis_plot.set_title(f"{ft_topic_name} | {ft_field_name}")
    axis_plot.set_xlabel("time [s]")
    axis_plot.grid(True, alpha=0.25)
    for event_time_s, event_label in event_rows[:40]:
        axis_plot.axvline(event_time_s, color="tab:red", alpha=0.15, linewidth=0.8)
        if abs(event_time_s - current_time_s) < 0.35:
            axis_plot.text(
                event_time_s,
                ymax,
                event_label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color="tab:red",
            )

    if ft_time_s.size:
        idx = nearest_index(ft_time_s, current_time_s)
        value = float(ft_values[idx])
        axis_plot.scatter([ft_time_s[idx]], [value], color="black", s=20, zorder=3)
        axis_plot.text(
            0.02,
            0.98,
            f"t={current_time_s:.2f}s\nvalue={value:.4g}",
            transform=axis_plot.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    figure.tight_layout()
    figure.canvas.draw()
    frame = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(figure.canvas.get_width_height()[::-1] + (4,))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    plt.close(figure)
    return frame_bgr


def export_side_by_side_video(
    path: Path,
    image_samples: list[tuple[float, np.ndarray]],
    ft_time_s: np.ndarray,
    ft_values: np.ndarray,
    event_rows: list[tuple[float, str]],
    ft_topic_name: str,
    ft_field_name: str,
    video_fps: float,
    video_width: int,
) -> None:
    if not image_samples or ft_time_s.size == 0 or ft_values.size == 0:
        return

    first_frame = render_ft_video_frame(
        image_samples[0][1],
        image_samples[0][0],
        ft_time_s,
        ft_values,
        event_rows,
        ft_topic_name,
        ft_field_name,
        video_width,
    )
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(video_fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        writer.write(first_frame)
        for current_time_s, image_bgr in image_samples[1:]:
            frame = render_ft_video_frame(
                image_bgr,
                current_time_s,
                ft_time_s,
                ft_values,
                event_rows,
                ft_topic_name,
                ft_field_name,
                video_width,
            )
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = open_bag_connection(args.bag)
    try:
        topics = load_topics(connection)
        ft_topics = [
            topic for topic in topics if topic.name.startswith(args.topic_prefix)
        ]
        if not ft_topics:
            raise RuntimeError(
                f"No topics with prefix {args.topic_prefix!r} found in {args.bag}"
            )

        raw_events = extract_event_series(connection, topics, args.event_topic)
        raw_images = (
            extract_image_samples(connection, topics, args.video_topic)
            if args.video_topic
            else []
        )
        bag_start_ns: int | None = None
        topic_samples: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        topic_summaries: list[dict[str, Any]] = []

        for topic in ft_topics:
            timestamps_ns, series = extract_topic_samples(connection, topic)
            if timestamps_ns.size == 0 or not series:
                continue
            if bag_start_ns is None:
                bag_start_ns = int(timestamps_ns[0])
            else:
                bag_start_ns = min(bag_start_ns, int(timestamps_ns[0]))
            topic_samples[topic.name] = (timestamps_ns, series)

        if not topic_samples:
            raise RuntimeError(
                "Found FT topics, but could not decode numeric payloads."
            )

        bag_start_ns = int(
            min(timestamps_ns[0] for timestamps_ns, _ in topic_samples.values())
        )
        event_series_s = [
            ((timestamp_ns - bag_start_ns) / 1e9, label)
            for timestamp_ns, label in raw_events
        ]
        image_series_s = [
            ((timestamp_ns - bag_start_ns) / 1e9, image_bgr)
            for timestamp_ns, image_bgr in raw_images
        ]
        write_events_csv(
            output_dir / f"{sanitize_topic_name(args.event_topic)}.csv",
            event_series_s,
        )
        event_metric_rows: list[dict[str, Any]] = []

        for topic_name, (timestamps_ns, series) in topic_samples.items():
            relative_time_s = (timestamps_ns - bag_start_ns) / 1e9
            topic_samples[topic_name] = (relative_time_s, series)

            topic_output_stem = sanitize_topic_name(topic_name)
            write_csv(output_dir / f"{topic_output_stem}.csv", relative_time_s, series)
            plot_topic(
                output_dir / f"{topic_output_stem}.png",
                topic_name,
                relative_time_s,
                series,
                event_series_s,
                args.plot_limit,
                args.max_event_lines,
            )

            summary = compute_topic_summary(topic_name, relative_time_s, series)
            event_windows = compute_event_windows(
                relative_time_s, series, event_series_s, args.window_seconds
            )
            event_metric_rows.extend(
                compute_event_metrics(
                    topic_name,
                    relative_time_s,
                    series,
                    event_series_s,
                    args.window_seconds,
                )
            )
            if event_windows:
                write_event_windows_csv(
                    output_dir / f"{topic_output_stem}__event_windows.csv",
                    event_windows,
                )
            summary["event_windows"] = event_windows
            topic_summaries.append(summary)

        shared_fields = (
            sorted(
                set.intersection(
                    *[set(series.keys()) for _, series in topic_samples.values()]
                )
            )
            if len(topic_samples) >= 2
            else []
        )
        plot_overview(
            output_dir / "ft_topic_comparison.png",
            topic_summaries,
            topic_samples,
            shared_fields,
        )
        plot_variant_comparison(
            output_dir / "ft_variant_comparison.png",
            topic_samples,
            event_series_s,
            args.max_event_lines,
        )
        write_event_metrics_csv(
            output_dir / "ft_event_metrics.csv",
            event_metric_rows,
        )
        write_interpretation_report(
            output_dir / "interpretation.txt",
            topic_summaries,
            event_metric_rows,
        )
        if args.video_topic:
            video_topic_data = topic_samples.get(args.video_ft_topic)
            if video_topic_data is None:
                raise RuntimeError(
                    f"Requested video FT topic {args.video_ft_topic!r} was not analyzed."
                )
            ft_time_s, ft_series = video_topic_data
            if args.video_ft_field not in ft_series:
                raise RuntimeError(
                    f"Requested video FT field {args.video_ft_field!r} not found in {args.video_ft_topic}."
                )
            export_side_by_side_video(
                output_dir / "ft_video_overlay.mp4",
                image_series_s,
                ft_time_s,
                ft_series[args.video_ft_field],
                event_series_s,
                args.video_ft_topic,
                args.video_ft_field,
                args.video_fps,
                args.video_width,
            )

        manifest = {
            "bag": str(args.bag),
            "topic_prefix": args.topic_prefix,
            "event_topic": args.event_topic,
            "video_topic": args.video_topic,
            "topics_analyzed": [summary["topic"] for summary in topic_summaries],
            "summary_count": len(topic_summaries),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(topic_summaries, indent=2), encoding="utf-8"
        )
        print(f"Wrote FT analysis to {output_dir}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
