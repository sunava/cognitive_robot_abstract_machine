import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean, pstdev


def _parse_json_if_possible(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if not ((text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))):
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def _round_or_none(value, ndigits=3):
    if value is None:
        return None
    return round(float(value), ndigits)


def _failure_summary(row):
    action_success = _to_bool(row.get("action_success"))
    overall_success = _to_bool(row.get("overall_success"))
    if action_success is False:
        ex_type = row.get("exception_type")
        return f"exception:{ex_type}" if ex_type else "exception"
    if overall_success is True:
        return "ok"

    failed = _parse_json_if_possible(row.get("geometric_failed_checks"))
    if isinstance(failed, list) and failed:
        return "+".join(str(x) for x in failed)

    fallbacks = []
    for key in ("distance_success", "target_intersection_success", "mixing_success"):
        value = _to_bool(row.get(key))
        if value is False:
            fallbacks.append(key)
    if fallbacks:
        return "+".join(fallbacks)
    return "unknown"


def _load_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_trial_summary(rows):
    result = []
    for idx, row in enumerate(rows, start=1):
        summary = {
            "run_id": idx,
            "action": row.get("action"),
            "container": row.get("container"),
            "tool": row.get("tool"),
            "action_success": _to_bool(row.get("action_success")),
            "geometric_success": _to_bool(row.get("geometric_success")),
            "overall_success": _to_bool(row.get("overall_success")),
            "failure_summary": _failure_summary(row),
            "distance_within_threshold_percent": _round_or_none(
                _to_float(row.get("distance_within_threshold_percent")), 2
            ),
            "inside_target_volume_ratio": _round_or_none(
                _to_float(row.get("inside_target_volume_ratio")), 3
            ),
            "mean_distance": _round_or_none(_to_float(row.get("mean_distance")), 4),
            "min_distance": _round_or_none(_to_float(row.get("min_distance")), 4),
            "mixing_success": _to_bool(row.get("mixing_success")),
            "num_points_executed": row.get("num_points_executed"),
            "pointer_stride": row.get("pointer_stride"),
        }
        result.append(summary)
    return result


def _safe_mean(values):
    return mean(values) if values else None


def _safe_std(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return pstdev(values)


def _bool_rate(values):
    xs = [1.0 for v in values if v is True] + [0.0 for v in values if v is False]
    return _safe_mean(xs)


def build_aggregated(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row.get("action"), row.get("tool"), row.get("container"))
        groups[key].append(row)

    aggregated = []
    for (action, tool, container), items in groups.items():
        action_success_vals = [_to_bool(r.get("action_success")) for r in items]
        geometric_success_vals = [_to_bool(r.get("geometric_success")) for r in items]
        overall_success_vals = [_to_bool(r.get("overall_success")) for r in items]

        dist_percent = [_to_float(r.get("distance_within_threshold_percent")) for r in items]
        dist_percent = [v for v in dist_percent if v is not None]

        inside_ratio = [_to_float(r.get("inside_target_volume_ratio")) for r in items]
        inside_ratio = [v for v in inside_ratio if v is not None]

        mean_d = [_to_float(r.get("mean_distance")) for r in items]
        mean_d = [v for v in mean_d if v is not None]

        aggregated.append(
            {
                "action": action,
                "tool": tool,
                "container": container,
                "n_trials": len(items),
                "action_success_rate": _round_or_none(_bool_rate(action_success_vals), 3),
                "geometric_success_rate": _round_or_none(_bool_rate(geometric_success_vals), 3),
                "overall_success_rate": _round_or_none(_bool_rate(overall_success_vals), 3),
                "distance_within_threshold_percent_mean": _round_or_none(_safe_mean(dist_percent), 2),
                "distance_within_threshold_percent_std": _round_or_none(_safe_std(dist_percent), 2),
                "inside_target_volume_ratio_mean": _round_or_none(_safe_mean(inside_ratio), 3),
                "inside_target_volume_ratio_std": _round_or_none(_safe_std(inside_ratio), 3),
                "mean_distance_mean": _round_or_none(_safe_mean(mean_d), 4),
                "mean_distance_std": _round_or_none(_safe_std(mean_d), 4),
            }
        )
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "experiment_metrics.csv"),
    )
    parser.add_argument(
        "--summary-out",
        default=os.path.join(os.path.dirname(__file__), "experiment_metrics_summary.csv"),
    )
    parser.add_argument(
        "--aggregated-out",
        default=os.path.join(os.path.dirname(__file__), "experiment_metrics_aggregated.csv"),
    )
    args = parser.parse_args()

    rows = _load_rows(args.input)
    summary = build_trial_summary(rows)
    aggregated = build_aggregated(rows)

    summary_fields = [
        "run_id",
        "action",
        "container",
        "tool",
        "action_success",
        "geometric_success",
        "overall_success",
        "failure_summary",
        "distance_within_threshold_percent",
        "inside_target_volume_ratio",
        "mean_distance",
        "min_distance",
        "mixing_success",
        "num_points_executed",
        "pointer_stride",
    ]
    aggregated_fields = [
        "action",
        "tool",
        "container",
        "n_trials",
        "action_success_rate",
        "geometric_success_rate",
        "overall_success_rate",
        "distance_within_threshold_percent_mean",
        "distance_within_threshold_percent_std",
        "inside_target_volume_ratio_mean",
        "inside_target_volume_ratio_std",
        "mean_distance_mean",
        "mean_distance_std",
    ]

    _write_csv(args.summary_out, summary_fields, summary)
    _write_csv(args.aggregated_out, aggregated_fields, aggregated)


if __name__ == "__main__":
    main()
