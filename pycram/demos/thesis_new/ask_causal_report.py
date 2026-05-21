"""
Ask data-backed causal questions over the robot-substitution report.

This script does not infer new facts from prior knowledge. It reads the CSV
artifacts produced by robot_substitution_causal_diagnosis.py and answers common
questions from those numbers.

Examples:
    python ask_causal_report.py "Was ist P(success | do(hsrb))?"
    python ask_causal_report.py "Warum ist do(hsrb) schlechter?"
    python ask_causal_report.py "Ist hsrb schlechter als pr2?"
    python ask_causal_report.py "morphology hsrb"
    python ask_causal_report.py "drive"
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path("/home/hassouna/cognitive_robot_abstract_machine")
CAUSAL_DIR = REPO_ROOT / "pycram/demos/thesis_new/records/causal_intervention"

SUMMARY_CSV = CAUSAL_DIR / "robot_substitution_causal_summary.csv"
PAIRWISE_CSV = CAUSAL_DIR / "robot_substitution_pairwise_effects.csv"
MECHANISM_CSV = CAUSAL_DIR / "robot_substitution_mechanism_summary.csv"
FEATURES_CSV = CAUSAL_DIR / "robot_morphology_features.csv"
ENRICHED_RESULTS_CSV = (
    CAUSAL_DIR / "raw_cutting_intervention_results_with_robot_features.csv"
)
RAW_RESULTS_CSV = CAUSAL_DIR / "raw_cutting_intervention_results.csv"

SUCCESS = "final_success"
ROBOT = "robot_name"
TASK = "task_name"
TASK_VALUE = "bread_cutting"
UNIT_COLUMNS = ["world_name", "seed", "bread_name"]

GEOMETRY_FEATURES = [
    "object_volume_aabb",
    "object_size_x",
    "object_size_y",
    "object_size_z",
    "object_world_x",
    "object_world_y",
    "object_world_z",
    "target_world_x",
    "target_world_y",
    "target_world_z",
    "object_yaw_rad",
    "object_yaw_relative_to_robot_rad",
    "object_yaw_relative_to_approach_rad",
    "cut_normal_world_yaw_rad",
    "cut_normal_relative_to_robot_rad",
    "cut_normal_relative_to_approach_rad",
    "cut_normal_approach_abs_angle_rad",
    "cut_normal_approach_parallel_score",
    "cut_normal_approach_perpendicular_score",
    "object_z_minus_robot_tool_frame_max_z_m",
    "object_top_z_minus_robot_tool_frame_max_z_m",
]

GEOMETRY_ALIASES = {
    "volume": "object_volume_aabb",
    "object_volume": "object_volume_aabb",
    "object_z": "object_world_z",
    "object_height": "object_world_z",
    "top_z": "object_top_z_minus_robot_tool_frame_max_z_m",
    "object_top_z": "object_top_z_minus_robot_tool_frame_max_z_m",
    "relative_top_z": "object_top_z_minus_robot_tool_frame_max_z_m",
    "cut_angle": "cut_normal_approach_abs_angle_rad",
    "angle": "cut_normal_approach_abs_angle_rad",
    "cut_normal_angle": "cut_normal_approach_abs_angle_rad",
    "parallel": "cut_normal_approach_parallel_score",
    "perpendicular": "cut_normal_approach_perpendicular_score",
}

ROBOT_ALIASES = {
    "hsr": "hsrb",
    "hsr-b": "hsrb",
    "hsr_b": "hsrb",
    "justin": "rollin_justin",
    "g1": "unitree_g1",
}


def normalize_robot(name: object) -> str:
    value = str(name).strip().lower()
    return ROBOT_ALIASES.get(value, value)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


@dataclass
class ReportData:
    summary: pd.DataFrame
    pairwise: pd.DataFrame
    mechanisms: pd.DataFrame
    features: pd.DataFrame
    results: pd.DataFrame

    @property
    def robots(self) -> list[str]:
        return sorted(self.summary[ROBOT].dropna().map(normalize_robot).unique())


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run robot_substitution_causal_diagnosis.py first."
        )


def load_report_data() -> ReportData:
    for path in [SUMMARY_CSV, PAIRWISE_CSV, MECHANISM_CSV]:
        require_file(path)

    summary = pd.read_csv(SUMMARY_CSV)
    pairwise = pd.read_csv(PAIRWISE_CSV)
    mechanisms = pd.read_csv(MECHANISM_CSV)
    features = pd.read_csv(FEATURES_CSV) if FEATURES_CSV.exists() else pd.DataFrame()

    results_path = (
        ENRICHED_RESULTS_CSV if ENRICHED_RESULTS_CSV.exists() else RAW_RESULTS_CSV
    )
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()

    for frame in [summary, mechanisms, features, results]:
        if not frame.empty and ROBOT in frame.columns:
            frame[ROBOT] = frame[ROBOT].map(normalize_robot)
    for column in ["robot_a", "robot_b"]:
        if column in pairwise.columns:
            pairwise[column] = pairwise[column].map(normalize_robot)

    if not results.empty:
        if TASK in results.columns:
            results = results[results[TASK] == TASK_VALUE].copy()
        results[SUCCESS] = as_bool(results[SUCCESS])
        results[ROBOT] = results[ROBOT].map(normalize_robot)
        results = results[results[ROBOT].isin(summary[ROBOT])].copy()

    return ReportData(summary, pairwise, mechanisms, features, results)


def find_robots(question: str, known_robots: list[str]) -> list[str]:
    normalized = question.lower()
    matches = []
    for robot in known_robots:
        match = re.search(rf"(?<![a-z0-9_]){re.escape(robot)}(?![a-z0-9_])", normalized)
        if match:
            matches.append((match.start(), robot))
    for alias, robot in ROBOT_ALIASES.items():
        match = re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", normalized)
        if match and robot in known_robots:
            matches.append((match.start(), robot))

    found = []
    for _, robot in sorted(matches, key=lambda item: item[0]):
        if robot not in found:
            found.append(robot)
    return found


def find_geometry_feature(question: str, data: ReportData) -> str | None:
    normalized = question.lower()
    available = [
        feature for feature in GEOMETRY_FEATURES if feature in data.results.columns
    ]
    candidates = []
    for alias, feature in GEOMETRY_ALIASES.items():
        if feature in available:
            match = re.search(
                rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", normalized
            )
            if match:
                candidates.append((match.start(), feature))
    for feature in available:
        match = re.search(
            rf"(?<![a-z0-9_]){re.escape(feature)}(?![a-z0-9_])", normalized
        )
        if match:
            candidates.append((match.start(), feature))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def find_bin_label(question: str) -> str | None:
    normalized = question.lower()
    for label in ["low", "mid", "middle", "medium", "high"]:
        if re.search(rf"(?<![a-z0-9_]){label}(?![a-z0-9_])", normalized):
            return "mid" if label in {"middle", "medium"} else label
    return None


def find_bin_labels(question: str) -> list[str]:
    normalized = question.lower()
    matches = []
    for label in ["low", "mid", "middle", "medium", "high"]:
        for match in re.finditer(rf"(?<![a-z0-9_]){label}(?![a-z0-9_])", normalized):
            normalized_label = "mid" if label in {"middle", "medium"} else label
            matches.append((match.start(), normalized_label))
    labels = []
    for _, label in sorted(matches, key=lambda item: item[0]):
        if label not in labels:
            labels.append(label)
    return labels


def make_geometry_bins(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        bins = pd.qcut(numeric, q=3, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(index=series.index, dtype=object)
    non_null_bins = sorted(pd.Series(bins).dropna().unique())
    if len(non_null_bins) <= 1 and numeric.nunique(dropna=True) >= 2:
        try:
            bins = pd.cut(
                numeric,
                bins=min(3, numeric.nunique(dropna=True)),
                labels=False,
                include_lowest=True,
            )
            non_null_bins = sorted(pd.Series(bins).dropna().unique())
        except ValueError:
            pass
    if len(non_null_bins) == 0:
        return pd.Series(index=series.index, dtype=object)
    if len(non_null_bins) == 1:
        mapping = {non_null_bins[0]: "all"}
    elif len(non_null_bins) == 2:
        mapping = {non_null_bins[0]: "low", non_null_bins[1]: "high"}
    else:
        mapping = {
            non_null_bins[0]: "low",
            non_null_bins[1]: "mid",
            non_null_bins[2]: "high",
        }
    return pd.Series(bins, index=series.index).map(mapping).astype(object)


def best_reference_robot(data: ReportData, excluded: str | None = None) -> str:
    rows = data.summary.copy()
    if excluded is not None:
        rows = rows[rows[ROBOT] != excluded]
    return str(rows.sort_values("success_probability", ascending=False).iloc[0][ROBOT])


def get_do(data: ReportData, robot: str) -> str:
    robot = normalize_robot(robot)
    rows = data.summary[data.summary[ROBOT] == robot]
    if rows.empty:
        return f"Unknown robot '{robot}'. Known robots: {', '.join(data.robots)}"
    row = rows.iloc[0]
    return (
        f"P(success | do(robot={robot})) = {row.success_probability:.4f} "
        f"({pct(row.success_probability)})\n"
        f"Successes: {int(row.successes)}/{int(row.runs)}\n"
        f"Source: {SUMMARY_CSV}"
    )


def get_effect(data: ReportData, robot_a: str, robot_b: str) -> str:
    robot_a = normalize_robot(robot_a)
    robot_b = normalize_robot(robot_b)
    rows = data.pairwise[
        (data.pairwise["robot_a"] == robot_a) & (data.pairwise["robot_b"] == robot_b)
    ]
    if rows.empty:
        reverse = data.pairwise[
            (data.pairwise["robot_a"] == robot_b)
            & (data.pairwise["robot_b"] == robot_a)
        ]
        if reverse.empty:
            return f"No paired effect found for {robot_a} and {robot_b}."
        row = reverse.iloc[0].copy()
        effect = -float(row.effect_a_minus_b)
        wins = int(row.losses_a)
        losses = int(row.wins_a)
        p_a = float(row.p_success_do_b)
        p_b = float(row.p_success_do_a)
        paired = int(row.paired_units)
    else:
        row = rows.iloc[0]
        effect = float(row.effect_a_minus_b)
        wins = int(row.wins_a)
        losses = int(row.losses_a)
        p_a = float(row.p_success_do_a)
        p_b = float(row.p_success_do_b)
        paired = int(row.paired_units)

    direction = "better" if effect > 0 else "worse" if effect < 0 else "same"
    return (
        f"Paired effect {robot_a} - {robot_b}: {effect:+.4f} "
        f"({pct(abs(effect))} absolute success-rate difference, {robot_a} is {direction}).\n"
        f"P(success | do({robot_a})) on paired units = {p_a:.4f}\n"
        f"P(success | do({robot_b})) on paired units = {p_b:.4f}\n"
        f"Wins/losses/ties for {robot_a}: {wins}/{losses}/{int(row.ties)}\n"
        f"Paired task units: {paired}\n"
        f"Source: {PAIRWISE_CSV}"
    )


def get_mechanism(data: ReportData, robot: str) -> str:
    robot = normalize_robot(robot)
    rows = data.mechanisms[data.mechanisms[ROBOT] == robot]
    if rows.empty:
        return f"No mechanism summary found for {robot}."
    row = rows.iloc[0]
    return (
        f"{robot} failures are mostly explained by "
        f"{row.failure_feasibility_reason_mode}: "
        f"{int(row.failure_feasibility_reason_mode_count)}/{int(row.failure_count)} "
        f"failures ({pct(row.failure_feasibility_reason_mode_share)}).\n"
        f"Robot decision mode on failures: {row.failure_robot_decision_mode} "
        f"({pct(row.failure_robot_decision_mode_share)}).\n"
        f"Decision reason mode on failures: {row.failure_decision_reason_mode} "
        f"({pct(row.failure_decision_reason_mode_share)}).\n"
        f"Mean collision failure count on failures: "
        f"{row.failure_collision_failure_count_mean:.2f}\n"
        f"Source: {MECHANISM_CSV}"
    )


def paired_losing_mechanism(data: ReportData, robot: str, reference: str) -> str:
    if data.results.empty:
        return "No raw/enriched result rows available for paired losing-unit diagnosis."
    robot = normalize_robot(robot)
    reference = normalize_robot(reference)
    needed = set(UNIT_COLUMNS + [ROBOT, SUCCESS, "feasibility_reason"])
    if not needed.issubset(data.results.columns):
        return "Result CSV does not contain enough columns for paired losing-unit diagnosis."

    pivot = data.results.pivot_table(
        index=UNIT_COLUMNS,
        columns=ROBOT,
        values=SUCCESS,
        aggfunc="mean",
    )
    if robot not in pivot.columns or reference not in pivot.columns:
        return f"No paired raw rows found for {robot} and {reference}."
    losing_units = pivot[(pivot[robot] == 0) & (pivot[reference] == 1)].index
    if len(losing_units) == 0:
        return f"{robot} has no losing units against {reference}."

    indexed = data.results.set_index(UNIT_COLUMNS)
    losing_robot_rows = indexed.loc[losing_units].reset_index()
    losing_robot_rows = losing_robot_rows[losing_robot_rows[ROBOT] == robot]
    counts = losing_robot_rows["feasibility_reason"].value_counts(dropna=False)
    lines = [
        f"On identical task units where {reference} succeeds and {robot} fails:",
        f"{robot} loses {len(losing_units)} paired units against {reference}.",
    ]
    for reason, count in counts.items():
        lines.append(
            f"- {reason}: {int(count)}/{len(losing_units)} ({pct(count / len(losing_units))})"
        )
    lines.append(
        f"Source: {ENRICHED_RESULTS_CSV if ENRICHED_RESULTS_CSV.exists() else RAW_RESULTS_CSV}"
    )
    return "\n".join(lines)


def get_why(data: ReportData, robot: str, reference: str | None = None) -> str:
    robot = normalize_robot(robot)
    if reference is None:
        reference = best_reference_robot(data, excluded=robot)
    else:
        reference = normalize_robot(reference)

    return "\n\n".join(
        [
            explain_why(data, robot, reference),
            get_do(data, robot),
            get_do(data, reference),
            get_effect(data, robot, reference),
            get_mechanism(data, robot),
            paired_losing_mechanism(data, robot, reference),
            compare_morphology(data, robot, reference),
            (
                "Data-backed causal reading:\n"
                f"do(robot={robot}) changes the success rate relative to "
                f"do(robot={reference}) on matched task units. The dominant mediator "
                "in the failed runs is the feasibility/collision-motion failure mode."
            ),
        ]
    )


def explain_why(data: ReportData, robot: str, reference: str) -> str:
    robot = normalize_robot(robot)
    reference = normalize_robot(reference)
    robot_summary = data.summary[data.summary[ROBOT] == robot]
    reference_summary = data.summary[data.summary[ROBOT] == reference]
    robot_mechanism = data.mechanisms[data.mechanisms[ROBOT] == robot]
    if robot_summary.empty or reference_summary.empty or robot_mechanism.empty:
        return f"Most likely explanation: not enough report data for {robot} vs {reference}."

    robot_summary = robot_summary.iloc[0]
    reference_summary = reference_summary.iloc[0]
    robot_mechanism = robot_mechanism.iloc[0]
    effect = float(robot_summary.success_probability) - float(
        reference_summary.success_probability
    )

    support = []
    if not data.results.empty and set(
        UNIT_COLUMNS + [ROBOT, SUCCESS, "feasibility_reason"]
    ).issubset(data.results.columns):
        pivot = data.results.pivot_table(
            index=UNIT_COLUMNS,
            columns=ROBOT,
            values=SUCCESS,
            aggfunc="mean",
        )
        if robot in pivot.columns and reference in pivot.columns:
            losing_units = pivot[(pivot[robot] == 0) & (pivot[reference] == 1)].index
            if len(losing_units):
                indexed = data.results.set_index(UNIT_COLUMNS)
                losing_rows = indexed.loc[losing_units].reset_index()
                losing_rows = losing_rows[losing_rows[ROBOT] == robot]
                reason_counts = losing_rows["feasibility_reason"].value_counts(
                    dropna=False
                )
                reason = reason_counts.index[0]
                count = int(reason_counts.iloc[0])
                support.append(
                    f"on the paired units where {reference} succeeds and {robot} fails, "
                    f"{count}/{len(losing_units)} failures are `{reason}`"
                )

    feature_reasons = strongest_feature_reasons(data, robot, reference)
    feature_sentence = ""
    if feature_reasons:
        feature_sentence = (
            " The strongest feature-level hints are " + ", ".join(feature_reasons) + "."
        )

    mechanism_sentence = (
        f"{int(robot_mechanism.failure_feasibility_reason_mode_count)}/"
        f"{int(robot_mechanism.failure_count)} {robot} failures are "
        f"`{robot_mechanism.failure_feasibility_reason_mode}`"
    )
    paired_sentence = f" Also, {support[0]}." if support else ""

    if effect < 0:
        claim = f"{robot} is worse than {reference}"
        effect_phrase = (
            f"changing the robot from {reference} to {robot} lowers success "
            f"by {abs(effect):.4f} ({pct(abs(effect))})"
        )
    elif effect > 0:
        claim = f"{robot} is better than {reference}"
        effect_phrase = (
            f"changing the robot from {reference} to {robot} increases success "
            f"by {abs(effect):.4f} ({pct(abs(effect))})"
        )
    else:
        claim = f"{robot} and {reference} are tied"
        effect_phrase = (
            f"changing the robot from {reference} to {robot} does not change success "
            "on the current intervention summary"
        )

    return (
        "Most likely explanation\n"
        "-----------------------\n"
        f"{claim} because {effect_phrase} on the intervention data, and the failures "
        f"are dominated by the motion feasibility mechanism: {mechanism_sentence}."
        f"{paired_sentence}{feature_sentence}\n"
        f"\n{feature_attribution_analysis(data, robot, reference, effect)}\n"
        "This is a data-backed explanation of the robot intervention effect; the morphology "
        "features are explanatory evidence, not independent do-interventions."
    )


def feature_attribution_analysis(
    data: ReportData, robot: str, reference: str, effect: float
) -> str:
    if data.features.empty:
        return "Feature attribution: no morphology feature table available."

    feature_columns = [
        "robot_drive_type",
        "robot_arm_count",
        "robot_base_bbox_width_m",
        "robot_base_bbox_depth_m",
        "robot_base_bbox_height_m",
        "robot_tool_frame_max_z_m",
        "robot_arm_reach_xy_max_m",
        "robot_hardware_dof_count",
    ]
    columns = [column for column in feature_columns if column in data.features.columns]
    joined = data.summary.merge(data.features[[ROBOT] + columns], on=ROBOT, how="inner")
    if joined.empty:
        return (
            "Feature attribution: no overlap between summary and morphology features."
        )

    robot_row = joined[joined[ROBOT] == robot]
    reference_row = joined[joined[ROBOT] == reference]
    if robot_row.empty or reference_row.empty:
        return (
            "Feature attribution: comparison robots not found in morphology features."
        )
    robot_row = robot_row.iloc[0]
    reference_row = reference_row.iloc[0]

    expected_sign = 1 if effect > 0 else -1 if effect < 0 else 0
    feature_scores: list[dict[str, object]] = []

    def add_score(
        name: str,
        raw_score: float,
        predicted_delta: float,
        detail: str,
        caveat: str = "",
    ) -> None:
        supports_effect = (
            expected_sign == 0
            or raw_score == 0
            or (predicted_delta * expected_sign > 0)
        )
        feature_scores.append(
            {
                "name": name,
                "score": abs(float(raw_score)) if supports_effect else 0.0,
                "predicted_delta": float(predicted_delta),
                "detail": detail,
                "caveat": caveat,
                "supports_effect": supports_effect,
            }
        )

    if (
        "robot_drive_type" in joined.columns
        and robot_row["robot_drive_type"] != reference_row["robot_drive_type"]
    ):
        groups = joined.groupby("robot_drive_type", dropna=False).apply(
            lambda group: group["successes"].sum() / group["runs"].sum(),
            include_groups=False,
        )
        robot_p = float(groups.loc[robot_row["robot_drive_type"]])
        reference_p = float(groups.loc[reference_row["robot_drive_type"]])
        delta = robot_p - reference_p
        add_score(
            "drive type",
            abs(delta),
            delta,
            f"{robot_row['robot_drive_type']} group={robot_p:.4f}, "
            f"{reference_row['robot_drive_type']} group={reference_p:.4f}",
            "weak if the grouped success rates are nearly equal",
        )

    if (
        "robot_arm_count" in joined.columns
        and robot_row["robot_arm_count"] != reference_row["robot_arm_count"]
    ):
        groups = joined.groupby("robot_arm_count", dropna=False).apply(
            lambda group: group["successes"].sum() / group["runs"].sum(),
            include_groups=False,
        )
        robot_p = float(groups.loc[robot_row["robot_arm_count"]])
        reference_p = float(groups.loc[reference_row["robot_arm_count"]])
        delta = robot_p - reference_p
        add_score(
            "arm count",
            abs(delta),
            delta,
            f"{robot_row['robot_arm_count']} arm(s) group={robot_p:.4f}, "
            f"{reference_row['robot_arm_count']} arm(s) group={reference_p:.4f}",
            "confounded with robot identity, reach, and base geometry in the current four-robot scope",
        )

    numeric_features = {
        "reach": ["robot_arm_reach_xy_max_m"],
        "base footprint": ["robot_base_bbox_width_m", "robot_base_bbox_depth_m"],
        "base height": ["robot_base_bbox_height_m"],
        "tool-frame height": ["robot_tool_frame_max_z_m"],
        "hardware dof": ["robot_hardware_dof_count"],
    }
    for name, numeric_columns in numeric_features.items():
        component_scores = []
        component_details = []
        component_predicted = []
        for column in numeric_columns:
            if column not in joined.columns:
                continue
            values = joined[[column, "success_probability"]].dropna()
            if len(values) < 3 or values[column].nunique() < 2:
                continue
            feature_range = float(values[column].max() - values[column].min())
            if feature_range == 0:
                continue
            corr = values[column].corr(values["success_probability"])
            if pd.isna(corr):
                continue
            robot_value = float(robot_row[column])
            reference_value = float(reference_row[column])
            normalized_delta = (robot_value - reference_value) / feature_range
            predicted_delta = float(corr) * normalized_delta
            component_scores.append(abs(predicted_delta))
            component_predicted.append(predicted_delta)
            component_details.append(
                f"{column}: {robot_value:.3f} vs {reference_value:.3f}, corr {corr:+.2f}"
            )
        if component_scores:
            add_score(
                name,
                sum(component_scores) / len(component_scores),
                sum(component_predicted) / len(component_predicted),
                "; ".join(component_details),
            )

    supported = [item for item in feature_scores if float(item["score"]) > 0]
    total = sum(float(item["score"]) for item in supported)
    lines = [
        "Natural feature analysis",
        "------------------------",
    ]
    if not supported or total == 0:
        lines.append(
            "No morphology feature gives a clear directional explanation for this pair."
        )
        return "\n".join(lines)

    lines.append(
        "Heuristic share among morphology explanations that point in the observed direction "
        "(not causal probabilities):"
    )
    for item in sorted(supported, key=lambda row: float(row["score"]), reverse=True):
        share = float(item["score"]) / total
        lines.append(
            f"- {item['name']}: {pct(share)} evidence share; {item['detail']}"
            + (f"; caveat: {item['caveat']}" if item["caveat"] else "")
        )

    rejected = [
        item
        for item in feature_scores
        if not item["supports_effect"] or float(item["score"]) == 0
    ]
    weak = [
        item
        for item in rejected
        if item["name"] == "drive type" or item["name"] == "arm count"
    ]
    for item in weak:
        lines.append(
            f"- {item['name']} is not selected as a main explanation here; {item['detail']}."
        )
    lines.append(
        "Interpretation: the hard causal statement is the robot substitution effect; "
        "these shares only rank which recorded morphology differences best align with that effect."
    )
    return "\n".join(lines)


def strongest_feature_reasons(
    data: ReportData, robot: str, reference: str
) -> list[str]:
    if data.features.empty:
        return []
    numeric_columns = [
        "robot_arm_reach_xy_max_m",
        "robot_base_bbox_width_m",
        "robot_base_bbox_depth_m",
        "robot_tool_frame_max_z_m",
        "robot_hardware_dof_count",
    ]
    columns = [
        column
        for column in numeric_columns + ["robot_arm_count", "robot_drive_type"]
        if column in data.features.columns
    ]
    joined = data.summary.merge(data.features[[ROBOT] + columns], on=ROBOT, how="inner")
    if joined.empty:
        return []
    robot_row = joined[joined[ROBOT] == robot]
    reference_row = joined[joined[ROBOT] == reference]
    if robot_row.empty or reference_row.empty:
        return []
    robot_row = robot_row.iloc[0]
    reference_row = reference_row.iloc[0]

    reasons: list[tuple[float, str]] = []

    if (
        "robot_arm_count" in joined.columns
        and robot_row["robot_arm_count"] != reference_row["robot_arm_count"]
    ):
        arm_groups = joined.groupby("robot_arm_count", dropna=False).apply(
            lambda group: group["successes"].sum() / group["runs"].sum(),
            include_groups=False,
        )
        robot_arm_p = float(arm_groups.loc[robot_row["robot_arm_count"]])
        reference_arm_p = float(arm_groups.loc[reference_row["robot_arm_count"]])
        delta = robot_arm_p - reference_arm_p
        reasons.append(
            (
                abs(delta),
                f"arm count ({robot}: {robot_row['robot_arm_count']}, "
                f"{reference}: {reference_row['robot_arm_count']}; grouped delta {delta:+.4f})",
            )
        )

    if (
        "robot_drive_type" in joined.columns
        and robot_row["robot_drive_type"] != reference_row["robot_drive_type"]
    ):
        drive_groups = joined.groupby("robot_drive_type", dropna=False).apply(
            lambda group: group["successes"].sum() / group["runs"].sum(),
            include_groups=False,
        )
        robot_drive_p = float(drive_groups.loc[robot_row["robot_drive_type"]])
        reference_drive_p = float(drive_groups.loc[reference_row["robot_drive_type"]])
        delta = robot_drive_p - reference_drive_p
        if abs(delta) >= 0.05:
            reasons.append(
                (
                    abs(delta),
                    f"drive type ({robot}: {robot_row['robot_drive_type']}, "
                    f"{reference}: {reference_row['robot_drive_type']}; grouped delta {delta:+.4f})",
                )
            )

    for column in numeric_columns:
        if column not in joined.columns:
            continue
        values = joined[[column, "success_probability"]].dropna()
        if len(values) < 3 or values[column].nunique() < 2:
            continue
        corr = values[column].corr(values["success_probability"])
        if pd.isna(corr) or abs(corr) < 0.5:
            continue
        robot_value = float(robot_row[column])
        reference_value = float(reference_row[column])
        robot_lower = robot_value < reference_value
        supports_worse = (corr > 0 and robot_lower) or (corr < 0 and not robot_lower)
        if supports_worse:
            direction = "lower" if robot_lower else "higher"
            reasons.append(
                (
                    abs(float(corr)),
                    f"{column} ({robot} is {direction}: {robot_value:.3f} vs "
                    f"{reference_value:.3f}, corr {corr:+.2f})",
                )
            )

    return [reason for _, reason in sorted(reasons, reverse=True)[:3]]


def get_morphology(data: ReportData, robot: str) -> str:
    robot = normalize_robot(robot)
    if data.features.empty:
        return "No morphology feature CSV found."
    rows = data.features[data.features[ROBOT] == robot]
    if rows.empty:
        return f"No morphology row found for {robot}."
    row = rows.iloc[0]
    if "robot_feature_extraction_ok" in row and not bool(
        row.robot_feature_extraction_ok
    ):
        return f"Morphology extraction failed for {robot}: {row.robot_feature_error}"

    fields = [
        "robot_class",
        "robot_drive_type",
        "robot_arm_count",
        "robot_manipulator_count",
        "robot_base_bbox_width_m",
        "robot_base_bbox_depth_m",
        "robot_base_bbox_height_m",
        "robot_tool_frame_max_z_m",
        "robot_arm_reach_xy_max_m",
        "robot_hardware_dof_count",
    ]
    lines = [f"Morphology for {robot}:"]
    for field in fields:
        if field in row.index:
            lines.append(f"- {field}: {row[field]}")
    lines.append(f"Source: {FEATURES_CSV}")
    return "\n".join(lines)


def compare_morphology(data: ReportData, robot: str, reference: str) -> str:
    if data.features.empty:
        return "No morphology feature CSV found for morphology comparison."
    robot = normalize_robot(robot)
    reference = normalize_robot(reference)
    rows = data.features[data.features[ROBOT].isin([robot, reference])].copy()
    if len(rows) < 2:
        return f"Not enough morphology rows to compare {robot} and {reference}."

    if "robot_feature_extraction_ok" in rows.columns:
        failed = rows[rows["robot_feature_extraction_ok"] != True]
        if not failed.empty:
            failed_robots = ", ".join(failed[ROBOT].astype(str))
            return f"Morphology extraction failed for: {failed_robots}."

    by_robot = rows.set_index(ROBOT)
    fields = [
        "robot_drive_type",
        "robot_arm_count",
        "robot_manipulator_count",
        "robot_base_bbox_width_m",
        "robot_base_bbox_depth_m",
        "robot_base_bbox_height_m",
        "robot_tool_frame_max_z_m",
        "robot_arm_reach_xy_max_m",
        "robot_hardware_dof_count",
    ]
    lines = [f"Morphology comparison {robot} vs {reference}:"]
    for field in fields:
        if field in by_robot.columns:
            lines.append(
                f"- {field}: {robot}={by_robot.loc[robot, field]}, "
                f"{reference}={by_robot.loc[reference, field]}"
            )

    notes = []
    if "robot_drive_type" in by_robot.columns:
        robot_drive = by_robot.loc[robot, "robot_drive_type"]
        reference_drive = by_robot.loc[reference, "robot_drive_type"]
        if robot_drive != reference_drive:
            notes.append(
                f"drive type differs ({robot}: {robot_drive}, {reference}: {reference_drive})"
            )
    if "robot_arm_count" in by_robot.columns:
        if (
            by_robot.loc[robot, "robot_arm_count"]
            != by_robot.loc[reference, "robot_arm_count"]
        ):
            notes.append("arm count differs")
    if "robot_arm_reach_xy_max_m" in by_robot.columns:
        reach_delta = float(by_robot.loc[robot, "robot_arm_reach_xy_max_m"]) - float(
            by_robot.loc[reference, "robot_arm_reach_xy_max_m"]
        )
        notes.append(f"reach proxy differs by {reach_delta:+.3f} m")
    if notes:
        lines.append("Feature differences to inspect: " + "; ".join(notes) + ".")
    evidence = morphology_evidence(data, robot, reference)
    if evidence:
        lines.extend(["", evidence])
    lines.append(
        "Reading: these features can explain or stratify the robot effect, "
        "but they are not separate do-interventions by themselves."
    )
    lines.append(f"Source: {FEATURES_CSV}")
    return "\n".join(lines)


def morphology_evidence(data: ReportData, robot: str, reference: str) -> str:
    feature_columns = [
        "robot_drive_type",
        "robot_arm_count",
        "robot_base_bbox_width_m",
        "robot_base_bbox_depth_m",
        "robot_base_bbox_height_m",
        "robot_tool_frame_max_z_m",
        "robot_arm_reach_xy_max_m",
        "robot_hardware_dof_count",
    ]
    if data.features.empty:
        return ""
    joined = data.summary.merge(
        data.features[
            [ROBOT]
            + [column for column in feature_columns if column in data.features.columns]
        ],
        on=ROBOT,
        how="inner",
    )
    if joined.empty:
        return ""

    robot_row = joined[joined[ROBOT] == robot]
    reference_row = joined[joined[ROBOT] == reference]
    if robot_row.empty or reference_row.empty:
        return ""
    robot_row = robot_row.iloc[0]
    reference_row = reference_row.iloc[0]

    lines = ["Feature evidence across current robots:"]

    if "robot_drive_type" in joined.columns:
        drive_groups = joined.groupby("robot_drive_type", dropna=False).apply(
            lambda group: pd.Series(
                {
                    "p_success": group["successes"].sum() / group["runs"].sum(),
                    "robots": ", ".join(sorted(group[ROBOT].astype(str))),
                }
            ),
            include_groups=False,
        )
        robot_drive = robot_row["robot_drive_type"]
        reference_drive = reference_row["robot_drive_type"]
        if robot_drive != reference_drive:
            robot_drive_p = drive_groups.loc[robot_drive, "p_success"]
            reference_drive_p = drive_groups.loc[reference_drive, "p_success"]
            drive_delta = float(robot_drive_p) - float(reference_drive_p)
            lines.append(
                f"- drive_type is weak here: {robot_drive}={float(robot_drive_p):.4f} "
                f"vs {reference_drive}={float(reference_drive_p):.4f} "
                f"(delta {drive_delta:+.4f}) across current robots."
            )

    if "robot_arm_count" in joined.columns:
        arm_groups = joined.groupby("robot_arm_count", dropna=False).apply(
            lambda group: pd.Series(
                {
                    "p_success": group["successes"].sum() / group["runs"].sum(),
                    "robots": ", ".join(sorted(group[ROBOT].astype(str))),
                }
            ),
            include_groups=False,
        )
        robot_arm = robot_row["robot_arm_count"]
        reference_arm = reference_row["robot_arm_count"]
        if robot_arm != reference_arm:
            robot_arm_p = arm_groups.loc[robot_arm, "p_success"]
            reference_arm_p = arm_groups.loc[reference_arm, "p_success"]
            lines.append(
                f"- arm_count supports the explanation: {robot_arm} arm(s)="
                f"{float(robot_arm_p):.4f} ({arm_groups.loc[robot_arm, 'robots']}) "
                f"vs {reference_arm} arm(s)={float(reference_arm_p):.4f} "
                f"({arm_groups.loc[reference_arm, 'robots']})."
            )

    numeric_columns = [
        "robot_base_bbox_width_m",
        "robot_base_bbox_depth_m",
        "robot_base_bbox_height_m",
        "robot_tool_frame_max_z_m",
        "robot_arm_reach_xy_max_m",
        "robot_hardware_dof_count",
    ]
    numeric_scores = []
    for column in numeric_columns:
        if column not in joined.columns:
            continue
        values = joined[[column, "success_probability"]].dropna()
        if len(values) < 3 or values[column].nunique() < 2:
            continue
        corr = values[column].corr(values["success_probability"])
        if pd.isna(corr):
            continue
        delta = float(robot_row[column]) - float(reference_row[column])
        numeric_scores.append(
            (abs(corr), corr, column, delta, robot_row[column], reference_row[column])
        )

    for _, corr, column, delta, robot_value, reference_value in sorted(
        numeric_scores, reverse=True
    )[:3]:
        direction = (
            "higher tends to succeed more"
            if corr > 0
            else "lower tends to succeed more"
        )
        robot_side = "lower" if delta < 0 else "higher" if delta > 0 else "same"
        lines.append(
            f"- {column}: {direction} in current robots (corr {corr:+.2f}); "
            f"{robot} is {robot_side} than {reference} "
            f"({robot_value} vs {reference_value})."
        )

    lines.append(
        "Caveat: this ranks explanatory feature associations over only the currently covered robots; "
        "it does not prove do(feature=value)."
    )
    return "\n".join(lines)


def get_drive_type(data: ReportData, robot: str | None = None) -> str:
    if data.features.empty or "robot_drive_type" not in data.features.columns:
        return "No robot_drive_type feature found. Re-run morphology feature extraction first."

    feature_rows = data.features[data.features[ROBOT].isin(data.summary[ROBOT])].copy()
    if "robot_feature_extraction_ok" in feature_rows.columns:
        feature_rows = feature_rows[
            feature_rows["robot_feature_extraction_ok"] == True
        ].copy()

    if robot is not None:
        robot = normalize_robot(robot)
        rows = feature_rows[feature_rows[ROBOT] == robot]
        if rows.empty:
            return f"No drive type found for {robot}."
        row = rows.iloc[0]
        return f"{robot} drive type: {row.robot_drive_type}\n" f"Source: {FEATURES_CSV}"

    joined = data.summary.merge(
        feature_rows[[ROBOT, "robot_drive_type"]],
        on=ROBOT,
        how="inner",
    )
    if joined.empty:
        return "No overlap between causal summary and drive-type features."

    rows = []
    for drive_type, group in joined.groupby("robot_drive_type", dropna=False):
        successes = int(group["successes"].sum())
        runs = int(group["runs"].sum())
        success_probability = successes / runs if runs else float("nan")
        robots = ", ".join(sorted(group[ROBOT].astype(str)))
        rows.append((str(drive_type), success_probability, successes, runs, robots))

    rows.sort(key=lambda item: item[1], reverse=True)
    lines = ["Success grouped by robot_drive_type:"]
    for drive_type, success_probability, successes, runs, robots in rows:
        lines.append(
            f"- {drive_type}: {success_probability:.4f} ({pct(success_probability)}), "
            f"{successes}/{runs}, robots: {robots}"
        )
    lines.extend(
        [
            "",
            "Reading: this is a morphology/feature association over robot interventions.",
            "It is not a clean do(robot_drive_type=...) effect unless drive type is varied independently from robot identity.",
            f"Sources: {SUMMARY_CSV} and {FEATURES_CSV}",
        ]
    )
    return "\n".join(lines)


def geometry_available_features(data: ReportData) -> list[str]:
    if data.results.empty:
        return []
    features = []
    for feature in GEOMETRY_FEATURES:
        if feature not in data.results.columns:
            continue
        values = pd.to_numeric(data.results[feature], errors="coerce")
        if values.notna().sum() > 0 and values.nunique(dropna=True) >= 2:
            features.append(feature)
    return features


def geometry_summary(data: ReportData) -> str:
    features = geometry_available_features(data)
    if not features:
        return "No task-geometry columns found in the result CSV."

    lines = [
        "Task-geometry effects",
        "---------------------",
        "Reading: because task geometry is generated by seeds and repeated across robots, "
        "these effects can be read causally under the randomization assumption.",
        "",
        "Largest low/mid/high success differences:",
    ]
    rows = []
    for feature in features:
        table = geometry_feature_table(data, feature)
        if table.empty or len(table) < 2:
            continue
        delta = float(
            table["success_probability"].max() - table["success_probability"].min()
        )
        best = table.sort_values("success_probability", ascending=False).iloc[0]
        worst = table.sort_values("success_probability", ascending=True).iloc[0]
        rows.append((delta, feature, best, worst, table))
    for delta, feature, best, worst, _ in sorted(
        rows, key=lambda item: item[0], reverse=True
    )[:8]:
        lines.append(
            f"- {feature}: range {delta:.4f}; best={best.geometry_bin} "
            f"{best.success_probability:.4f}, worst={worst.geometry_bin} "
            f"{worst.success_probability:.4f}"
        )
    lines.append("")
    lines.append(
        "Ask for details, e.g. `why geometry object_top_z high`, `effect geometry volume high low`, or `geometry object_z by robot`."
    )
    return "\n".join(lines)


def geometry_feature_table(
    data: ReportData, feature: str, robot: str | None = None
) -> pd.DataFrame:
    if data.results.empty or feature not in data.results.columns:
        return pd.DataFrame()
    rows = data.results.copy()
    if robot is not None:
        rows = rows[rows[ROBOT] == normalize_robot(robot)].copy()
    rows = rows.dropna(subset=[feature, SUCCESS]).copy()
    if rows.empty:
        return pd.DataFrame()
    rows["geometry_bin"] = make_geometry_bins(rows[feature])
    rows = rows.dropna(subset=["geometry_bin"])
    if rows.empty:
        return pd.DataFrame()
    grouped = (
        rows.groupby("geometry_bin", observed=False)
        .agg(
            success_probability=(SUCCESS, "mean"),
            successes=(SUCCESS, "sum"),
            runs=(SUCCESS, "size"),
            min_value=(feature, "min"),
            max_value=(feature, "max"),
        )
        .reset_index()
    )
    grouped["successes"] = grouped["successes"].astype(int)
    return grouped.sort_values("geometry_bin")


def get_geometry_feature(data: ReportData, feature: str, by_robot: bool = False) -> str:
    feature = normalize_geometry_feature(feature, data)
    if feature is None:
        return "Unknown geometry feature. Try `geometry` to list available effects."

    if by_robot:
        lines = [
            f"Geometry effect by robot for {feature}",
            "--------------------------------"[: 29 + len(feature)],
        ]
        for robot in data.robots:
            table = geometry_feature_table(data, feature, robot=robot)
            if table.empty:
                continue
            lines.append(f"\n{robot}:")
            for _, row in table.iterrows():
                lines.append(
                    f"- {row.geometry_bin}: {row.success_probability:.4f} "
                    f"({int(row.successes)}/{int(row.runs)}), "
                    f"value range [{row.min_value:.4f}, {row.max_value:.4f}]"
                )
        lines.append(
            "\nInterpretation: this shows robot sensitivity to the same randomized task-geometry variable."
        )
        return "\n".join(lines)

    table = geometry_feature_table(data, feature)
    if table.empty:
        return f"No usable rows for geometry feature {feature}."
    lines = [
        f"Geometry effect for {feature}",
        "-----------------------------",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"- {row.geometry_bin}: P(success | geometry_bin={row.geometry_bin}) = "
            f"{row.success_probability:.4f} ({int(row.successes)}/{int(row.runs)}), "
            f"value range [{row.min_value:.4f}, {row.max_value:.4f}]"
        )
    lines.append(
        "Causal reading: under seed randomization/balancing, this approximates "
        f"P(success | do({feature}=bin))."
    )
    return "\n".join(lines)


def normalize_geometry_feature(feature: str | None, data: ReportData) -> str | None:
    if feature is None:
        return None
    value = feature.strip().lower()
    value = GEOMETRY_ALIASES.get(value, value)
    return value if value in geometry_available_features(data) else None


def get_geometry_effect(data: ReportData, feature: str, bin_a: str, bin_b: str) -> str:
    feature = normalize_geometry_feature(feature, data)
    if feature is None:
        return "Unknown geometry feature. Try `geometry` to list available effects."
    table = geometry_feature_table(data, feature)
    if table.empty:
        return f"No usable rows for geometry feature {feature}."
    by_bin = table.set_index("geometry_bin")
    if bin_a not in by_bin.index or bin_b not in by_bin.index:
        return f"Known bins for {feature}: {', '.join(map(str, by_bin.index))}"
    row_a = by_bin.loc[bin_a]
    row_b = by_bin.loc[bin_b]
    effect = float(row_a.success_probability - row_b.success_probability)
    direction = "better" if effect > 0 else "worse" if effect < 0 else "same"
    mechanism = geometry_bin_mechanism(data, feature, bin_a)
    if effect < 0:
        explanation = (
            f"The `{bin_a}` {feature} bin is worse than `{bin_b}`: success drops by "
            f"{abs(effect):.4f} ({pct(abs(effect))})."
        )
    elif effect > 0:
        explanation = (
            f"The `{bin_a}` {feature} bin is better than `{bin_b}`: success increases by "
            f"{abs(effect):.4f} ({pct(abs(effect))})."
        )
    else:
        explanation = f"The `{bin_a}` and `{bin_b}` bins have the same success rate."
    return (
        "Most likely geometry answer\n"
        "---------------------------\n"
        f"{explanation} "
        f"This estimates P(success | do({feature}={bin_a})) versus "
        f"P(success | do({feature}={bin_b})) under the seed-randomization assumption.\n"
        f"{mechanism}\n\n"
        f"Geometry effect {feature}: {bin_a} - {bin_b} = {effect:+.4f} "
        f"({pct(abs(effect))} absolute success-rate difference, {bin_a} is {direction}).\n"
        f"{bin_a}: {row_a.success_probability:.4f} ({int(row_a.successes)}/{int(row_a.runs)}), "
        f"value range [{row_a.min_value:.4f}, {row_a.max_value:.4f}]\n"
        f"{bin_b}: {row_b.success_probability:.4f} ({int(row_b.successes)}/{int(row_b.runs)}), "
        f"value range [{row_b.min_value:.4f}, {row_b.max_value:.4f}]\n"
        "Causal reading: this is a task-geometry effect under the seed-randomization assumption."
    )


def geometry_bin_mechanism(data: ReportData, feature: str, bin_label: str) -> str:
    if data.results.empty or "feasibility_reason" not in data.results.columns:
        return "No mechanism column is available for this geometry bin."
    rows = data.results.dropna(subset=[feature, SUCCESS]).copy()
    rows["geometry_bin"] = make_geometry_bins(rows[feature]).astype(str)
    selected = rows[rows["geometry_bin"] == bin_label]
    failed = selected[~selected[SUCCESS]]
    if failed.empty:
        return f"No failures were observed in the `{bin_label}` bin."
    counts = failed["feasibility_reason"].value_counts(dropna=False)
    reason = counts.index[0]
    count = int(counts.iloc[0])
    return (
        f"In failed `{bin_label}` runs, the dominant mechanism is `{reason}` "
        f"({count}/{len(failed)}, {pct(count / len(failed))})."
    )


def why_geometry(data: ReportData, feature: str, bin_label: str | None = None) -> str:
    feature = normalize_geometry_feature(feature, data)
    if feature is None:
        return "Unknown geometry feature. Try `geometry` to list available effects."
    table = geometry_feature_table(data, feature)
    if table.empty:
        return f"No usable rows for geometry feature {feature}."
    if bin_label is None:
        bin_label = str(table.sort_values("success_probability").iloc[0].geometry_bin)
    if bin_label not in set(table["geometry_bin"].astype(str)):
        return (
            f"Known bins for {feature}: {', '.join(table['geometry_bin'].astype(str))}"
        )

    rows = data.results.dropna(subset=[feature, SUCCESS]).copy()
    rows["geometry_bin"] = make_geometry_bins(rows[feature]).astype(str)
    selected = rows[rows["geometry_bin"] == bin_label]
    other = rows[rows["geometry_bin"] != bin_label]
    selected_p = float(selected[SUCCESS].mean())
    other_p = float(other[SUCCESS].mean()) if len(other) else float("nan")
    effect = selected_p - other_p
    selected_range = (selected[feature].min(), selected[feature].max())

    failed = selected[~selected[SUCCESS]]
    mechanism = ""
    if "feasibility_reason" in failed.columns and len(failed):
        counts = failed["feasibility_reason"].value_counts(dropna=False)
        reason = counts.index[0]
        count = int(counts.iloc[0])
        mechanism = (
            f" In failed runs from this bin, the dominant mechanism is `{reason}` "
            f"({count}/{len(failed)}, {pct(count / len(failed))})."
        )

    robot_lines = []
    for robot in data.robots:
        robot_rows = selected[selected[ROBOT] == robot]
        if len(robot_rows):
            robot_lines.append(
                (
                    robot,
                    float(robot_rows[SUCCESS].mean()),
                    int(robot_rows[SUCCESS].sum()),
                    len(robot_rows),
                )
            )
    robot_lines.sort(key=lambda item: item[1])

    lines = [
        "Most likely geometry explanation",
        "--------------------------------",
        f"For `{feature}`, the `{bin_label}` bin has P(success)={selected_p:.4f}; "
        f"all other bins have P(success)={other_p:.4f}, difference {effect:+.4f}.",
        f"Value range for `{bin_label}`: [{selected_range[0]:.4f}, {selected_range[1]:.4f}]."
        + mechanism,
        "Causal reading: because this task geometry is generated by seeds and repeated across robots, "
        f"this approximates P(success | do({feature}={bin_label})) under the randomization assumption.",
    ]
    if robot_lines:
        lines.append("")
        lines.append("Robot sensitivity in this bin:")
        for robot, p_success, successes, runs in robot_lines:
            lines.append(f"- {robot}: {p_success:.4f} ({successes}/{runs})")
    return "\n".join(lines)


def list_robots(data: ReportData) -> str:
    rows = data.summary.sort_values("success_probability", ascending=False)
    lines = ["Robots in current causal report:"]
    for _, row in rows.iterrows():
        lines.append(
            f"- {row[ROBOT]}: P(success | do(robot={row[ROBOT]})) = "
            f"{row.success_probability:.4f} ({int(row.successes)}/{int(row.runs)})"
        )
    return "\n".join(lines)


def help_text() -> str:
    return (
        "Ask one of these:\n"
        "- do hsrb\n"
        "- effect hsrb pr2\n"
        "- why hsrb pr2\n"
        "- why hsrb\n"
        "- mechanism hsrb\n"
        "- morphology hsrb\n"
        "- drive\n"
        "- drive hsrb\n"
        "- geometry\n"
        "- geometry object_z\n"
        "- geometry object_z by robot\n"
        "- why geometry object_top_z high\n"
        "- effect geometry volume high low\n"
        "- list\n\n"
        "Natural wording also works for common questions, e.g. "
        "'Warum ist do(hsrb) schlechter?' or 'Ist hsrb schlechter als pr2?'."
    )


def answer(question: str, data: ReportData) -> str:
    question = question.strip()
    lower = question.lower()
    robots = find_robots(lower, data.robots)

    if lower in {"help", "-h", "--help", ""}:
        return help_text()
    if lower in {"list", "robots", "summary"}:
        return list_robots(data)
    if lower == "geometry":
        return geometry_summary(data)

    words = lower.split()
    if words:
        command = words[0]
        if command == "do" and len(robots) >= 1:
            return get_do(data, robots[0])
        if command == "effect" and len(robots) >= 2:
            return get_effect(data, robots[0], robots[1])
        if command == "why" and len(robots) >= 1:
            reference = robots[1] if len(robots) >= 2 else None
            return get_why(data, robots[0], reference)
        if command == "mechanism" and len(robots) >= 1:
            return get_mechanism(data, robots[0])
        if command == "morphology" and len(robots) >= 1:
            return get_morphology(data, robots[0])
        if command == "drive":
            return get_drive_type(data, robots[0] if robots else None)
        if command == "geometry":
            feature = find_geometry_feature(lower, data)
            if feature is None:
                return geometry_summary(data)
            labels = find_bin_labels(lower)
            if len(labels) >= 2:
                return get_geometry_effect(data, feature, labels[0], labels[1])
            return get_geometry_feature(
                data, feature, by_robot=("by robot" in lower or "per robot" in lower)
            )
        if command == "effect" and "geometry" in words:
            feature = find_geometry_feature(lower, data)
            labels = [
                word
                for word in words
                if word in {"low", "mid", "middle", "medium", "high"}
            ]
            labels = [
                "mid" if label in {"middle", "medium"} else label for label in labels
            ]
            if feature is not None and len(labels) >= 2:
                return get_geometry_effect(data, feature, labels[0], labels[1])
        if command == "why" and "geometry" in words:
            feature = find_geometry_feature(lower, data)
            if feature is not None:
                return why_geometry(data, feature, find_bin_label(lower))

    if any(word in lower for word in ["warum", "why", "schlechter", "worse"]):
        if "geometry" in lower or any(alias in lower for alias in GEOMETRY_ALIASES):
            feature = find_geometry_feature(lower, data)
            if feature is not None:
                return why_geometry(data, feature, find_bin_label(lower))
        if len(robots) >= 2:
            return get_why(data, robots[0], robots[1])
        if len(robots) == 1:
            return get_why(data, robots[0])
    if "p(success" in lower or "do(" in lower or " do " in f" {lower} ":
        if len(robots) >= 1:
            return get_do(data, robots[0])
    if any(word in lower for word in ["besser", "better", "effect", "effekt"]):
        if len(robots) >= 2:
            return get_effect(data, robots[0], robots[1])
    if any(word in lower for word in ["mechanism", "mechanismus", "failure", "fehler"]):
        if len(robots) >= 1:
            return get_mechanism(data, robots[0])
    if any(word in lower for word in ["morphology", "morphologie", "reach", "arm"]):
        if len(robots) >= 1:
            return get_morphology(data, robots[0])
    if any(word in lower for word in ["drive", "differential", "diff", "omni"]):
        return get_drive_type(data, robots[0] if robots else None)
    if "geometry" in lower or find_geometry_feature(lower, data) is not None:
        feature = find_geometry_feature(lower, data)
        if feature is None:
            return geometry_summary(data)
        return get_geometry_feature(
            data, feature, by_robot=("by robot" in lower or "per robot" in lower)
        )

    return "I could not map that question to the report tables.\n\n" + help_text()


def main() -> None:
    data = load_report_data()
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(answer(question, data))
        return

    print(help_text())
    print("\nType 'quit' to exit.")
    while True:
        try:
            question = input("\ncausal> ").strip()
        except EOFError:
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        print(answer(question, data))


if __name__ == "__main__":
    main()
