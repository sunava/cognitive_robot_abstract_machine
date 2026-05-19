#!/usr/bin/env python3
"""
Estimate causal effects from a CSV using simple backdoor-adjusted g-computation.

This is a generic CSV adaptation of the causal-diagnosis idea: for each candidate
cause column, fit a supervised model for the target using that cause plus the
other observed covariates, then estimate what the target would be under
interventions such as do(cause = low) and do(cause = high).

The estimates are causal only under the usual observational assumptions:
the relevant confounders are present in the CSV, the chosen cause is upstream of
the target, and there is enough overlap in the data to compare interventions.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PYCRAM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    PYCRAM_ROOT / "demos" / "thesis_new" / "records" / "cut_all_breads_results.csv"
)
DEFAULT_OUTPUT_DIR = PYCRAM_ROOT / "demos" / "thesis_new" / "records" / "causality"
DEFAULT_TARGET_COLUMN = "final_success"
DEFAULT_CUT_CAUSE_COLUMNS = [
    "knowledge_query_success",
    "knowledge_cutting_tool",
    "knowledge_cutting_position",
    "knowledge_repetition",
    "required_prerequisite",
    "prerequisite_satisfied_initially",
    "autonomous_execution_feasible",
    "object_aabb_min_x",
    "object_aabb_min_y",
    "object_aabb_min_z",
    "object_aabb_max_x",
    "object_aabb_max_y",
    "object_aabb_max_z",
    "object_size_x",
    "object_size_y",
    "object_size_z",
    "object_volume_aabb",
    "target_world_x",
    "target_world_y",
    "target_world_z",
    "support_yaw_rad",
    "support_size_x",
    "support_size_y",
    "support_size_z",
    "object_world_x",
    "object_world_y",
    "object_world_z",
    "object_quat_x",
    "object_quat_y",
    "object_quat_z",
    "object_quat_w",
    "object_roll_rad",
    "object_pitch_rad",
    "object_yaw_rad",
    "anchor_local_x",
    "anchor_local_y",
    "anchor_local_z",
    "anchor_norm_x",
    "anchor_norm_y",
    "anchor_norm_z",
    "cut_normal_local_x",
    "cut_normal_local_y",
    "cut_normal_local_z",
    "cut_normal_world_x",
    "cut_normal_world_y",
    "cut_normal_world_z",
    "cut_normal_world_yaw_rad",
    "technique_name",
    "slice_thickness_m",
    "num_cuts_x",
    "pointer_stride",
]

DEFAULT_IGNORE_SUBSTRINGS = (
    "id",
    "uuid",
    "name",
    "description",
    "reason",
    "timestamp",
    "time_s",
    "date",
    "path",
    "outcome",
    "final",
    "success",
)


@dataclass(frozen=True)
class OutcomeSpec:
    kind: str
    positive_label: object | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn ranked intervention effects from a CSV."
    )
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_CSV,
        help=f"Input CSV file. Defaults to {DEFAULT_INPUT_CSV}.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_COLUMN,
        help="Outcome/effect column to explain, e.g. final_success or execution_time_s.",
    )
    parser.add_argument(
        "--causes",
        nargs="*",
        help="Candidate cause columns. Defaults to numeric/low-cardinality columns.",
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        help="Columns to ignore in addition to obvious identifiers/text columns.",
    )
    parser.add_argument(
        "--positive-label",
        help="Positive class for a binary target. Defaults to True/1/success when present.",
    )
    parser.add_argument(
        "--max-categorical-levels",
        type=int,
        default=20,
        help="Maximum unique values for a categorical cause/covariate.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Optional row sample size for faster experimentation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "cut_all_breads_causal_effects.csv",
        help="Output CSV with ranked causal-effect estimates.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "cut_all_breads_causal_effects_report.txt",
        help="Output text report.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=7,
        help="Random seed for sampling and model fitting.",
    )
    return parser.parse_args()


def is_probably_identifier(column: str) -> bool:
    lowered = column.lower()
    return any(part in lowered for part in DEFAULT_IGNORE_SUBSTRINGS)


def coerce_boolean_like(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)

    lowered = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "success", "succeeded", "ok"}
    falsy = {"false", "0", "no", "n", "failed", "failure", "blocked"}
    known = lowered.dropna().unique()

    if len(known) <= 2 and set(known).issubset(truthy | falsy):
        return lowered.map(lambda value: 1 if value in truthy else 0)

    return series


def infer_outcome(
    series: pd.Series, positive_label: str | None
) -> tuple[pd.Series, OutcomeSpec]:
    series = coerce_boolean_like(series)

    if positive_label is not None:
        return (series.astype(str) == str(positive_label)).astype(int), OutcomeSpec(
            kind="binary", positive_label=positive_label
        )

    non_null = series.dropna()
    unique_values = list(pd.unique(non_null))
    if len(unique_values) == 2:
        preferred = [True, 1, "1", "true", "True", "success", "succeeded", "ok"]
        positive = next(
            (value for value in preferred if value in unique_values), unique_values[-1]
        )
        return (series == positive).astype(int), OutcomeSpec(
            kind="binary", positive_label=positive
        )

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() < max(10, int(0.5 * len(series))):
        raise ValueError(
            f"Target {series.name!r} is not numeric or binary. Use a numeric/binary target."
        )
    return numeric, OutcomeSpec(kind="continuous")


def usable_columns(
    data: pd.DataFrame,
    target: str,
    explicit_causes: Iterable[str] | None,
    ignored_columns: set[str],
    max_categorical_levels: int,
) -> list[str]:
    if explicit_causes:
        missing = [column for column in explicit_causes if column not in data.columns]
        if missing:
            raise ValueError(f"Unknown cause columns: {', '.join(missing)}")
        return [column for column in explicit_causes if column != target]

    candidates = []
    for column in data.columns:
        if (
            column == target
            or column in ignored_columns
            or is_probably_identifier(column)
        ):
            continue
        non_null = data[column].dropna()
        if non_null.empty:
            continue
        if pd.api.types.is_numeric_dtype(non_null) and not pd.api.types.is_bool_dtype(
            non_null
        ):
            candidates.append(column)
            continue
        if non_null.nunique() <= max_categorical_levels:
            candidates.append(column)
    return candidates


def build_preprocessor(
    data: pd.DataFrame, feature_columns: list[str], max_categorical_levels: int
):
    numeric_columns = []
    categorical_columns = []

    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(
            data[column]
        ) and not pd.api.types.is_bool_dtype(data[column]):
            numeric_columns.append(column)
        elif data[column].nunique(dropna=True) <= max_categorical_levels:
            categorical_columns.append(column)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_columns),
            ("cat", categorical_pipe, categorical_columns),
        ],
        remainder="drop",
    )


def fit_target_model(
    data: pd.DataFrame,
    target_values: pd.Series,
    feature_columns: list[str],
    outcome: OutcomeSpec,
    max_categorical_levels: int,
    random_state: int,
) -> tuple[Pipeline, dict]:
    valid_mask = target_values.notna()
    x = data.loc[valid_mask, feature_columns].copy()
    y = target_values.loc[valid_mask]

    preprocessor = build_preprocessor(data, feature_columns, max_categorical_levels)
    if outcome.kind == "binary":
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        stratify = None

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=random_state, stratify=stratify
    )
    pipeline.fit(x_train, y_train)

    metrics = {"rows_used": int(len(x)), "features_used": int(len(feature_columns))}
    if len(x_test) > 0:
        predictions = pipeline.predict(x_test)
        if outcome.kind == "binary":
            metrics["accuracy"] = float(accuracy_score(y_test, predictions))
            if len(set(y_test)) == 2:
                probabilities = pipeline.predict_proba(x_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        else:
            metrics["mae"] = float(mean_absolute_error(y_test, predictions))
            metrics["r2"] = float(r2_score(y_test, predictions))

    pipeline.fit(x, y)
    return pipeline, metrics


def predict_expected_target(
    model: Pipeline, rows: pd.DataFrame, outcome: OutcomeSpec
) -> np.ndarray:
    if outcome.kind == "binary":
        return model.predict_proba(rows)[:, 1]
    return model.predict(rows)


def intervention_values(
    series: pd.Series, max_categorical_levels: int
) -> tuple[str, object, object] | None:
    non_null = series.dropna()
    if non_null.empty:
        return None

    if pd.api.types.is_numeric_dtype(non_null) and not pd.api.types.is_bool_dtype(
        non_null
    ):
        low = float(non_null.quantile(0.25))
        high = float(non_null.quantile(0.75))
        if math.isclose(low, high):
            return None
        return "numeric_q25_q75", low, high

    counts = non_null.astype(str).value_counts()
    if len(counts) < 2 or len(counts) > max_categorical_levels:
        return None
    reference = counts.index[0]
    comparison = counts.index[1]
    return "categorical_top2", reference, comparison


def estimate_effects(
    data: pd.DataFrame,
    model: Pipeline,
    outcome: OutcomeSpec,
    feature_columns: list[str],
    cause_columns: list[str],
    max_categorical_levels: int,
) -> pd.DataFrame:
    baseline_rows = data[feature_columns].copy()
    baseline_expected = predict_expected_target(model, baseline_rows, outcome)
    baseline_mean = float(np.mean(baseline_expected))
    results = []

    for cause in cause_columns:
        values = intervention_values(data[cause], max_categorical_levels)
        if values is None:
            continue
        intervention_type, low_value, high_value = values

        low_rows = baseline_rows.copy()
        high_rows = baseline_rows.copy()
        low_rows[cause] = low_value
        high_rows[cause] = high_value

        low_expected = float(np.mean(predict_expected_target(model, low_rows, outcome)))
        high_expected = float(
            np.mean(predict_expected_target(model, high_rows, outcome))
        )
        effect = high_expected - low_expected

        results.append(
            {
                "cause": cause,
                "intervention_type": intervention_type,
                "low_or_reference_value": low_value,
                "high_or_comparison_value": high_value,
                "expected_target_at_low_or_reference": low_expected,
                "expected_target_at_high_or_comparison": high_expected,
                "average_causal_effect": effect,
                "absolute_effect": abs(effect),
                "baseline_expected_target": baseline_mean,
                "non_null_rows": int(data[cause].notna().sum()),
                "unique_values": int(data[cause].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(results).sort_values("absolute_effect", ascending=False)


def write_report(
    report_path: Path,
    csv_path: Path,
    target: str,
    outcome: OutcomeSpec,
    cause_columns: list[str],
    metrics: dict,
    effects: pd.DataFrame,
) -> None:
    lines = [
        "CSV Causal Effect Report",
        "========================",
        "",
        f"Input CSV: {csv_path}",
        f"Target/effect: {target}",
        f"Target kind: {outcome.kind}",
    ]
    if outcome.positive_label is not None:
        lines.append(f"Positive label: {outcome.positive_label}")
    lines.extend(
        [
            f"Candidate causes: {len(cause_columns)}",
            "",
            "Model diagnostics:",
            json.dumps(metrics, indent=2, sort_keys=True),
            "",
            "Interpretation:",
            "Each row compares the model's expected target under two interventions.",
            "For numeric causes this is do(cause = 75th percentile) minus do(cause = 25th percentile).",
            "For categorical causes this is do(cause = second-most-common value) minus do(cause = most-common value).",
            "These are causal estimates only if the CSV includes the relevant confounders.",
            "",
            "Top effects:",
        ]
    )

    if effects.empty:
        lines.append(
            "No estimable effects. Check that cause columns vary and are usable."
        )
    else:
        display_columns = [
            "cause",
            "low_or_reference_value",
            "high_or_comparison_value",
            "expected_target_at_low_or_reference",
            "expected_target_at_high_or_comparison",
            "average_causal_effect",
            "absolute_effect",
        ]
        lines.append(effects[display_columns].head(20).to_string(index=False))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.csv)
    if args.sample and len(data) > args.sample:
        data = data.sample(args.sample, random_state=args.random_state).reset_index(
            drop=True
        )

    if args.target not in data.columns:
        raise ValueError(f"Target column {args.target!r} does not exist in {args.csv}")

    ignored_columns = set(args.ignore)
    target_values, outcome = infer_outcome(data[args.target], args.positive_label)
    explicit_causes = args.causes
    if explicit_causes is None and args.csv.resolve() == DEFAULT_INPUT_CSV.resolve():
        explicit_causes = [
            column
            for column in DEFAULT_CUT_CAUSE_COLUMNS
            if column in data.columns and data[column].notna().any()
        ]

    cause_columns = usable_columns(
        data=data,
        target=args.target,
        explicit_causes=explicit_causes,
        ignored_columns=ignored_columns,
        max_categorical_levels=args.max_categorical_levels,
    )
    if not cause_columns:
        raise ValueError("No usable cause columns found. Pass --causes explicitly.")

    feature_columns = cause_columns.copy()
    model, metrics = fit_target_model(
        data=data,
        target_values=target_values,
        feature_columns=feature_columns,
        outcome=outcome,
        max_categorical_levels=args.max_categorical_levels,
        random_state=args.random_state,
    )
    effects = estimate_effects(
        data=data,
        model=model,
        outcome=outcome,
        feature_columns=feature_columns,
        cause_columns=cause_columns,
        max_categorical_levels=args.max_categorical_levels,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.output, index=False)
    write_report(
        args.report, args.csv, args.target, outcome, cause_columns, metrics, effects
    )

    print(f"Rows loaded: {len(data):,}")
    print(f"Target: {args.target} ({outcome.kind})")
    print(f"Candidate causes: {len(cause_columns)}")
    print(f"Wrote ranked effects: {args.output}")
    print(f"Wrote report: {args.report}")
    if not effects.empty:
        print()
        print(
            effects[["cause", "average_causal_effect", "absolute_effect"]]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
