#!/usr/bin/env python3
"""
Unified causal query model for the bread-cutting CSV.

This file gives you a small API for the questions we discussed:

    model.ask_do({"object_size_z": 0.078})
    model.ask_change("object_size_z", low=0.065031, high=0.078038)
    model.ask_robot_substitution(source_robot="hsrb", target_robot="pr2")

Two estimators are combined:

1. G-computation model
   A random forest predicts final_success from pre-action variables. A do-query
   is answered by copying the population rows, setting intervention columns to
   requested values, and averaging predicted success.

2. JPT-stratified robot substitution
   For robot replacement questions, a JPT learns scene strata from geometry.
   The source robot's scene distribution is kept fixed, and the target robot's
   success rate is estimated inside matching JPT strata.

These are observational causal estimates. They are useful for thesis-level
analysis, but they still rely on assumptions: no important hidden confounders,
reasonable overlap between robots/scenes, and pre-action variables only.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jpt.trees import JPT
from jpt.variables import NumericVariable
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "probabilistic_model" / "src"))

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
    / "query_model"
)

TARGET_COLUMN = "final_success_numeric"

G_COMPUTATION_FEATURES = [
    "robot_name",
    "knowledge_query_success",
    "knowledge_cutting_tool",
    "knowledge_cutting_position",
    "knowledge_repetition",
    "required_prerequisite",
    "prerequisite_satisfied_initially",
    "autonomous_execution_feasible",
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

JPT_SCENE_FEATURES = [
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

JPT_MIN_SAMPLES_PER_LEAF = 25
SMOOTHING_STRENGTH = 2.0


@dataclass
class DoQueryResult:
    query_type: str
    intervention: dict[str, Any]
    population_filter: dict[str, Any] | None
    rows_used: int
    baseline_success_probability: float
    intervened_success_probability: float
    effect: float
    estimator: str


@dataclass
class RobotSubstitutionResult:
    query_type: str
    source_robot: str
    target_robot: str
    rows_used: int
    source_rows: int
    target_rows: int
    rf_source_scene_success_probability: float
    rf_target_on_source_scene_success_probability: float
    rf_effect: float
    jpt_source_scene_success_probability: float
    jpt_target_on_source_scene_success_probability: float
    jpt_effect: float
    jpt_strata: int
    common_support_weight: float


class CausalQueryModel:
    def __init__(self, csv_path: Path = INPUT_CSV, random_state: int = 7):
        self.csv_path = Path(csv_path)
        self.random_state = random_state
        self.data = self._load_data()
        self.feature_columns = [
            column
            for column in G_COMPUTATION_FEATURES
            if column in self.data.columns and self.data[column].notna().any()
        ]
        self.pipeline: Pipeline | None = None
        self.metrics: dict[str, float] = {}

    def _load_data(self) -> pd.DataFrame:
        data = pd.read_csv(self.csv_path)
        data[TARGET_COLUMN] = data["final_success"].astype(bool).astype(float)

        for column in set(G_COMPUTATION_FEATURES + JPT_SCENE_FEATURES):
            if column in data.columns and column != "robot_name":
                converted = pd.to_numeric(data[column], errors="coerce")
                if converted.notna().sum() >= data[column].notna().sum() * 0.5:
                    data[column] = converted

        return data.reset_index(drop=True)

    def fit(self) -> "CausalQueryModel":
        x = self.data[self.feature_columns].copy()
        y = self.data[TARGET_COLUMN].copy()

        numeric_columns = [
            column
            for column in self.feature_columns
            if pd.api.types.is_numeric_dtype(x[column])
            and not pd.api.types.is_bool_dtype(x[column])
        ]
        categorical_columns = [
            column for column in self.feature_columns if column not in numeric_columns
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_columns,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="ignore", sparse_output=False
                                ),
                            ),
                        ]
                    ),
                    categorical_columns,
                ),
            ],
            remainder="drop",
        )
        model = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        stratify = y if y.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=self.random_state, stratify=stratify
        )
        self.pipeline.fit(x_train, y_train)
        predictions = self.pipeline.predict(x_test)
        probabilities = self.pipeline.predict_proba(x_test)[:, 1]
        self.metrics = {
            "rows": float(len(self.data)),
            "features": float(len(self.feature_columns)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

        self.pipeline.fit(x, y)
        return self

    def _require_fit(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError("Call fit() before asking causal queries.")
        return self.pipeline

    def _population(
        self, population_filter: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        rows = self.data.copy()
        if population_filter:
            for column, value in population_filter.items():
                if column not in rows.columns:
                    raise ValueError(f"Unknown population filter column: {column}")
                rows = rows[rows[column] == value]
        if rows.empty:
            raise ValueError(f"Population filter produced no rows: {population_filter}")
        return rows.reset_index(drop=True)

    def _expected_success(self, rows: pd.DataFrame) -> float:
        pipeline = self._require_fit()
        probabilities = pipeline.predict_proba(rows[self.feature_columns])[:, 1]
        return float(np.mean(probabilities))

    def ask_do(
        self,
        intervention: dict[str, Any],
        population_filter: dict[str, Any] | None = None,
    ) -> DoQueryResult:
        rows = self._population(population_filter)
        missing = [
            column for column in intervention if column not in self.feature_columns
        ]
        if missing:
            raise ValueError(
                "Intervention contains unknown or non-model columns: "
                + ", ".join(missing)
            )

        baseline = self._expected_success(rows)
        intervened_rows = rows.copy()
        for column, value in intervention.items():
            intervened_rows[column] = value
        intervened = self._expected_success(intervened_rows)

        return DoQueryResult(
            query_type="do",
            intervention=intervention,
            population_filter=population_filter,
            rows_used=len(rows),
            baseline_success_probability=baseline,
            intervened_success_probability=intervened,
            effect=intervened - baseline,
            estimator="random_forest_g_computation",
        )

    def ask_change(
        self,
        variable: str,
        low: Any,
        high: Any,
        population_filter: dict[str, Any] | None = None,
    ) -> dict[str, DoQueryResult | float]:
        low_result = self.ask_do({variable: low}, population_filter)
        high_result = self.ask_do({variable: high}, population_filter)
        return {
            "low": low_result,
            "high": high_result,
            "high_minus_low": high_result.intervened_success_probability
            - low_result.intervened_success_probability,
        }

    def ask_robot_substitution(
        self,
        source_robot: str,
        target_robot: str,
    ) -> RobotSubstitutionResult:
        source_rows = self._population({"robot_name": source_robot})
        target_rows = self._population({"robot_name": target_robot})

        rf_source = self.ask_do(
            {"robot_name": source_robot},
            population_filter={"robot_name": source_robot},
        )
        rf_target = self.ask_do(
            {"robot_name": target_robot},
            population_filter={"robot_name": source_robot},
        )

        jpt_estimates = self._jpt_robot_substitution(source_robot, target_robot)
        return RobotSubstitutionResult(
            query_type="robot_substitution",
            source_robot=source_robot,
            target_robot=target_robot,
            rows_used=jpt_estimates["rows_used"],
            source_rows=len(source_rows),
            target_rows=len(target_rows),
            rf_source_scene_success_probability=rf_source.intervened_success_probability,
            rf_target_on_source_scene_success_probability=rf_target.intervened_success_probability,
            rf_effect=rf_target.intervened_success_probability
            - rf_source.intervened_success_probability,
            jpt_source_scene_success_probability=jpt_estimates["source_probability"],
            jpt_target_on_source_scene_success_probability=jpt_estimates[
                "target_probability"
            ],
            jpt_effect=jpt_estimates["target_probability"]
            - jpt_estimates["source_probability"],
            jpt_strata=jpt_estimates["jpt_strata"],
            common_support_weight=jpt_estimates["common_support_weight"],
        )

    def _jpt_robot_substitution(
        self, source_robot: str, target_robot: str
    ) -> dict[str, float | int]:
        data = self.data[
            self.data["robot_name"].isin([source_robot, target_robot])
        ].copy()
        scene_features = [
            column
            for column in JPT_SCENE_FEATURES
            if column in data.columns and data[column].notna().any()
        ]
        for column in scene_features:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(
            subset=scene_features + [TARGET_COLUMN, "robot_name"]
        ).reset_index(drop=True)

        if data["robot_name"].nunique() != 2:
            raise ValueError(
                f"Need rows for both {source_robot!r} and {target_robot!r}."
            )

        variables = [
            NumericVariable(column, precision=self._jpt_precision(data, column))
            for column in scene_features
        ]
        model = JPT(variables=variables, min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF)
        model.fit(data[scene_features])
        leaf_to_rows = self._assign_rows_to_jpt_leaves(model, data, scene_features)

        global_rates = data.groupby("robot_name")[TARGET_COLUMN].mean().to_dict()
        source_total = int((data["robot_name"] == source_robot).sum())
        source_probability = 0.0
        target_probability = 0.0
        common_support_weight = 0.0

        for row_indices in leaf_to_rows.values():
            stratum = data.iloc[row_indices]
            source_count = int((stratum["robot_name"] == source_robot).sum())
            if source_count == 0:
                continue

            target_count = int((stratum["robot_name"] == target_robot).sum())
            source_successes = float(
                stratum.loc[stratum["robot_name"] == source_robot, TARGET_COLUMN].sum()
            )
            target_successes = float(
                stratum.loc[stratum["robot_name"] == target_robot, TARGET_COLUMN].sum()
            )
            source_weight = source_count / source_total
            source_rate = self._smoothed_rate(
                source_successes, source_count, global_rates[source_robot]
            )
            target_rate = self._smoothed_rate(
                target_successes, target_count, global_rates[target_robot]
            )
            source_probability += source_weight * source_rate
            target_probability += source_weight * target_rate
            if target_count > 0:
                common_support_weight += source_weight

        return {
            "rows_used": int(len(data)),
            "source_probability": float(source_probability),
            "target_probability": float(target_probability),
            "jpt_strata": int(len(leaf_to_rows)),
            "common_support_weight": float(common_support_weight),
        }

    @staticmethod
    def _jpt_precision(data: pd.DataFrame, column: str) -> float:
        standard_deviation = float(data[column].std())
        if not np.isfinite(standard_deviation) or standard_deviation <= 0:
            return 0.001
        return max(standard_deviation * 0.01, 1e-5)

    @staticmethod
    def _smoothed_rate(successes: float, count: float, prior_rate: float) -> float:
        return float(
            (successes + SMOOTHING_STRENGTH * prior_rate) / (count + SMOOTHING_STRENGTH)
        )

    @staticmethod
    def _assign_rows_to_jpt_leaves(
        jpt_model: JPT,
        data: pd.DataFrame,
        scene_features: list[str],
    ) -> dict[int, list[int]]:
        internal_variables = list(jpt_model.varnames.values())
        data_array = data[scene_features].astype(np.float64).values
        leaf_to_row_indices: dict[int, list[int]] = {}
        for row_index, row_values in enumerate(data_array):
            row_as_dict = {
                variable: float(value)
                for variable, value in zip(internal_variables, row_values)
            }
            leaf_node = next(jpt_model.apply(row_as_dict))
            leaf_to_row_indices.setdefault(id(leaf_node), []).append(row_index)
        return leaf_to_row_indices


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    if isinstance(result, dict):
        return {
            key: (
                result_to_dict(value)
                if hasattr(value, "__dataclass_fields__")
                else value
            )
            for key, value in result.items()
        }
    return result


def main() -> None:
    model = CausalQueryModel().fit()

    queries = {
        "hsrb_to_pr2": result_to_dict(model.ask_robot_substitution("hsrb", "pr2")),
        "pr2_to_hsrb": result_to_dict(model.ask_robot_substitution("pr2", "hsrb")),
        "larger_object_size_z": result_to_dict(
            model.ask_change("object_size_z", low=0.065031, high=0.078038)
        ),
        "higher_object_world_z": result_to_dict(
            model.ask_change("object_world_z", low=0.81, high=0.966814)
        ),
        "model_metrics": model.metrics,
    }

    output_path = OUTPUT_DIR / "example_queries.json"
    write_json(output_path, queries)

    print("CausalQueryModel fitted.")
    print(json.dumps(queries, indent=2, sort_keys=True))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
