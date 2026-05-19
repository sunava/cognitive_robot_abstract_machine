#!/usr/bin/env python3
"""
JPT-based causal comparison for bread cutting: PR2 vs HSRB.

This script answers the counterfactual-style question:

    If the same bread-cutting scene were executed with PR2 instead of HSRB,
    would success become more likely?

It uses a JPT as the backdoor stratification model: the JPT learns regions of
similar scene geometry, then PR2 and HSRB are compared within those same learned
regions. Categorical robot identity is encoded as a numeric intervention
variable:

    robot_is_pr2 = 1.0 for PR2, 0.0 for HSRB

The effect variable is:

    final_success_numeric = 1.0 for success, 0.0 for failure

The script trains a JPT on scene variables, uses JPT leaves as adjustment
strata, and compares:

    P(success | do(robot_is_pr2 = 1))
    P(success | do(robot_is_pr2 = 0))
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from jpt.trees import JPT
from jpt.variables import NumericVariable


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
OUTPUT_DIR = PROJECT_ROOT / "pycram" / "demos" / "thesis_new" / "records" / "causality"
RESULTS_CSV = OUTPUT_DIR / "jpt_pr2_vs_hsrb_causal_effect.csv"
REPORT_TXT = OUTPUT_DIR / "jpt_pr2_vs_hsrb_report.txt"
JPT_MODEL_PATH = OUTPUT_DIR / "jpt_pr2_vs_hsrb_scene_strata.json"

EFFECT_VARIABLE = "final_success_numeric"
ROBOT_CAUSE_VARIABLE = "robot_is_pr2"

SCENE_ADJUSTMENT_VARIABLES = [
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

CAUSE_VARIABLES = [ROBOT_CAUSE_VARIABLE]
ALL_VARIABLES = [EFFECT_VARIABLE, ROBOT_CAUSE_VARIABLE] + SCENE_ADJUSTMENT_VARIABLES

JPT_MIN_SAMPLES_PER_LEAF = 25
SMOOTHING_STRENGTH = 2.0


def load_robot_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT_CSV)
    data = data[data["robot_name"].isin(["pr2", "hsrb"])].copy()
    data[EFFECT_VARIABLE] = data["final_success"].astype(bool).astype(float)
    data[ROBOT_CAUSE_VARIABLE] = (data["robot_name"] == "pr2").astype(float)

    for column in SCENE_ADJUSTMENT_VARIABLES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=ALL_VARIABLES).reset_index(drop=True)
    if data.empty:
        raise RuntimeError("No usable PR2/HSRB rows after numeric filtering.")

    print(f"Loaded {len(data):,} PR2/HSRB bread-cutting rows.")
    print(pd.crosstab(data["robot_name"], data["final_success"]).to_string())
    return data


def variable_precision(data: pd.DataFrame, column: str) -> float:
    if column in (EFFECT_VARIABLE, ROBOT_CAUSE_VARIABLE):
        return 0.05
    standard_deviation = float(data[column].std())
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return 0.001
    return max(standard_deviation * 0.01, 1e-5)


def build_jpt_variable_list(data: pd.DataFrame) -> list[NumericVariable]:
    return [
        NumericVariable(column, precision=variable_precision(data, column))
        for column in SCENE_ADJUSTMENT_VARIABLES
    ]


def train_or_load_jpt(data: pd.DataFrame) -> tuple[JPT, list[NumericVariable]]:
    variables = build_jpt_variable_list(data)
    if JPT_MODEL_PATH.exists():
        print(f"Loading JPT model from {JPT_MODEL_PATH}.")
        model = JPT(variables=variables, min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF)
        model = model.load(str(JPT_MODEL_PATH))
        print(f"JPT leaves: {len(model.leaves)}")
        return model, variables

    print(
        f"Training JPT scene stratifier on {len(data):,} rows and "
        f"{len(SCENE_ADJUSTMENT_VARIABLES)} scene variables."
    )
    model = JPT(variables=variables, min_samples_leaf=JPT_MIN_SAMPLES_PER_LEAF)
    model.fit(data[SCENE_ADJUSTMENT_VARIABLES])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(JPT_MODEL_PATH))
    print(f"JPT leaves: {len(model.leaves)}")
    return model, variables


def assign_rows_to_jpt_leaves(
    jpt_model: JPT, data: pd.DataFrame
) -> dict[int, list[int]]:
    internal_variables = list(jpt_model.varnames.values())
    variable_names = [variable.name for variable in internal_variables]
    data_array = data[variable_names].astype(np.float64).values

    leaf_to_row_indices: dict[int, list[int]] = {}
    for row_index, row_values in enumerate(data_array):
        row_as_dict = {
            variable: float(value)
            for variable, value in zip(internal_variables, row_values)
        }
        leaf_node = next(jpt_model.apply(row_as_dict))
        leaf_to_row_indices.setdefault(id(leaf_node), []).append(row_index)

    print(f"Assigned rows to {len(leaf_to_row_indices)} unique JPT leaves.")
    return leaf_to_row_indices


def smoothed_rate(successes: float, count: float, prior_rate: float) -> float:
    return float(
        (successes + SMOOTHING_STRENGTH * prior_rate) / (count + SMOOTHING_STRENGTH)
    )


def estimate_robot_intervention_from_jpt_strata(
    jpt_model: JPT,
    data: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    leaf_to_rows = assign_rows_to_jpt_leaves(jpt_model, data)
    global_rates = data.groupby("robot_name")[EFFECT_VARIABLE].mean().to_dict()

    p_success_do_pr2 = 0.0
    p_success_do_hsrb = 0.0
    common_support_weight = 0.0
    stratum_rows = []

    for stratum_index, row_indices in enumerate(leaf_to_rows.values(), start=1):
        stratum = data.iloc[row_indices]
        stratum_weight = len(stratum) / len(data)
        by_robot = stratum.groupby("robot_name")[EFFECT_VARIABLE].agg(["count", "sum"])

        pr2_count = (
            float(by_robot.loc["pr2", "count"]) if "pr2" in by_robot.index else 0.0
        )
        pr2_successes = (
            float(by_robot.loc["pr2", "sum"]) if "pr2" in by_robot.index else 0.0
        )
        hsrb_count = (
            float(by_robot.loc["hsrb", "count"]) if "hsrb" in by_robot.index else 0.0
        )
        hsrb_successes = (
            float(by_robot.loc["hsrb", "sum"]) if "hsrb" in by_robot.index else 0.0
        )

        pr2_rate = smoothed_rate(pr2_successes, pr2_count, global_rates["pr2"])
        hsrb_rate = smoothed_rate(hsrb_successes, hsrb_count, global_rates["hsrb"])
        p_success_do_pr2 += stratum_weight * pr2_rate
        p_success_do_hsrb += stratum_weight * hsrb_rate

        has_common_support = pr2_count > 0 and hsrb_count > 0
        if has_common_support:
            common_support_weight += stratum_weight

        stratum_rows.append(
            {
                "stratum": stratum_index,
                "rows": len(stratum),
                "weight": stratum_weight,
                "pr2_rows": int(pr2_count),
                "hsrb_rows": int(hsrb_count),
                "pr2_success_rate_smoothed": pr2_rate,
                "hsrb_success_rate_smoothed": hsrb_rate,
                "pr2_minus_hsrb": pr2_rate - hsrb_rate,
                "common_support": has_common_support,
            }
        )

    estimates = {
        "p_success_do_hsrb": p_success_do_hsrb,
        "p_success_do_pr2": p_success_do_pr2,
        "pr2_minus_hsrb": p_success_do_pr2 - p_success_do_hsrb,
        "jpt_strata": len(leaf_to_rows),
        "common_support_weight": common_support_weight,
    }
    return estimates, pd.DataFrame(stratum_rows)


def write_outputs(
    data: pd.DataFrame, estimates: dict[str, float], strata: pd.DataFrame
) -> None:
    observed = (
        data.groupby("robot_name")[EFFECT_VARIABLE]
        .agg(["count", "mean"])
        .rename(columns={"mean": "observed_success_rate"})
    )

    results = pd.DataFrame(
        [
            {
                "comparison": "do(robot_is_pr2=1) - do(robot_is_pr2=0)",
                **estimates,
                "observed_pr2_success_rate": float(
                    observed.loc["pr2", "observed_success_rate"]
                ),
                "observed_hsrb_success_rate": float(
                    observed.loc["hsrb", "observed_success_rate"]
                ),
                "observed_pr2_rows": int(observed.loc["pr2", "count"]),
                "observed_hsrb_rows": int(observed.loc["hsrb", "count"]),
                "jpt_strata": int(estimates["jpt_strata"]),
                "common_support_weight": float(estimates["common_support_weight"]),
            }
        ]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    strata.to_csv(OUTPUT_DIR / "jpt_pr2_vs_hsrb_strata.csv", index=False)

    lines = [
        "JPT PR2 vs HSRB Causal Report",
        "==============================",
        "",
        f"Input CSV: {INPUT_CSV}",
        f"Rows used: {len(data)}",
        "",
        "Observed success rates:",
        observed.to_string(),
        "",
        "JPT-stratified backdoor causal estimate:",
        f"P(success | do(robot = HSRB)): {estimates['p_success_do_hsrb']:.4f}",
        f"P(success | do(robot = PR2)) : {estimates['p_success_do_pr2']:.4f}",
        f"PR2 - HSRB effect           : {estimates['pr2_minus_hsrb']:.4f}",
        f"JPT scene strata            : {int(estimates['jpt_strata'])}",
        f"Common-support stratum mass : {estimates['common_support_weight']:.4f}",
        "",
        "Interpretation:",
        "A positive effect means PR2 is estimated to be more likely to succeed",
        "than HSRB under comparable scene geometry in this CSV.",
        "",
        "Caveat:",
        "The JPT learns scene strata from observed geometry. Within each stratum,",
        "robot-specific success rates are smoothed and averaged by stratum mass.",
        "This is a JPT-derived observational causal estimate, not a physical",
        "simulator or guaranteed counterfactual truth.",
    ]
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(results.to_string(index=False))
    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {REPORT_TXT}")


def main() -> None:
    data = load_robot_data()
    jpt_model, _ = train_or_load_jpt(data)
    estimates, strata = estimate_robot_intervention_from_jpt_strata(jpt_model, data)
    write_outputs(data, estimates, strata)


if __name__ == "__main__":
    main()
