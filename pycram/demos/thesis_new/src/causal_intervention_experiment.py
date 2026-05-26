#!/usr/bin/env python3
"""
Controlled interventional experiment for thesis causal learning.

Goal
----
Create data where causal questions are identifiable by design, not only by
observational assumptions.

Robot substitution intervention:

    do(robot_name = r)

For each task, environment, and seed, the same task instances are run for every
robot. Existing task logs contain one row per object/target. After execution,
rows are paired by:

    task_name + environment_name + seed + task_instance_id

This creates a proper interventional block:

    causal_instance_id, robot_name, final_success

where the scene is held fixed by the seed/environment and only the robot is
changed. That is the dataset you can use to claim genuine simulated
interventions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RECORDS_DIR = Path(__file__).resolve().parents[1] / "records"
EXPERIMENT_DIR = RECORDS_DIR / "causal_intervention"
RAW_INTERVENTION_RESULTS = EXPERIMENT_DIR / "raw_cutting_intervention_results2.csv"
RAW_RESULTS_BY_TASK = {
    "cut": EXPERIMENT_DIR / "raw_cutting_intervention_results2.csv",
    "pour": EXPERIMENT_DIR / "raw_pouring_intervention_results2.csv",
    "wipe": EXPERIMENT_DIR / "raw_wiping_intervention_results2.csv",
    "mix": EXPERIMENT_DIR / "raw_mixing_intervention_results2.csv",
}
RESULTS_ENV_BY_TASK = {
    "cut": "THESIS_CUT_RESULTS_CSV_PATH",
    "pour": "THESIS_POUR_RESULTS_CSV_PATH",
    "wipe": "THESIS_WIPE_RESULTS_CSV_PATH",
    "mix": "THESIS_MIX_RESULTS_CSV_PATH",
}
MANIFEST_CSV = EXPERIMENT_DIR / "robot_substitution_manifest.csv"
PAIRED_DATASET_CSV = EXPERIMENT_DIR / "paired_robot_substitution_dataset.csv"
PAIRWISE_EFFECTS_CSV = EXPERIMENT_DIR / "paired_robot_substitution_effects.csv"
SUMMARY_JSON = EXPERIMENT_DIR / "robot_substitution_summary.json"
RUN_STATUS_CSV = EXPERIMENT_DIR / "robot_substitution_run_status.csv"
DEFAULT_CHILD_RUN_TIMEOUT_S = 1800
RUN_STATUS_COLUMNS = [
    "timestamp_utc",
    "status",
    "causal_experiment_id",
    "task_name",
    "environment_name",
    "seed",
    "robot_name",
    "returncode",
    "message",
]
PAIRED_DATASET_BY_TASK = {
    task_name: EXPERIMENT_DIR / f"paired_{task_name}_robot_substitution_dataset.csv"
    for task_name in RAW_RESULTS_BY_TASK
}
PAIRWISE_EFFECTS_BY_TASK = {
    task_name: EXPERIMENT_DIR / f"paired_{task_name}_robot_substitution_effects.csv"
    for task_name in RAW_RESULTS_BY_TASK
}
SUMMARY_BY_TASK = {
    task_name: EXPERIMENT_DIR / f"{task_name}_robot_substitution_summary.json"
    for task_name in RAW_RESULTS_BY_TASK
}

DEFAULT_ROBOTS = (
    "pr2",
    "hsrb",
    "tiago",
    "stretch",
    "armar7",
    "rollin_justin",
    "unitree_g1",
    "garmi"
)
DEFAULT_ENVIRONMENTS = ("apartment", "kitchen", "isr")
DEFAULT_SEEDS = tuple(range(910001, 910011))
DEFAULT_TASKS = ("cut",)
TASK_ALIASES = {
    "cut": "cut",
    "cutting": "cut",
    "bread_cutting": "cut",
    "pour": "pour",
    "pouring": "pour",
    "wipe": "wipe",
    "wiping": "wipe",
    "space_wiping": "wipe",
    "mix": "mix",
    "mixing": "mix",
    "bowl_mixing": "mix",
}


@dataclass(frozen=True)
class InterventionRun:
    causal_experiment_id: str
    intervention_family: str
    treatment_variable: str
    treatment_value: str
    task_name: str
    environment_name: str
    seed: int
    robot_name: str
    expected_instance_key: str


def normalize_robot_name(robot_name: object) -> str:
    mapping = {
        "justin": "rollin_justin",
        "g1": "unitree_g1",
    }
    value = str(robot_name).strip().lower()
    return mapping.get(value, value)


def normalize_environment_name(environment_name: object) -> str:
    return str(environment_name).strip().lower()


def normalize_task_name(task_name: object) -> str:
    value = str(task_name).strip().lower()
    if value not in TASK_ALIASES:
        supported = ", ".join(sorted(TASK_ALIASES))
        raise ValueError(f"Unsupported task '{task_name}'. Supported: {supported}")
    return TASK_ALIASES[value]


def raw_results_path_for_task(task_name: object) -> Path:
    return RAW_RESULTS_BY_TASK[normalize_task_name(task_name)]


def build_manifest(
    *,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    robots: tuple[str, ...] = DEFAULT_ROBOTS,
    environments: tuple[str, ...] = DEFAULT_ENVIRONMENTS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> pd.DataFrame:
    runs = []
    for task_name in tasks:
        normalized_task = normalize_task_name(task_name)
        for environment_name in environments:
            normalized_environment = normalize_environment_name(environment_name)
            for seed in seeds:
                causal_experiment_id = (
                    f"robot_substitution:{normalized_task}:{normalized_environment}:{seed}"
                )
                for robot_name in robots:
                    normalized_robot = normalize_robot_name(robot_name)
                    runs.append(
                        InterventionRun(
                            causal_experiment_id=causal_experiment_id,
                            intervention_family="robot_substitution",
                            treatment_variable="robot_name",
                            treatment_value=normalized_robot,
                            task_name=normalized_task,
                            environment_name=normalized_environment,
                            seed=int(seed),
                            robot_name=normalized_robot,
                            expected_instance_key=(
                                f"{normalized_task}:{normalized_environment}:{seed}:<task_instance_id>"
                            ),
                        )
                    )
    return pd.DataFrame([asdict(run) for run in runs])


def write_manifest(manifest: pd.DataFrame) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_CSV, index=False)
    print(f"Wrote intervention manifest: {MANIFEST_CSV}")
    print(f"Planned runs: {len(manifest)}")


def run_key(*, task_name: object, environment_name: object, seed: object, robot_name: object):
    return (
        normalize_task_name(task_name),
        normalize_environment_name(environment_name),
        int(seed),
        normalize_robot_name(robot_name),
    )


def manifest_row_key(row) -> tuple[str, str, int, str]:
    return run_key(
        task_name=row.task_name,
        environment_name=row.environment_name,
        seed=row.seed,
        robot_name=row.robot_name,
    )


def append_run_status(row, status: str, *, returncode: int | str = "", message: str = "") -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    status_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "causal_experiment_id": row.causal_experiment_id,
        "task_name": normalize_task_name(row.task_name),
        "environment_name": normalize_environment_name(row.environment_name),
        "seed": int(row.seed),
        "robot_name": normalize_robot_name(row.robot_name),
        "returncode": returncode,
        "message": message,
    }
    status_df = pd.DataFrame([status_row], columns=RUN_STATUS_COLUMNS)
    status_df.to_csv(
        RUN_STATUS_CSV,
        mode="a",
        header=not RUN_STATUS_CSV.exists() or RUN_STATUS_CSV.stat().st_size == 0,
        index=False,
    )


def load_run_status_events() -> pd.DataFrame:
    if not RUN_STATUS_CSV.exists() or RUN_STATUS_CSV.stat().st_size == 0:
        return pd.DataFrame(columns=RUN_STATUS_COLUMNS)
    events = pd.read_csv(RUN_STATUS_CSV)
    for column in RUN_STATUS_COLUMNS:
        if column not in events.columns:
            events[column] = ""
    events["seed"] = pd.to_numeric(events["seed"], errors="coerce").astype("Int64")
    events["task_name"] = events["task_name"].map(normalize_task_name)
    events["environment_name"] = events["environment_name"].map(
        normalize_environment_name
    )
    events["robot_name"] = events["robot_name"].map(normalize_robot_name)
    return events


def latest_status_by_run() -> dict[tuple[str, str, int, str], str]:
    events = load_run_status_events()
    latest = {}
    for event in events.itertuples():
        if pd.isna(event.seed):
            continue
        latest[
            run_key(
                task_name=event.task_name,
                environment_name=event.environment_name,
                seed=event.seed,
                robot_name=event.robot_name,
            )
        ] = str(event.status)
    return latest


def infer_completed_run_keys_from_raw_results(
    manifest: pd.DataFrame,
) -> set[tuple[str, str, int, str]]:
    inferred = set()
    for task_name in sorted(manifest["task_name"].map(normalize_task_name).unique()):
        raw_path = raw_results_path_for_task(task_name)
        if not raw_path.exists() or raw_path.stat().st_size == 0:
            continue
        try:
            raw = load_raw_results(raw_path)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        raw = raw[raw["task_name_normalized"] == task_name].copy()
        if raw.empty or "task_instance_id" not in raw.columns:
            continue
        raw = raw.dropna(
            subset=[
                "environment_name_normalized",
                "seed",
                "robot_name_normalized",
                "task_instance_id",
            ]
        )
        counts = (
            raw.groupby(
                [
                    "environment_name_normalized",
                    "seed",
                    "robot_name_normalized",
                ]
            )["task_instance_id"]
            .nunique()
            .reset_index(name="row_count")
        )
        block_max = counts.groupby(
            ["environment_name_normalized", "seed"]
        )["row_count"].transform("max")
        block_robot_count = counts.groupby(
            ["environment_name_normalized", "seed"]
        )["robot_name_normalized"].transform("nunique")
        completed = counts[
            (counts["row_count"] > 0)
            & (counts["row_count"] == block_max)
            & (block_robot_count > 1)
        ]
        for row in completed.itertuples():
            inferred.add(
                run_key(
                    task_name=task_name,
                    environment_name=row.environment_name_normalized,
                    seed=row.seed,
                    robot_name=row.robot_name_normalized,
                )
            )
    manifest_keys = {manifest_row_key(row) for row in manifest.itertuples()}
    return inferred & manifest_keys


def remove_raw_rows_for_run(row) -> int:
    raw_path = raw_results_path_for_task(row.task_name)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return 0
    raw = pd.read_csv(raw_path)
    if raw.empty:
        return 0
    seed = pd.to_numeric(raw.get("seed"), errors="coerce")
    robot = raw.get("robot_name", pd.Series("", index=raw.index)).map(
        normalize_robot_name
    )
    environment = raw.get("world_name", pd.Series("", index=raw.index)).map(
        normalize_environment_name
    )
    task = raw.get("task_name", pd.Series("", index=raw.index)).map(
        normalize_task_name
    )
    key = manifest_row_key(row)
    matching = (
        (task == key[0])
        & (environment == key[1])
        & (seed == key[2])
        & (robot == key[3])
    )
    removed = int(matching.sum())
    if removed:
        raw.loc[~matching].to_csv(raw_path, index=False)
    return removed


def execute_manifest(manifest: pd.DataFrame, *, resume: bool = True) -> None:
    latest_status = latest_status_by_run() if resume else {}
    completed_keys = {
        key for key, status in latest_status.items() if status == "completed"
    }
    inferred_keys = set()
    if resume:
        inferred_keys = infer_completed_run_keys_from_raw_results(manifest)
        completed_keys |= inferred_keys
        remaining = len(manifest) - len(
            {manifest_row_key(row) for row in manifest.itertuples()} & completed_keys
        )
        print(
            "[causal-resume] "
            f"status-completed={len(completed_keys - inferred_keys)} "
            f"raw-inferred-completed={len(inferred_keys)} "
            f"remaining={remaining} "
            f"status-log={RUN_STATUS_CSV}"
        )

    for index, row in manifest.iterrows():
        key = manifest_row_key(row)
        if resume and key in completed_keys:
            print(
                "[causal-resume] skip completed "
                f"{index + 1}/{len(manifest)} task={row.task_name} "
                f"env={row.environment_name} seed={row.seed} robot={row.robot_name}"
            )
            continue
        if resume:
            removed = remove_raw_rows_for_run(row)
            if removed:
                print(
                    "[causal-resume] removed incomplete raw rows "
                    f"task={row.task_name} env={row.environment_name} "
                    f"seed={row.seed} robot={row.robot_name} rows={removed}"
                )
        print(
            "[causal-run] "
            f"{index + 1}/{len(manifest)} "
            f"task={row.task_name} env={row.environment_name} "
            f"seed={row.seed} robot={row.robot_name}"
        )
        append_run_status(row, "started")
        result = run_task_intervention_isolated(
            task_name=row.task_name,
            seed=int(row.seed),
            robot_name=row.robot_name,
            environment_name=row.environment_name,
        )
        if result.returncode:
            removed = remove_raw_rows_for_run(row)
            append_run_status(
                row,
                "failed",
                returncode=int(result.returncode),
                message=(
                    "child process returned non-zero; "
                    f"removed_raw_rows={removed}"
                ),
            )
        else:
            append_run_status(row, "completed", returncode=0)


def run_cutting_intervention_isolated(
    *, seed: int, robot_name: str, environment_name: str
) -> subprocess.CompletedProcess:
    return run_task_intervention_isolated(
        task_name="cut",
        seed=seed,
        robot_name=robot_name,
        environment_name=environment_name,
    )


def run_task_intervention_isolated(
    *, task_name: str, seed: int, robot_name: str, environment_name: str
) -> subprocess.CompletedProcess:
    normalized_task = normalize_task_name(task_name)
    thesis_new_dir = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[4]
    local_python_paths = [
        repo_root / "pycram",
        thesis_new_dir,
        repo_root / "pycram" / "src",
        repo_root / "giskardpy" / "src",
        repo_root / "semantic_digital_twin" / "src",
        repo_root / "krrood" / "src",
        repo_root / "random_events" / "src",
        repo_root / "probabilistic_model" / "src",
        repo_root / "physics_simulators" / "src",
    ]
    existing_python_paths = [
        Path(path)
        for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if path
    ]
    child_python_paths = local_python_paths + existing_python_paths
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    code = (
        "import faulthandler, sys; "
        "faulthandler.enable(); "
        "faulthandler.dump_traceback_later(180, repeat=False); "
        f"sys.path[:0] = {[str(path) for path in local_python_paths]!r}; "
        "from demos.thesis_new.src.demo_runners import run_thesis_demo; "
        "run_thesis_demo("
        f"task_name={normalized_task!r}, "
        f"seed={int(seed)!r}, "
        f"robot_name={str(robot_name)!r}, "
        f"environment_name={str(environment_name)!r}"
        ")"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in child_python_paths)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp")
    env[RESULTS_ENV_BY_TASK[normalized_task]] = str(
        raw_results_path_for_task(normalized_task)
    )
    child_run_timeout_s = int(
        os.environ.get("THESIS_CAUSAL_CHILD_TIMEOUT_S", DEFAULT_CHILD_RUN_TIMEOUT_S)
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            timeout=child_run_timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(
            "[causal-run] child wall-clock timeout; continuing "
            f"(task={normalized_task}, seed={seed}, robot={robot_name}, "
            f"environment={environment_name}, timeout_s={child_run_timeout_s})"
        )
        result = subprocess.CompletedProcess(
            args=[sys.executable, "-c", code],
            returncode=124,
        )
    if result.returncode:
        print(
            "[causal-run] child failed; continuing "
            f"(task={normalized_task}, seed={seed}, robot={robot_name}, "
            f"environment={environment_name}, "
            f"returncode={result.returncode})"
        )
    return result


def load_raw_results(raw_results_path: Path = RAW_INTERVENTION_RESULTS) -> pd.DataFrame:
    if not raw_results_path.exists():
        raise FileNotFoundError(
            f"Raw intervention results do not exist yet: {raw_results_path}"
        )
    results = pd.read_csv(raw_results_path)
    results["seed"] = pd.to_numeric(results["seed"], errors="coerce").astype("Int64")
    results["robot_name_normalized"] = results["robot_name"].map(normalize_robot_name)
    results["environment_name_normalized"] = results["world_name"].map(
        normalize_environment_name
    )
    if "task_instance_id" not in results.columns:
        for candidate in ("bread_name", "target_name", "bowl_name"):
            if candidate in results.columns:
                results["task_instance_id"] = results[candidate]
                break
    results["task_name_normalized"] = results["task_name"].map(normalize_task_name)
    return results


def build_paired_dataset(
    *,
    task_name: str = "cut",
    manifest_path: Path = MANIFEST_CSV,
    raw_results_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    normalized_task = normalize_task_name(task_name)
    if raw_results_path is None:
        raw_results_path = raw_results_path_for_task(normalized_task)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist yet: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["task_name"].map(normalize_task_name) == normalized_task].copy()
    manifest["seed"] = pd.to_numeric(manifest["seed"], errors="coerce").astype("Int64")
    manifest["robot_name_normalized"] = manifest["robot_name"].map(normalize_robot_name)
    manifest["environment_name_normalized"] = manifest["environment_name"].map(
        normalize_environment_name
    )
    manifest["task_name_normalized"] = manifest["task_name"].map(normalize_task_name)
    if manifest.empty:
        raise RuntimeError(f"No manifest rows for task '{normalized_task}'.")

    results = load_raw_results(raw_results_path)
    results = results[results["task_name_normalized"] == normalized_task].copy()
    merged = results.merge(
        manifest[
            [
                "causal_experiment_id",
                "intervention_family",
                "treatment_variable",
                "treatment_value",
                "task_name_normalized",
                "seed",
                "robot_name_normalized",
                "environment_name_normalized",
            ]
        ],
        on=[
            "task_name_normalized",
            "seed",
            "robot_name_normalized",
            "environment_name_normalized",
        ],
        how="inner",
    )

    if merged.empty:
        raise RuntimeError(
            "No rows matched the manifest. Run the experiment first or check seeds/robots/environments."
        )

    merged["causal_instance_id"] = (
        merged["task_name_normalized"].astype(str)
        + ":"
        + merged["environment_name_normalized"].astype(str)
        + ":"
        + merged["seed"].astype(str)
        + ":"
        + merged["task_instance_id"].astype(str)
    )
    merged["do_variable"] = merged["treatment_variable"]
    merged["do_value"] = merged["treatment_value"]
    merged["is_interventional"] = True
    merged["final_success_numeric"] = merged["final_success"].astype(bool).astype(int)

    block_sizes = merged.groupby("causal_instance_id")["do_value"].nunique()
    complete_robot_count = manifest["robot_name_normalized"].nunique()
    complete_blocks = block_sizes[block_sizes == complete_robot_count].index
    merged["complete_intervention_block"] = merged["causal_instance_id"].isin(
        complete_blocks
    )

    complete = merged[merged["complete_intervention_block"]].copy()
    pairwise_effects = pd.DataFrame()
    if not complete.empty:
        outcome_table = complete.pivot_table(
            index="causal_instance_id",
            columns="do_value",
            values="final_success_numeric",
            aggfunc="max",
        )
        rows = []
        robots = sorted(manifest["robot_name_normalized"].dropna().unique().tolist())
        for source_robot in robots:
            for target_robot in robots:
                if source_robot == target_robot:
                    continue
                if (
                    source_robot not in outcome_table.columns
                    or target_robot not in outcome_table.columns
                ):
                    continue
                delta = outcome_table[target_robot] - outcome_table[source_robot]
                rows.append(
                    {
                        "source_robot": source_robot,
                        "target_robot": target_robot,
                        "causal_estimand": f"E[Y(do({target_robot})) - Y(do({source_robot}))]",
                        "paired_ate": float(delta.mean()),
                        "instances": int(delta.notna().sum()),
                        "source_success_rate": float(
                            outcome_table[source_robot].mean()
                        ),
                        "target_success_rate": float(
                            outcome_table[target_robot].mean()
                        ),
                        "target_better_count": int((delta > 0).sum()),
                        "source_better_count": int((delta < 0).sum()),
                        "same_outcome_count": int((delta == 0).sum()),
                    }
                )
        pairwise_effects = pd.DataFrame(rows)

    summary = {
        "manifest_runs": int(len(manifest)),
        "matched_rows": int(len(merged)),
        "causal_instances": int(merged["causal_instance_id"].nunique()),
        "complete_causal_instances": int(len(complete_blocks)),
        "robots_per_complete_block": int(complete_robot_count),
        "robots": sorted(manifest["robot_name_normalized"].dropna().unique().tolist()),
        "environments": sorted(
            manifest["environment_name_normalized"].dropna().unique().tolist()
        ),
        "seeds": [
            int(seed) for seed in sorted(manifest["seed"].dropna().unique().tolist())
        ],
        "success_rate_by_robot": {
            str(robot): float(rate)
            for robot, rate in merged.groupby("do_value")["final_success_numeric"]
            .mean()
            .items()
        },
    }
    if not pairwise_effects.empty:
        summary["paired_ate_by_substitution"] = {
            f"{row.source_robot}->{row.target_robot}": float(row.paired_ate)
            for row in pairwise_effects.itertuples()
        }

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    paired_dataset_csv = PAIRED_DATASET_BY_TASK[normalized_task]
    pairwise_effects_csv = PAIRWISE_EFFECTS_BY_TASK[normalized_task]
    summary_json = SUMMARY_BY_TASK[normalized_task]
    merged.to_csv(paired_dataset_csv, index=False)
    if not pairwise_effects.empty:
        pairwise_effects.to_csv(pairwise_effects_csv, index=False)
    if normalized_task == "cut":
        merged.to_csv(PAIRED_DATASET_CSV, index=False)
        if not pairwise_effects.empty:
            pairwise_effects.to_csv(PAIRWISE_EFFECTS_CSV, index=False)
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if normalized_task == "cut":
        SUMMARY_JSON.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return merged, summary


def parse_csv_list(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_seed_range(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value:
        return default
    if ":" in value:
        start, stop = value.split(":", 1)
        return tuple(range(int(start), int(stop)))
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="cut",
        help="Comma-separated tasks: cut,wipe,mix. Default: cut.",
    )
    parser.add_argument(
        "--robots",
        default="pr2,hsrb",
        help="Comma-separated treatment robots. Default keeps the experiment small: pr2,hsrb.",
    )
    parser.add_argument(
        "--environments",
        default="apartment",
        help="Comma-separated environments. Default: apartment.",
    )
    parser.add_argument(
        "--seeds",
        default="910001:910006",
        help="Comma-separated seeds or Python-style start:stop range. Default: 910001:910006.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the intervention manifest. Without this, only the manifest is written.",
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Build paired dataset from existing raw results and manifest.",
    )
    args = parser.parse_args()

    manifest = build_manifest(
        tasks=tuple(
            normalize_task_name(task)
            for task in parse_csv_list(args.tasks, DEFAULT_TASKS)
        ),
        robots=tuple(
            normalize_robot_name(robot)
            for robot in parse_csv_list(args.robots, DEFAULT_ROBOTS)
        ),
        environments=tuple(
            normalize_environment_name(environment)
            for environment in parse_csv_list(args.environments, DEFAULT_ENVIRONMENTS)
        ),
        seeds=parse_seed_range(args.seeds, DEFAULT_SEEDS),
    )
    write_manifest(manifest)

    if args.execute:
        execute_manifest(manifest)

    if args.build_dataset:
        for task_name in sorted(manifest["task_name"].dropna().unique()):
            try:
                _, summary = build_paired_dataset(task_name=task_name)
            except (RuntimeError, FileNotFoundError) as exc:
                print(f"Could not build paired dataset for {task_name} yet: {exc}")
                print("Run with --execute first, then rerun with --build-dataset.")
                print(
                    "Expected raw intervention results at: "
                    f"{raw_results_path_for_task(task_name)}"
                )
            else:
                print(json.dumps(summary, indent=2, sort_keys=True))
                print(
                    "Wrote paired interventional dataset: "
                    f"{PAIRED_DATASET_BY_TASK[task_name]}"
                )
                print(
                    f"Wrote paired causal effects: {PAIRWISE_EFFECTS_BY_TASK[task_name]}"
                )
                print(f"Wrote summary: {SUMMARY_BY_TASK[task_name]}")


if __name__ == "__main__":
    main()
