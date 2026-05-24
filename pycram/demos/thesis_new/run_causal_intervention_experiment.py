"""
Run the controlled causal intervention experiment from PyCharm.

This file intentionally has no command-line arguments. Press Run in PyCharm to
execute the planned do(robot_name=...) runs.

Default plan:
    tasks       = cut, mix, wipe
    environments = kitchen, apartment, isr
    seeds       = 910001..910005
    robots      = tiago, justin, stretch, hsrb, pr2, armar7, g1

This creates true paired intervention data:
    same task + same environment + same seed + same task_instance_id
    only robot_name changes

With RESUME=True, restarting this file skips runs already completed in the
causal status log. Existing raw CSV runs are also used once to infer completed
manifest rows before the status log exists.
"""

import os

from src.causal_intervention_experiment import (
    build_manifest,
    execute_manifest,
    normalize_environment_name,
    normalize_robot_name,
    normalize_task_name,
    write_manifest,
)

TASKS = ("pour",)
ROBOTS = (
    "tiago",
    "justin",
    "hsrb",
    "pr2",
    "armar7",
    "g1",
    "garmi",
    "stretch"
)
ROBOTS=("pr2", )
ENVIRONMENTS = (
    "apartment",
    "isr",
    "kitchen"
)
SEEDS = tuple(range(910001, 910006))
RESUME = True
CHILD_RUN_TIMEOUT_S = 1800


def main() -> None:
    os.environ["THESIS_CAUSAL_CHILD_TIMEOUT_S"] = str(CHILD_RUN_TIMEOUT_S)
    manifest = build_manifest(
        tasks=tuple(normalize_task_name(task) for task in TASKS),
        robots=tuple(normalize_robot_name(robot) for robot in ROBOTS),
        environments=tuple(
            normalize_environment_name(environment) for environment in ENVIRONMENTS
        ),
        seeds=SEEDS,
    )
    write_manifest(manifest)
    execute_manifest(manifest, resume=RESUME)


if __name__ == "__main__":
    main()
