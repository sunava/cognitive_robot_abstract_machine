"""
Run the controlled causal intervention experiment from PyCharm.

This file intentionally has no command-line arguments. Press Run in PyCharm to
execute the planned do(robot_name=...) runs.

Default plan:
    tasks       = pour
    environments = kitchen, apartment, isr
    seeds       = 910001..910005
    robots      = tiago, justin, stretch, hsrb, pr2, armar7, g1, garmi

Additional replay:
    tasks       = cut, mix
    environments = kitchen, apartment, isr
    seeds       = 910001..910005
    robots      = garmi

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

TASKS = ( "mix",)
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
ROBOTS = ("stretch",)
ENVIRONMENTS = (
    "apartment",
    "isr",
    "kitchen"
)
ENVIRONMENTS = (
    "kitchen",
)
SEEDS = tuple(range(910001, 910006))
RESUME = True
CHILD_RUN_TIMEOUT_S = 3000
ENABLE_POUR_PARTICLES = True
POUR_PARTICLE_COUNT = 32
POUR_PARTICLE_RADIUS = 0.007


def main() -> None:
    os.environ["THESIS_CAUSAL_CHILD_TIMEOUT_S"] = str(CHILD_RUN_TIMEOUT_S)
    if ENABLE_POUR_PARTICLES:
        os.environ["THESIS_POUR_PARTICLES"] = "1"
        os.environ["THESIS_POUR_PARTICLE_COUNT"] = str(POUR_PARTICLE_COUNT)
        os.environ["THESIS_POUR_PARTICLE_RADIUS"] = str(POUR_PARTICLE_RADIUS)
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
