import argparse
import json
import subprocess
import sys

from .demo_cut_all_breads_retry import main_cutting

from .demo_mix_all_bowls_retry import main_mixing

from .demo_wipe_all_spaces_retry import main_wiping
from .world_setup import resolve_robot_name
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

THESIS_DEMO_RUNNERS = {
    "cut": main_cutting,
    "cutting": main_cutting,
    "mix": main_mixing,
    "mixing": main_mixing,
    "wipe": main_wiping,
    "wiping": main_wiping,
}


def get_thesis_demo_runner(task_name):
    normalized = str(task_name).strip().lower()
    if normalized not in THESIS_DEMO_RUNNERS:
        supported = ", ".join(sorted(THESIS_DEMO_RUNNERS))
        raise ValueError(
            f"Unsupported thesis_new demo '{task_name}'. Supported: {supported}"
        )
    return THESIS_DEMO_RUNNERS[normalized]


def run_thesis_demo(
    task_name, *, seed=None, robot_name=None, environment_name=None, **runner_kwargs
):
    runner = get_thesis_demo_runner(task_name)
    resolved_robot_name = resolve_robot_name(robot_name)
    return runner(
        seed=seed,
        robot_name=resolved_robot_name,
        environment_name=environment_name,
        **runner_kwargs,
    )


def run_thesis_demo_isolated(
    task_name, *, seed=None, robot_name=None, environment_name=None, **runner_kwargs
):
    command = [
        sys.executable,
        "-m",
        __name__,
        "--task",
        str(task_name),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if robot_name is not None:
        command.extend(["--robot", str(robot_name)])
    if environment_name is not None:
        command.extend(["--environment", str(environment_name)])
    if runner_kwargs:
        command.extend(["--runner-kwargs-json", json.dumps(runner_kwargs)])

    result = subprocess.run(command)
    if result.returncode:
        print(
            "[demo isolated] child failed; continuing "
            f"(task={task_name}, robot={robot_name}, environment={environment_name}, "
            f"returncode={result.returncode})"
        )
    return result


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--robot", default=None)
    parser.add_argument("--environment", default=None)
    parser.add_argument("--runner-kwargs-json", default="{}")
    args = parser.parse_args()

    runner_kwargs = json.loads(args.runner_kwargs_json)
    run_thesis_demo(
        args.task,
        seed=args.seed,
        robot_name=args.robot,
        environment_name=args.environment,
        **runner_kwargs,
    )


if __name__ == "__main__":
    _main()
