"""Showcase demo for dependency-ordered query and resolution.

This orchestrates the four thesis task demos in one narrative run:

- cutting
- mixing
- wiping
- pouring

The individual task demos already perform the heavy lifting. This file makes the
resolution story explicit for presentation: which slots are open, which subsystem
grounds them, and which concrete demo is launched.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    from thesis_new import run_thesis_demo_isolated
except ImportError:
    pycram_root = Path(__file__).resolve().parents[2]
    if str(pycram_root) not in sys.path:
        sys.path.insert(0, str(pycram_root))
    from demos.thesis_new import run_thesis_demo_isolated


@dataclass(frozen=True)
class ShowcaseTask:
    task_name: str
    title: str
    seed_offset: int
    runner_kwargs: dict = field(default_factory=dict)
    open_slots: tuple[str, ...] = ()
    resolution_trace: tuple[str, ...] = ()


SHOWCASE_TASKS: tuple[ShowcaseTask, ...] = (
    ShowcaseTask(
        task_name="cut",
        title="Cutting: object-specific tool and stroke grounding",
        seed_offset=0,
        runner_kwargs={"object_kind": "apple"},
        open_slots=(
            "target object instance",
            "cutting affordance and food-on relation",
            "tool class and mounted tool",
            "cutting position and repetition",
            "navigation pose around the object",
            "arm/tool attempt ordering",
        ),
        resolution_trace=(
            "KRROOD/SPARQL query resolves cutting knowledge for the object class.",
            "Semantic Digital Twin resolves object geometry and support relation.",
            "CostmapLocation resolves a collision-free robot pose.",
            "Execution tries primary and fallback arm/tool bindings.",
        ),
    ),
    ShowcaseTask(
        task_name="mix",
        title="Mixing: container/tool/motion profile grounding",
        seed_offset=100,
        runner_kwargs={"container_kind": "pot"},
        open_slots=(
            "container instance",
            "mixing tool",
            "motion pattern",
            "required prerequisite or assistance",
            "pickup/navigation pose",
            "retry arm",
        ),
        resolution_trace=(
            "Knowledge query resolves tool and motion profile.",
            "World geometry resolves container bounds and stirring volume.",
            "Navigation costmap resolves a reachable stance.",
            "Retry policy switches arm when the first execution binding fails.",
        ),
    ),
    ShowcaseTask(
        task_name="wipe",
        title="Wiping: surface-region and contact-path grounding",
        seed_offset=200,
        open_slots=(
            "surface/pose target",
            "reachable wipe side",
            "costmap preview",
            "navigation pose",
            "arm and contact path",
        ),
        resolution_trace=(
            "World sampling creates candidate dirty/contact regions.",
            "Height and environment filters reject invalid targets.",
            "Costmaps resolve reachable navigation poses.",
            "Execution binds arm and path for the selected surface.",
        ),
    ),
    ShowcaseTask(
        task_name="pour",
        title="Pouring: source-tool/target-container/side grounding",
        seed_offset=300,
        open_slots=(
            "target bowl instance",
            "mounted cup/source object",
            "pour side",
            "navigation pose",
            "arm/tool binding",
            "retry side",
        ),
        resolution_trace=(
            "World model resolves bowl and mounted cup candidates.",
            "Geometry checks bind source opening relative to target opening.",
            "CostmapLocation resolves pickup stance.",
            "Retry policy changes pour side when the first binding fails.",
        ),
    ),
)


def _select_tasks(names: Iterable[str] | None) -> list[ShowcaseTask]:
    if not names:
        return list(SHOWCASE_TASKS)
    requested = {name.strip().lower() for name in names}
    selected = [task for task in SHOWCASE_TASKS if task.task_name in requested]
    missing = requested - {task.task_name for task in selected}
    if missing:
        supported = ", ".join(task.task_name for task in SHOWCASE_TASKS)
        raise ValueError(
            f"Unknown showcase task(s): {sorted(missing)}. Supported: {supported}"
        )
    return selected


def _print_task_story(
    task: ShowcaseTask, *, robot_name: str, environment_name: str, seed: int
) -> None:
    print("\n" + "=" * 88)
    print(task.title)
    print("=" * 88)
    print(
        f"launch: task={task.task_name} robot={robot_name} environment={environment_name} seed={seed}"
    )
    if task.runner_kwargs:
        print(f"variant kwargs: {task.runner_kwargs}")

    print("\nOpen designator/query slots:")
    for slot in task.open_slots:
        print(f"  - {slot}")

    print("\nResolution pipeline:")
    for index, step in enumerate(task.resolution_trace, start=1):
        print(f"  {index}. {step}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a narrated thesis showcase across cutting, mixing, wiping, and pouring."
    )
    parser.add_argument("--robot", default="pr2")
    parser.add_argument("--environment", default="apartment")
    parser.add_argument("--seed", type=int, default=910001)
    parser.add_argument(
        "--task",
        action="append",
        choices=[task.task_name for task in SHOWCASE_TASKS],
        help="Run only selected task(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the query/resolution story without launching child demos.",
    )
    args = parser.parse_args()

    selected_tasks = _select_tasks(args.task)

    print("\nDependency-ordered query/resolution showcase")
    print("This demo sells the thesis claim: one symbolic task description becomes")
    print("a concrete execution by resolving object, knowledge, geometry, scene, and")
    print("controller bindings at runtime.")

    for task in selected_tasks:
        task_seed = args.seed + task.seed_offset
        _print_task_story(
            task,
            robot_name=args.robot,
            environment_name=args.environment,
            seed=task_seed,
        )
        if args.dry_run:
            print("\n[dry-run] child demo not launched")
            continue

        run_thesis_demo_isolated(
            task.task_name,
            seed=task_seed,
            robot_name=args.robot,
            environment_name=args.environment,
            **task.runner_kwargs,
        )


if __name__ == "__main__":
    main()
