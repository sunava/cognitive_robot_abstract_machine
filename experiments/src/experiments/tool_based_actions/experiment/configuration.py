"""
Configuration of the tool-based action experiment.

An experiment is a grid of trials: every configured task is run once per seed. Each
trial spawns its targets at seeded random poses, so rerunning a trial specification
reproduces the exact same scene.

The configuration is a plain dataclass, so
:func:`krrood.adapters.json_serializer.to_json` and
:func:`krrood.adapters.json_serializer.from_json` serialize it without any custom code,
e.g. for handing it to a trial subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from krrood.utils import recursive_subclasses
from typing_extensions import List, Type

from experiments.tool_based_actions.experiment.task_definitions import (
    ToolTaskDefinition,
)


@dataclass(frozen=True)
class TrialSpecification:
    """
    One fully reproducible trial of the experiment grid.
    """

    task: Type[ToolTaskDefinition]
    """
    The tool-based task the trial runs.
    """

    seed: int
    """
    Seed that fixes the sampled scene of this trial.
    """

    environment_name: str
    """
    Name of the environment the trial runs in.
    """

    @property
    def identifier(self) -> str:
        """
        :return: A unique, human-readable identifier of this trial.
        """
        return f"{self.task.task_name()}:{self.environment_name}:{self.seed}"


@dataclass(frozen=True)
class ExperimentConfiguration:
    """
    The full configuration of one experiment campaign.
    """

    tasks: List[Type[ToolTaskDefinition]] = field(
        default_factory=lambda: recursive_subclasses(ToolTaskDefinition)
    )
    """
    The tasks to run.
    """

    seeds: List[int] = field(default_factory=lambda: [910001, 910002, 910003])
    """
    The seeds to run every task with.
    """

    environment_name: str = "apartment"
    """
    Name of the environment the trials run in, recorded with every result.
    """

    minimum_targets_per_trial: int = 2
    """
    Smallest number of targets a trial spawns.
    """

    maximum_targets_per_trial: int = 30
    """
    Largest number of targets a trial spawns.
    """

    targets_per_square_meter: float = 12.0
    """
    Target density on the spawn surfaces, clamped to the per-trial minimum and maximum.
    """

    target_clearance: float = 0.35
    """
    Minimum center distance in meters between two spawned targets.
    """

    footprint_clearance: float = 0.03
    """
    Minimum free gap in meters between the footprints of two spawned targets.
    """

    scale_choices: List[float] = field(
        default_factory=lambda: [0.8, 1.0, 1.2, 1.4, 1.6]
    )
    """
    Uniform scale factors a spawned target is randomly sized with.
    """

    footprint_safety_factor: float = 1.08
    """
    Factor the measured target footprint radius is inflated by during placement.
    """

    maximum_spawn_height: float = 1.35
    """
    Highest surface top in meters, in the world frame, targets are spawned on.
    """

    full_body_motion: bool = True
    """
    Allow the robot base to drive along during tool motions instead of keeping it fixed
    at the navigated pose.
    """

    tool_path_pointer_stride: int = 10
    """
    Keep every Nth sampled tool path waypoint for execution.
    """

    collision_avoidance: bool = True
    """
    Avoid collisions with the environment during motions.

    The tool motions themselves exempt the acting arm's manipulator, including the
    mounted tool, so the tool can touch its target.
    """

    surface_names: List[str] = field(
        default_factory=lambda: [
            "island_countertop",
            "countertop",
            "table_area_main",
        ]
    )
    """
    Names of the support surface bodies targets are spawned on.
    """

    surface_margin: float = 0.15
    """
    Distance in meters kept from every surface edge when spawning.
    """

    spawn_height_offset: float = 0.05
    """
    Height in meters above a surface top at which targets are spawned.
    """

    results_file: Path = field(
        default_factory=lambda: Path(__file__).parent
        / "records"
        / "tool_based_experiment_results.jsonl"
    )
    """
    File the trial results are appended to, one JSON object per line.
    """

    trial_timeout: timedelta = timedelta(hours=1)
    """
    Wall-clock limit for a single trial process, sized for the density-based target
    counts.
    """

    def build_trial_specifications(self) -> List[TrialSpecification]:
        """
        :return: The full trial grid, one specification per task and seed.
        """
        return [
            TrialSpecification(
                task=task,
                seed=seed,
                environment_name=self.environment_name,
            )
            for task in self.tasks
            for seed in self.seeds
        ]
