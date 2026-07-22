"""
Configuration of the tool-based action experiment.

An experiment is a grid of trials: every configured task is run once per seed. Each
trial spawns its targets at seeded random poses, so rerunning a trial specification
reproduces the exact same scene.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

from krrood.adapters.json_serializer import SubclassJSONSerializer
from semantic_digital_twin.world_description.geometry import BoundingBox
from typing_extensions import Any, Dict, List, Tuple


class ToolBasedTask(enum.Enum):
    """
    The tool-based composite tasks the experiment can run.
    """

    CUTTING = "cutting"
    MIXING = "mixing"
    POURING = "pouring"
    WIPING = "wiping"


@dataclass(frozen=True)
class SpawnRegion:
    """
    An axis-aligned rectangle on a support surface in which targets are spawned.

    Built from a measured :class:`~semantic_digital_twin.world_description.geometry.BoundingBox`
    via :meth:`from_bounding_box`, but kept as a lightweight, world-frame numeric region so
    the seeded sampler and its tests stay independent of a live world.
    """

    minimum_x: float
    """
    Lower X bound of the region in the world frame.
    """

    maximum_x: float
    """
    Upper X bound of the region in the world frame.
    """

    minimum_y: float
    """
    Lower Y bound of the region in the world frame.
    """

    maximum_y: float
    """
    Upper Y bound of the region in the world frame.
    """

    height: float
    """
    Z coordinate in the world frame at which targets are spawned.
    """

    def contains(self, x: float, y: float) -> bool:
        """
        :param x: X coordinate in the world frame.
        :param y: Y coordinate in the world frame.
        :return: True if the point lies inside the region.
        """
        return (
            self.minimum_x <= x <= self.maximum_x
            and self.minimum_y <= y <= self.maximum_y
        )

    def grid_capacity(self, clearance: float) -> int:
        """
        :param clearance: Minimum distance in meters between two targets.
        :return: A conservative number of targets that provably fit into the region,
            based on an axis-aligned grid packing.
        """
        columns = int((self.maximum_x - self.minimum_x) / clearance) + 1
        rows = int((self.maximum_y - self.minimum_y) / clearance) + 1
        return columns * rows

    def inset(self, margin: float) -> SpawnRegion:
        """
        :param margin: Distance in meters to shrink the region by on every side.
        :return: The shrunken region at the same height.
        """
        return SpawnRegion(
            minimum_x=self.minimum_x + margin,
            maximum_x=self.maximum_x - margin,
            minimum_y=self.minimum_y + margin,
            maximum_y=self.maximum_y - margin,
            height=self.height,
        )

    def is_empty(self) -> bool:
        """
        :return: True if the region contains no points.
        """
        return self.maximum_x <= self.minimum_x or self.maximum_y <= self.minimum_y

    def area(self) -> float:
        """
        :return: The area of the region in square meters.
        """
        if self.is_empty():
            return 0.0
        return (self.maximum_x - self.minimum_x) * (self.maximum_y - self.minimum_y)

    @classmethod
    def from_bounding_box(
        cls, bounding_box: BoundingBox, margin: float, height_offset: float
    ) -> SpawnRegion:
        """
        Build a spawn region from a measured world-frame bounding box.

        :param bounding_box: The surface's bounding box in the world frame.
        :param margin: Distance in meters kept from every surface edge.
        :param height_offset: Height in meters above the surface top at which targets
            are spawned.
        :return: The inset spawnable region on top of the surface.
        """
        return cls(
            minimum_x=bounding_box.min_x + margin,
            maximum_x=bounding_box.max_x - margin,
            minimum_y=bounding_box.min_y + margin,
            maximum_y=bounding_box.max_y - margin,
            height=bounding_box.max_z + height_offset,
        )


@dataclass(frozen=True)
class TrialSpecification:
    """
    One fully reproducible trial of the experiment grid.
    """

    task: ToolBasedTask
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
        return f"{self.task.value}:{self.environment_name}:{self.seed}"


@dataclass(frozen=True)
class ExperimentConfiguration(SubclassJSONSerializer):
    """
    The full configuration of one experiment campaign.
    """

    tasks: Tuple[ToolBasedTask, ...] = tuple(ToolBasedTask)
    """
    The tasks to run.
    """

    seeds: Tuple[int, ...] = (910001, 910002, 910003)
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

    scale_choices: Tuple[float, ...] = (0.8, 1.0, 1.2, 1.4, 1.6)
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

    surface_names: Tuple[str, ...] = (
        "island_countertop",
        "countertop",
        "table_area_main",
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

    trial_timeout: float = 3600.0
    """
    Wall-clock limit in seconds for a single trial process, sized for the density-based
    target counts.
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

    def to_json(self) -> Dict[str, Any]:
        """
        :return: This configuration as a JSON-serializable dict, e.g. to hand it to a
            trial subprocess.
        """
        return {
            **super().to_json(),
            "tasks": [task.value for task in self.tasks],
            "seeds": list(self.seeds),
            "environment_name": self.environment_name,
            "minimum_targets_per_trial": self.minimum_targets_per_trial,
            "maximum_targets_per_trial": self.maximum_targets_per_trial,
            "targets_per_square_meter": self.targets_per_square_meter,
            "target_clearance": self.target_clearance,
            "footprint_clearance": self.footprint_clearance,
            "scale_choices": list(self.scale_choices),
            "footprint_safety_factor": self.footprint_safety_factor,
            "maximum_spawn_height": self.maximum_spawn_height,
            "full_body_motion": self.full_body_motion,
            "tool_path_pointer_stride": self.tool_path_pointer_stride,
            "collision_avoidance": self.collision_avoidance,
            "surface_names": list(self.surface_names),
            "surface_margin": self.surface_margin,
            "spawn_height_offset": self.spawn_height_offset,
            "results_file": str(self.results_file),
            "trial_timeout": self.trial_timeout,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> ExperimentConfiguration:
        """
        :param data: A dict produced by :meth:`to_json`.
        :return: The deserialized configuration.
        """
        return cls(
            tasks=tuple(ToolBasedTask(task) for task in data["tasks"]),
            seeds=tuple(data["seeds"]),
            environment_name=data["environment_name"],
            minimum_targets_per_trial=data["minimum_targets_per_trial"],
            maximum_targets_per_trial=data["maximum_targets_per_trial"],
            targets_per_square_meter=data["targets_per_square_meter"],
            target_clearance=data["target_clearance"],
            footprint_clearance=data["footprint_clearance"],
            scale_choices=tuple(data["scale_choices"]),
            footprint_safety_factor=data["footprint_safety_factor"],
            maximum_spawn_height=data["maximum_spawn_height"],
            full_body_motion=data["full_body_motion"],
            tool_path_pointer_stride=data["tool_path_pointer_stride"],
            collision_avoidance=data["collision_avoidance"],
            surface_names=tuple(data["surface_names"]),
            surface_margin=data["surface_margin"],
            spawn_height_offset=data["spawn_height_offset"],
            results_file=Path(data["results_file"]),
            trial_timeout=data["trial_timeout"],
        )
