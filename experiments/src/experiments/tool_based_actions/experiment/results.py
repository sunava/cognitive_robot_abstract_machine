"""
Result recording for the tool-based action experiment.

Results are stored as JSON lines so runs can append concurrently and a crashed campaign
can resume from what is already on disk. Per-target results aggregate into
:class:`TaskReliability` rows, an
:class:`~experiments.experiment_definitions.ExperimentResult`, for rendering the campaign
summary as a table for a paper.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import MISSING, dataclass, fields
from pathlib import Path

from krrood.adapters.json_serializer import JSON_TYPE_NAME, SubclassJSONSerializer
from krrood.exceptions import DataclassException
from typing_extensions import Any, Dict, List, Optional, Set

from experiments.experiment_definitions import ExperimentResult
from experiments.tool_based_actions.experiment.configuration import (
    ToolBasedTask,
    TrialSpecification,
)


@dataclass
class IncompatibleResultRecord(DataclassException):
    """
    Raised when a stored result line does not match the current :class:`TargetResult`
    schema, typically because the results file was written by an older version of the
    experiment.
    """

    missing_fields: List[str]
    """
    Fields the current schema requires that the stored record does not carry.
    """

    unexpected_fields: List[str]
    """
    Fields the stored record carries that the current schema does not know.
    """

    def error_message(self) -> str:
        return (
            f"Result record does not match the current TargetResult schema: missing "
            f"fields {self.missing_fields}, unexpected fields {self.unexpected_fields}."
        )

    def suggest_correction(self) -> str:
        return "Archive or delete the results file to start a fresh campaign."


@dataclass(frozen=True)
class TargetResult(SubclassJSONSerializer):
    """
    The outcome of one tool action on one target of a trial.

    Persisted as one JSON line; aggregated into :class:`TaskReliability` rows for the
    campaign summary table.
    """

    trial_identifier: str
    """
    Identifier of the trial the target belongs to.
    """

    task: ToolBasedTask
    """
    The task that was performed.
    """

    seed: int
    """
    Seed of the trial's scene.
    """

    robot_name: str
    """
    Name of the robot that performed the action.
    """

    environment_name: str
    """
    Name of the environment the action ran in.
    """

    target_name: str
    """
    Name of the target the action acted on.
    """

    target_x: float
    """
    X coordinate of the target in the world frame.
    """

    target_y: float
    """
    Y coordinate of the target in the world frame.
    """

    target_yaw: float
    """
    Rotation in radians of the target around the world Z axis.
    """

    target_scale: float
    """
    Uniform scale factor the target was spawned with.
    """

    surface_name: str
    """
    Name of the surface the target was spawned on.
    """

    success: bool
    """
    True if the action completed without an error.
    """

    duration: float
    """
    Wall-clock duration of the action in seconds.
    """

    failure_reason: Optional[str] = None
    """
    Compact description of the failure, or None on success.
    """

    def to_json(self) -> Dict[str, Any]:
        """
        :return: This result as a JSON-serializable dict.
        """
        return {
            **super().to_json(),
            "trial_identifier": self.trial_identifier,
            "task": self.task.value,
            "seed": self.seed,
            "robot_name": self.robot_name,
            "environment_name": self.environment_name,
            "target_name": self.target_name,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "target_yaw": self.target_yaw,
            "target_scale": self.target_scale,
            "surface_name": self.surface_name,
            "success": self.success,
            "duration": self.duration,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> TargetResult:
        """
        :param data: A dict produced by :meth:`to_json`.
        :return: The deserialized result.
        """
        return cls(
            trial_identifier=data["trial_identifier"],
            task=ToolBasedTask(data["task"]),
            seed=data["seed"],
            robot_name=data["robot_name"],
            environment_name=data["environment_name"],
            target_name=data["target_name"],
            target_x=data["target_x"],
            target_y=data["target_y"],
            target_yaw=data["target_yaw"],
            target_scale=data["target_scale"],
            surface_name=data["surface_name"],
            success=data["success"],
            duration=data["duration"],
            failure_reason=data["failure_reason"],
        )

    @classmethod
    def _validate_record_matches_schema(cls, record: Dict[str, Any]) -> None:
        """
        Check that a deserialized record carries exactly the fields of this class.

        :param record: The deserialized JSON record.
        :raises IncompatibleResultRecord: If required fields are missing or unknown
            fields are present.
        """
        payload_fields = record.keys() - {JSON_TYPE_NAME}
        field_names = {field.name for field in fields(cls)}
        required_field_names = {
            field.name
            for field in fields(cls)
            if field.default is MISSING and field.default_factory is MISSING
        }
        missing_fields = sorted(required_field_names - payload_fields)
        unexpected_fields = sorted(payload_fields - field_names)
        if missing_fields or unexpected_fields:
            raise IncompatibleResultRecord(missing_fields, unexpected_fields)


@dataclass
class TaskReliability(ExperimentResult):
    """
    One per-task reliability row of the campaign summary, renderable as a paper table
    via :class:`~experiments.experiment_definitions.ExperimentsTable` and
    :class:`~experiments.experiment_definitions.TypstRenderer`.
    """

    task: str
    """
    Name of the task the row aggregates.
    """

    successes: int
    """
    Number of targets the task succeeded on.
    """

    total: int
    """
    Total number of targets attempted for the task.
    """

    @classmethod
    def from_results(cls, results: List[TargetResult]) -> List[TaskReliability]:
        """
        :param results: The target results to aggregate.
        :return: One reliability row per task, ordered by task name.
        """
        successes = Counter(result.task.value for result in results if result.success)
        totals = Counter(result.task.value for result in results)
        return [
            cls(task=name, successes=successes.get(name, 0), total=totals[name])
            for name in sorted(totals)
        ]


@dataclass
class ResultRecorder:
    """
    Appends target results to a JSON lines file and answers resume queries.
    """

    results_file: Path
    """
    The file results are appended to.
    """

    def record(self, result: TargetResult) -> None:
        """
        Append one result to the results file.

        :param result: The result to persist.
        """
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with self.results_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result.to_json()) + "\n")

    def load_results(self) -> List[TargetResult]:
        """
        :return: All results recorded so far, oldest first.
        """
        if not self.results_file.exists():
            return []
        lines = self.results_file.read_text(encoding="utf-8").splitlines()
        return [self._load_record(line) for line in lines if line.strip()]

    @staticmethod
    def _load_record(line: str) -> TargetResult:
        """
        :param line: One JSON line from the results file.
        :return: The deserialized result.
        :raises IncompatibleResultRecord: If the line was written by an older schema.
        """
        record = json.loads(line)
        TargetResult._validate_record_matches_schema(record)
        return TargetResult.from_json(record)

    def completed_trial_identifiers(self) -> Set[str]:
        """
        :return: Identifiers of trials that already have at least one recorded
            result.
        """
        return {result.trial_identifier for result in self.load_results()}

    def is_completed(self, specification: TrialSpecification) -> bool:
        """
        :param specification: The trial to check.
        :return: True if the trial already has recorded results.
        """
        return specification.identifier in self.completed_trial_identifiers()
