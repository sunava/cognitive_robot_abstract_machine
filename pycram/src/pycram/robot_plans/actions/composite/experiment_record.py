from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ExperimentRecord:
    data: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _pose_to_dict(pose) -> dict:
        return {
            "position": pose.to_position().to_np()[:3].tolist(),
            "quaternion": pose.to_quaternion().to_np().tolist(),
        }

    @classmethod
    def from_action(
        cls,
        action,
        num_points_sampled: int,
        num_points_executed: int,
        pointer_stride: int,
    ) -> "ExperimentRecord":
        container_name = (
            str(action.container.name)
            if hasattr(action, "container") and action.container is not None
            else None
        )
        tool_name = (
            str(action.tool.root.name)
            if action.tool is not None and getattr(action.tool, "root", None) is not None
            else None
        )
        container_pose = (
            cls._pose_to_dict(action.container.global_pose)
            if hasattr(action, "container") and action.container is not None
            else None
        )
        data = {
            "action": action.__class__.__name__,
            "container": container_name,
            "tool": tool_name,
            "robot_pose": cls._pose_to_dict(action.context.robot.root.global_pose),
            "container_pose": container_pose,
            "num_points_sampled": int(num_points_sampled),
            "num_points_executed": int(num_points_executed),
            "pointer_stride": int(pointer_stride),
        }
        return cls(data=data)

    def set(self, key: str, value: Any) -> "ExperimentRecord":
        self.data[key] = value
        return self

    def update(self, values: Dict[str, Any]) -> "ExperimentRecord":
        self.data.update(values)
        return self

    def mark_action_success(self, success: bool) -> "ExperimentRecord":
        self.data["action_success"] = bool(success)
        return self

    def mark_exception(self, exc: Exception) -> "ExperimentRecord":
        self.data["exception_type"] = type(exc).__name__
        self.data["exception_message"] = str(exc)
        return self

    def finalize_geometric(self) -> "ExperimentRecord":
        geometric_flags = []
        geometric_checks = {}
        for key in (
            "distance_success",
            "target_intersection_success",
            "cutting_success",
            "mixing_success",
        ):
            if key in self.data:
                value = bool(self.data[key])
                geometric_flags.append(value)
                geometric_checks[key] = value

        self.data["geometric_success"] = (
            bool(all(geometric_flags)) if len(geometric_flags) > 0 else None
        )
        self.data["geometric_checks"] = geometric_checks
        self.data["geometric_failed_checks"] = [
            key for key, value in geometric_checks.items() if not value
        ]
        self.data["overall_success"] = bool(self.data.get("action_success", False)) and (
            self.data["geometric_success"] is not False
        )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)
