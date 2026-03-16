import copy
import numpy as np

from pycram.datastructures.enums import Arms

DEFAULT_TOOL_MOUNTS = {
    "cut": {
        Arms.RIGHT: {
            "x": 0.0,
            "y": 0.0,
            "z": 0.08,
            "roll": 0.0,
            "pitch": -np.pi / 2,
            "yaw": 0.0,
        },
        Arms.LEFT: {
            "x": 0.0,
            "y": 0.0,
            "z": -0.08,
            "roll": np.pi,
            "pitch": np.pi / 2,
            "yaw": 0.0,
        },
    },
    "mix": {
        Arms.RIGHT: {
            "x": 0.0,
            "y": 0.0,
            "z": -0.08,
            "roll": 0.0,
            "pitch": np.pi / 2,
            "yaw": 0.0,
        },
        Arms.LEFT: {
            "x": 0.0,
            "y": 0.0,
            "z": 0.08,
            "roll": 0.0,
            "pitch": -np.pi / 2,
            "yaw": 0.0,
        },
    },
    "wipe": {
        Arms.RIGHT: {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": np.pi / 2,
            "yaw": 0.0,
        },
        Arms.LEFT: {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": np.pi / 2,
            "yaw": 0.0,
        },
    },
}

# Adjust these per robot as needed. Values here are merged over the defaults above.
ROBOT_TOOL_MOUNT_OVERRIDES = {
    "pr2": {},
    "tiago": {},
    "armar7": {
        "cut": {
            Arms.RIGHT: {
                "x": 0.0,
                "y": -0.15,
                "z": 0.03,
                "roll": -np.pi,
                "pitch": 0,
                "yaw": -np.pi / 2,
            },
            Arms.LEFT: {
                "x": 0.0,
                "y": -0.15,
                "z": 0.03,
                "roll": np.pi,
                "pitch": 0,
                "yaw": -np.pi / 2,
            },
        },
        "mix": {
            Arms.RIGHT: {
                "x": 0.0,
                "y": 0.15,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0,
                "yaw": np.pi / 2,
            },
            Arms.LEFT: {
                "x": 0.0,
                "y": 0.15,
                "z": 0.0,
                "roll": 0,
                "pitch": 0,
                "yaw": np.pi / 2,
            },
        },
        "wipe": {
            Arms.RIGHT: {
                "x": 0.0,
                "y": 0,
                "z": 0,
                "roll": 0.0,
                "pitch": -np.pi / 2,
                "yaw": 0.0,
            },
            Arms.LEFT: {
                "x": 0.0,
                "y": 0,
                "z": 0,
                "roll": 0.0,
                "pitch": np.pi / 2,
                "yaw": 0.0,
            },
        },
    },
    "hsrb": {
        "cut": {
            Arms.LEFT: {
                "x": 0.08,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": np.pi,
                "yaw": np.pi,
            },
        },
        "mix": {
            Arms.LEFT: {
                "x": 0.08,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": -np.pi,
                "yaw": 0.0,
            },
        },
        "wipe": {
            Arms.LEFT: {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
    },
    "stretch": {},
}


def get_tool_mount_pose_kwargs(task_name, robot_name, arm):
    task_key = str(task_name).strip().lower()
    robot_key = str(robot_name).strip().lower()
    if task_key not in DEFAULT_TOOL_MOUNTS:
        supported = ", ".join(sorted(DEFAULT_TOOL_MOUNTS))
        raise ValueError(
            f"Unsupported tool mount task '{task_name}'. Supported: {supported}"
        )

    base = copy.deepcopy(DEFAULT_TOOL_MOUNTS[task_key][arm])
    overrides = (
        ROBOT_TOOL_MOUNT_OVERRIDES.get(robot_key, {}).get(task_key, {}).get(arm, {})
    )
    base.update(overrides)
    return base
