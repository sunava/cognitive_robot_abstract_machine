"""
Exercise lightweight HTTP contracts without importing optional numerical engines.
"""

from dataclasses import dataclass
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
import sys


# %% optional dependency isolation
@dataclass
class OptionalEngineImports(MetaPathFinder):
    """
    Fail imports that would require the optional contact or liquid simulation.
    """

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        """
        Reject numerical engine packages while allowing the lightweight HTTP API.
        """
        if fullname.split(".")[0] in {"numpy", "scipy", "mujoco"}:
            raise ModuleNotFoundError(fullname)
        return None


sys.meta_path.insert(0, OptionalEngineImports())
from cramera import laboratory_physics_session, laboratory_robot_session, server

assert laboratory_physics_session.PhysicsRoute.PIPETTING
assert laboratory_robot_session.RobotPhysicsRoute.PIPETTING
print("Lightweight station API imports successfully")
