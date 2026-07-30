"""
Hooks that attach the bridge to a running coraplex/giskardpy demo.

.. warning:: Every world access (forward kinematics for poses etc.) happens on
   the *simulation* thread itself, inside the ``Executor.tick`` hook. Reading
   the world from a separate sampler thread corrupts the native solver's heap
   — the HTTP handlers therefore only ever serve the last finished snapshot
   dict.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing_extensions import Any, Protocol, runtime_checkable

from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.plan import Plan
from giskardpy.executor import Executor
from semantic_digital_twin.adapters.mesh import MeshParser
from semantic_digital_twin.world import World

from cram_viz.live.bridge import Bridge, TaskStatus

logger = logging.getLogger(__name__)


class HookAlreadyInstalledError(RuntimeError):
    """
    Raised when a hook is installed twice in the same process.

    Each ``install_*`` function monkeypatches a class method; installing it again would
    stack another bridge on top of the first and, once
    :func:`cram_viz.live.runner.start` goes on to bind the HTTP port a second time, fail
    with ``Address already in use``.
    """


@runtime_checkable
class _InstalledHook(Protocol):
    """
    Structural type for a method this module has already monkeypatched.
    """

    is_bridge_hook: bool


def _require_not_installed(patched_method: Any, hook_name: str) -> None:
    """
    Guard clause: raise if ``patched_method`` was already wrapped by this module.
    """
    if isinstance(patched_method, _InstalledHook):
        raise HookAlreadyInstalledError(
            "%s is already installed in this process" % hook_name
        )


# %% tick hook --------------------------------------------------------------------
def install_tick_hook(bridge: Bridge, plan_snapshot_tick_interval: int = 5) -> None:
    """
    Bind the bridge to the executing world and snapshot on every sim tick.

    The plan tree is only re-walked every plan_snapshot_tick_interval ticks.

    :raises HookAlreadyInstalledError: if this hook is already installed.
    """
    _require_not_installed(Executor.tick, "Executor.tick")
    original_tick = Executor.tick

    def tick(self, *args: Any, **kwargs: Any) -> None:
        """
        Run the real tick, then bind/snapshot the bridge off its result.
        """
        result = original_tick(self, *args, **kwargs)
        if bridge.world is None:
            bridge.world = self.context.world
            bridge._bind()
            logger.info(
                "attached to world (robot=%s, %d joints)",
                type(bridge.robot).__name__ if bridge.robot else "?",
                len(bridge._connections),
            )
        try:
            bridge.apply_moves()  # viewer drags land in the real world here
            bridge.snapshot()
            bridge.observe_chart(self.motion_statechart)
            bridge._ticks += 1
            if bridge._ticks % plan_snapshot_tick_interval == 0:
                bridge.snapshot_plan()
        except Exception:
            # boundary guard: a visualization bug must never take the robot
            # demo down — this is the single intentional broad except
            logger.exception("bridge snapshot failed this tick")
        return result

    tick.is_bridge_hook = True
    Executor.tick = tick


# %% plan hooks --------------------------------------------------------------------
def install_plan_hooks(bridge: Bridge) -> None:
    """
    Follow the coraplex plan: which plan executes, and which plan nodes the currently
    running giskard executable belongs to.

    Both hooks fire on the thread that runs the plan (the same one that ticks the
    executor), so they may touch plan objects directly.

    :raises HookAlreadyInstalledError: if this hook is already installed.
    """
    _require_not_installed(Plan.perform, "Plan.perform")
    original_perform = Plan.perform

    def perform(self, *args: Any, **kwargs: Any) -> Any:
        """
        Capture the plan the moment it starts performing.
        """
        bridge._plan = self
        bridge.snapshot_plan()
        return original_perform(self, *args, **kwargs)

    perform.is_bridge_hook = True
    Plan.perform = perform

    original_execute = GiskardExecutable.execute

    def execute(self, *args: Any, **kwargs: Any) -> None:
        """
        Track this executable's motion group and freeze its final status.
        """
        bridge.bind_motion_group(self)
        try:
            result = original_execute(self, *args, **kwargs)
        except BaseException:
            bridge.freeze_motion_group(self, TaskStatus.FAILED)
            bridge.snapshot_plan()
            raise
        bridge.freeze_motion_group(self, TaskStatus.SUCCEEDED)
        bridge.snapshot_plan()
        return result

    execute.is_bridge_hook = True
    GiskardExecutable.execute = execute


# %% mesh hook --------------------------------------------------------------------
def install_mesh_hook(bridge: Bridge) -> None:
    """
    Remember every mesh an object is built from, so the bridge can serve its geometry to
    the viewer.

    All mesh formats go through ``MeshParser.parse``; the body name matches the mesh
    file's basename.

    :raises HookAlreadyInstalledError: if this hook is already installed.
    """
    _require_not_installed(MeshParser.parse, "MeshParser.parse")
    original_parse = MeshParser.parse

    def parse(self, *args: Any, **kwargs: Any) -> World:
        """
        Remember this mesh's file path before parsing it.
        """
        if self.file_path:
            bridge._mesh_files[Path(self.file_path).name.lower()] = self.file_path
        return original_parse(self, *args, **kwargs)

    parse.is_bridge_hook = True
    MeshParser.parse = parse
