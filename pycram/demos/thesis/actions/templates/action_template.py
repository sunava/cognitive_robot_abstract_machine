"""Action template abstraction composed of phase specs and conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

from pycram.demos.thesis.actions.phase_runtime import (
    ActionProgram,
    AnchorResolver,
    ParamResolver,
    PhaseSpec,
)


class Condition(Protocol):
    """Callable condition for pre/post checks."""

    def __call__(self, world: Any, **kwargs: Any) -> bool: ...


@dataclass(frozen=True)
class PreconditionFailedError(RuntimeError):
    """Raised when action preconditions are not satisfied."""

    action_name: str

    def __str__(self) -> str:
        return f"Preconditions failed for action={self.action_name}."


@dataclass(frozen=True)
class ActionTemplate:
    """Template describing phases plus pre/post conditions."""

    name: str
    phases: Tuple[PhaseSpec, ...]
    preconditions: Tuple[Condition, ...] = ()
    postconditions: Tuple[Condition, ...] = ()

    def bind(
        self,
        anchors: AnchorResolver,
        params: ParamResolver,
        world: Any,
        **kwargs: Any,
    ) -> ActionProgram:
        """Bind anchors/params and validate conditions to build a program."""
        for c in self.preconditions:
            if not c(world, **kwargs):
                raise PreconditionFailedError(self.name)
        inst = tuple(
            ph.bind(anchors, params, world=world, **kwargs) for ph in self.phases
        )
        return ActionProgram(phases=inst)
