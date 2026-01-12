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
    def __call__(self, world: Any, **kwargs: Any) -> bool: ...


@dataclass(frozen=True)
class ActionTemplate:
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
        for c in self.preconditions:
            if not c(world, **kwargs):
                raise RuntimeError(f"precondition failed for action={self.name}")
        inst = tuple(
            ph.bind(anchors, params, world=world, **kwargs) for ph in self.phases
        )
        return ActionProgram(phases=inst)
