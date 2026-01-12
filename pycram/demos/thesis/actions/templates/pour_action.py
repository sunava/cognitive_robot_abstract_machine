from __future__ import annotations

from typing import Any

from pycram.demos.thesis.actions.action_template import ActionTemplate
from pycram.demos.thesis.actions.phase_runtime import PhaseSpec, PrimitiveFamily


def _always(world: Any, **kwargs: Any) -> bool:
    return True


POUR_TEMPLATE = ActionTemplate(
    name="pour",
    preconditions=(_always,),
    postconditions=(_always,),
    phases=(
        PhaseSpec(
            family=PrimitiveFamily.MATERIAL_TRANSFER_DISCHARGE,
            anchor_key="pour_boundary_anchor",
            param_key="pour_discharge",
        ),
        PhaseSpec(
            family=PrimitiveFamily.MATERIAL_TRANSFER_SHAKE,
            anchor_key="pour_boundary_anchor",
            param_key="pour_shake",
        ),
    ),
)
