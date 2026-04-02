"""Pour action template with discharge and shake phases."""

from __future__ import annotations

from typing import Any

from pycram.demos.thesis.actions.templates.action_template import ActionTemplate
from pycram.demos.thesis.actions.phase_runtime import (
    AnchorKey,
    ParamKey,
    PhaseSpec,
    PrimitiveFamily,
)


def _always(world: Any, **kwargs: Any) -> bool:
    """Trivial condition that always evaluates to True."""
    return True


POUR_TEMPLATE = ActionTemplate(
    name="pour",
    preconditions=(_always,),
    postconditions=(_always,),
    phases=(
        PhaseSpec(
            family=PrimitiveFamily.MATERIAL_TRANSFER_DISCHARGE,
            anchor_key=AnchorKey.POUR_BOUNDARY,
            param_key=ParamKey.POUR_DISCHARGE_SPEC,
        ),
        PhaseSpec(
            family=PrimitiveFamily.MATERIAL_TRANSFER_SHAKE,
            anchor_key=AnchorKey.POUR_BOUNDARY,
            param_key=ParamKey.POUR_SHAKE_SPEC,
        ),
    ),
)
