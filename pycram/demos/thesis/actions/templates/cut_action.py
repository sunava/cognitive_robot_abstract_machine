"""Cut action template with alignment, cut, and retreat phases."""

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


CUT_TEMPLATE = ActionTemplate(
    name="cut",
    preconditions=(_always,),
    postconditions=(_always,),
    phases=(
        PhaseSpec(
            family=PrimitiveFamily.SURFACE_SCRUB_CIRCLE,
            anchor_key=AnchorKey.TOOL_CONTACT,
            param_key=ParamKey.CUT_ALIGN_SPEC,
        ),
        PhaseSpec(
            family=PrimitiveFamily.SEPARATION_CONTACT,
            anchor_key=AnchorKey.CUT_PLANE,
            param_key=ParamKey.CUT_SPEC,
        ),
        PhaseSpec(
            family=PrimitiveFamily.SURFACE_SCRUB_CIRCLE,
            anchor_key=AnchorKey.TOOL_CONTACT,
            param_key=ParamKey.CUT_RETREAT_SPEC,
        ),
    ),
)
