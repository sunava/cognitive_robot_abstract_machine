from __future__ import annotations

from typing import Any

from pycram.demos.thesis.actions.templates.action_template import ActionTemplate
from pycram.demos.thesis.actions.phase_runtime import PhaseSpec, PrimitiveFamily


def _always(world: Any, **kwargs: Any) -> bool:
    return True


CUT_TEMPLATE = ActionTemplate(
    name="cut",
    preconditions=(_always,),
    postconditions=(_always,),
    phases=(
        PhaseSpec(
            family=PrimitiveFamily.SURFACE_SCRUB_CIRCLE,
            anchor_key="tool_contact_anchor",
            param_key="cut_align_scrub",
        ),
        PhaseSpec(
            family=PrimitiveFamily.SEPARATION_CONTACT,
            anchor_key="cut_plane_anchor",
            param_key="cut_sep",
        ),
        PhaseSpec(
            family=PrimitiveFamily.SURFACE_SCRUB_CIRCLE,
            anchor_key="tool_contact_anchor",
            param_key="cut_retreat_scrub",
        ),
    ),
)
