"""
Placement-pattern geometry for Anforderungsprofil 1.

Given a recipe (box and part geometry), compute the default grid pattern
of parts laid out inside the box. The pattern is computed once and can
then be shifted by a global, clamped X/Y offset. No further editing — by
design.

This module is intentionally self-contained (pure Python, no
dependencies) so it can later be fed by the real SDT or a database
without touching the web layer. See :mod:`recipes` for the pluggable
data source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Size:
    """
    A 2-D footprint in millimetres (top-down view).
    """

    width: float
    """
    Extent along the x axis.
    """

    height: float
    """
    Extent along the y axis.
    """


@dataclass(frozen=True)
class Recipe:
    """
    One placement recipe: a part placed into a box, with a packing gap.
    """

    id: str
    """
    Unique identifier of the recipe.
    """

    name: str
    """
    Human-readable display name.
    """

    box: Size
    """
    Inner size of the box the parts are placed into.
    """

    part: Size
    """
    Footprint of the part.
    """

    gap: float = 10.0
    """
    Minimum spacing between parts and to the box wall, in millimetres.
    """

@dataclass(frozen=True)
class Placement:
    """
    A single placed part, position = lower-left corner in box coordinates.
    """

    x: float
    """
    Lower-left corner along the x axis.
    """

    y: float
    """
    Lower-left corner along the y axis.
    """

    width: float
    """
    Extent along the x axis.
    """

    height: float
    """
    Extent along the y axis.
    """


def _fitting_count(span: float, item: float, gap: float) -> int:
    """
    How many items of size ``item`` (plus ``gap`` spacing on all sides) fit
    along ``span``.
    """
    if item <= 0:
        return 0
    return max(int((span - gap) // (item + gap)), 0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_pattern(recipe: Recipe, offset_x: float = 0.0,
                    offset_y: float = 0.0) -> dict:
    """
    Compute the default centred grid pattern for ``recipe`` and apply a clamped
    global offset.

    Returns a JSON-serialisable dict describing the box, the part, the
    grid and every placement, plus the allowed offset range so the UI
    can bound its sliders.
    """
    box, part, gap = recipe.box, recipe.part, recipe.gap

    columns = _fitting_count(box.width, part.width, gap)
    rows = _fitting_count(box.height, part.height, gap)

    placements: list[Placement] = []
    if columns and rows:
        block_width = columns * part.width + (columns - 1) * gap
        block_height = rows * part.height + (rows - 1) * gap
        base_x = (box.width - block_width) / 2.0
        base_y = (box.height - block_height) / 2.0

        max_offset_x = max(base_x - gap, 0.0)
        max_offset_y = max(base_y - gap, 0.0)
        clamped_x = _clamp(offset_x, -max_offset_x, max_offset_x)
        clamped_y = _clamp(offset_y, -max_offset_y, max_offset_y)

        for row in range(rows):
            for column in range(columns):
                placements.append(Placement(
                    x=base_x + clamped_x + column * (part.width + gap),
                    y=base_y + clamped_y + row * (part.height + gap),
                    width=part.width,
                    height=part.height,
                ))
    else:
        max_offset_x = max_offset_y = 0.0
        clamped_x = clamped_y = 0.0

    return {
        "recipe": {"id": recipe.id, "name": recipe.name},
        "box": asdict(box),
        "part": asdict(part),
        "gap": gap,
        "columns": columns,
        "rows": rows,
        "count": len(placements),
        "offset": {"x": clamped_x, "y": clamped_y},
        "offset_range": {"max_x": max_offset_x, "max_y": max_offset_y},
        "placements": [asdict(placement) for placement in placements],
    }
