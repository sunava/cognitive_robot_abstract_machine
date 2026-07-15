"""
Placement-pattern geometry.

A :class:`Shape` is the footprint of a single part (for example one of the
black rails) and comes from the shape catalog (:mod:`catalog`). A
:class:`Pattern` arranges one shape as a grid inside a box; the resulting
placements can be shifted by a global, clamped X/Y offset.

This module is intentionally self-contained (pure Python, no dependencies) so
it can later be fed by the real SDT or a database without touching the web
layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

AUTO_FIT = 0
"""
Sentinel for :attr:`Pattern.rows` / :attr:`Pattern.columns`: place as many
shapes along that axis as fit into the box.
"""

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
class Shape:
    """
    A single part footprint from the shape catalog.
    """

    id: str
    """
    Unique identifier of the shape.
    """

    name: str
    """
    Human-readable display name.
    """

    size: Size
    """
    Footprint of the part.
    """


@dataclass(frozen=True)
class Pattern:
    """
    A placement pattern: one catalog shape repeated as a grid inside a box.

    ..note:: ``rows``/``columns`` set to :data:`AUTO_FIT` mean "as many as
        fit"; explicit values are clamped to what actually fits.
    """

    id: str
    """
    Unique identifier of the pattern.
    """

    name: str
    """
    Human-readable display name.
    """

    box: Size
    """
    Inner size of the box the shapes are placed into.
    """

    shape: Shape
    """
    The catalog shape this pattern repeats.
    """

    rows: int = AUTO_FIT
    """
    Requested number of rows, :data:`AUTO_FIT` for as many as fit.
    """

    columns: int = AUTO_FIT
    """
    Requested number of columns, :data:`AUTO_FIT` for as many as fit.
    """

    gap: float = 10.0
    """
    Minimum spacing between shapes and to the box wall, in millimetres.
    """

@dataclass(frozen=True)
class Placement:
    """A single placed shape, position = lower-left corner in box
    coordinates."""

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


def _effective_count(requested: int, fitting: int) -> int:
    """
    Resolve a requested row/column count against what actually fits.
    """
    if requested == AUTO_FIT:
        return fitting
    return min(max(requested, 0), fitting)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_placements(pattern: Pattern, offset_x: float = 0.0,
                       offset_y: float = 0.0) -> dict:
    """
    Compute the centred grid of shapes for ``pattern`` and apply a clamped
    global offset.

    Returns a JSON-serialisable dict describing the box, the shape, the
    grid and every placement, plus the allowed offset range so the UI
    can bound its sliders.
    """
    box, shape, gap = pattern.box, pattern.shape.size, pattern.gap

    columns = _effective_count(pattern.columns,
                               _fitting_count(box.width, shape.width, gap))
    rows = _effective_count(pattern.rows,
                            _fitting_count(box.height, shape.height, gap))

    placements: list[Placement] = []
    if columns and rows:
        block_width = columns * shape.width + (columns - 1) * gap
        block_height = rows * shape.height + (rows - 1) * gap
        base_x = (box.width - block_width) / 2.0
        base_y = (box.height - block_height) / 2.0

        max_offset_x = max(base_x - gap, 0.0)
        max_offset_y = max(base_y - gap, 0.0)
        clamped_x = _clamp(offset_x, -max_offset_x, max_offset_x)
        clamped_y = _clamp(offset_y, -max_offset_y, max_offset_y)

        for row in range(rows):
            for column in range(columns):
                placements.append(Placement(
                    x=base_x + clamped_x + column * (shape.width + gap),
                    y=base_y + clamped_y + row * (shape.height + gap),
                    width=shape.width,
                    height=shape.height,
                ))
    else:
        max_offset_x = max_offset_y = 0.0
        clamped_x = clamped_y = 0.0

    return {
        "pattern": {"id": pattern.id, "name": pattern.name},
        "box": asdict(box),
        "shape": {"id": pattern.shape.id, "name": pattern.shape.name,
                  **asdict(shape)},
        "gap": gap,
        "columns": columns,
        "rows": rows,
        "count": len(placements),
        "offset": {"x": clamped_x, "y": clamped_y},
        "offset_range": {"max_x": max_offset_x, "max_y": max_offset_y},
        "placements": [asdict(placement) for placement in placements],
    }
