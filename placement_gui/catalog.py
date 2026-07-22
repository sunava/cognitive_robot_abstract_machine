"""
Data source for shapes and patterns.

The rest of the app depends only on the :class:`Catalog` interface; the
storage backend lives entirely in this file. See :mod:`placement` for the
geometry and :mod:`app` for the web layer.

Two backends
------------
* :class:`DemoCatalog` — built-in sample shapes and patterns, no database, no
  extra dependencies. Saved patterns live in memory for the lifetime of the
  server. Enable with the ``--demo`` flag on ``app.py`` or by setting
  ``PLACEMENT_DEMO=1``.
* :class:`PostgresCatalog` — the real backend. Reads the ``shapes`` and
  ``patterns`` tables via the ``psycopg`` (v3) driver. Run
  ``python3 db_init.py`` once to create and seed them.

Connection (PostgreSQL backend)
-------------------------------
No credentials live in code. Connection settings come from the standard libpq
environment variables (``PGHOST``, ``PGPORT``, ``PGDATABASE``, ``PGUSER``,
``PGPASSWORD``), or set ``PLACEMENT_DB_DSN`` to a full connection string, e.g.::

    export PLACEMENT_DB_DSN="postgresql://user:pass@dbhost:5432/placement"
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from placement import Orientation, Pattern, Shape, Size


class UnknownShapeError(Exception):
    """
    Raised when a pattern references a shape id that is not in the catalog.
    """

    def __init__(self, shape_id: str):
        super().__init__(f"unknown shape '{shape_id}'")
        self.shape_id = shape_id
        """
        The shape id that could not be resolved.
        """


class Catalog(ABC):
    """
    Read access to the shape catalog and read/write access to the stored
    patterns.
    """

    @property
    @abstractmethod
    def is_demo(self) -> bool:
        """
        True when this catalog serves built-in sample data instead of a
        database.
        """

    @abstractmethod
    def list_shapes(self) -> list[Shape]:
        """
        All shapes available for building patterns.
        """

    @abstractmethod
    def get_shape(self, shape_id: str) -> Shape | None:
        """
        The shape with the given id, or ``None`` if unknown.
        """

    @abstractmethod
    def list_patterns(self) -> list[Pattern]:
        """
        All stored placement patterns.
        """

    @abstractmethod
    def get_pattern(self, pattern_id: str) -> Pattern | None:
        """
        The pattern with the given id, or ``None`` if unknown.
        """

    @abstractmethod
    def save_pattern(self, pattern: Pattern) -> None:
        """
        Insert the pattern, or update it if its id already exists.

        :raises UnknownShapeError: if the pattern references a shape
            that is not in the catalog.
        """


def _sample_shapes() -> list[Shape]:
    """
    The built-in sample shapes; same values that ``db_init.py`` seeds into
    PostgreSQL.
    """
    return [
        # Rail-shaped parts (long thin bars) matching the black parts in the
        # requirements doc.
        Shape(id="rail-long", name="Rail long", size=Size(480.0, 40.0)),
        Shape(id="rail-slim", name="Rail slim", size=Size(520.0, 34.0)),
        Shape(id="bracket", name="Bracket", size=Size(120.0, 80.0)),
    ]


def _sample_patterns(shapes: dict[str, Shape]) -> list[Pattern]:
    """
    The built-in sample patterns; same values that ``db_init.py`` seeds into
    PostgreSQL.
    """
    box = Size(600.0, 400.0)
    return [
        Pattern(id="rails-long", name="Rails long", box=box,
                shape=shapes["rail-long"], gap=12.0),
        Pattern(id="rails-slim", name="Rails slim", box=box,
                shape=shapes["rail-slim"], gap=12.0),
    ]


@dataclass
class DemoCatalog(Catalog):
    """
    In-memory catalog with built-in sample data, for demonstrating the UI
    without a database.
    """

    _shapes: dict[str, Shape] = field(default_factory=dict)
    """
    Sample shapes keyed by id.
    """

    _patterns: dict[str, Pattern] = field(default_factory=dict)
    """
    Stored patterns keyed by id; saved patterns live here until the server
    stops.
    """

    def __post_init__(self):
        self._shapes = {shape.id: shape for shape in _sample_shapes()}
        self._patterns = {pattern.id: pattern
                          for pattern in _sample_patterns(self._shapes)}

    @property
    def is_demo(self) -> bool:
        return True

    def list_shapes(self) -> list[Shape]:
        return list(self._shapes.values())

    def get_shape(self, shape_id: str) -> Shape | None:
        return self._shapes.get(shape_id)

    def list_patterns(self) -> list[Pattern]:
        return list(self._patterns.values())

    def get_pattern(self, pattern_id: str) -> Pattern | None:
        return self._patterns.get(pattern_id)

    def save_pattern(self, pattern: Pattern) -> None:
        if pattern.shape.id not in self._shapes:
            raise UnknownShapeError(pattern.shape.id)
        self._patterns[pattern.id] = pattern


_SELECT_SHAPES = "SELECT id, name, width, height FROM shapes"
_SELECT_PATTERNS = """
SELECT p.id, p.name, p.box_width, p.box_height, p.rows, p.columns, p.gap,
       p.orientation, p.flipped_placements,
       s.id, s.name, s.width, s.height
FROM patterns p JOIN shapes s ON s.id = p.shape_id
"""
_UPSERT_PATTERN = """
INSERT INTO patterns (id, name, box_width, box_height, shape_id, rows, columns,
                      gap, orientation, flipped_placements)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    name               = EXCLUDED.name,
    box_width          = EXCLUDED.box_width,
    box_height         = EXCLUDED.box_height,
    shape_id           = EXCLUDED.shape_id,
    rows               = EXCLUDED.rows,
    columns            = EXCLUDED.columns,
    gap                = EXCLUDED.gap,
    orientation        = EXCLUDED.orientation,
    flipped_placements = EXCLUDED.flipped_placements;
"""


@dataclass
class PostgresCatalog(Catalog):
    """
    Catalog backed by the PostgreSQL ``shapes`` and ``patterns`` tables.
    """

    dsn: str = ""
    """
    Connection string; empty means "use the libpq environment variables".
    """

    @property
    def is_demo(self) -> bool:
        return False

    def _connect(self):
        """
        Open a new PostgreSQL connection.

        ``psycopg`` is imported lazily so demo mode works without the
        driver installed. One connection per request is fine for this
        low-traffic HMI prototype; add a pool here if volume ever grows.
        """
        import psycopg  # lazy: only needed for the real backend
        return psycopg.connect(self.dsn) if self.dsn else psycopg.connect()

    @staticmethod
    def _row_to_shape(row) -> Shape:
        shape_id, name, width, height = row
        return Shape(id=shape_id, name=name,
                     size=Size(float(width), float(height)))

    @staticmethod
    def _row_to_pattern(row) -> Pattern:
        (pattern_id, name, box_width, box_height, rows, columns, gap,
         orientation, flipped_placements,
         shape_id, shape_name, shape_width, shape_height) = row
        return Pattern(
            id=pattern_id,
            name=name,
            box=Size(float(box_width), float(box_height)),
            shape=Shape(id=shape_id, name=shape_name,
                        size=Size(float(shape_width), float(shape_height))),
            rows=int(rows),
            columns=int(columns),
            gap=float(gap),
            orientation=Orientation(orientation),
            flipped_placements=tuple(json.loads(flipped_placements or "[]")),
        )

    def list_shapes(self) -> list[Shape]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(_SELECT_SHAPES + " ORDER BY id")
            return [self._row_to_shape(row) for row in cursor.fetchall()]

    def get_shape(self, shape_id: str) -> Shape | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(_SELECT_SHAPES + " WHERE id = %s", (shape_id,))
            row = cursor.fetchone()
            return self._row_to_shape(row) if row else None

    def list_patterns(self) -> list[Pattern]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(_SELECT_PATTERNS + " ORDER BY p.id")
            return [self._row_to_pattern(row) for row in cursor.fetchall()]

    def get_pattern(self, pattern_id: str) -> Pattern | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(_SELECT_PATTERNS + " WHERE p.id = %s",
                           (pattern_id,))
            row = cursor.fetchone()
            return self._row_to_pattern(row) if row else None

    def save_pattern(self, pattern: Pattern) -> None:
        if self.get_shape(pattern.shape.id) is None:
            raise UnknownShapeError(pattern.shape.id)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(_UPSERT_PATTERN, (
                pattern.id, pattern.name,
                pattern.box.width, pattern.box.height,
                pattern.shape.id, pattern.rows, pattern.columns, pattern.gap,
                pattern.orientation.value,
                json.dumps(list(pattern.flipped_placements)),
            ))


def demo_requested() -> bool:
    """
    True when the demo (no-database) backend was requested via the
    ``PLACEMENT_DEMO`` environment variable.
    """
    return os.environ.get("PLACEMENT_DEMO", "").strip() not in (
        "", "0", "false", "False")


def create_catalog() -> Catalog:
    """
    Build the catalog selected by the environment: demo data or PostgreSQL.
    """
    if demo_requested():
        return DemoCatalog()
    return PostgresCatalog(dsn=os.environ.get("PLACEMENT_DB_DSN", ""))
