"""
Recipe data source.

The rest of the app depends only on the :class:`RecipeSource` interface; the
storage backend lives entirely in this file. See :mod:`placement` for the
geometry and :mod:`app` for the web layer.

Two backends
------------
* :class:`DemoRecipes` — built-in sample recipes, no database, no extra
  dependencies. Great for showing the UI on any machine. Enable with the
  ``--demo`` flag on ``app.py`` or by setting ``PLACEMENT_DEMO=1``.
* :class:`PostgresRecipes` — the real backend. Reads the ``recipes`` table via
  the ``psycopg`` (v3) driver. Run ``python3 db_init.py`` once to create and
  seed it.

Connection (PostgreSQL backend)
-------------------------------
No credentials live in code. Connection settings come from the standard libpq
environment variables (``PGHOST``, ``PGPORT``, ``PGDATABASE``, ``PGUSER``,
``PGPASSWORD``), or set ``PLACEMENT_DB_DSN`` to a full connection string, e.g.::

    export PLACEMENT_DB_DSN="postgresql://user:pass@dbhost:5432/placement"
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from placement import Recipe, Size


class RecipeSource(ABC):
    """
    Read access to the stored placement recipes.
    """

    @property
    @abstractmethod
    def is_demo(self) -> bool:
        """
        True when this source serves built-in sample data instead of a
        database.
        """

    @abstractmethod
    def list_recipes(self) -> list[Recipe]:
        """
        All stored recipes.
        """

    @abstractmethod
    def get_recipe(self, recipe_id: str) -> Recipe | None:
        """
        The recipe with the given id, or ``None`` if unknown.
        """


@dataclass
class DemoRecipes(RecipeSource):
    """
    Built-in sample recipes, for demonstrating the UI without a database.

    Rail-shaped parts (long thin bars) matching the black parts in the
    requirements doc; both boxes have the same size, as required.
    """

    @property
    def is_demo(self) -> bool:
        return True

    def list_recipes(self) -> list[Recipe]:
        box = Size(600.0, 400.0)
        return [
            Recipe(id="A", name="Part A", box=box,
                   part=Size(480.0, 40.0), gap=12.0),
            Recipe(id="B", name="Part B", box=box,
                   part=Size(520.0, 34.0), gap=12.0),
        ]

    def get_recipe(self, recipe_id: str) -> Recipe | None:
        return next((recipe for recipe in self.list_recipes()
                     if recipe.id == recipe_id), None)


_SELECT = ("SELECT id, name, box_width, box_height, part_width, part_height, "
           "gap FROM recipes")


@dataclass
class PostgresRecipes(RecipeSource):
    """
    Recipes backed by the PostgreSQL ``recipes`` table.
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
    def _row_to_recipe(row) -> Recipe:
        (recipe_id, name, box_width, box_height, part_width, part_height,
         gap) = row
        return Recipe(
            id=recipe_id,
            name=name,
            box=Size(float(box_width), float(box_height)),
            part=Size(float(part_width), float(part_height)),
            gap=float(gap),
        )

    def list_recipes(self) -> list[Recipe]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(_SELECT + " ORDER BY id")
            return [self._row_to_recipe(row) for row in cursor.fetchall()]

    def get_recipe(self, recipe_id: str) -> Recipe | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(_SELECT + " WHERE id = %s", (recipe_id,))
            row = cursor.fetchone()
            return self._row_to_recipe(row) if row else None


def demo_requested() -> bool:
    """
    True when the demo (no-database) backend was requested via the
    ``PLACEMENT_DEMO`` environment variable.
    """
    return os.environ.get("PLACEMENT_DEMO", "").strip() not in (
        "", "0", "false", "False")


def create_recipe_source() -> RecipeSource:
    """
    Build the recipe source selected by the environment: demo data or
    PostgreSQL.
    """
    if demo_requested():
        return DemoRecipes()
    return PostgresRecipes(dsn=os.environ.get("PLACEMENT_DB_DSN", ""))
