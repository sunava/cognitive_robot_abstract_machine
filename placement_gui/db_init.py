#!/usr/bin/env python3
"""
Create and seed the PostgreSQL ``recipes`` table.

Run once::

    python3 db_init.py

Re-running is safe: the schema uses ``IF NOT EXISTS`` and the seed upserts, so
existing rows are refreshed rather than duplicated.

Uses the same connection settings as ``recipes.py`` — the standard libpq
environment variables (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD) or a
full ``PLACEMENT_DB_DSN`` connection string.
"""

from __future__ import annotations

import os
import sys

import psycopg

_DSN = os.environ.get("PLACEMENT_DB_DSN") or ""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recipes (
    id          TEXT PRIMARY KEY,
    name        TEXT             NOT NULL,
    box_width   DOUBLE PRECISION NOT NULL,
    box_height  DOUBLE PRECISION NOT NULL,
    part_width  DOUBLE PRECISION NOT NULL,
    part_height DOUBLE PRECISION NOT NULL,
    gap         DOUBLE PRECISION NOT NULL DEFAULT 10.0
);
"""

# (id, name, box_width, box_height, part_width, part_height, gap) — the two
# objects from the requirements, sorted into boxes of the same size.
_SEED = [
    ("A", "Part A", 600.0, 400.0, 480.0, 40.0, 12.0),
    ("B", "Part B", 600.0, 400.0, 520.0, 34.0, 12.0),
]

_UPSERT = """
INSERT INTO recipes (id, name, box_width, box_height, part_width, part_height, gap)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    name        = EXCLUDED.name,
    box_width   = EXCLUDED.box_width,
    box_height  = EXCLUDED.box_height,
    part_width  = EXCLUDED.part_width,
    part_height = EXCLUDED.part_height,
    gap         = EXCLUDED.gap;
"""


def main() -> int:
    try:
        connection = psycopg.connect(_DSN) if _DSN else psycopg.connect()
    except psycopg.OperationalError as exc:
        print(f"Could not connect to PostgreSQL: {exc}", file=sys.stderr)
        print(
            "Set PGHOST/PGDATABASE/PGUSER/PGPASSWORD (or PLACEMENT_DB_DSN) and "
            "make sure the server is reachable.",
            file=sys.stderr,
        )
        return 1

    with connection:
        with connection.cursor() as cursor:
            cursor.execute(_SCHEMA)
            cursor.executemany(_UPSERT, _SEED)
    connection.close()
    print(f"recipes table ready; seeded/updated {len(_SEED)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
