#!/usr/bin/env python3
"""
Create and seed the PostgreSQL ``shapes`` and ``patterns`` tables.

Run once::

    python3 db_init.py

Re-running is safe: the schema uses ``IF NOT EXISTS`` and the seeds upsert, so
existing rows are refreshed rather than duplicated.

Uses the same connection settings as ``catalog.py`` — the standard libpq
environment variables (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD) or a
full ``PLACEMENT_DB_DSN`` connection string.
"""

from __future__ import annotations

import os
import sys

import psycopg

_DSN = os.environ.get("PLACEMENT_DB_DSN") or ""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shapes (
    id     TEXT PRIMARY KEY,
    name   TEXT             NOT NULL,
    width  DOUBLE PRECISION NOT NULL,
    height DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS patterns (
    id         TEXT PRIMARY KEY,
    name       TEXT             NOT NULL,
    box_width  DOUBLE PRECISION NOT NULL,
    box_height DOUBLE PRECISION NOT NULL,
    shape_id   TEXT             NOT NULL REFERENCES shapes(id),
    rows       INTEGER          NOT NULL DEFAULT 0,
    columns    INTEGER          NOT NULL DEFAULT 0,
    gap        DOUBLE PRECISION NOT NULL DEFAULT 10.0
);
"""

# (id, name, width, height) — same values as the DemoCatalog samples.
_SHAPE_SEED = [
    ("rail-long", "Rail long", 480.0, 40.0),
    ("rail-slim", "Rail slim", 520.0, 34.0),
    ("bracket", "Bracket", 120.0, 80.0),
]

# (id, name, box_width, box_height, shape_id, rows, columns, gap);
# rows/columns 0 = as many as fit.
_PATTERN_SEED = [
    ("rails-long", "Rails long", 600.0, 400.0, "rail-long", 0, 0, 12.0),
    ("rails-slim", "Rails slim", 600.0, 400.0, "rail-slim", 0, 0, 12.0),
]

_UPSERT_SHAPE = """
INSERT INTO shapes (id, name, width, height)
VALUES (%s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    name   = EXCLUDED.name,
    width  = EXCLUDED.width,
    height = EXCLUDED.height;
"""

_UPSERT_PATTERN = """
INSERT INTO patterns (id, name, box_width, box_height, shape_id, rows, columns, gap)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    name       = EXCLUDED.name,
    box_width  = EXCLUDED.box_width,
    box_height = EXCLUDED.box_height,
    shape_id   = EXCLUDED.shape_id,
    rows       = EXCLUDED.rows,
    columns    = EXCLUDED.columns,
    gap        = EXCLUDED.gap;
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
            cursor.executemany(_UPSERT_SHAPE, _SHAPE_SEED)
            cursor.executemany(_UPSERT_PATTERN, _PATTERN_SEED)
    connection.close()
    print(f"shapes/patterns tables ready; seeded/updated "
          f"{len(_SHAPE_SEED)} shapes and {len(_PATTERN_SEED)} patterns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
