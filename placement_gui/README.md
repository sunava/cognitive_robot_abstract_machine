# Placement Pattern GUI

A web GUI that visualizes **placement patterns** of parts inside a box,
intended for display in the browser of a **SIMATIC HMI MTP1500 Unified
Comfort** panel.

A **shape** is the footprint of a single part (one of the black rails) and
lives in the **shape catalog** (database table `shapes`). A **pattern**
arranges one catalog shape as a grid inside a box and lives in the `patterns`
table. The GUI has two tabs:

- **Placement** — pick a stored pattern, see the computed placements on an
  HTML canvas, and shift the whole pattern via **X/Y offset** sliders.
- **Pattern Editor** — pick a shape from the catalog (with previews), define a
  new pattern (box size, rows/columns — `0` means "as many as fit" — gap and
  orientation) with a live preview, and **save** it. Saved patterns immediately
  show up in the Placement tab; saving under the same name updates the same
  pattern. The **0°/90° orientation** toggle rotates all parts and recomputes
  how many fit; **tapping a single part** in the preview turns it by 180°, so
  its head end faces the other way (like alternating rows in a real box).

---

## Quick start — Demo mode (no database)

The fastest way to see the UI. Uses built-in sample shapes and patterns,
**no database and no extra packages required** — only Python 3:

```bash
cd placement_gui
python3 app.py --demo
```

Then open <http://localhost:8000/> in a browser. A **"Demo data"** badge appears
next to the title so it's clear the data is sample data, not live data.
Patterns saved in demo mode live in memory until the server stops.

---

## Production — PostgreSQL backend

The real backend reads shapes and patterns from PostgreSQL.

**1. Install the driver** (one time; needs pip):

```bash
pip install -r requirements.txt
```

**2. Point it at your database** via the standard libpq environment variables:

```bash
export PGHOST=dbhost PGPORT=5432 PGDATABASE=placement PGUSER=app PGPASSWORD=secret
# …or a single connection string instead:
# export PLACEMENT_DB_DSN="postgresql://app:secret@dbhost:5432/placement"
```

No credentials are stored in code — they only come from the environment.

**3. Create and seed the tables** (one time; safe to re-run):

```bash
python3 db_init.py
```

**4. Run the server:**

```bash
python3 app.py --host 0.0.0.0 --port 8000
```

Open on the panel at `http://<server-ip>:8000/`. The startup line prints which
backend is active (`[PostgreSQL]` or `[DEMO data …]`).

---

## Architecture

```
HMI panel (Chromium)  --HTTP-->  app.py  (separate machine)
       browser                   ├─ static/       HTML/CSS/JS canvas GUI (two tabs)
                                  ├─ placement.py  pattern geometry (pure stdlib)
                                  └─ catalog.py    shapes + patterns data source
                                        ├─ DemoCatalog     : built-in samples (--demo)
                                        └─ PostgresCatalog : shapes/patterns tables via psycopg
```

- **Web + geometry layer** is pure Python standard library (`http.server`).
- **Only the data source** (`catalog.py`) touches a database, and only in the
  PostgreSQL backend. The `psycopg` import is lazy, so demo mode needs no driver.
- The editor preview and the placement view share **one geometry code path**
  on the server (`placement.compute_placements`).

## Files

| File | Purpose |
|------|---------|
| `app.py` | Stdlib HTTP server: serves `static/` and the JSON API. `--demo` flag. |
| `placement.py` | `Shape`/`Pattern` dataclasses + centred grid computation and offset range. |
| `catalog.py` | `Catalog` interface with `DemoCatalog` and `PostgresCatalog` backends. |
| `db_init.py` | Creates and seeds the PostgreSQL `shapes` and `patterns` tables. |
| `requirements.txt` | The one dependency: `psycopg` (PostgreSQL backend only). |
| `static/` | `index.html`, `style.css`, `app.js` — the two-tab canvas GUI. |
| `test_*.py` | Pytest suites for geometry, demo catalog, and the JSON API. |

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/status` | `{ "demo": true \| false }` — which backend is active. |
| `GET /api/shapes` | Shape catalog: `id, name, width, height`. |
| `GET /api/patterns` | Stored patterns: `id, name, box, shape_id, rows, columns, gap`. |
| `POST /api/patterns` | Save a pattern (JSON body as above, id derived from the name). |
| `GET /api/placements?pattern=ID&offset_x=0&offset_y=0` | Computed placements: box, shape, grid, count, positions, allowed offset range. |
| `GET /api/preview?shape=ID&box_width=&box_height=&rows=&columns=&gap=&orientation=&flipped=` | Placements for an unsaved pattern (editor live preview). |

On a database error the API returns HTTP `503` with `{ "error": … }` instead of
dropping the request, so the UI degrades gracefully.

## Database schema

`db_init.py` creates:

```sql
CREATE TABLE shapes (
    id     TEXT PRIMARY KEY,
    name   TEXT             NOT NULL,
    width  DOUBLE PRECISION NOT NULL,
    height DOUBLE PRECISION NOT NULL
);

CREATE TABLE patterns (
    id                 TEXT PRIMARY KEY,
    name               TEXT             NOT NULL,
    box_width          DOUBLE PRECISION NOT NULL,
    box_height         DOUBLE PRECISION NOT NULL,
    shape_id           TEXT             NOT NULL REFERENCES shapes(id),
    rows               INTEGER          NOT NULL DEFAULT 0,     -- 0 = as many as fit
    columns            INTEGER          NOT NULL DEFAULT 0,     -- 0 = as many as fit
    gap                DOUBLE PRECISION NOT NULL DEFAULT 10.0,
    orientation        TEXT             NOT NULL DEFAULT 'original',  -- or 'rotated' (90°)
    flipped_placements TEXT             NOT NULL DEFAULT '[]'   -- JSON list of 180°-turned indices
);
```

…and seeds three shapes (two rails + a bracket) and two rail patterns in a
600 × 400 mm box.

## Tests

Pure stdlib, run with pytest from this directory:

```bash
pytest placement_gui
```

## Assumptions in the prototype (adjust as needed)

- 2-D top-down view; rectangular shape footprints. Real geometries come later
  from the SDT — swap `catalog.py` accordingly, the rest is untouched.
- Default pattern = centred grid; offset clamped to the box walls.
- Patterns repeat a single shape; mixed-shape patterns would extend
  `Pattern`/`compute_placements`.
- Parts are packed as rectangular footprints. The 90° orientation genuinely
  changes how many fit; the per-part 180° turn keeps the same footprint and is
  stored for the placement process (real interlocking of head shapes needs the
  actual part geometry from the SDT).
