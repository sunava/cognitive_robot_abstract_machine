# Placement Pattern GUI (Anforderungsprofil 1)

A web GUI that visualizes a **default placement pattern** for a part inside a
box, intended for display in the browser of a **SIMATIC HMI MTP1500 Unified
Comfort** panel (6AV2128-3QB06-0AX1).

The pattern is computed on the server, drawn on an HTML canvas, and can be
shifted as a whole via **X/Y offset** sliders. The two objects from the
requirements are switched via the recipe buttons in the header; both boxes
have the same size. **No further editing is available — by design** (more
editing capabilities can be sold later if the customer needs them).

---

## Quick start — Demo mode (no database)

The fastest way to see the UI. Uses built-in sample recipes, **no database and
no extra packages required** — only Python 3:

```bash
cd placement_gui
python3 app.py --demo
```

Then open <http://localhost:8000/> in a browser. A **"Demo data"** badge appears
next to the title so it's clear the recipes are samples, not live data.

---

## Production — PostgreSQL backend

The real backend reads recipes from a PostgreSQL `recipes` table.

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

**3. Create and seed the table** (one time; safe to re-run):

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
       browser                   ├─ static/       HTML/CSS/JS canvas GUI
                                  ├─ placement.py  pattern computation (pure stdlib)
                                  └─ recipes.py    data source
                                        ├─ DemoRecipes     : built-in samples (--demo)
                                        └─ PostgresRecipes : recipes table via psycopg
```

## Files

| File | Purpose |
|------|---------|
| `app.py` | Stdlib HTTP server: serves `static/` and the JSON API. `--demo` flag. |
| `placement.py` | Computes the centred grid pattern + allowed offset range. |
| `recipes.py` | `RecipeSource` interface with demo and PostgreSQL backends. |
| `db_init.py` | Creates and seeds the PostgreSQL `recipes` table. |
| `requirements.txt` | The one dependency: `psycopg` (PostgreSQL backend only). |
| `static/` | `index.html`, `style.css`, `app.js` — the canvas GUI. |
| `test_*.py` | Pytest suites for geometry, recipe source, and the JSON API. |

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/status` | `{ "demo": true \| false }` — which backend is active. |
| `GET /api/recipes` | List of recipes: `id, name, box, part, gap`. |
| `GET /api/pattern?recipe=A&offset_x=0&offset_y=0` | Computed pattern: box, part, grid, count, all placements, allowed offset range. |

On a database error the API returns HTTP `503` with `{ "error": … }` instead of
dropping the request, so the UI degrades gracefully and shows a clear message.

## Database schema

`db_init.py` creates:

```sql
CREATE TABLE recipes (
    id          TEXT PRIMARY KEY,
    name        TEXT             NOT NULL,
    box_width   DOUBLE PRECISION NOT NULL,
    box_height  DOUBLE PRECISION NOT NULL,
    part_width  DOUBLE PRECISION NOT NULL,
    part_height DOUBLE PRECISION NOT NULL,
    gap         DOUBLE PRECISION NOT NULL DEFAULT 10.0
);
```

…and seeds the two recipes (Part A and Part B) in a 600 × 400 mm box.

## Tests

Pure stdlib, run with pytest:

```bash
pytest placement_gui
```

## The open technical questions (from the requirements)

1. **Web server stack** → Python stdlib `http.server` — a zero-dependency web
   layer that runs on any machine with Python 3. If auth, WebSockets or scale
   are ever needed, only `app.py` is swapped (e.g. for FastAPI); geometry and
   data source stay untouched.
2. **SDT access vs. database for recipes** → **PostgreSQL**, fully encapsulated
   behind the `RecipeSource` interface in `recipes.py` (+ `db_init.py`). If the
   SDT should become the source later, only a new `RecipeSource` implementation
   is added; web layer and geometry are untouched.
3. **Writing inputs back** (offset → SDT/DB) → currently the offset is only
   applied and the pattern returned. To persist it, add a `POST` endpoint in
   `app.py` plus a `save_offset` method on `RecipeSource` — the seam is
   prepared by the interface.

## Assumptions in the prototype (adjust as needed)

- Box inner size **600 × 400 mm**; both boxes the same size (per requirements).
- Part A 480 × 40 mm, Part B 520 × 34 mm, minimum spacing 12 mm.
- 2-D top-down view; rectangular parts. Real geometries come later from the SDT.
- Default pattern = centred grid; offset clamped to the box walls.

..note:: An extended version with a shape catalog and a pattern editor tab
(rotation of parts, saving patterns from the GUI) exists in the git history of
this branch, should the customer buy more editing capabilities later.
