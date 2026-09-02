# Changelog — Plan view & live constraints (branch `cramera-port`)

_2026-09-02 — 20 commits (`1ba0b54a4` … `1ae600f4d`), ~1300 lines._

A readable, live plan view for the cramera dashboard, plus the ability to attach
natural-language constraints to a running robot plan and have them compiled to real
giskardpy motion-statechart goals.

---

## Plan tab — readable step list
The Plan tab no longer shows the vis node graph; it renders the executed plan as a
readable step list.

- **Numbered, collapsible step tree.** Structural containers (Sequential / Parallel /
  Underspecified) are flattened away, so there are no empty rows; actions are numbered
  (1, 2, 3.1, 3.3.1 …) and motions nest under their action as muted leaves.
- **Conditions removed** from this view entirely — they are internal checks that never
  execute, so they were pure noise (and used to wedge parent steps on "running").
- **Strong visual hierarchy.** Top-level "phases" render as cards with an accent bar;
  nested steps sit under a vertical guide line with connector ticks (tree look).
- **Live status with a monotonic "done" progression.** A step's status is derived from
  its subtree (all children done → done, any running → running, any failed → failed).
  The bridge makes completion _sticky_: a node that has run and gone idle reads as
  SUCCEEDED (raw coraplex status otherwise reverts expanded nodes to CREATED, so only the
  root ever ended "done").
- **Expand-all / collapse-all** toolbar directly above the tree; collapse state persists
  across the 700 ms live refresh.
- **Layout fixes:** the list starts below the tab bar (first row no longer clipped), the
  graph's zoom/fullscreen overlays are hidden in this view, and the divider between the
  constraints column and the tree is draggable (width persists in localStorage).
- The Plan tab now attaches to the live bridge like the Statechart/Transforms tabs
  (`"live": "plan"` was missing, so it used to show "no plan tree — run onboarding").

## Constraints — natural language → giskardpy
A constraints palette lives in a fixed left column of the Plan tab.

- Write a constraint in plain English (or use the presets), **drag it onto any step**;
  it is compiled and POSTed to the live bridge, and shows as a ⛓ chip on the step.
- **Rule-based translation** (regex, not an LLM) covering: orientation
  (upright/level/flat/tilt/spill/steady) → `VectorsAligned`; gaze
  (look/watch/observe/keep-in-view/focus) → `PointingAt`; height
  (above/below/off the table/keep low) → `HeightMonitor`; distance
  (away from/clearance/avoid/keep clear) → `DistanceMonitor`.
- **Length parser**: `10 cm` / `0.1 m` / `5mm` in the text sets the thresholds/limits.
- **Object resolution**: the object is taken from the sentence (milk, bowl, spoon, tray,
  flask, vial, beaker, …) or falls back to the step's target.
- **ⓘ help panel** documenting all mappings in English, for colleagues.
- Unmatched phrasings show **"no match"** and are not sent. Gripper open/close is
  intentionally _not_ mapped yet (the joint can't be resolved reliably across robots).

## Bridge — `/constraint` endpoint + live compilation
- New **`AttachConstraintRequest` + `POST /constraint`** on the live bridge, mirroring
  `MoveRequest` / `POST /move` (validated on the HTTP thread, drained on the sim tick).
- **`_inject_constraint` compiles each constraint to a REAL giskardpy node** against the
  live world — resolves link/object names to live `Body` objects and builds
  `VectorsAligned` / `PointingAt` / `HeightMonitor` / `DistanceMonitor` with live
  `Point3` / `Vector3`.
- **Terminal-visible confirmation**: an immediate "received" print on POST, and a
  "compiled against the live world" print on the next tick (with the resolved body names
  and the running motion statechart) — so you can verify it end-to-end.
- **Known limitation (by design):** the node is _not_ hot-added to the currently ticking
  MotionStatechart — a compiled chart binds several fixed-size updaters, so `add_node`
  mid-run crashes the executor (verified). Constraints are queued for the next motion
  activation instead.

## Motion statechart
- **Fixed the status ring widths.** Per-status ring width was silently forced to 2 px
  (with node scaling active, a group-level `borderWidth` beat the per-node one), so every
  status ring drew identically. Rings now render at their real width (running/failed
  thickest, not-started thinnest).

## Plan Builder — constraints on the plan structure
The Plan Builder (`plan_builder.html`) gets the same natural-language constraints as the
Plan view, now attachable to the plan you are composing.

- **Constraints palette** in the left column: type a constraint ("milk must stay
  upright", "keep the bowl above the table"), each card shows the giskardpy goal it
  compiles to (or **no match**). Same rule-based mapping as the Plan view (`ⓘ` explains it).
- **Drag a constraint onto a plan step** → it attaches as a chip on that step (⛓ text +
  goal), removable with ×. Constraints are part of the step, so they move/save with it.
- **Live push:** while the embedded 3D scene is running, attaching a constraint also POSTs
  it to the scaffold's bridge (`/constraint`), so it takes effect on the next activation —
  exactly like the Plan view.
- **Written into the generated demo:** steps with constraints emit a documented
  `# constraints` comment block + a `CONSTRAINTS = [...]` metadata list (step, text, goal,
  params), so the plan file records them.

## Dev tooling
- **`start_demo.sh [demo.py]`** and **`start_viewer.sh`** — source the ROS workspace, set
  the cramera backend, and launch the demo (+ live bridge on :8765) / the web viewer on
  http://localhost:8711. Two terminals, then "◉ Live" → Plan tab.

---

### Files
`live/bridge.py`, `live/http.py`, `knowledge/views/plan_tree.py`,
`web/panels/graph/panel.js`, `web/panels/graph/graph.js`, `web/app.css`,
`web/plan_view.html` + `web/plan_sample.json` (standalone version of the plan view),
`web/plan_builder.html` + `web/plan_builder.js` + `web/plan_builder.css` (Plan Builder),
`start_demo.sh`, `start_viewer.sh`.
