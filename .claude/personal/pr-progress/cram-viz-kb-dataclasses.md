# `cram-viz-kb-dataclasses` (fix-my-pr / viz-kb-dataclasses, draft PR sunava#32)

All 7 planned commits landed and pushed; the item's implementation is
complete. Split across two PRs — see "PR split" below — because #31 merged
early, mid-implementation.

## Open question (surfaced in the plan, not yet answered)

T44 has no description anywhere in the plan's triage notes. Never resolved
this session (no access to the upstream `cram2#485` PR, out of this
session's GitHub scope) — still needs the user or a later pass.

## PR split — #31 merged early after its first commit

PR #31 (this branch, same name) was merged into `cram-viz-integration` by
the user directly on GitHub, but only the first commit (`enums.py`,
`622db038`) had landed on it at that point — the other six commits below
were pushed to the same branch afterward and were never part of that merge.
Discovered when re-checking PR #31's state before starting the next commit
and finding it `merged`/`closed` with only 1 commit, while the local branch
had 7.

Fix: opened a new draft PR, #32, same branch (`cram-viz-kb-dataclasses`) →
`cram-viz-integration`. Since `622db038` is common ancestry (it's in
`cram-viz-integration`'s history via #31's merge), #32's diff is exactly the
remaining six commits — no rebase or new branch needed. `plan.yaml`'s
`pull_request_number` now points at #32.

## Commits landed (all pushed, suite green after each — 195 passed on
`test/cram_viz_test`)

1. `enums.py` — `ArmSide` (+ `UNKNOWN`, replacing the `"n/a"` fallback,
   test-first), `NodeGroup` (replaces `PlanNodeGroup` + every bare group
   string across the package), `EdgeKind`. **Merged via #31.**
2. `subgraph.py` (not `views/base.py` — that import path made
   `graph_payload.py` → `views.base` → `views/__init__.py` →
   `graph_payload.py` a circular import) — `GraphNode`/`DetailEntry`/
   `GraphEdge`/`LegendEntry`/`SubgraphAccumulator`, plus a payload dataclass
   per view (`KnowledgeGraphPayload`, `SubgraphViewPayload` shared by
   package/subpackage/class views, `UrdfViewPayload`, `PlanViewPayload`,
   `ChartViewPayload`, `UnknownViewPayload`), each with `to_payload()`.
   `server.py`'s `_json()` now calls `.to_payload()` generically via
   `hasattr` before `json.dumps` — the one call site outside the package.
3. `presets.py` — `Preset` dataclass, `ARCH_PRESETS` → `Tuple[Preset, ...]`.
4. `architecture_scan.py` — `ArchitectureScanner` class (the four module
   constants become class attributes); `scan()`/`load()` return a new
   `ArchitectureScan` dataclass built directly from typed
   `Package`/`PythonClass`/`PackageDependency`, removing the
   dict-then-`Package(**entry)` indirection (T29). `_subpackage_of` moved
   from `EpisodeKnowledgeBase` onto the scanner. New test coverage:
   `TestArchitectureScanner` (dict-free `scan()`, cache round-trip via
   `load()`).
5. `views/architecture.py`'s `CLASS_CAP`/`SUBCLASS_CAP` → class attributes
   on a new `ArchitectureViews` class (`package_view`/`subpackage_view`/
   `class_view` classmethods, replacing the bare `_package_view` etc.
   functions); `architecture_entities.py`'s `PythonClass.bases` → bare
   `tuple` → `Tuple[str, ...]`.
6. `scene_bundle.py` — `SceneBundle`, `ParsedUrdf`/`UrdfJoint` replacing
   `load_scene()`/`load_urdf()`'s tuple returns; every call site (3
   production modules, 6 test monkeypatches) updated to attribute access.
7. `eql_session.py` — `QueryResult` dataclass replacing `run_query()`'s
   dict, private `_RenderedRows` replacing `_result_rows()`'s 3-tuple. Rows
   themselves stay `List[Dict[str, Any]]` (genuinely dynamic per-query
   shape, not a fixed type this package owns).

Every commit's JSON wire shape verified byte-identical by driving the real
`server.py` HTTP endpoints (`/api/kb`, `/api/kb/view` for each tab,
`/api/kb/expand`, `/api/eql`) against a fixture scene bundle — not just
through the test suite, since `to_payload()` is new code with no direct
frontend-consumption test coverage.

`scripts/format_docstrings.py` run on every modified file, per commit.

## Next

Nothing left in this item's own scope. Watch PR #32 for review; T44 stays
open for the user or a later pass. `viz-param-docs` (blocked on this item,
`viz-bridge-injection` and `viz-onboard-dataclasses`, all now done) can
start once #32 lands.
