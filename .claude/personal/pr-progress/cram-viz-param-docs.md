# `cram-viz-param-docs` (fix-my-pr / viz-param-docs — kicked off, draft PR sunava#33)

## Plan (approved via plan mode this session)

Mechanical sweep: add `:param <name>:` lines for every parameter missing one
across `cram_viz/src/cram_viz/{live,onboard,knowledge}/`, `server.py` and
`body_geometry.py`. Full plan text is in the roadmap section for this item
(`.claude/personal/plans/fix-my-pr/roadmap.md`).

An AST scan off `origin/cram-viz-integration`'s tip found **104 functions
across 19 files** needing `:param:` additions (close to the manifest's "~95"
estimate). Breakdown, commit sequence, and the forwarding-`*args`/`**kwargs`
convention are all in the roadmap section — not duplicating here.

Purely additive documentation, no behavior/signature/type change anywhere,
so no new tests are needed.

## Status

- Branch `cram-viz-param-docs` created off `origin/cram-viz-integration`,
  bootstrap commit pushed.
- Draft PR [sunava#33](https://github.com/sunava/cognitive_robot_abstract_machine/pull/33)
  opened against `cram-viz-integration`, no `bug` label (docs-admin item).
- `plan_item_bootstrap.py open` + `record` run; manifest's `viz-param-docs`
  entry now has `pull_request_number: 33`, `status: in_progress`.
- Subscribed to tracking issue #19.

## Next

Implement the 4-commit sequence from the plan (live/ → onboard/ → knowledge/
→ server.py+body_geometry.py), running `python -m pytest test/cram_viz_test -q`
and `scripts/format_docstrings.py` after each. Re-run the AST param-coverage
scan at the end to confirm 0 remaining gaps.

## Open question carried over, not this item's to resolve

T18, T23, T41 (this item's own review threads) have no description anywhere
in the plan's triage notes, same situation as T44 on the now-done
`viz-kb-dataclasses`. Proceeding without the original thread text since this
item's own `plan.yaml` notes fully describe the required change.

---

# `cram-viz-kb-dataclasses` (fix-my-pr / viz-kb-dataclasses — DONE, merged via sunava#32)

Item complete. PR #32 merged into `cram-viz-integration` on 2026-08-08 with
all 7 planned commits plus one review fixup (8 total on the branch). Split
across two PRs — see "PR split" below — because #31 merged early,
mid-implementation.

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
8. Review fixup (post-merge-of-#31 PR #32 review): `QueryResult` renamed to
   `RenderResult` per sunava's own review comment on PR #32 ("might be
   confusing with eql methods") — `Query`/`Match`/`Evaluable` are krrood's
   own EQL vocabulary, and dropping "Query" from this package's own result
   type removes the ambiguity. Replied to the comment and resolved the
   thread.

Every commit's JSON wire shape verified byte-identical by driving the real
`server.py` HTTP endpoints (`/api/kb`, `/api/kb/view` for each tab,
`/api/kb/expand`, `/api/eql`) against a fixture scene bundle — not just
through the test suite, since `to_payload()` is new code with no direct
frontend-consumption test coverage.

`scripts/format_docstrings.py` run on every modified file, per commit.

## Next

Nothing left in this item's own scope — it is done. T44 was never
identified (no access to upstream `cram2#485` this session); still open for
the user or a later pass. `viz-param-docs` (blocked on this item,
`viz-bridge-injection` and `viz-onboard-dataclasses`, all now done) is
unblocked and can start.
