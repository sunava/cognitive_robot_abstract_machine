# `cram-viz-reply-sheet` (fix-my-pr / viz-reply-sheet — content written, draft PR sunava#34 awaiting self-review)

## Plan (approved via plan mode this session)

Compile one paste-ready reply per review thread on cram2#485 into
`cram_viz/docs/pr-485-review-replies.md`, from material `roadmap.md`
already contains (already-fixed commits, design-defense rationale, the PRs
that closed each remaining thread). No source changes, no tests — this
item is text compilation, not code. Full plan text is in the roadmap
section for this item (`.claude/personal/plans/fix-my-pr/roadmap.md`).

Both dependencies (`viz-param-docs` #33, `viz-bundle-urdf-reuse` #28)
confirmed merged live via `check_dependency_readiness.py` before starting.

## What actually landed

`cram_viz/docs/pr-485-review-replies.md` (584 lines), one commit. Rather
than just restating the roadmap's paraphrase, fetched all 11 closing pull
requests' actual live descriptions (#20, #22, #23, #24, #26, #27, #28, #29,
#30, #21, plus #32/#33 already read earlier this session) to write each
reply from verified specifics — file paths, dataclass names, test names —
not from memory. Structure: an index table (all 49 documented `T` codes,
group, status, closing PR), then one section per code organized by the
roadmap's own Group A–E buckets, plus a combined section for `viz-param-docs`'s
T18/T23/T41 (one shared reply, since the plan's own notes describe them as
one mechanical sweep).

**Three gaps flagged in the sheet itself, not silently papered over:**
- **T44** — still no description anywhere (flagged by two prior items,
  still unresolved). Sheet has a `NEEDS INPUT` placeholder instead of a
  fabricated reply.
- **T13/T36** — never mentioned anywhere in `roadmap.md`'s triage. A full
  `grep -oE '\bT[0-9]+\b'` sweep found only 49 distinct codes (`T1`–`T51`
  minus these two), while the reviewer-count table sums to 51. Flagged so
  the user can reconcile against GitHub's actual thread count.
- **T16 is only partially closed** — found while writing the reply, not in
  the original triage. PR #27's own description says it fixed
  `get_chart()`'s residue but explicitly left `Bridge.get_state`,
  `get_plan`, `status` and `object_catalog` as untouched
  `Dict[str, Any]` methods; no item in this plan converts those four. The
  T16 reply says so honestly instead of claiming full closure.

**Verification done**: a script confirmed all 49 documented codes appear
exactly once (46 as individual `###` headings + T18/T23/T41 sharing one
combined reply); all three pre-plan Group A commit hashes (`c60d2eb1c`,
`fb984ff90`, `a58c9065a`) verified to exist with matching commit messages
via `git log --all` after an unshallow fetch.

**Deliverable shape was a judgment call, confirmed working**:
`cram_viz/docs/pr-485-review-replies.md` — no existing convention for this
kind of document in the repo (`cram_viz/` has only a root `README.md`).
Whether this PR ever merges into `cram-viz-integration`, or just serves as
a reviewable diff to copy from and close unmerged, is still left to the
user.

## Status

Branch `cram-viz-reply-sheet`, draft PR
[sunava#34](https://github.com/sunava/cognitive_robot_abstract_machine/pull/34)
against `cram-viz-integration`, no `bug` label (not a bug-fix item). Content
commit pushed; PR description updated with the same summary and checked
test-plan boxes.

## Next

Implementation is done. Per personal-notes convention the PR stays draft
until the user self-reviews it — in particular checking whether T44's
placeholder can be filled in, whether T13/T36 turn out to be real missing
threads, and whether T16's partial-closure framing is acceptable to post
as-is or needs its own follow-up PR first. Once merged (or otherwise
resolved), this is the last item before `cram-viz-integration` → cram2#485
is ready for re-review.

---

# `cram-viz-param-docs` (fix-my-pr / viz-param-docs — DONE, merged via sunava#33)

## Plan (approved via plan mode this session)

Mechanical sweep: add `:param <name>:` lines for every parameter missing one
across `cram_viz/src/cram_viz/{live,onboard,knowledge}/`, `server.py` and
`body_geometry.py`. Full plan text is in the roadmap section for this item
(`.claude/personal/plans/fix-my-pr/roadmap.md`).

An AST scan off `origin/cram-viz-integration`'s tip found **104 functions
across 19 files** needing `:param:` additions (close to the manifest's "~95"
estimate).

## Commits landed (all pushed, suite green after each — 195 passed on
`test/cram_viz_test`)

1. `live/` package — `bridge.py` (27), `hooks.py` (4), `http.py` (3).
2. `onboard/` package — `demo.py` (21), `bundle_urdf.py` (4).
3. `knowledge/` package — `knowledge_base.py`, `eql_session.py`,
   `graph_payload.py`, `scene_bundle.py`, `subgraph.py`,
   `architecture_scan.py`, `views/__init__.py`, `views/plan_tree.py`,
   `views/kinematics.py`, `views/architecture.py` (36 total).
4. `server.py` (7, including `Handler.__init__`, which had no docstring at
   all before this) + `body_geometry.py` (2).

Final AST re-scan across every swept file: **0 remaining functions with
missing `:param:` coverage**. `python -m pytest test/cram_viz_test -q` green
after every commit (195 passed throughout — no behavior change anywhere).

`scripts/format_docstrings.py` run on every modified file, per commit. Its
docformatter pass has a recurring quirk on `:param format: ...` lines
specifically (drops the space after the colon before an inline `` `` ``
literal, e.g. `:param format:``printf``-style...`) — hit twice
(`live/http.py`'s and `server.py`'s `log_message`) and once more on
`graph_payload.py`'s `number_format` param; all three fixed by hand
immediately after the formatting pass, verified via a grep for the pattern
across every swept file before each commit.

## Status

- Branch `cram-viz-param-docs`, draft PR
  [sunava#33](https://github.com/sunava/cognitive_robot_abstract_machine/pull/33)
  against `cram-viz-integration`, no `bug` label (docs-admin item).
- All 4 planned commits pushed; manifest still shows `pull_request_number: 33`.
- Subscribed to tracking issue #19.

## Next

Implementation is done; PR description updated with the commit breakdown and
checked test-plan boxes. Per personal-notes convention the PR stays draft
until the user self-reviews it. Per this plan's own convention (matching
every other item's manifest entry), `status` only flips to `done` once
GitHub confirms the PR merged — `/plan-dashboard fix-my-pr`'s
`sync_manifest_status.py` auto-corrects that the next time it runs against
live state, so no manual edit is needed here. Once merged,
`viz-reply-sheet` (depends on this item + `viz-bundle-urdf-reuse`, both then
done) becomes unblockable.

## Open question carried over, not this item's to resolve

T18, T23, T41 (this item's own review threads) have no description anywhere
in the plan's triage notes, same situation as T44 on the now-done
`viz-kb-dataclasses`. Proceeded without the original thread text since this
item's own `plan.yaml` notes fully described the required change.

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
