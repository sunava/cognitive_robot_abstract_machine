# Reply sheet for cram2/cognitive_robot_abstract_machine#485

Paste-ready replies for the 51 review threads left `CHANGES_REQUESTED` by
LucaKro and Narenvasant on 2026-07-29. Compiled from the `fix-my-pr` plan's
own triage (`roadmap.md`) and verified against every closing pull request's
actual description on the `sunava` fork — not re-derived from memory.

**How to use this sheet.** This session has no read access to the `cram2`
repository, so it cannot fetch the real GitHub thread permalinks or comment
IDs — every entry below is keyed by the same internal `T<n>` code the triage
used, with the reviewer's paraphrased complaint restated so you can match it
to the right thread by content. Copy the "Reply" text under each code onto
its matching thread on cram2#485.

## Two things to check before treating this as complete

1. **T44 has no recorded description anywhere** in this plan's history —
   flagged unresolved by two prior sessions already, still true. There's a
   `NEEDS INPUT` placeholder for it below instead of a fabricated reply;
   fill it in once you have the original thread text.
2. **T13 and T36 never appear anywhere in `roadmap.md`'s triage.** A full
   sweep of every `T<n>` code mentioned in that file turns up only 49
   distinct codes (`T1`–`T51` minus these two), while the reviewer-count
   table at the top of the roadmap sums to exactly 51 unresolved threads.
   Nobody flagged this before this session. Two explanations are equally
   plausible and this session can't tell which without `cram2` access:
   either two threads were never triaged at all, or the numbering has a
   legitimate gap from an earlier pass. **Please reconcile the 49 codes
   below against the actual 51 threads on GitHub** — if two threads have no
   match here, they need triage from scratch.

A third thing, found while compiling this sheet rather than in the original
triage: **T16 is only partially closed.** PR #27's own description is
explicit that it fixed `get_chart()`'s residue but left `Bridge.get_state`,
`get_plan`, `status` and `object_catalog` as `Dict[str, Any]` wire-boundary
methods, untouched, on purpose — "not named in this item, left untouched."
No other item in this plan converts those four. T16's reply below reflects
that honestly (partially fixed) rather than claiming full closure.

## Index

| T# | Group | Status | Closed by |
| --- | --- | --- | --- |
| T1 | A | Already fixed (pre-plan) | commit `c60d2eb1c` |
| T2 | B | Design defended | reply only |
| T3 | C | Fixed | PR #22 |
| T4 | B | Design defended | reply only |
| T5 | D | Design defended (partial gap acknowledged) | reply only |
| T6 | B | Design defended | reply only |
| T7 | C | Fixed | PR #22 |
| T8 | C | Fixed | PR #22 |
| T9 | C | Fixed | PR #22 |
| T10 | C | Fixed | PR #22 |
| T11 | B | Design defended | reply only |
| T12 | C | Fixed (wire format) | PR #26 |
| **T13** | — | **Unaccounted for — see note above** | — |
| T14 | C | Fixed | PR #22 |
| T15 | B | Design defended | reply only |
| T16 | D | **Partially fixed** | PR #27 (residue remains) |
| T17 | D | Fixed | PR #29 |
| T18 | docs | Fixed | PR #33 |
| T19 | A | Already fixed (pre-plan) | — |
| T20 | A | Already fixed (pre-plan) | commit `fb984ff90` |
| T21 | C | Fixed | PR #22 |
| T22 | D | Fixed | PR #24 |
| T23 | docs | Fixed | PR #33 |
| T24 | D | Fixed | PR #27 |
| T25 | A | Already fixed (pre-plan) | commit `a58c9065a` |
| T26 | A | Already fixed (pre-plan) | — |
| T27 | C | Fixed | PR #30 |
| T28 | D | Fixed | PR #32 |
| T29 | D | Fixed | PR #32 |
| T30 | C | Fixed | PR #22 |
| T31 | D | Design defended | reply only |
| T32 | C | Fixed | PR #22 |
| T33 | E | Fixed | PR #28 (+ PR #21 upstream) |
| T34 | D | Fixed | PR #28 |
| T35 | B | Design defended | reply only |
| **T36** | — | **Unaccounted for — see note above** | — |
| T37 | D | Fixed | PR #24 |
| T38 | C | Fixed | PR #22 |
| T39 | D | Fixed | PR #24 |
| T40 | D | Fixed | PR #32 |
| T41 | docs | Fixed | PR #33 |
| **T44** | D | **NEEDS INPUT — no description found** | — |
| T42 | C | Fixed | PR #22 |
| T43 | Bug | Fixed | PR #20 |
| T45 | D | Fixed | PR #31 / #32 |
| T46 | E | Fixed | PR #30 |
| T47 | C | Fixed | PR #22 |
| T48 | Bug + E | Fixed | PR #20 (splicing) + PR #32 (dataclass) |
| T49 | B | Design defended | reply only |
| T50 | B | Design defended | reply only |
| T51 | E | Fixed | PR #23 |

---

## The two bugs (neither raised by a reviewer, both sit inside raised threads)

### T43 — attach/detach plan nodes mis-grouped

> Paraphrase: the plan-node legend/grouping doesn't seem to distinguish
> attach/detach actions from other plan nodes.

**Reply:**

Good catch, and it turned out to be a real bug rather than a display quirk.
`PLAN_GROUPS` (then in `kb.py`, now `knowledge/views/plan_tree.py`) keyed
attach/detach colouring on `AttachmentNode`/`DetachmentNode`, but coraplex's
actual classes are `AttachNode`/`DetachNode`
(`coraplex/src/coraplex/plans/attachment_nodes.py:42,52`), and the lookup is
`type(node).__name__` — so neither key ever matched, in either the recorded
or the live plan view. The identical dict was duplicated client-side as
`PLAN_GROUP` in `panels/graph/panel.js` with the same wrong keys, so both
paths silently rendered attach/detach nodes as "Other plan node" and the
`{"group": "object", "label": "Attach / detach"}` legend row was dead.
Fixed in both places by #20, test-first: two new failing-then-passing tests
in `test_kb.py::TestPlanGroups` for the recorded/live grouping, and a new
`test/cram_viz_test/js/test_graph_panel.js` exercising the live plan tab
end-to-end through the frontend's own bus/fetch contract.

### T48 (part) — EQL preset names spliced unescaped into query source

> Paraphrase: (part of a broader concern about the EQL presets/query
> construction).

**Reply (bug portion — the dataclass portion is under T48 in Group E below):**

`get_presets()` built query text with plain `%s` string interpolation of
object/episode names into single-quoted EQL source that reaches
`eval(compile(...))`. A name containing an apostrophe produced a broken
`SyntaxError`. Fixed in #20 by switching all three splice sites to
`repr(name)`, which always yields a valid, correctly-escaped Python string
literal — test-first, via `test_kb.py::TestPresetSafety` covering all three
sites (object-name preset, both episode-name presets). To be clear about
scope: this is a correctness bug, not a security issue — the server binds
`127.0.0.1` and the entire point of the EQL panel is running user-typed
`eval`, so no privilege boundary is crossed either way.

---

## Group A — already fixed before this plan started (reply only)

### T1 — "best name best email"

**Reply:** Already correct — `pyproject.toml:7-9` has the right name/email;
this was fixed in commit `c60d2eb1c` prior to this round of triage. Nothing
further needed here.

### T19 — exception handling too broad

**Reply:** This is resolved — the broad `except` is gone;
`bundle_urdf.py:198-200` now guards the file check with `os.path.isfile`
directly rather than catching a broad exception around it.

### T20 — abbreviations in the copy helpers

**Reply:** Fixed in commit `fb984ff90`: `src`/`dst` → `source`/`destination`,
`ext` → `suffix`, `txt` → `mesh_text` throughout the copy helpers.

### T25 — unclear `pose7`-style naming

**Reply:** Renamed to `_pose_as_position_quaternion`
(`onboard/demo.py:372`) in commit `a58c9065a`, and it now delegates to
`semantic_digital_twin`'s own `to_position_quaternion_list` rather than
hand-rolling the conversion.

### T26 — no local/closure-based methods

**Reply:** Addressed — the closures in question became proper `LiveHooks`
dataclass methods (`live/hooks.py:36-111`), each documented and covered by
tests, rather than nested functions capturing state via closure.

---

## Group B — the current design is correct (reply only, no code change)

### T6 — `do_OPTIONS`/`do_GET`/`do_POST` naming

**Reply:** These names are required, not a style choice —
`BaseHTTPRequestHandler` dispatches incoming requests via
`getattr(self, "do_" + command)`, so renaming any of them breaks routing
entirely. The uppercase verbs mirror RFC 7231's HTTP method tokens
(`http.py:128`), which is also why they're uppercase rather than
snake_case. Leaving this as-is.

### T15 — use `World.controlled_connections` instead of the current filter

**Reply:** `World.controlled_connections` isn't equivalent to what the
bridge needs, so this isn't a straightforward swap. It filters on
`is_controlled` → `has_hardware_interface`, but sem_dt's own docstring
(`connections.py:105`) notes "a door hinge is also active but cannot be
controlled" — the bridge needs to animate every `ActiveConnection1DOF`,
including doors and drawers, which are exactly the objects these demo
scenes manipulate. Switching to `controlled_connections` would silently
stop animating them. Separately, that property also returns n-DOF
connections, while `bridge.py:954` specifically needs `.position` off a
1-DOF connection. Keeping the current filter.

### T11 — `world=None`/`robot=None` handling

**Reply:** Both are real, reachable states rather than defensive
over-engineering. `runner.start()` binds the port and installs hooks
*before* the demo builds its world, and `hooks.py:61-62` attaches on the
first tick — so `world=None` genuinely occurs during that startup window.
`robot=None` is also legitimate: a world can exist with no
`AbstractRobot` in it. Both checks stay.

### T2 — the exception guard's scope

**Reply:** The guard is narrower and more deliberate than it might look:
it wraps exactly one call, `bridge.observe_tick` (`hooks.py:63-69`),
catches `Exception` (not `BaseException`, so it won't swallow
`KeyboardInterrupt`/`SystemExit`), always logs via `logger.exception` for
visibility, and never touches or masks the real tick's own return value.
One honest caveat worth naming in the reply itself: `bridge.attach` at
`hooks.py:62` currently sits *outside* this guard — that's worth folding
in for consistency, and is a fair follow-up if you want to file it
separately.

### T4 — make `Executor` subclassable

**Reply:** `Executor` (`giskardpy/executor.py:85`) already *is*
subclass-friendly — `Ros2Executor` subclasses it today. The actual blocker
is injection, not extensibility: executors are constructed at fixed call
sites with no factory or config seam
(`coraplex/plans/executables.py:323`, `locations/backends.py:165`,
`pose_validator.py:258`). A subclass can't be substituted at any of those
sites without a coraplex-side change to how executors are constructed,
which is out of scope for this PR.

### T35 — provide a `to_urdf`/`export_urdf`-style helper upstream

**Reply:** This one's a genuine gap, not a design disagreement — nothing in
`semantic_digital_twin` currently copies a mesh plus its side assets into
an output tree. There's no `to_urdf`/`export_urdf` anywhere, and
`MeshParser` only wraps a path into a `Mesh` object; it doesn't touch the
filesystem. These helpers exist in `cram_viz` specifically because the
browser's `URDFLoader.js` needs raw mesh assets sitting on disk next to a
rewritten URDF — that's a `cram_viz`-specific need, not something
`semantic_digital_twin` currently provides for any consumer.

### T49 — nested `.gitignore` files shouldn't be used

**Reply:** The premise doesn't hold — nested `.gitignore` files are a core,
standard git feature, not a `cram_viz`-specific pattern. There are already
three precedents on `main`: `segmind/.gitignore`,
`krrood/plugins/pycharm/pyroles-pycharm/.gitignore`, and
`.claude/claude_reviews/.gitignore`. Keeping the nested file as-is.

### T50 — `requirements.txt` is effectively empty/unnecessary

**Reply:** It isn't empty — it's 318 bytes of comment that's been there
since before this review round, and it's load-bearing:
`pyproject.toml:19` and `:40` declare `dynamic = ["dependencies"]`, which
reads this file at build time. `segmind/requirements.txt` is a genuinely
empty file and is the real comparison point if you want one. One honest
caveat: the file's comment claims the package is stdlib-only, and that's
now partly stale since `onboard/demo.py:39-42` hard-imports
`semantic_digital_twin` at module top — worth a follow-up comment update,
separate from this thread's actual ask.

---

## Group C — small, low-risk fixes

### T3 — `cram-viz-live` console script entry point

**Reply:** Fixed in PR #22 — the console script pointed at
`cram_viz.live.__main__:main`, which only worked because `__main__.py`
re-imported `main` from `runner.py`. Repointed directly to
`cram_viz.live.runner:main`.

### T7 — move the live-mode usage docstring out of the module

**Reply:** Fixed in PR #22 — `live/__init__.py`'s usage-tutorial docstring
(the two ways to start live mode, the port, the Live button) moved into a
new "Live mode" section of `cram_viz/README.md`; the module docstring now
carries only the one-line summary and its own short note.

### T8 — comment narrating a past bug/fix

**Reply:** Fixed in PR #22 — trimmed the robot-scene panel comment down
from narrating the historical bug and its fix to the one sentence that
documents the current invariant.

### T9 — narrow the resolver's exception handling

**Reply:** Fixed in PR #22 — `bundle_urdf.py`'s `_resolve_package_uri`
caught bare `Exception` around both resolver calls; narrowed to
`(ParsingError, OSError)` and `(ImportError, LookupError, OSError)`
respectively (the `OSError` covers `ament_index_python` raising
`EnvironmentError` when installed but `AMENT_PREFIX_PATH` is unset — this
is exercised in CI, which runs in a ROS-enabled container). Also added a
test for the previously-uncovered no-hints, no-ROS `package://` path.

### T10 — hoist the `PackageUriResolver` import

**Reply:** Fixed in PR #22 — the import was local to every call; moved to
module level.

### T12 — de-abbreviate the live-bridge `sig` field

**Reply:** Fixed in PR #26, as a dedicated wire-format change split out
from the other small fixes: renamed `sig` → `signature` on the three
`live/bridge.py` dataclasses that carry it (`PlanSnapshot`,
`_ChartStructure`, `ChartSnapshot`) and their construction/read sites,
updated `live/http.py`'s API-contract docstring, updated the frontend
reader in `web/panels/graph/panel.js`, and updated every test assertion
that referenced the old key. The local variable in
`_serialize_chart_structure` was already spelled `signature`; only the
field name needed to change.

### T14 — drop the dead `or []`

**Reply:** Fixed in PR #22 — dropped the dead `or []` fallback on
`World.connections` in both `live/bridge.py` and `onboard/demo.py`; the
property never actually returns `None`.

### T21 — de-abbreviate `link()`'s parameters

**Reply:** Fixed in PR #22 — the architecture-view `link()` helper's
`src`/`dst` parameters are now `source`/`target`, matching the sibling
`package_deps` loop directly above it.

### T27 — `kb` is an abbreviation

**Reply:** Fixed in PR #30 — `kb.py` was renamed to a `knowledge/` package
(`cram_viz/src/cram_viz/knowledge/`), split into ~13 single-scoped modules
along its actual call graph. `kb` was an abbreviation of "knowledge base",
which our style guide flags for a module name.

### T30 / T47 — `KB` → a real name

**Reply:** Fixed in PR #22 — `KB` renamed to `EpisodeKnowledgeBase`, and
its accessors `get_kb`/`reset_kb`/`_kb` renamed to
`get_knowledge_base`/`reset_knowledge_base`/`_knowledge_base`, with every
call site updated (`server.py`, the graph-view helpers, tests). Note that
`kb.py` the *file* wasn't renamed until the later split (PR #30, T27) —
this thread was specifically about the class/accessor names.

### T32 — replace the hand-written `EQL_FACTORIES` dict

**Reply:** Fixed in PR #22 — replaced the hand-written 23-entry
`EQL_FACTORIES` dict with krrood's own
`entity_query_language.scope.eql_factory_namespace()`. One caveat worth
flagging honestly: `eql_factory_namespace()` deliberately omits
builtin-shadowing names (`max`, `sum`, `min`, …), so they're now reachable
only as `eql.max` etc., not as bare names. No shipped preset uses them
bare, so this is safe for us today, but it is a user-visible behavior
change for anyone typing a bare `max(...)` into the EQL panel.

### T38 — use a real ABC instead of a local Protocol

**Reply:** Fixed in PR #22 — replaced the locally-defined
`IsEvaluable` `Protocol` (which was really just a runtime `hasattr` check)
with krrood's actual `Evaluable` ABC
(`krrood/entity_query_language/evaluable.py:21`), which `Query` and
`Match` already implement — exactly what `run_query()` returns.

### T42 — turn `PLAN_GROUPS`/`PLAN_LEGEND` into real types

**Reply:** Fixed in PR #22 — `PLAN_GROUPS`/`PLAN_LEGEND` were bare-string
dicts/lists; replaced with a `PlanNodeGroup(str, Enum)` (matching this
module's existing `ArmSide` pattern) and a `PlanLegendEntry` dataclass,
serialized via `dataclasses.asdict` at the one call site so the JSON
payload is byte-identical — pinned by a new test asserting the exact
legend value before the refactor landed.

---

## Group D — medium refactors

### T22 / T39 — `Recorder` should be a dataclass

**Reply:** Fixed in PR #24 — `onboard/demo.py`'s `Recorder` class had a
65-line `__init__` that was pure attribute assignment, with per-field
docstrings already present, so the conversion to `@dataclass` was
mechanical: same field names, types and docstrings,
`field(default_factory=...)` for the mutable containers so instances don't
share state. Added a regression test (`TestRecorderMutableDefaults`)
specifically guarding against a `field(default=[])` mistake in that kind
of conversion.

### T37 — `bundle_urdf()`'s dict return value

**Reply:** Fixed in PR #24 — replaced the ten-key dict `bundle_urdf()`
returned with a `BundleReport` dataclass, updating all three call sites
(`bundle_urdf.py`'s own `main()`, `demo.py`'s bundling loop, and the test
assertions) from subscript to attribute access.

### T28 — the "funny tuples," two of them untyped

**Reply:** Fixed in PR #32 — every bare-tuple return became a named
dataclass: `scan_architecture`/`load_architecture`'s 3-tuple →
`ArchitectureScan`, `load_scene`'s 2-tuple → `SceneBundle`,
`load_urdf`'s 2-tuple → `ParsedUrdf`/`UrdfJoint`, `_result_rows`'s 3-tuple
→ a private `_RenderedRows`. The two bare-untyped ones: the view-builder
closure that returned a plain `tuple` now returns a
`GraphNode`/`DetailEntry`/`GraphEdge` triple via a `SubgraphAccumulator`,
and `PythonClass.bases` is now `Tuple[str, ...]` instead of a bare
`tuple`.

### T29 — duplicated architecture-scan logic, `ArchitectureScan` type

**Reply:** Fixed in PR #32 — added an `ArchitectureScanner` class wrapping
the scan/load functions as methods, and `scan()`/`load()` now build a new
`ArchitectureScan` dataclass directly from typed `Package`/`PythonClass`/
`PackageDependency` objects, removing the dict-then-`Package(**entry)`
indirection that had been duplicated verbatim in two places.

### T44 — *(NEEDS INPUT)*

**Reply:** *Not yet written.* No description of this thread's actual
content has ever been recorded in this plan's tracking, across three
separate sessions that touched this area. Please paste the original
review-comment text here (or reply to me with it) so a real reply can be
drafted — everything else on this sheet is ready to post, but this one
isn't yet.

### T40 — nine bare `Dict[str, Any]` view payloads

**Reply:** Fixed in PR #32 — every view builder (`graph_payload()`, the
package/subpackage/class views, the URDF view, the plan view, the chart
view, `view_payload`'s error case, and `run_query()`) now returns its own
payload dataclass (`KnowledgeGraphPayload`, `PackageViewPayload` /
`SubpackageViewPayload` / `ClassViewPayload`, `UrdfViewPayload`,
`PlanViewPayload`, `ChartViewPayload`, `RenderResult`), each with a
`to_payload()` producing exactly the same JSON shape the frontend got
before. `server.py`'s `_json()` now calls `.to_payload()` generically
before `json.dumps`, the one call-site change outside the package.

### T45 — substring heuristics instead of real types

**Reply:** Fixed via PR #31 (the `enums.py` commit, merged directly into
`cram-viz-integration`) and completed in PR #32 — `Gripper.side`/
`Arm.side` were plain `str` with substring checks like `"gripper" in
part`/`"left" in part`; replaced end-to-end with an `ArmSide(str, Enum)`,
which also adds an explicit `UNKNOWN` member replacing the previous
`"n/a"` string fallback.

### T16 — remaining `Dict[str, Any]` wire-boundary methods (partial)

**Reply:** Partially addressed, and I want to be upfront that it isn't
fully closed yet. 12 dataclasses and 4 enums landed for this area before
this review round. Of the remaining residue, PR #27 fixed `get_chart()`
specifically — it was calling `asdict(chart)` and then immediately
discarding and hand-rewriting the edges into `{"from", "to", "kind"}`
dicts; that mapping now lives on `ChartEdgeEntry.to_payload()` instead.
However, `Bridge`'s other four wire-boundary methods —`get_state`,
`get_plan`, `status`, and `object_catalog` — are still bare
`Dict[str, Any]` returns; PR #27 deliberately left them out of scope, and
no other item in this round of work covers them. I'd like to track that as
a explicit follow-up rather than imply it's done — happy to scope it as
its own small PR if that's useful.

### T24 — module-level `BRIDGE` global

**Reply:** Fixed in PR #27 — `live/http.py`'s `BridgeRequestHandler` and
`live/runner.py`'s `start()` read/wrote the module-level `BRIDGE` singleton
directly. `live/hooks.py` already injected the bridge as a dataclass
field, so this extends the same pattern: `BridgeRequestHandler` now takes
`bridge` in its constructor (captured before delegating to
`BaseHTTPRequestHandler.__init__`, since `socketserver` instantiates the
handler class per request), built via
`functools.partial(BridgeRequestHandler, bridge=bridge)`; `runner.start()`
binds the global to a local `bridge` name once instead of reading it six
times. `start()`'s own public signature is unchanged intentionally — it's
the one documented entry point, and `hooks.install_*` is hardwired to the
same singleton regardless, so a caller-supplied bridge wouldn't receive
observations anyway. `hooks.py`'s own three `install_*` functions still
reference the global directly — that was explicitly out of this item's
scope, not missed.

### T17 — hand-rolled body-extent scanning

**Reply:** Fixed in PR #29 — `BodyExtent.of` scanned for the first `Box`
or `Mesh` shape and read `.scale` directly, silently returning `None` for
bodies made only of `Sphere`/`Cylinder` shapes. Switched to
`ShapeCollection.scale`, which computes a scale for any shape type via
each shape's own `local_frame_bounding_box`. Same visual-then-collision
order as before; `None` only when both collections are genuinely empty.

### T34 — `xacro_to_urdf_text()` shells out to a CLI

**Reply:** Fixed in PR #28 — replaced the `subprocess.run(["xacro", ...])`
call (which requires a sourced ROS environment on `PATH` — confirmed no
`xacro` CLI is even present in our CI sandbox) with
`URDFParser.from_xacro(path).urdf`, which expands xacro in-process using
the `xacro` Python package, already a declared `semantic_digital_twin`
dependency. One caveat: `from_xacro` runs `hacky_urdf_parser_fix`, so the
resulting text is normalized (round-tripped through `ElementTree`) rather
than byte-identical to the source — verified this doesn't break
`bundle_urdf.py`'s regex-based mesh/link/joint extraction, with a new
end-to-end test bundling a real `.xacro` fixture.

### T31 — dicts should serialize via krrood's `SubclassJSONSerializer`

**Reply:** The dicts in question did become dataclasses (via
`viz-kb-dataclasses`), but I'd like to keep `dataclasses.asdict` for
serialization rather than switch to `SubclassJSONSerializer`.
`to_json`/`_from_json` there aren't automatic — each subclass has to
hand-write its own fields — and it stamps a `type` key onto the output
that the JS frontend doesn't expect or handle. `asdict` is the simpler,
correct tool for a wire format the panels read directly; happy to
reconsider if there's a reason to want the serializer's extra structure
that I'm missing.

### T5 — `monkey_patch.py` doesn't fully solve the polymorphism concern

**Reply:** Agreed it's a partial answer, and I want to be honest about
where the gap remains. `monkey_patch.py` (new, documented, unit-tested)
does isolate the *mechanism* of monkeypatching cleanly, but four
`setattr` call sites still remain in `hooks.py` (lines 125, 140, 141,
156). All four targets are constructed at fixed call sites with no
factory or config seam — the identical finding as T4's `Executor`
discussion. Proper polymorphism here needs upstream observer hooks added
in coraplex/giskardpy/`semantic_digital_twin`, which is out of scope for
this PR.

---

## Group E — large structural changes

### T46 — split the 2056-line `kb.py`

**Reply:** Fixed in PR #30 — `kb.py` (2056 lines, only 155 lines of tests
before this round) is now a `knowledge/` package split along its actual
call graph into 13 single-scoped modules (`scene_bundle.py`, `entities.py`,
`architecture_entities.py`, `architecture_scan.py`, `knowledge_base.py`,
`eql_session.py`, `graph_payload.py`, `presets.py`, and a `views/`
subpackage). Gated on a characterization-test pass first (PR #25) so a bad
cut couldn't silently break the graph panels; pure structural move, no
behavior change, suite green after every commit.

### T48 — EQL presets should be a real type

**Reply:** Fixed in PR #32 — `get_presets()`'s list of dicts is now a
`Preset` dataclass, with `ARCH_PRESETS` typed as `Tuple[Preset, ...]`. As a
bonus, this is also what permanently removes the apostrophe-splicing bug
this thread's presets were vulnerable to (see the T48/bug entry above for
the interim `repr()` fix that landed first).

### T33 — duplicated package-resolver logic

**Reply:** Fixed in PR #28, after closing the one real upstream gap first
in PR #21 (`semantic_digital_twin`): `bundle_urdf.py`'s
`_resolve_package_uri()` had a hand-rolled `ament_index_python` call plus a
hand-rolled prefix-path search duplicating
`semantic_digital_twin.adapters.package_resolver`. PR #21 added
`PrefixPathPackageLocator` to `ROSPackageLocator`'s default chain
upstream, benefiting every `semantic_digital_twin` consumer, not just this
one. With that in place, `PackageUriResolver().resolve(uri)` alone covers
the same three tiers, so the duplicated fallback and `_search_root_candidates()`
were deleted entirely.

### T51 — point the scenes submodule at the `cram2` org

**Reply:** Fixed in PR #23 — `cram-scenes` moved to `cram2/cram-scenes`
(previously `sunava/cram-scenes`). Retargeted the three sites that
hardcoded the old fork URL (`.gitmodules`, `paths.py`'s docstring, and the
README's scene-bundle section). The pinned submodule commit is unchanged
and verified to exist in the new remote's history, so this was a URL-only
change with no rebase needed.

---

## `viz-param-docs` bucket — T18, T23, T41

No description of these three threads was ever recorded in this plan's
triage either, but unlike T44, the plan's own item notes fully described
the needed change without it ("finish the `:param:` docstrings... roughly
95 functions"), so this reply is written from that basis.

**Reply (use for all three of T18 / T23 / T41):**

Fixed in PR #33 — completed the mechanical `:param:` docstring sweep
across `cram_viz`. An AST scan found 104 functions across 19 files (under
`live/`, `onboard/`, `knowledge/`, `server.py`, and `body_geometry.py`)
missing at least one parameter's `:param:` line; every one of them now has
full coverage, including `server.py`'s `Handler.__init__`, which had no
docstring at all before this. Purely additive documentation — no behavior,
signature, or type changes, so no test changes were needed; the existing
suite (195 tests) stayed green throughout.
