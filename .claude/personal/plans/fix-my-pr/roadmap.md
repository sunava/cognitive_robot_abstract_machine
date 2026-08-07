# fix-my-pr — why this plan exists

[cram2/cognitive_robot_abstract_machine#485](https://github.com/cram2/cognitive_robot_abstract_machine/pull/485)
adds `cram_viz`, a browser-based visualization workspace member, from
`sunava:cram-viz-integration` into `cram2:main`.

It cannot merge — but **not** because of CI. All 21 checks pass, including
`test_each_lib (cram_viz)`, and the PR is `MERGEABLE`. `mergeStateStatus` is
`BLOCKED` purely because `reviewDecision` is `CHANGES_REQUESTED`: LucaKro and
Narenvasant both requested changes on 2026-07-29 and never re-reviewed.

Of 85 review threads, **51 are unresolved**, and every single one dates from
that 2026-07-29 round:

| Reviewer | Unresolved |
| --- | --- |
| AbdelrhmanBassiouny | 30 |
| LucaKro | 18 |
| sunava (own notes) | 2 |
| Narenvasant | 1 |

Roughly twenty commits landed between 2026-07-31 and 2026-08-03 aimed squarely
at this feedback — "worked on comments from the first review round", "adresses
review from bass using enums", "cleaned up abbreviations", "moved imports as
top-level", plus a large local-code-review sweep merged as fork PR #16.
**None of those commits resolved the threads they answered.** 16 of the 51 are
already flagged `outdated` by GitHub. So the PR reads as entirely unaddressed
even where it isn't.

The job is therefore not "fix 51 findings". It is: triage all 51 against the
branch tip, fix what is genuinely open, and hand over reply text for the rest.

## Standing conventions for this plan

- **The cram2 PR is read-only.** No session on this plan comments on it, replies
  to a thread, or resolves anything. This matches both the user's explicit
  choice and AGENTS.md's rule against touching upstream PRs. Replies are
  drafted into a reply sheet; the user posts them.
- **Stacked branches.** Each item branches off `cram-viz-integration`, opens a
  PR on the `sunava` fork for the user's own review, then merges down — the
  same pattern as `cram-viz-midnight` / fork PR #16 last round.
- Per the user's personal notes: PRs open as **drafts**, bug-fix PRs carry the
  **`bug`** label, and a PR goes back to draft after any push.
- Per-branch working notes keep living in `.claude/personal/pr-progress/<branch>.md`;
  that mechanism is independent of this plan and is not duplicated here.

# %% The two bugs

Neither was raised by a reviewer. Neither is covered by a test. Both sit inside
threads the reviewers *did* raise, so fixing them closes review comments too.

## BUG-1 — attach/detach plan nodes are silently mis-grouped

`kb.py:1638-1644` keys `PLAN_GROUPS` on `"AttachmentNode"` and
`"DetachmentNode"`. coraplex's real classes are `AttachNode` and `DetachNode`
(`coraplex/src/coraplex/plans/attachment_nodes.py:42,52`). The lookup key is
`type(node).__name__` — set at `onboard/demo.py:503` and `live/bridge.py:1103` —
so neither entry ever matches. Attach and detach nodes fall through to `"ind"`
and render as "Other plan node", in both the recorded and the live path, and the
`{"group": "object", "label": "Attach / detach"}` legend row at `kb.py:1653` is
dead.

## BUG-2 — scene names are spliced into EQL source

`get_presets()` (`kb.py:2020-2045`) builds query text with
`"the(entity(obj).where(obj.name == '%s'))" % first_object.name`, and that string
reaches `eval(compile(...))` at `kb.py:1138`. An object name containing an
apostrophe yields a syntactically broken preset.

Scope this honestly when replying: it is **not** a security hole. The server
binds `127.0.0.1` (`server.py:196`) and the EQL panel's entire purpose is
executing user-typed queries, so `eval` is by design and no privilege boundary
is crossed. It is a correctness bug, and T48's dataclass presets remove it.

# %% Triage of all 51 threads

**AB** = AbdelrhmanBassiouny, **LK** = LucaKro, **NV** = Narenvasant,
**VH** = sunava's own notes.

## Group A — already fixed by the post-review commits (5). Reply only.

| # | ~ | Thread | Evidence |
| --- | --- | --- | --- |
| T1 | VH | "best name best email" | `pyproject.toml:7-9` is correct (`c60d2eb1c`) |
| T19 | AB | too broad exception | gone; `bundle_urdf.py:198-200` guards with `os.path.isfile` |
| T20 | AB | abbreviations in copy helpers | `src/dst`→`source/destination`, `ext`→`suffix`, `txt`→`mesh_text` (`fb984ff90`) |
| T25 | AB | `pose7 ...?` | renamed `_pose_as_position_quaternion`, `demo.py:372`, delegating to sem_dt's `to_position_quaternion_list` (`a58c9065a`) |
| T26 | LK | no local methods | closures became `LiveHooks` dataclass methods, `hooks.py:36-111`, documented and tested |

## Group B — the right answer is a reply, not a change (8)

| # | ~ | The answer |
| --- | --- | --- |
| T6 | AB | `do_OPTIONS`/`do_GET`/`do_POST` are dispatched by `BaseHTTPRequestHandler` via `getattr(self,'do_'+command)`. Renaming breaks routing; the uppercase verbs at `http.py:128` are RFC header values. |
| T15 | LK | `World.controlled_connections` is **not** equivalent. It filters on `is_controlled` → `has_hardware_interface`. sem_dt's own docstring (`connections.py:105`) says "A door hinge is also active but cannot be controlled." The bridge needs every `ActiveConnection1DOF`; the property would stop animating doors and drawers — the objects these demos manipulate. It also returns n-DOF connections, but `bridge.py:954` needs `.position`. |
| T11 | LK | `world=None` is a real reachable state: `runner.start()` binds the port and installs hooks before the demo builds its world, and `hooks.py:61-62` attaches on the first tick. `robot=None` is legitimate — a world may have no `AbstractRobot`. |
| T2 | NV/LK | The guard wraps exactly one call, `bridge.observe_tick` (`hooks.py:63-69`), catches `Exception` not `BaseException`, always `logger.exception`s, and never touches the real tick's return value. Caveat: `bridge.attach` at `:62` sits *outside* it — worth folding in so the reply is honest. |
| T4 | AB | `Executor` (`giskardpy/executor.py:85`) *is* subclass-friendly — `Ros2Executor` already subclasses it. The blocker is injection: executors are constructed at fixed sites (`coraplex/plans/executables.py:323`, `locations/backends.py:165`, `pose_validator.py:258`) with no factory or config seam. A subclass cannot be substituted without an upstream coraplex change. |
| T35 | LK | Genuine gap. Nothing in sem_dt copies a mesh plus its side assets into an output tree; there is no `to_urdf`/`export_urdf` anywhere, and `MeshParser` only wraps a path into a `Mesh`. These helpers exist because the browser `URDFLoader.js` needs raw assets on disk beside a rewritten URDF. |
| T49 | AB | Premise is wrong — nested `.gitignore` files are core git. Three precedents already on `main`: `segmind/.gitignore`, `krrood/plugins/pycharm/pyroles-pycharm/.gitignore`, `.claude/claude_reviews/.gitignore`. |
| T50 | AB | Not empty — 318 bytes of comment, present since before the review. Load-bearing: `pyproject.toml:19,40` declares `dynamic=["dependencies"]` reading this file. `segmind/requirements.txt` is a genuinely-empty precedent. Caveat: the "stdlib-only" claim is now partly stale, since `demo.py:39-42` hard-imports `semantic_digital_twin` at module top. |

## Group C — small, low-risk fixes (13) → `viz-small-fixes`, `viz-wire-rename`

T14 (`or []` — `World.connections` is `list(...)`, never `None`), T8, T7, T3,
T10, T9, T21, T30/T47, T27, T43, T42, T38, T32; plus T12 separately as a
wire-format change.

T32 caveat for the reply: `eql_factory_namespace()` deliberately omits
builtin-shadowing names (`max`, `sum`, …), exposing them as `eql.max`. The
current hand-written dict exposes them flat. No shipped preset uses them, so the
swap is safe, but a user-typed bare `max(...)` would change meaning.

T38 detail: the current code replaced `hasattr(result, "evaluate")` with
`isinstance(result, IsEvaluable)` against a **locally defined**
`runtime_checkable` Protocol — which at runtime is still just a `hasattr` check.
krrood's real `Evaluable` ABC lives at
`krrood/src/krrood/entity_query_language/evaluable.py:21` and is implemented by
`Query` and `Match`, exactly what `run_query` gets back.

## Group D — medium refactors (13)

T22/T39 (`Recorder` → dataclass; `__init__` is 65 lines of pure attribute
assignment with per-field docstrings already), T37 (`BundleReport`), T28 (six
"funny tuples", two of them bare untyped `tuple`), T29 (`ArchitectureScan`,
duplicated verbatim at `kb.py:548` and `:699`), T44, T40 (nine payload dicts),
T45 (substring heuristics → types), T16 (mostly done: 12 dataclasses and 4 enums
landed; residue is the five `Dict[str,Any]` wire-boundary methods and
`get_chart` rewriting `source`/`target` into `from`/`to` after `asdict`), T24
(`BRIDGE` global), T17 (`ShapeCollection.scale` handles every shape type;
`BodyExtent.of` returns `None` for spheres and cylinders), T34, T31, T5.

On T31 and T5 the recommendation is a reply rather than a change:

- **T31**: the dicts *did* become dataclasses; serialization is
  `dataclasses.asdict`. krrood's `SubclassJSONSerializer.to_json` is not
  automatic — each subclass hand-writes its fields — and it stamps a `type` key
  the JS frontend does not expect. `asdict` is the right tool for a wire format
  the panels read.
- **T5**: `monkey_patch.py` (new, documented, unit-tested) isolates the
  *mechanism*, but four `setattr` sites remain (`hooks.py:125,140,141,156`). All
  four targets are constructed at fixed call sites with no injection seam, the
  same finding as T4. Proper polymorphism needs upstream observer hooks in
  coraplex/giskardpy/sem_dt.

## Group E — large (4)

T46 (the split), T48 (presets), T33 (resolver reuse), T51 (submodule org).

# %% The kb.py split

Full split along the existing `# %%` markers, landing together with the T27
rename since both touch the same import sites:

| New module | From `kb.py` | ~Lines |
| --- | --- | --- |
| `scene_bundle.py` | 80-165 | 86 |
| `entities.py` | 168-380 | 213 |
| `architecture_entities.py` | 382-477 | 96 |
| `architecture_scan.py` | 479-756 | 278 |
| `knowledge_base.py` | 769-1026 | 258 |
| `eql_session.py` | 1029-1221 | 193 |
| `graph_payload.py` | 1223-1459 | 237 |
| `views/base.py` | 1461-1493 | 33 |
| `views/__init__.py` | 1495-1636 | 142 |
| `views/plan.py` | 1638-1724 | 87 |
| `views/kinematics.py` | 1726-1824 | 99 |
| `views/architecture.py` | 1826-1973 | 148 |
| `presets.py` | 1975-2044 | 70 |

The local Protocols at 47-77 are deleted (T38); the `__main__` smoke test at
2046-2056 becomes a test.

**Why characterization tests come first.** `kb.py` is the least-tested and
most-complained-about file in the package:

| Module | Lines | Test | Test lines |
| --- | --- | --- | --- |
| `kb.py` | 2056 | `test_kb.py` | **155** |
| `live/bridge.py` | 1285 | `test_live_bridge.py` | 650 |
| `onboard/demo.py` | 1054 | `test_onboard.py` | 372 |
| `live/hooks.py` | 156 | `test_live_hooks.py` | 196 |

155 lines cannot catch a bad cut, and a silent break in the graph panels is the
likely failure mode. `viz-kb-characterization` is a hard gate on `viz-kb-split`.

# %% Reusable APIs found during triage

Verified against the worktree, so no future session has to re-derive them.

- **`krrood.entity_query_language.scope.eql_factory_namespace()`**
  (`scope.py:94`) — returns the EQL factory namespace as a dict, auto-including
  every public name in `factories`. Replaces the 23-entry hand-written
  `EQL_FACTORIES`. Not re-exported from the package `__init__`; import the
  submodule.
- **`krrood.entity_query_language.evaluable.Evaluable`** (`evaluable.py:21`) — the
  real ABC, implemented by `Query` and `Match`. Its `tolist()` would also let
  `_result_rows` drop its `iter()`/`TypeError` dance.
- **`krrood.adapters.json_serializer.SubclassJSONSerializer`**
  (`json_serializer.py:110`) — opt-in requires inheriting it, overriding
  `to_json` from `super().to_json()`, and overriding the classmethod `_from_json`
  (not `from_json`). A plain dataclass that does not inherit it **fails** rather
  than falling back.
- **`ShapeCollection.scale`** (`shape_collection.py:210`) — the closest ready-made
  answer to `_box_size`; handles every shape type. Also `combined_mesh` (`:132`),
  `min_point` (`:220`), `max_point` (`:226`), and `BoundingBox`
  (`geometry.py:1119`). `KinematicStructureEntity.combined_mesh` exists
  (`world_entity.py:331`) but there is no `bounding_box` property yet — adding one
  is the reviewer's third option, with `center_of_mass` (`:337`) as precedent.
- **`URDFParser.from_xacro`** (`adapters/urdf.py:161-182`) — expands xacro
  in-process; `xacro` is already a declared sem_dt dependency. Strictly better
  than shelling out to the CLI.
- **`adapters/package_resolver.py`** — `CompositePathResolver`, `FileUriResolver`,
  `SearchPathFileResolver`, `PackageUriResolver`, `ROSPackageLocator`,
  `AmentPackageLocator`. Covers all but one of `bundle_urdf.py`'s resolver tiers.
- **`World.controlled_connections`** (`world.py:728`) — exists, but does *not* fit
  the bridge's need. See Group B, T15.

# %% Live state and the #18 trade-off

The fork carries parallel work the review threads say nothing about:

| Fork PR | Branch | Ahead / behind | State |
| --- | --- | --- | --- |
| [#18](https://github.com/sunava/cognitive_robot_abstract_machine/pull/18) | `warehouse-viz-features` | 19 / **0** | `MERGEABLE`, updated 2026-08-05 |
| [#15](https://github.com/sunava/cognitive_robot_abstract_machine/pull/15) | `cram-viz-code-review-fixes` | 2 / **371** | `CONFLICTING`, stale since 2026-07-30 |
| [#14](https://github.com/sunava/cognitive_robot_abstract_machine/pull/14) | `cram-viz-bug` | 2 / **371** | `CONFLICTING`, stale since 2026-07-29 |

**#18 is the live constraint.** Fully rebased and mergeable, it edits `kb.py`
(+77/-52), `demo.py` (+334/-74) and `bundle_urdf.py` (+33/-12) — including new
`scene_id` parameters on `scene_dir`, `load_scene` and `load_urdf`, the exact
signatures `viz-kb-dataclasses` converts. Splitting `kb.py` into 13 modules makes
#18 unmergeable and forces a 19-commit rebase over a file that no longer exists.

**Decision: refactor first, rebase #18 afterwards.** Recorded here so the cost is
a known trade rather than a surprise later.

**#15 and #14 are excluded by decision**, noted only so a later session does not
rediscover them and assume they were missed. #15 does contain four test files
absent from the branch — `test_robot_scene_panel.js` (336),
`test_graph_panel.js` (300), `test_eql_panel.js` (272), `test_bundle_urdf.py`
(165) — and they are deliberately not being mined. `viz-kb-characterization`
writes fresh tests instead.

# %% Verification

Per AGENTS.md, every fix is test-first, and both bugs get a failing test before
the fix.

- `python -m pytest test/cram_viz_test -q` — baseline before touching anything,
  green after each item.
- Any wire-format change (T12 `sig`, T16 `from`/`to`, T31) must update
  `panels/graph/panel.js` in the same commit and keep the node tests
  (`test/cram_viz_test/js/`) green.
- `scripts/format_docstrings.py` on every modified file.
- Report suite results honestly in each PR description; the previous round's
  "115 passed, 2 failed" with the failures explained is the right precedent.

## Standing scope note

Waves 2 and 3 are a large diff on an already-large PR, and
`semdt-prefix-path-locator` puts a `semantic_digital_twin` change into a
`cram_viz` initiative. Both were chosen deliberately, and both are defensible —
these were the reviewers' headline objections. If review latency ever matters
more than completeness, wave 1 plus the Group B replies already answer 26 of the
51 threads and could ship alone.

## `viz-onboard-dataclasses` — Recorder and BundleReport (T22/T39, T37)

Closes T22/T39 (`Recorder`) and T37 (`bundle_urdf()`'s return dict), both
flagged as Group D medium refactors. Scoped to exactly these two shapes —
`Recorder`'s own internal dict/list-typed fields (`actions`, `frames`, etc.)
stay as-is; a different Group D finding on a different module covers those.

- **`Recorder` → `@dataclass`** (`onboard/demo.py`): the existing 65-line
  `__init__` is pure attribute assignment with per-field docstrings already
  present, so the conversion is mechanical — same field names, types and
  docstrings, `field(default_factory=...)` for the mutable containers.
  Confirmed via `git grep` that every `Recorder()` call site (in `demo.py`
  and `test_onboard.py`) takes no constructor args and only does attribute
  get/set afterward, so no call site needs to change.
- **`bundle_urdf()`'s dict → `BundleReport`** (`onboard/bundle_urdf.py`):
  the ten-key dict (`name, urdf, source, links, joints, movable_joints,
  meshes_copied, mesh_exts, refs_rewritten, missing`) becomes a dataclass.
  Three call sites move from subscript to attribute access: `bundle_urdf.py`'s
  own `main()`, `demo.py`'s bundling loop (lines ~868-903), and
  `test_onboard.py`'s `TestBundleUrdf` assertions (updating those assertions
  to attribute access is the TDD anchor — they fail against the dict-
  returning code first, then pass once `BundleReport` lands).
  `typing_extensions.Any` is dropped from `bundle_urdf.py`'s imports once its
  only use (the `Dict[str, Any]` return annotation) is gone.
- New test added to `test_onboard.py`: two independently constructed
  `Recorder()` instances must have distinct (`is not`) list/dict attribute
  objects — the regression guard against writing `field(default=[])` instead
  of `default_factory` during the conversion, the one real hazard
  `AGENTS.md`'s mutable-default-argument rule flags for this refactor.

No wire-format change: both `scene.json`/`trajectory.json` payloads stay
plain dicts built from the new dataclasses' fields, so no
`panels/graph/panel.js` update is needed here (unlike T12/T16/T31).

Dependency `viz-small-fixes` (PR #22) was confirmed merged into
`cram-viz-integration` before starting — verified live via
`check_dependency_readiness.py`, not assumed from the manifest.

Branch: `cram-viz-onboard-dataclasses`, based on `cram-viz-integration`.
Draft PR: [sunava#24](https://github.com/sunava/cognitive_robot_abstract_machine/pull/24).

## `viz-kb-characterization` — characterization tests for `kb.py` before splitting it

Hard gate for `viz-kb-split`: `kb.py` (2060 lines) has 248 lines of tests, but
`graph_payload()` and `expand_node()` — the two functions a bad module cut is
most likely to silently break — were only partially pinned (`graph_payload`)
or not tested at all (`expand_node()`, no tests whatsoever). No production
code changes; `kb.py`'s behavior is already correct, the tests just describe
it. Written fresh, per this plan's own convention — fork PR #15's test files
are deliberately not read or mined.

- **`graph_payload()`** (`TestGraphPayloadStructure`, new): the robot→arm→
  gripper node/edge chain, the episode chain (`precedes`/`performed by`/
  `picks`/`places at`), object detail lines, the architecture cluster
  (`cram` root, package/subpackage `contains` edges, `package_deps` `imports`
  edges), both branches of the `link()` grounding-edge guard clause (present:
  the anchor episode → `coraplex.plans` `planned by`, since that subpackage
  exists in the fixture; absent: no edge ever targets
  `giskardpy.motion_statechart` or `semantic_digital_twin`, since neither
  package exists in the fixture architecture), the plan-tree cluster
  (`plan` node, `executed by`/`spans` edges), and the status string — derived
  from `len(payload["nodes"])`/`len(kb.get_knowledge_base().joints)`/
  `len(kb.get_knowledge_base().classes)` rather than a hardcoded second copy.
- **`expand_node()`** (`TestExpandNode`, new): every dispatch branch (robot →
  `_urdf_view`, `"plan"` → `_plan_view`, package/subpackage/class → their
  views, unknown id → `None`), plus `_class_view`'s three real branches —
  internal-base resolution, external-base fallback (`ext:<name>`), and
  subclass listing — and the `CLASS_CAP`/`SUBCLASS_CAP` truncation notes
  (exercised via 151/81 synthetic `PythonClass` instances written directly
  onto the `EpisodeKnowledgeBase` singleton, not 151 real fixture files —
  characterizing the payload-building logic's cap behavior, not the
  architecture scanner, which `test_architecture_scan` already covers).

**Fixture change**: the existing dataset (`coraplex.plans.Plan`,
`krrood.eql.Entity`) had zero inheritance relationships, so `_class_view`'s
real branches were untestable. Added two small real files (decided with the
user rather than assumed): `coraplex/src/coraplex/plans/typed_plan.py`
(`TypedPlan(Plan)` — in-repo subclass + internal-base resolution) and
`krrood/src/krrood/errors.py` (`EqlError(Exception)` — external-base
fallback). Both add one class each to their package's top-level view, which
existing tests don't assert an exact count against.

Dependency `viz-bugs` (PR #20) confirmed merged into `cram-viz-integration`
before starting, via `check_dependency_readiness.py` against live GitHub
state, not assumed from the manifest.

Branch: `cram-viz-kb-characterization`, based on `cram-viz-integration`.
Draft PR: [sunava#25](https://github.com/sunava/cognitive_robot_abstract_machine/pull/25).

## `viz-bridge-injection` — remove the module-level `BRIDGE` global from `http.py`/`runner.py` (T24, T16 residue)

Closes T24 and the remaining `get_chart()` part of T16's residue (the "five
`Dict[str,Any]` wire-boundary methods" part of T16 stays out of scope — this
item's own notes name only the `get_chart()` hand-rewrite).

- **`live/http.py`**: `BridgeRequestHandler` read/wrote the module-level
  `BRIDGE` singleton at 7 sites across `do_GET`/`_send_mesh`/`do_POST`.
  `live/hooks.py` already injects the bridge as a dataclass field
  (`LiveHooks.bridge`) — `BaseHTTPRequestHandler` subclasses can't take a
  constructor argument the way a plain dataclass can, since `socketserver`
  instantiates the handler *class* per request
  (`RequestHandlerClass(request, client_address, self)`). Fixed by giving
  `BridgeRequestHandler` an `__init__(self, *args, bridge: Bridge, **kwargs)`
  that captures `bridge` before delegating to
  `BaseHTTPRequestHandler.__init__` (which runs the request synchronously, so
  `self.bridge` must be set first), and building the handler in `serve()` via
  `functools.partial(BridgeRequestHandler, bridge=bridge)` — the standard way
  to pass extra constructor state through `socketserver`'s per-request
  handler instantiation. `serve(bridge: Bridge, port: int = DEFAULT_PORT)`
  now requires `bridge` explicitly (no default; it's an internal function
  called from exactly one place).
- **`live/runner.py`**: `start()` read/wrote `BRIDGE` 6 times (reuse guard,
  world binding, the `serve(port)` call, return). `start()` keeps its
  existing signature (`world`, `port`) — it is the one public, documented
  entry point (`README.md`: `from cram_viz.live.runner import start;
  start()`), and adding a `bridge` parameter would be speculative: the
  `hooks.install_*` functions are hardwired to the same global singleton via
  `_LIVE_HOOKS = LiveHooks(bridge=BRIDGE)`, so a caller-supplied different
  bridge would not receive tick/plan/mesh observations anyway. Instead,
  `start()` binds `BRIDGE` to a local `bridge` name once at the top of the
  function and uses `bridge.` for the rest of the body, mirroring
  `hooks.py`'s own single-binding-site pattern, and passes it into
  `serve(bridge, port)`.
- **Explicitly out of scope**, per the item's own notes: `hooks.py`'s three
  `install_*` functions still call `BRIDGE.claim_hook(...)` directly against
  the same global (lines 122/137/153) — not named in this item, left
  untouched.
- **T16 residue**: `Bridge.get_chart()` called `asdict(chart)` (which already
  serializes `ChartEdgeEntry.source`/`target` as-is), then immediately
  discarded `payload["edges"]` and hand-rewrote it by re-walking
  `chart.edges` a second time into `{"from": ..., "to": ..., "kind": ...}`
  dicts (`from`/`to` because `from` is a Python keyword and can't be a
  dataclass field name). Added `ChartEdgeEntry.to_payload() -> Dict[str,
  str]` so the wire-shape mapping lives on the dataclass itself, independently
  testable, instead of being duplicated inline in `get_chart()`. No wire
  format change — the `/chart` JSON shape returned to the frontend is
  unchanged, confirmed by the existing `test_live_bridge.py`
  `TestChartSnapshot::test_structure_and_states` assertion staying green
  without modification.

**New test coverage** (both `http.py`'s `serve()`/`BridgeRequestHandler` and
`runner.py`'s `start()` had zero prior test coverage — `test_server.py`
despite its name tests a different module, `cram_viz/server.py`):

- `test_live_bridge.py`: new direct unit test for `ChartEdgeEntry.to_payload()`.
- New `test_live_http.py`, modeled on `test_server.py`'s real-server-on-an-
  ephemeral-port fixture idiom: `serve(bridge, 0)` against a real `Bridge()`,
  one test per endpoint, plus a test spinning up two `serve()` calls against
  two independently constructed `Bridge()` instances to prove
  `BridgeRequestHandler` no longer reads a shared global.
- New `test_live_runner.py` covering `start()`'s control flow (reuse guard,
  world binding, hook installation, server assignment) via `monkeypatch` —
  `hooks.install_*` monkey-patches coraplex/giskardpy process-globally with
  no uninstall, so tests substitute no-ops rather than calling them for real,
  and substitute a fresh `Bridge()` for `runner.BRIDGE` per test so no test
  touches or dirties the real process singleton.

Dependency `viz-wire-rename` (PR #26) confirmed merged into
`cram-viz-integration` before starting, via `check_dependency_readiness.py`
against live GitHub state.

Branch: `cram-viz-bridge-injection`, based on `cram-viz-integration`.
Draft PR: [sunava#27](https://github.com/sunava/cognitive_robot_abstract_machine/pull/27).

## `viz-bundle-urdf-reuse` — drop bundle_urdf.py's duplicated resolver stack and xacro subprocess (T33, T34)

Both dependencies confirmed merged via `check_dependency_readiness.py` against live
GitHub state before starting: `semdt-prefix-path-locator` (PR #21, merged into
`main`, and `main`'s `c77d2db8` confirmed an ancestor of `cram-viz-integration`)
and `viz-onboard-dataclasses` (PR #24, merged into `cram-viz-integration`).

**T33**: `bundle_urdf.py`'s `_resolve_package_uri()` duplicated
`adapters/package_resolver.py` with a hand-rolled `ament_index_python` call plus
a hand-rolled prefix-path search (`_search_root_candidates()`, walking
`AMENT_PREFIX_PATH`/`ROS_PACKAGE_PATH`/`CMAKE_PREFIX_PATH` and
`~/*_ws/install`, `~/*/install`, `/opt/ros/*`). PR #21 added `PrefixPathPackageLocator`
to `ROSPackageLocator`'s default chain (`AmentPackageLocator`,
`ROSPackagePathLocator`, `PrefixPathPackageLocator`), so `PackageUriResolver().resolve(uri)`
alone now covers exactly the same three tiers. `_search_root_candidates()` and the
hand-rolled fallback in `_resolve_package_uri()` are deleted; the function
delegates entirely to `PackageUriResolver().resolve(uri)`, catching
`(ParsingError, OSError)` as before. `import glob` is dropped, now unused.
`TestResolveUri.test_an_unresolvable_package_uri_is_unresolved`'s docstring is
updated to describe the single delegated call rather than the old three-tier
narration; its assertion (`None` with no env vars set) is unchanged.

**T34**: `xacro_to_urdf_text()` shelled out to the `xacro` CLI via
`subprocess.run(["xacro", path], ...)`, requiring a sourced ROS environment on
`PATH` — confirmed no `xacro` CLI is even present in this sandbox, so the
pre-change implementation could not run here at all. Replaced with
`URDFParser.from_xacro(path).urdf` (`semantic_digital_twin/adapters/urdf.py:161-182`),
which expands xacro in-process via the `xacro` Python package, already a
declared `semantic_digital_twin` dependency (`requirements.txt:7`) — no CLI
required. `import subprocess` and the `XACRO_ERROR_TAIL` constant are dropped,
both used only by the old implementation.

**Caveat carried over and addressed**: `from_xacro` runs `hacky_urdf_parser_fix`
before returning `.urdf`, round-tripping the XML through `xml.etree.ElementTree`
(dropping `<transmission>`/`<gazebo>` sections), so the text is normalized
rather than raw. Verified `ElementTree.tostring` still emits double-quoted
attributes, so `bundle_urdf.py`'s regex-based `MESH_REFERENCE_PATTERN`/
`LINK_PATTERN`/`JOINT_PATTERN` extraction is structurally unaffected — proven
with a new end-to-end test bundling a `.xacro` source the same way the existing
`TestBundleUrdf` tests bundle a plain `.urdf` source. Actually loading the
bundled output in the browser's `URDFLoader.js` is outside what a pytest run
can confirm; this PR is backend-only (no `panels/graph/panel.js` touch, no
wire-format change), so that remains a manual spot-check noted in the PR
description rather than an automated step here.

New test coverage in `test_onboard.py`: `TestXacroToUrdfText` (new, proves the
in-process expansion against a macro-free xacro fixture, deriving expected
link/joint names from `bundler.LINK_PATTERN`/`JOINT_PATTERN` rather than a
hardcoded second copy) and `TestBundleUrdf.test_a_xacro_source_is_bundled_like_a_urdf_source`
(new, end-to-end).

No other call site referenced `_search_root_candidates`, `xacro_to_urdf_text`,
or `XACRO_ERROR_TAIL` outside `bundle_urdf.py` itself (`git grep` confirmed).

Branch: `cram-viz-bundle-urdf-reuse`, based on `cram-viz-integration`.
Draft PR: [sunava#28](https://github.com/sunava/cognitive_robot_abstract_machine/pull/28).

## `viz-semdt-geometry` — use `ShapeCollection.scale` instead of hand-rolled body extents (T17)

`cram_viz/src/cram_viz/body_geometry.py`'s `BodyExtent.of(body)` scans
`body.visual`/`body.collision` for the first `Box` or `Mesh` shape and reads
`.scale` off it directly. Every other shape type (`Sphere`, `Cylinder`) has
no `.scale` attribute, so a body made only of those shapes silently reports
`None` — the live bridge falls back to a default placeholder box
(`_box_size` in `live/bridge.py`) and the onboarder omits the object's
`height` entirely (`onboard/demo.py`), even though the shape's real size is
knowable.

`semantic_digital_twin`'s `ShapeCollection.scale` property
(`shape_collection.py:210`) already computes a scale for any shape type, via
each shape's own `local_frame_bounding_box` (`Box`, `Mesh`, `Sphere`, and
`Cylinder` all implement it). The fix is:

```python
@classmethod
def of(cls, body: Body) -> Optional[BodyExtent]:
    for shape_collection in (body.visual, body.collision):
        if not shape_collection.shapes:
            continue
        scale = shape_collection.scale
        return cls(x=float(scale.x), y=float(scale.y), z=float(scale.z))
    return None
```

Same visual-before-collision order as today; returns `None` only when both
collections are completely empty. Drops the now-unused `Box`/`Mesh` import.
Both call sites are unchanged, they already handle a `None` result.

**Note on item scope vs. the codebase**: the item's own notes mention
"spheres, cylinders and capsules", but `semantic_digital_twin`'s
`geometry.py` currently defines only `Sphere` and `Cylinder` as concrete
shapes besides `Box`/`Mesh` — no `Capsule` class exists. Test coverage
targets the shapes that actually exist.

**A consequence surfaced during planning, not stated in the item's own
notes**: `ShapeCollection.scale` → `as_bounding_box_collection_at_origin` →
`BoundingBox.transform_to_origin` reads
`self.origin.reference_frame._world.transform(...)` — a shape only
participates if its `origin.reference_frame` belongs to a real `World`. Two
existing `test_live_bridge.py` tests (`test_an_object_without_a_mesh_is_catalogued_as_a_sized_box`,
`test_an_object_with_unscaled_shapes_falls_back_to_the_default_size`) prove
`_box_size`'s behavior via a lightweight mimic (`PublishedBody`/`ShapeSet`)
built from a bare, world-less `Box`. Under the fix, that shape is filtered
out of the bounding-box collection entirely, so `ShapeCollection.scale`
raises `ValueError` (`min()` on an empty list) instead of returning the
`None`/default-size the test expects. Both tests move to a real `World` +
`Body` fixture, following `test_shape_collections.py`'s idiom; the mimic
stays for the two tests in that file that never reach geometry.

**New test file** `test/cram_viz_test/test_body_geometry.py` — no dedicated
test file existed for `body_geometry.py` before this item. Written
test-first per AGENTS.md: `Sphere`/`Cylinder` cases were confirmed to fail
against the pre-fix code (proving the bug) before the fix landed. Covers
`Box`, `Mesh` (regression), `Sphere`, `Cylinder` (new coverage — previously
`None`), the no-shapes-at-all case (still `None`), the
visual-preferred-over-collision order, and `BodyExtent.rounded()` (untouched
but previously untested). `onboard/demo.py`'s `build_scene()` call site is
not independently tested — it is one line inside an already-untested
200-line function with no `TestBuildScene` today, and the behavior change is
fully covered at the `BodyExtent.of` level.

Dependency `viz-bridge-injection` (PR #27) confirmed merged into
`cram-viz-integration` before starting, via `check_dependency_readiness.py`
against live GitHub state, not assumed from the manifest.

Filed under Group D ("medium refactors"), not one of the two dedicated
bug-fix items, so per plan convention this PR does not carry the
personal-notes `bug` label.

**Flag surfaced during research, not addressed by this item**: fork PR #18
(`warehouse-viz-features`) shows as closed, unmerged as of a live check
during this item's planning (`closed_at: 2026-08-07T07:57:20Z`), contradicting
this roadmap's earlier recorded "MERGEABLE, updated 2026-08-05" and the
"refactor first, rebase #18 afterward" decision that assumed it would still
be there to rebase. Noted here so `viz-kb-split`, which relies on that
decision, does not rediscover this as a surprise.

Branch: `cram-viz-semdt-geometry`, based on `cram-viz-integration`.
Draft PR: [sunava#29](https://github.com/sunava/cognitive_robot_abstract_machine/pull/29).

## `viz-kb-split` — rename `kb.py` to a `knowledge` package and split it into ~13 modules (T46, T27)

The item's own notes bundle T27 (a rename) with T46 (the split) but never
record T27's target name. Asked the user directly this session: the new
package is named **`knowledge`**, not `knowledge_base` — `knowledge_base.py`
is reserved as the inner submodule holding the `EpisodeKnowledgeBase` class,
which would collide with the package name if the package itself were also
called `knowledge_base`. `kb` is an abbreviation of "knowledge base",
which `AGENTS.md`'s no-abbreviations rule flags for a module name. The
sibling `cram_viz/src/cram_viz/live/` package (`bridge.py`/`hooks.py`/
`http.py`/`runner.py`) is the closest landed precedent for how a split
package is organized here, though `git log --follow` on `live/bridge.py`
shows it was authored as a package from day one — there is no in-repo
precedent for the *incremental extraction* this item needed, so the split
was designed from the source's actual call graph instead.

**The roadmap's original line-range table (above, "The kb.py split"
section) is now stale.** Several sibling PRs (`viz-bugs`, `viz-small-fixes`,
`viz-wire-rename`, `viz-bridge-injection`, `viz-onboard-dataclasses`) merged
into `cram-viz-integration` since that table was written and shifted line
numbers throughout `kb.py`. A fresh read of the full 2060-line file on
`origin/cram-viz-integration` found two call-graph corrections the file's
own (also stale) `# %%` section headers get wrong:

- `_measurement_line()` sits textually inside "scan the CRAM architecture"
  but is only ever called from `graph_payload()` — it moved to
  `graph_payload.py`, not `architecture_scan.py`.
- `_count_plan_nodes()` sits textually inside "the graph-panel tabs" next to
  `_plan_view()`, but its one call site is inside `graph_payload()`, not
  `_plan_view()` — it moved to `graph_payload.py`, not `views/plan.py`.

### Target layout

```
cram_viz/src/cram_viz/knowledge/
    __init__.py            # package docstring (from kb.py's) + thin re-export shim
    scene_bundle.py
    entities.py
    architecture_entities.py
    architecture_scan.py
    knowledge_base.py
    eql_session.py
    graph_payload.py
    presets.py
    views/
        __init__.py         # view_payload() / expand_node() dispatchers, _chart_view
        base.py
        plan.py
        kinematics.py
        architecture.py
```

`views/architecture.py` and `views/kinematics.py`'s references to
`EpisodeKnowledgeBase` are type-hints only — imported under `TYPE_CHECKING`
per `AGENTS.md`.

### Test strategy

`test/cram_viz_test/test_kb.py` (593 lines) imports only
`from cram_viz import kb` (never `from cram_viz.kb import X`) and touches
exactly 13 attributes. All 13 stay re-exported from `knowledge/__init__.py`
throughout the split, so the test file changes only once — in the first
(rename) commit, renamed to `test_knowledge.py` with `kb.` replaced by
`knowledge.` — not once per extraction commit. Splitting the test file
itself into per-module files (mirroring `live/`'s
`test_live_bridge.py`/... convention) is deferred; the item's own notes
name only the source split.

The `if __name__ == "__main__":` preset smoke block (iterates
`get_presets()`, runs each through `run_query()`, logs OK/FAIL) becomes a
real pytest test, `TestPresetSmoke`, per the item's own notes — then the
block is deleted from source.

No production behavior changes in this item — it is a pure structural
move, so the existing characterization coverage (from `viz-kb-characterization`,
#25) is the safety net for every commit; no new failing-test-first cycle is
needed per move, except for the smoke-test conversion itself.

### Commit sequence (suite green after each)

One `git mv kb.py knowledge/__init__.py` rename commit (updating the 4
external references: `server.py`, `conftest.py`, `test_kb.py` →
`test_knowledge.py`, `README.md:126`), then bottom-up extraction commits in
dependency order: `entities` → `architecture_entities` → `scene_bundle` →
`views/base` → `architecture_scan` → `knowledge_base` → (`eql_session`,
`presets`, `views/architecture`, `views/plan`, `views/kinematics` — mutually
independent) → `graph_payload` → `views/__init__` dispatcher (folding
`knowledge/__init__.py` down to a thin re-export shim) → the
`TestPresetSmoke` conversion.

### Also found, not part of this item's scope

`warehouse-viz-features` (fork PR #18) — the item that depends on this one
and shares the "refactor first, rebase after" trade-off recorded above in
"Live state and the #18 trade-off" — was found **closed unmerged** as of
2026-08-07T07:57:20Z, contradicting this document's recorded "MERGEABLE, 0
behind, updated 2026-08-05" status and `plan.yaml`'s `status: in_progress`
for that item. Nothing in `viz-kb-split` depends on #18, so this doesn't
block or change this item's plan — flagged here so a future session doesn't
rediscover the same drift and re-derive it. `warehouse-viz-features`'s own
manifest entry and status are unchanged by this note; they should be
corrected when that item is next touched.

Branch: `cram-viz-kb-split`, based on `cram-viz-integration`.
Draft PR: [sunava#30](https://github.com/sunava/cognitive_robot_abstract_machine/pull/30).

### Review feedback (2026-08-07) — one fixed here, five deferred to `viz-kb-dataclasses`

The user's own review on PR #30 left 6 comments. One was in this item's own
scope and is fixed: `views/plan.py` renamed to `views/plan_tree.py` (module
docstring updated, three import sites updated) since "plan" collided with
coraplex's own `Plan`/`PlanNode` types — this module renders the serialized
plan-node tree recorded in a scene bundle, not a coraplex `Plan` itself.
Identifiers (`PlanNodeGroup`, `_plan_view`, etc.) were left unchanged; only
the module name was in question.

The other five are all "no global variables" / "`Dict[str, Any]` payloads are
confusing" — exactly what `viz-kb-dataclasses` (T40/T42/T44/T45/T28/T29/T48)
is already scoped to fix, so they were left as open, unresolved threads on
PR #30 rather than fixed piecemeal here (this item is a pure structural move
with no behavior change). Concretely, when `viz-kb-dataclasses` starts, fold
these into its plan:

- `views/architecture.py`'s `CLASS_CAP`/`SUBCLASS_CAP` and
  `architecture_scan.py`'s `DESCRIPTION_LENGTH_LIMIT`/
  `ARCHITECTURE_CACHE_VERSION`/`SKIP_DIRS`/`PKG_DESCRIPTIONS` — module-level
  constants the reviewer wants off module globals (pre-existing, carried over
  verbatim from `kb.py`, not introduced by the split).
- `views/architecture.py`'s `_package_view`/`_subpackage_view`/`_class_view`
  and `views/plan_tree.py`'s `_plan_view` — and by extension every other view
  builder and `graph_payload()`/`view_payload()`/`expand_node()` — return
  bare `Dict[str, Any]` payloads with string keys; the reviewer wants real
  types.
- Collecting `PlanNodeGroup` (in `views/plan_tree.py`) and `ArmSide` (in
  `entities.py`) into a shared enums module was also raised and deferred here
  — pick their new home as part of the dataclass pass rather than moving them
  twice.
