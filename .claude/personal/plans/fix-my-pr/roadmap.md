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
