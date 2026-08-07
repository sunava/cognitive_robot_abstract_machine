# `cram-viz-kb-dataclasses` (fix-my-pr / viz-kb-dataclasses, draft PR sunava#31)

Plan approved this session; full detail is in `roadmap.md`'s
`viz-kb-dataclasses` section. Only the bootstrap commit is pushed so far —
implementation has not started.

## Open question (surfaced in the plan, not yet answered)

T44 has no description anywhere in the plan's triage notes. Can't resolve it
without access to the upstream `cram2#485` PR, which is out of this session's
GitHub scope. Proceeding without it per the approved plan; ask the user or
pick it up in a later pass once identified.

## Plan (commit sequence, suite green after each)

1. `enums.py` (new) — `ArmSide` moved from `entities.py` + new `UNKNOWN`
   member (replaces the `"n/a"` fallback in `knowledge_base.py:145`,
   test-first), `NodeGroup` (replaces `PlanNodeGroup` + every bare group
   string), `EdgeKind` (`PROP`/`TYPE`).
2. `views/base.py` — `GraphNode`/`DetailEntry`/`GraphEdge` dataclasses +
   `SubgraphAccumulator` replacing `_view()`'s bare-tuple closure.
   `GraphEdge.to_payload()` maps `source`/`target` → `from`/`to` (mirrors
   `ChartEdgeEntry.to_payload()` from `viz-bridge-injection`).
3. `presets.py` — `Preset` dataclass, `ARCH_PRESETS` → `Tuple[Preset, ...]`.
4. `architecture_scan.py` — new `ArchitectureScanner` class (constants as
   class attributes, answers the "no global variables" review threads);
   `scan()`/`load()` return a new `ArchitectureScan` dataclass built from
   real `Package`/`PythonClass`/`PackageDependency` objects directly
   (removes the dict-then-convert indirection — this is T29). New test-first
   coverage for the dict-free construction.
5. `entities.py`/`architecture_entities.py` — `Gripper.side`/`Arm.side` →
   `ArmSide` (T45); `PythonClass.bases` → `Tuple[str, ...]` (was bare
   `tuple`). `views/architecture.py` — `CLASS_CAP`/`SUBCLASS_CAP` off-global.
6. `scene_bundle.py` — `SceneBundle`, `ParsedUrdf`/`UrdfJoint` dataclasses
   replacing `load_scene()`/`load_urdf()`'s tuple returns (T28).
7. `eql_session.py` — `QueryResult` dataclass replacing `run_query()`'s dict
   and `_result_rows()`'s tuple.
8. Per-view payload dataclasses (`KnowledgeGraphPayload`,
   `PackageViewPayload`/`SubpackageViewPayload`/`ClassViewPayload`,
   `UrdfViewPayload`, `PlanViewPayload`, `ChartViewPayload`) — T40's "nine
   payload dicts". Each gets `to_payload()`; `server.py`'s `_json()` switches
   to calling it before `json.dumps` (the one call site outside the package).
9. `scripts/format_docstrings.py` on every modified file.

`test_knowledge.py` (sole test file) assertions move from dict-subscript to
attribute/dataclass-equality access in the same commit as each conversion —
no separate failing-test-first cycle needed except for the two items called
out above (T28/T40/T29/T45/enums are pure structural moves, per
`viz-kb-split`'s own precedent).

## Done so far

- Branch `cram-viz-kb-dataclasses` created off `origin/cram-viz-integration`,
  bootstrap commit pushed.
- Draft PR sunava#31 opened, no `bug` label (Group D refactor, not a bug fix).
- `roadmap.md`'s `viz-kb-dataclasses` section written with the full design
  and the T44/T42/thread-resolution flags.

## Next

Start commit 1 (`enums.py`) test-first, then work down the sequence above.
Run `python -m pytest test/cram_viz_test -q` after each commit.
