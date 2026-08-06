# `viz-kb-characterization` (fix-my-pr plan, item viz-kb-characterization)

Branch: `cram-viz-kb-characterization`, based on `cram-viz-integration`.
Draft PR: https://github.com/sunava/cognitive_robot_abstract_machine/pull/25
Tracking issue: #19. Dependency `viz-bugs` (PR #20) confirmed merged.

## Plan (approved)

Tests-only PR — no production `kb.py` changes. Add characterization tests to
`test/cram_viz_test/test_kb.py` for `graph_payload()` and `expand_node()`
before `viz-kb-split` cuts `kb.py` into ~13 modules. Full plan reasoning is in
this plan's `roadmap.md` section (same content, appended via
`plan_item_bootstrap.py record`).

1. Fixture: add two files exercising inheritance (currently untestable —
   zero base/subclass relationships in the dataset):
   - `test/cram_viz_test/dataset/architecture/coraplex/src/coraplex/plans/typed_plan.py`
     — `TypedPlan(Plan)` (in-repo subclass/base).
   - `test/cram_viz_test/dataset/architecture/krrood/src/krrood/errors.py`
     — `EqlError(Exception)` (external base).
2. `TestGraphPayloadStructure` (new class in `test_kb.py`): robot/arm/gripper
   chain, episode chain edges, object detail lines, architecture cluster
   (packages/subpackages/import edges), `link()` grounding-edge guard clause
   (both present branch via `coraplex.plans` and absent branch via
   `giskardpy`/`semantic_digital_twin`, neither in the fixture), plan-tree
   cluster, status string (derived, not hardcoded).
3. `TestExpandNode` (new class in `test_kb.py`): every dispatch branch
   (robot/plan/package/subpackage/class/unknown), `_class_view`'s
   internal-base, external-base (`ext:`), and subclass-listing branches, and
   `CLASS_CAP`/`SUBCLASS_CAP` truncation notes (via synthetic `PythonClass`
   lists written directly onto the KB singleton, not real files).
4. `python -m pytest test/cram_viz_test -q` green throughout;
   `scripts/format_docstrings.py` on modified/added files.
5. Update PR description with final before/after test counts.

## Done so far

- Branch created, empty-commit bootstrapped, pushed.
- Draft PR #25 opened against `cram-viz-integration`.
- `plan.yaml`/`roadmap.md` recorded (`in_progress`, branch, PR number).
- Added both fixture files (step 1); confirmed via a scratch inspection
  script that `_class_view`'s internal-base, external-base (`ext:`), and
  subclass-listing branches now all fire against real fixture data.
- `TestGraphPayloadStructure` (8 tests) and `TestExpandNode` (11 tests)
  written in `test/cram_viz_test/test_kb.py`, all values taken from actual
  payloads (not hand-derived), per steps 2-3.
- Suite green throughout: `test_kb.py` 26 → 45 tests (155 → 593 lines);
  full `cram_viz` suite 144 → 163 passed. No production `kb.py` changes.
- `docformatter` had to be `uv pip install`ed into the workspace venv first
  (not a declared dependency); `scripts/format_docstrings.py` then run on
  all three modified/added files (step 4).
- Commit authored as `sunava <hassouna@uni-bremen.de>` (amended once locally
  before pushing — the sandbox's global git identity defaults to
  `Claude <noreply@anthropic.com>`, which AGENTS.md forbids as a commit
  author; nothing had been pushed yet at that point).
- PR #25 description updated with final before/after test counts (step 5)
  and pushed. Left in draft, as it already was.

## Next

- Nothing outstanding for this item. Ready for the user's own review of
  draft PR #25; this plan's cram2 PR stays untouched throughout.
