# Kickoff of `fix-my-pr` / `viz-onboard-dataclasses`

This session's own branch (`claude/plan-item-kickoff-viz-onboard-9fxn0e`) is
the kickoff session, not the item's implementation branch. The item itself
lives on `cram-viz-onboard-dataclasses` (draft PR
[sunava#24](https://github.com/sunava/cognitive_robot_abstract_machine/pull/24),
base `cram-viz-integration`) — bootstrapped and `record`ed into
`fix-my-pr`'s `plan.yaml`/`roadmap.md` as `in_progress`.

## Plan (approved)

Two mechanical dataclass conversions, closing review threads T22/T39 and T37:

1. **`BundleReport`** dataclass in `cram_viz/src/cram_viz/onboard/bundle_urdf.py`,
   replacing `bundle_urdf()`'s ten-key return dict. Update `main()`'s
   dict-subscript reads, `demo.py`'s bundling-loop call site
   (~lines 868-903), and `test_onboard.py`'s `TestBundleUrdf` assertions
   (subscript → attribute — write these first so they fail, then land the
   dataclass). Drop the now-unused `Any` import from `bundle_urdf.py`.
2. **`Recorder` → `@dataclass`** in `cram_viz/src/cram_viz/onboard/demo.py` —
   same field names/types/docstrings as the current 65-line `__init__`,
   `field(default_factory=...)` for every mutable container. No call site
   changes (every `Recorder()` use takes no args). Add one new test:
   two `Recorder()` instances must have distinct list/dict attribute objects
   (guards against `field(default=[])` by mistake).
3. Run `python -m pytest test/cram_viz_test -q` throughout, keep green.
   `scripts/format_docstrings.py` on every modified file before opening for
   review.

Full plan detail: see the `viz-onboard-dataclasses` section this session
appended to `fix-my-pr`'s `roadmap.md`.

## Done so far

- Branch `cram-viz-onboard-dataclasses` created off `origin/cram-viz-integration`,
  pushed with an empty bootstrap commit.
- Draft PR #24 opened on `sunava/cognitive_robot_abstract_machine`.
- `plan.yaml`/`roadmap.md` updated and pushed (item now `in_progress`, branch
  and PR number recorded).
- Confirmed dependency `viz-small-fixes` (PR #22) merged into
  `cram-viz-integration` before branching.

## Next

- Implement Part 1 (`BundleReport`) test-first, then Part 2 (`Recorder`
  dataclass + the new mutable-default regression test).
- Push commits to `cram-viz-onboard-dataclasses`, keep PR #24's description
  in sync, keep it in draft.
- Republish `/plan-dashboard fix-my-pr` after implementation lands.
