## Plan

Resolve `fix-my-pr` plan item `viz-submodule-org` (T51): the
`cram-scenes` submodule was blocked on `cram-scenes` moving into the
`cram2` org. Verified `cram2/cram-scenes` now exists and contains the
exact pinned submodule commit, so this is a URL-only retarget, no rebase.

## Done

- Found a third hardcoded-URL site (`cram_viz/README.md:22`) beyond the two
  `plan.yaml`'s note listed (`.gitmodules:3`, `paths.py:7`).
- Made the three one-line edits, verified `git submodule update --init`
  clones from the new URL and checks out the same pinned commit
  (`54df924`).
- Committed (as the human author, per AGENTS.md) and pushed this branch.
- Opened draft PR [#23](https://github.com/sunava/cognitive_robot_abstract_machine/pull/23)
  against `cram-viz-integration`.
- Updated `fix-my-pr`'s `plan.yaml`: `viz-submodule-org` is now
  `status: in_progress`, `pull_request_number: 23`, blocker cleared.

## Next

- Could not run `python -m pytest test/cram_viz_test -q` or
  `scripts/format_docstrings.py` in this sandbox — the project's Python
  dependencies (pytest, typing_extensions, etc.) aren't installed and
  installing the full monorepo is out of scope for this fix. No test
  references the changed strings, and the submodule-clone verification
  above is the direct proof this specific change works.
- Waiting on review/CI on PR #23, then merge into `cram-viz-integration`
  and flip `viz-submodule-org.status` to `done` (matching the `viz-bugs`/
  PR #20 convention: `done` means merged into `cram-viz-integration`, not
  just opened).
