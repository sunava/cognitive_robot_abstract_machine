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
  against `cram-viz-integration`; sunava marked it ready and merged it.
- Updated `fix-my-pr`'s `plan.yaml`: `viz-submodule-org` is now
  `status: done`, `pull_request_number: 23`, blocker cleared.

## Next

Item complete — PR #23 merged into `cram-viz-integration`. Nothing further
for `viz-submodule-org`; the couldn't-run-pytest caveat from before merge
(sandbox lacks the project's Python deps) is moot now that it's landed.
