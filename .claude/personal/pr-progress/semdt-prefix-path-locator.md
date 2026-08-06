## semdt-prefix-path-locator (fix-my-pr plan, track: upstream)

PR: https://github.com/sunava/cognitive_robot_abstract_machine/pull/21 (draft)

**Done:**
- Added `PrefixPathPackageLocator` to
  `semantic_digital_twin/adapters/package_resolver.py`, searching
  `AMENT_PREFIX_PATH`/`CMAKE_PREFIX_PATH` plus `~/*_ws/install`, `~/*/install`,
  `/opt/ros/*` directly on disk (no ROS tooling required).
- Wired it into `ROSPackageLocator`'s default `locators` list.
- New test-first file `test/semantic_digital_twin_test/test_adapters/test_package_resolver.py`
  (9 tests, all passing). Confirmed the 3 failures + 11 errors in the rest of
  `test_adapters` are pre-existing (missing real ROS packages in this sandbox),
  identical on `main`.
- `format_docstrings.py` run on both changed files - no changes needed.
- Committed as Vanessa Hassouna (per AGENTS.md authorship rule), pushed, PR
  opened as draft, subscribed to PR activity.

**Next:**
- Watching PR #21 for CI + review comments (subscribed). Drive to green since
  this is a PR I created.
- Once merged: this item's `status` in `fix-my-pr`'s plan.yaml should move from
  `not_started` to `done`, and `pull_request_number: 21` recorded - do this via
  the plan-dashboard tooling / `/plan-dashboard fix-my-pr` conventions, not by
  hand-editing here.
- Unblocks `viz-bundle-urdf-reuse` (depends on this item): dropping
  `bundle_urdf.py`'s now-redundant `_search_root_candidates`/env-var-reading
  resolver stack in `cram_viz`. Out of scope for this PR.
