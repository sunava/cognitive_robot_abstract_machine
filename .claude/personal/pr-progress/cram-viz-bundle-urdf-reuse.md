# cram-viz-bundle-urdf-reuse (PR #28) — viz-bundle-urdf-reuse, fix-my-pr plan

Closes review threads T33 and T34 against cram2#485. Full plan recorded in
`fix-my-pr`'s `roadmap.md` under the `viz-bundle-urdf-reuse` heading; this is
the working summary.

## Plan
1. **T34** (do first, the "clean half"): add `TestXacroToUrdfText` to
   `test_onboard.py` (new, fails today — no `xacro` CLI in this sandbox to
   exercise the current subprocess-based implementation). Replace
   `bundle_urdf.py`'s `xacro_to_urdf_text()` body with
   `URDFParser.from_xacro(path).urdf`; drop `import subprocess` and
   `XACRO_ERROR_TAIL`. Add `TestBundleUrdf.test_a_xacro_source_is_bundled_like_a_urdf_source`
   (new, end-to-end, proves `hacky_urdf_parser_fix`'s ElementTree round-trip
   doesn't break the regex-based mesh rewriting).
2. **T33**: update `TestResolveUri.test_an_unresolvable_package_uri_is_unresolved`'s
   docstring only (assertion unchanged) to describe one delegated call instead
   of three fallback tiers. Delete `_search_root_candidates()`; simplify
   `_resolve_package_uri()` to delegate entirely to
   `PackageUriResolver().resolve(uri)` (now covers the same three tiers via
   PR #21's `PrefixPathPackageLocator`). Drop `import glob`.
3. `uv sync --extra dev --active` (this sandbox has neither `pytest` nor
   `xacro` installed yet), then `python -m pytest test/cram_viz_test -q` green
   throughout. `scripts/format_docstrings.py` on both modified files.
4. Update PR #28's description from the WIP placeholder to the real summary
   (suite results, files touched) once the suite is green.

Open item, decided rather than left dangling: the roadmap's "verify the browser
viewer still loads it" caveat is addressed via the automated structural test
(`test_a_xacro_source_is_bundled_like_a_urdf_source`) plus a caveat note in the
PR description, not a manual `/run` browser check - this is a backend-only
change (no `panels/graph/panel.js` touch, no wire-format change). Flag to the
user if this should change.

## Done
- Branch `cram-viz-bundle-urdf-reuse` created off `cram-viz-integration`,
  bootstrap commit pushed (author corrected to `sunava <hassouna@uni-bremen.de>`
  after it was accidentally created under the harness's default `Claude
  <noreply@anthropic.com>` identity - amended, then force-pushed).
- Dependencies `semdt-prefix-path-locator` (PR #21) and `viz-onboard-dataclasses`
  (PR #24) confirmed merged via `check_dependency_readiness.py` against live
  GitHub state.
- Plan recorded in `roadmap.md`.
- **T34 implemented**: `TestXacroToUrdfText` added (proven failing first - no
  `xacro` CLI in this sandbox), then `xacro_to_urdf_text()` replaced with
  `URDFParser.from_xacro(path).urdf`; dropped `import subprocess` and
  `XACRO_ERROR_TAIL`. `TestBundleUrdf.test_a_xacro_source_is_bundled_like_a_urdf_source`
  added (end-to-end, passes).
- **T33 implemented**: `TestResolveUri.test_an_unresolvable_package_uri_is_unresolved`'s
  docstring updated (assertion unchanged); `_search_root_candidates()` deleted;
  `_resolve_package_uri()` simplified to delegate to
  `PackageUriResolver().resolve(uri)`; dropped `import glob`.
- Environment set up (`uv sync --extra dev --active`; `black`/`docformatter`
  pinned versions from `.pre-commit-config.yaml` installed separately since
  they're pre-commit-only deps, not part of the `dev` extra).
  `python -m pytest test/cram_viz_test -q` — 184 passed (182 baseline + 2 new),
  0 failures. `scripts/format_docstrings.py` run on both modified files.
- Real commit pushed (`8f013bc9`), PR #28 description updated with the full
  summary, subscribed to PR activity, 60-minute check-in scheduled.

## Next
- Item complete pending review/CI. Watch for CI results and any review
  comments via the PR subscription; nothing else planned for this item.
