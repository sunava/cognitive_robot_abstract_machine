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

Open item noted in the plan, defaulted rather than blocked on: whether to also
manually load a bundled xacro-sourced scene in the browser viewer (via `/run`)
to fully address the roadmap's "verify the browser viewer still loads it"
caveat, versus relying on the new automated structural test plus a caveat note
in the PR description. Currently defaulting to the latter (backend-only
change, no `panels/graph/panel.js` touch) — flag to the user if this should
change.

## Done
- Branch `cram-viz-bundle-urdf-reuse` created off `cram-viz-integration`,
  bootstrap commit pushed.
- Draft PR #28 opened (WIP placeholder description).
- Dependencies `semdt-prefix-path-locator` (PR #21) and `viz-onboard-dataclasses`
  (PR #24) confirmed merged via `check_dependency_readiness.py` against live
  GitHub state.
- Plan recorded in `roadmap.md`.

## Next
- Step 1 above: write the failing `TestXacroToUrdfText` test, then implement
  the `from_xacro` swap.
