# `fix-my-pr` / `viz-onboard-dataclasses`

Draft PR [sunava#24](https://github.com/sunava/cognitive_robot_abstract_machine/pull/24),
branch `cram-viz-onboard-dataclasses`, base `cram-viz-integration`. Closes
review threads T22/T39 (`Recorder`) and T37 (`BundleReport`).

## Done

- `BundleReport` dataclass added to `cram_viz/src/cram_viz/onboard/bundle_urdf.py`,
  replacing `bundle_urdf()`'s ten-key return dict. Updated `main()`'s reads,
  `demo.py`'s bundling loop (~lines 868-903), and `test_onboard.py`'s
  `TestBundleUrdf` assertions (subscript → attribute; written first so they
  failed against the dict-returning code, then passed once the dataclass
  landed). Dropped the now-unused `Any` import from `bundle_urdf.py`.
- `Recorder` converted to `@dataclass` in `demo.py` — same field
  names/types/docstrings as the previous 65-line `__init__`,
  `field(default_factory=...)` for every mutable container. No call site
  changes needed. Added `TestRecorderMutableDefaults` (two `Recorder()`
  instances must not share their mutable field objects) as the regression
  guard for the one real hazard this conversion carries.
- Environment note for whoever picks this up next: the sandboxed session
  python3 is 3.11, but this repo's `pyproject.toml` requires `>=3.12,<3.13`
  (`dataclasses.make_dataclass()`'s `module=` kwarg, used deep in krrood's
  class-diagram introspection, is 3.12+ only). Built a Python 3.12 venv at
  `/root/.venvs/cram-viz-onboard` with `semantic_digital_twin`, `krrood`,
  `cram_viz`, `giskardpy`, `coraplex`, `probabilistic_model` installed
  `--no-deps -e`, plus their actual runtime imports installed piecemeal
  (not `pip install -r requirements.txt` — several transitive deps in those
  files fail to build under this environment's setuptools/distutils, e.g.
  `dnutils`, `polytope`, `arff`). That venv is local to this container and
  won't survive a fresh session/container.
- Suite: `python -m pytest test/cram_viz_test -q` (via that venv,
  `--ignore=test/cram_viz_test/test_live_hooks.py` — that file needs
  `matplotlib`/`pandas`/`PyQT5`/etc. through `coraplex` → `giskardpy`'s QP
  solver plotting, unrelated to this change) — 152 passed (151 baseline + 1
  new test). Same 4 pre-existing failures as baseline (`test_kb.py`'s EQL
  query tests, `test_server.py`'s EQL roundtrip — all missing the `jpt`
  package deep in `probabilistic_model`, unrelated to `Recorder`/
  `BundleReport`). `scripts/format_docstrings.py` run on all three modified
  files.
- Commit `7264cc4f`, authored `sunava <hassouna@uni-bremen.de>` — the
  session's git config was set to `Claude <noreply@anthropic.com>` (also
  true of the earlier bootstrap commit `6f7d5224` on this branch, not yet
  fixed), so used `git commit --author` for this one rather than touching
  global config.
- PR #24 description updated to match; still draft, per personal-notes
  convention.

## Next

- User review of the diff itself (2 files + 1 test file, scoped exactly to
  T22/T39/T37).
- Once reviewed, this stays a draft until told to mark ready — per
  personal-notes convention, no bug label needed (this is a refactor, not a
  bug fix).
- After merge: re-run `/plan-dashboard fix-my-pr` to pick up the merge.
