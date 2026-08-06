# cram-viz-wire-rename (T12) — plan/progress

**Plan**: rename the abbreviated `sig` field to `signature` everywhere it
crosses the live-bridge wire: `live/bridge.py`'s three dataclasses
(`PlanSnapshot`, `_ChartStructure`, `ChartSnapshot`), `live/http.py`'s API
contract docstring, `web/panels/graph/panel.js`'s reader, and both test
suites that assert on the key (`test_live_bridge.py`, `js/test_graph_panel.js`).

**Done**:
- Opened draft PR sunava#26 against `cram-viz-integration` as a bootstrap
  (empty commit), subscribed to its activity.
- On the first CI check-in, found the bootstrap commit had no real diff and
  CI had never triggered (0 workflow runs) — the rename itself had not
  actually been implemented yet despite this note previously claiming so.
  Did the real implementation and pushed it as a second commit (`8ad32922`):
  renamed the field + all read/construction sites in `live/bridge.py`,
  updated `live/http.py`'s docstring, `panels/graph/panel.js`
  (`live.sig` → `live.signature`), `test_live_bridge.py` (6 assertions) and
  `test_graph_panel.js`'s live-mode mock.
- `uv run pytest test/cram_viz_test -q` (incl. the JS suite via
  `test_web_assets.py`): 144 passed.
- Ran `black` on the touched Python files. `docformatter` could not be
  installed this session — PyPI fetches for that one package
  (`docformatter-1.7.8`) consistently timed out through the proxy after
  several retries — so `scripts/format_docstrings.py` (which shells out to
  it) couldn't run. No docstring *text* changed in this diff (only the field
  name above each docstring), so the gap is low-risk, but it should be
  re-run once docformatter is installable.

**Next**:
- Confirm CI actually goes green on #26 now that there's a real commit for
  it to run against.
- Try `scripts/format_docstrings.py` again when PyPI access is reliable.
- Address any review feedback that comes in.
- Once merged into `cram-viz-integration`, mark `viz-wire-rename` `done` in
  the `fix-my-pr` manifest — `viz-bridge-injection` depends on it.
