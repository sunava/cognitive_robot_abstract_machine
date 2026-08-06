# cram-viz-wire-rename (T12) — plan/progress

**Plan**: rename the abbreviated `sig` field to `signature` everywhere it
crosses the live-bridge wire: `live/bridge.py`'s three dataclasses
(`PlanSnapshot`, `_ChartStructure`, `ChartSnapshot`), `live/http.py`'s API
contract docstring, `web/panels/graph/panel.js`'s reader, and both test
suites that assert on the key (`test_live_bridge.py`, `js/test_graph_panel.js`).

**Done**:
- Renamed the field + all read/construction sites in `live/bridge.py`.
- Updated `live/http.py`'s docstring.
- Updated `panels/graph/panel.js` (`live.sig` → `live.signature`).
- Updated `test_live_bridge.py` (6 assertions) and `test_graph_panel.js`'s
  live-mode mock.
- Opened draft PR sunava#26 against `cram-viz-integration`, subscribed to
  its activity.

**Next**:
- Confirm CI is green on #26 (full `test/cram_viz_test` suite incl. the
  JS suite via `test_web_assets.py`).
- Address any review feedback that comes in.
- Once merged into `cram-viz-integration`, mark `viz-wire-rename` `done` in
  the `fix-my-pr` manifest — `viz-bridge-injection` depends on it.
