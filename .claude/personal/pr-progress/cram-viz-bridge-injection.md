# cram-viz-bridge-injection — viz-bridge-injection (fix-my-pr plan)

Draft PR: sunava#27, based on cram-viz-integration. Implemented, suite green,
still draft (per convention, drafts flip to ready only when explicitly told
to).

## Plan (see full design in the approved kickoff plan; summarized here)

1. `http.py`: drop `BRIDGE` import. Give `BridgeRequestHandler` an
   `__init__(self, *args, bridge: Bridge, **kwargs)` that sets `self.bridge`
   before calling `super().__init__(*args, **kwargs)`. Replace the 7
   `BRIDGE.` call sites in `do_GET`/`_send_mesh`/`do_POST` with `self.bridge.`.
   `serve(bridge: Bridge, port: int = DEFAULT_PORT)` builds the handler via
   `functools.partial(BridgeRequestHandler, bridge=bridge)`.
2. `runner.py`: `start()` keeps its existing signature (`world`, `port`) —
   it's the documented public entry point. Bind `bridge = BRIDGE` once at the
   top of the function (mirroring `hooks.py`'s
   `_LIVE_HOOKS = LiveHooks(bridge=BRIDGE)`), use `bridge.` for the rest of
   the body, pass `bridge` into `serve(bridge, port)`.
3. `bridge.py`: add `ChartEdgeEntry.to_payload() -> Dict[str, str]` returning
   `{"from": self.source, "to": self.target, "kind": self.kind}`; simplify
   `get_chart()` to `payload["edges"] = [edge.to_payload() for edge in
   chart.edges]` instead of hand-rewriting inline (T16 residue).
4. Tests first (TDD): `ChartEdgeEntry.to_payload()` unit test in
   `test_live_bridge.py`; new `test_live_http.py` (real server on an
   ephemeral port via `serve(bridge, 0)`, one test per endpoint, plus a
   two-independent-bridges test proving no shared global); new
   `test_live_runner.py` covering `start()`'s control flow via `monkeypatch`
   (substitute `runner.BRIDGE` with a fresh `Bridge()` per test, no-op the
   `hooks.install_*` calls, spy on `serve`) — `hooks.install_*` monkey-patches
   coraplex/giskardpy process-globally with no uninstall, so real calls must
   not happen in tests.
5. Explicitly out of scope: `hooks.py`'s three `install_*` functions'
   `BRIDGE.claim_hook(...)` calls stay as-is (not named in this item's
   notes); the other four `Dict[str,Any]` wire-boundary methods on `Bridge`
   stay as-is (this item's notes narrow T16's residue to just `get_chart()`).
6. No wire-format change — `/chart`'s JSON shape is unchanged, so no
   `panels/graph/panel.js` update needed.

## Done so far

- Branch created, empty bootstrap commit pushed, draft PR #27 opened.
- `plan.yaml` item flipped to `in_progress`, roadmap section recorded.
- All three sets of failing tests written first (confirmed failing against
  the pre-change code), then `bridge.py`/`http.py`/`runner.py` implemented
  per the plan above, one file at a time, tests green after each.
- Full suite green: `uv run pytest test/cram_viz_test -q` — 181 passed (was
  172 before this branch's 9 new tests).
- `scripts/format_docstrings.py` run on all 6 modified/added files
  (`docformatter` had to be `uv pip install`ed this session — it wasn't
  present, unlike the note on PR #26 which found the same gap).
- Committed as `Vanessa Hassouna <33067562+sunava@users.noreply.github.com>`
  (matching every prior commit in this repo's history) rather than this
  environment's global git config default (`Claude <noreply@anthropic.com>`,
  which the bootstrap commit was accidentally made under) — used
  `git -c user.name=... -c user.email=...` per AGENTS.md's authorship rule,
  without touching the global config itself. The bootstrap commit's own
  wrong authorship was left as-is rather than rewriting already-pushed
  history unasked; flagged to the user in chat.
- Pushed, PR #27 description updated with the real diff and suite numbers.

## Next

- Nothing blocking — implementation is complete and the suite is green.
- If the user wants PR #27 marked ready for review, do that explicitly (per
  personal notes, drafts flip to ready only on explicit request, and a PR
  goes back to draft after any further push).
- Downstream: `viz-semdt-geometry` depends on this item; it can start once
  this PR merges into `cram-viz-integration`.
