## Branch `cramera-port` — porting the cramera EQL-console stack

Source: the cramera/montessori stack in the AbdelrhmanBassiouny fork,
#169 -> #170 -> #164 -> #165 -> #167 -> #168 (tip mirrored locally as
`cramera-voice-stack`). No PR opened yet.

**Plan.** Port feature by feature onto `cramera-world-visualization` rather than
merging: the two branches are divergent generations of cramera (the stack builds
on the monkey-patch bridge our branch replaced with `LiveVisualization`), so a
merge was 39 conflicts / ~110 hunks and was abandoned.

**Done (2 commits).**
- #170 infra: DetectionEvent/AbstractDetector as Symbols, krrood verbalization
  (match groups, cardinality, unimportable-import scope), coraplex costmap and
  designator symbol graphs, their tests.
- The EQL console at the stack's generation: knowledge layer (query_runner,
  query_vocabulary, queryable_knowledge, query_domain, query_verbalization,
  workspace_classes, question_matching, database_evaluation, replay), keeping our
  transforms view / bundle-signature cache / local-scene shadowing; web modules
  (completion, suggestions, question_display, voice, answer_table, query_source,
  preset_groups, replay, highlight_arrow); endpoints on both servers
  (/api/eql/vocabulary, /api/eql/members, /api/question, and /presets, /eql,
  /vocabulary, /members, /question on the live bridge); Bridge query surface +
  live/query.py; rapidfuzz requirement; the stack's tests for all of it.

**Not ported, and why.** #164 (where-is highlighting) and #165 (event replay)
are features of the Montessori demo, and `experiments/src/experiments/montessori`
exists only on the stack — bringing them means bringing that package (#169/#202)
first. Replay buttons will render for rows carrying timestamps but the live
bridge has no `/replay`: our own recording layer would have to serve the clip.

**Next.** Verify in a working environment (see below), then decide whether to
bring the montessori package for the live-demo features.

**Verified.** `cram-env` now has every workspace package (it was missing
coraplex/cramera/segmind/experiments earlier), plus rapidfuzz and ruff, which I
installed - ruff is not in any extra but krrood shells out to it.

    test/cramera_test  ->  749 passed, 22 skipped   (skips = JS tests, no node)
    test/segmind_test  ->   45 passed,  1 skipped
    test/krrood_test   -> 2156 passed,  6 skipped

Fixed while getting there: Cache-Control no-cache -> no-store (a real bug the
stack had already fixed); dropped the ported tests for endpoints this branch has
no demo for (/run, /events, /replay, the live model catalog) and held /info to
the fields the bridge reports; restored this branch's segmind detector test,
whose ported version imports HoleContactDetector and so interrupted the whole
segmind collection.

**Open question for the developer.** `MINIMUM_SIMILARITY` (70) in
question_matching lets "what is the weather like today" match "what is in the
scene?" at 71.0 on the fixture scene - the PR's calibration holds for the
montessori wordings, not for short generic ones. Not touched: the number was
chosen deliberately.

**A scene to try it on.** ~/.local/share/cramera-demo/run.sh serves the test
fixture (13 presets, no download). The ten real recordings already in
~/.cramera/scenes answer 10 presets each; pr2_breakfast in the scenes submodule
is the richest real one. Not Franka_Montessori: 20 of its 25 presets are
requires_live and need the montessori package.
**Statecharts in recordings (c1d8fa054).** Serialization moved out of the live
bridge into `cramera/live/chart_structure.py` + `chart_observer.py`, shared by the
bridge and the onboarder; a scene bundle now carries `statecharts.json`. The
observer answers two questions, not one: `snapshot()` is the chart's current
state, `change()` is what a viewer that already holds the last one needs. A
recording must not dedupe the wire's way -- a recorded tick with nothing to say
means the chart stopped, not that it stood still -- which had left 85 of 971
frames carrying a chart; deduping in `RecordedStatecharts` keeps the file at 25 KB.

**Segmind watch set (same commit).** A body entered it only while free, so an
object a demo starts off fixed inside a drawer (the bullet-world spoon) was never
seen picked up -- 2 of 3 transports detected. The world already separates the
cases: `Spoon` is `IsPerceivable`, `Drawer` is `Furniture`, so the watch set
follows that annotation too.

**Next.**
- Live questions: no demo calls `register_query_source`; wire `DetectedEvents`
  plus a live query source into one.
- Re-record `pr2_breakfast_detected` and `bullet_world_detected` so they gain
  statecharts and the spoon pick-up the new rule sees. `g1_warehouse_wrench` is
  already re-recorded (971 frames, 6 charts, 4 pick/place events).
- `~/.cramera/scenes/index.json` still defaults to the non-existent `pr2_scuess`.
- Onboarding a second scene while one runs dies with a raw
  `OSError: Address already in use` on bridge port 8765; deserves a real message.
