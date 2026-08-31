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

**Cannot be verified here.** No environment on this machine has coraplex,
cramera or segmind installed (krrood and semantic_digital_twin only), `test/conftest.py`
needs objgraph, and there is no node for the JS tests. Everything above was
checked statically (parse + pyflakes clean, no undefined names, endpoint/JS
contract matched by hand) and black-formatted; none of it has been executed.
