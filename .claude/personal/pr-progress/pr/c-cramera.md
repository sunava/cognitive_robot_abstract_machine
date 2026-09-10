## Branch `pr/c-cramera` — the filtered cramera PR

Cut from `cramera-port` (30284460) onto `origin/main` (7be79555): only what the
cramera package needs, nothing of the demo/robot/experiment work that rides on
`cramera-port`. No PR opened yet.

**Structure (2 commits).**
1. Integration layer + its tests: coraplex (WorldVisualization/backends,
   plan callbacks with on_motion_tick, Designator+costmaps as Symbols,
   Attach/DetachNode, grasp robot-relative default, testing/demonstrations),
   semantic_digital_twin (numeric.py, numeric_global_pose, urdf material
   colors + relative mesh resolution, rerun log_current_state, predicates
   constant), segmind detectors (mirrors open mini PR #40 - merges clean once
   it lands; rerun/visualization layer likewise overlaps #36).
2. The cramera package, test/cramera_test, CI job (+node), version sync,
   workspace membership, .gitignore/.gitmodules + scenes submodule,
   start_demo.sh/start_viewer.sh.

**Deliberately excluded.** krrood verbalization changes (own mini PR
material; cramera imports only main-side krrood APIs - verified statically),
tf_publisher Ctrl+C fix (own PR), garmi/armar7 robot work, coraplex demos
(warehouse/chemistry/garmi/generated), experiments demos, montessori, root
scratch files (`-` wav, test.srdf, CHANGELOG_plan_constraints.md,
run_cramera.sh).

**Validation.** Static import check clean (every workspace symbol cramera and
the integration layer import resolves in-tree). Full uv-sync env built in the
session container; test runs: see below.

**Review done.** Fan-out review of all 279 files complete; ~700 findings in
.claude/claude_reviews/pr-c-cramera-20260910-090155.md. Plan approved (full
plan in /root/.claude/plans/vivid-noodling-naur.md of the session container;
mirrored by the report). Tranche 1 committed (d8933e02): 3 failing tests fixed
(--par CSS var, warehouse worlds/meshes removed, plan-view live expectation),
wiring (requirements.txt negation, poetry entry, shell-script robustness,
python floor 3.11, setup-node). Five fix agents running for tranche B
(behavior bugs) split by area: coraplex / sdt+segmind / cramera-live /
cramera-server+knowledge / web-JS.

**Next.**
- finish pytest runs (cramera / coraplex / segmind / sdt touched tests)
- /local-code-review with focus: docstrings + param docs everywhere, no
  abbreviations, no global variables, object-oriented design
- fix findings, push, open draft PR (bug label not needed - feature PR)
