## fix/docker-bootstrap-stdlib-only - cram2/cognitive_robot_abstract_machine#818 (draft, bug)

Plan: update_docker on cram2 main fails since #807 (typing cleanup): .github/docker/setup_workspace.py runs
with the bare ros:jazzy Python and imported typing_extensions (unused Optional). Remove the import, add a
test that the script imports only stdlib modules.

Done 2026-10-28:
- 0be22fe1f (sunava fork branch, PR from sunava:fix/docker-bootstrap-stdlib-only): import removed,
  test/cognitive_robot_abstract_machine_test/test_docker_workspace_setup.py (fails on main with
  {'typing_extensions'}, passes with the fix); krrood format + pre-commit hooks pass; python3 -I -S check.
- Context: user's PR #814 is red only from main's krrood failure (NamedTemporaryFile(directory=), fix #816 by
  LucaKro, approved) and the GARMI demo CollisionViolatedError (fix #815 by Tigul, draft).

Next:
- wait for CI on #818 and reviews; mark ready only when the user says so.
- update_docker itself only runs on push to main touching .github/docker/*, so the fix is proven in
  the image build only after merge.
