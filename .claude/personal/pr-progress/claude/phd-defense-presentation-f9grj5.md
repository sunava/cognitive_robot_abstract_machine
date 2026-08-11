# PhD defense presentation (CRAM-style)

## Plan
Build an interactive HTML presentation for Vanessa's PhD defense that runs
like a CRAM plan execution (slides = action designators, task-tree overlay,
PLAN SUCCEEDED finale). Content drafted from her public work (AKGs, action
cores, generalized cutting plans, PyCRAM/CRAM); thesis-exact wording pending.

## Done
- `presentations/phd_defense/index.html` + README committed and pushed
  (commits 073ce04e, 592c023d on claude/phd-defense-presentation-f9grj5).
- Smoke-tested with Playwright (no JS errors; screenshots checked).
- Published as artifact:
  https://claude.ai/code/artifact/669034ee-9fbc-449c-bf0c-3c511333eaa4
- Grounded slides in published work via web-search snippets (paper pages
  themselves are egress-blocked): Recipe1M+ 76.5% verb coverage +
  neuro-symbolic LLM fallback; action groups; SOMA/SOMA_HOME/SOMA_DFL;
  corrected publication list (Frontiers 2025 is Kümpel et al. with
  Hassouna as co-author; ICSR 2024 shared first authorship; AKR3@ESWC
  2024; K-CAP 2021 — author order of the latter two still unverified).

## Blocked / next
- Thesis PDF on Google Drive is unreachable from this environment (egress
  proxy blocks drive.google.com and uni-bremen.de). All thesis-specific
  content (exact title, date/committee, RQs, evaluation numbers, results,
  figures, publication list) is marked with class="todo" in the HTML and
  listed in the README. Next session: get the PDF (e.g. committed to the
  branch or pasted content) and replace the placeholders.
- No PR opened (not requested).
