# PhD Defense Presentation — "the talk as a CRAM plan"

A self-contained HTML presentation (`index.html`) that behaves like a CRAM plan
execution: every slide is an action designator, the talk has a live task tree,
and the final slide reports `PLAN SUCCEEDED` with the real talk runtime.

Open `index.html` in any browser. No build step, no network access needed.

## Controls

| Key | Action |
| --- | --- |
| `→` / `Space` / `PageDown` / click | next slide |
| `←` / `PageUp` | previous slide |
| `T` | toggle the plan / task-tree overlay (click a node to jump) |
| `N` | toggle speaker notes (bottom panel) |
| `F` | fullscreen |
| `Home` / `End` | first / last main slide |

Backup slides live after the final slide — reach them with `→` past
"Thank you" or via the task tree (`T`).

## Filling in thesis content

Every placeholder is marked with `class="todo"` (rendered with a dashed amber
underline; hover shows what belongs there). Current TODO list:

- [ ] Exact thesis title (title slide)
- [ ] Defense date and committee (title slide)
- [ ] Exact wording of the research questions (RQ slide)
- [ ] Robots, objects, run counts of the evaluation (setup slide)
- [ ] Three headline results with real numbers (results slide)
- [ ] Contributions exactly as numbered in the thesis
- [ ] Limitations as discussed in the thesis
- [ ] A still image of the real robot execution (parameter-resolution slide)
- [ ] The central results figure (results slide)
- [ ] Complete publication list, verify author order (backup slide)

Speaker notes are the `data-notes` attribute on each `<section class="slide">`.

## Export to PDF

Print the page (Ctrl+P) — a print stylesheet lays out one slide per page.
