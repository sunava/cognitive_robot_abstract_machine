# Branch: claude/thesis-presentation-02p426 — defense deck

Two deliverables now:
1. LOCAL deck (primary, user request "lass uns das lokal machen"): defense/
   on this branch — deck HTML + deck_player.js (rendering/URDF/material core
   reused from cramera panel.js, vendor/ copied from cram-viz-integration
   branch). Loads full bundles (meshes+textures+full trajectory) from
   defense/scenes (clone/symlink of cram2/cram-scenes, gitignored). Title
   slide background + episodes slide share ONE viewer, re-parented on slide
   change (DeckPlayer.onSlide); title = autoRotate ambient. Run:
   cd defense && git clone https://github.com/cram2/cram-scenes scenes &&
   python3 -m http.server 8123. Committed + pushed.
2. Artifact (shareable fallback, self-contained v3 vertex-color renderer):
   https://claude.ai/code/artifact/f96ca126-240d-4cc6-8f91-f9b14cb20de6

Done:
- 20-slide "cram_viewer" deck (earlier sessions).
- Slide 14 "recorded episodes": real cramera scene bundles
  (cram2/cram-scenes PR2/HSR/TIAGO_Apartment), trajectories downsampled
  ~10 fps as sparse joint tracks (~210 KB) and replayed with a JS FK
  player: robot switcher, plan-step chips, scrub/play/speed, orbit camera,
  base-path ribbon, light/dark themes. Later slides renumbered 15-21.
- v2/v3 after user feedback ("not as good as my cramera viewer"): raw-WebGL
  renderer inside the deck (no libs) with REAL URDF visual meshes.
  v3 (current): per-vertex smooth normals (int8) + per-vertex colors
  sampled from the DAE baseColorTextures at UVs (scipy cKDTree carries
  colors through fast-simplification decimation, cap 1300 tris/link);
  robots get their true textured look (PR2 blue head display, HSR dark
  body, TIAGo white), env colors from textures or URDF materials; smooth
  two-light lambert + mild gamma, planar contact shadows, checkerboard
  floor shader, wheel zoom, object legend in HUD (like cramera viewer).
  meshlib.json ~6.5 MB / 260k tris shared across scenes; deck ~6.9 MB.
  Walls/windows compiled with w:1 flag but skipped at draw (front-cull
  attempt left edge artifacts) -> future ghosted-walls toggle possible.
  Apartment FK cached; SwiftShader detected -> dpr 0.8 (headless preview
  is slow, any real GPU is fine).
- Scripts in session scratchpad: compile_scene.py (tracks),
  compile_meshes_v3.py (meshlib v3; compile_meshes.py = v2, obsolete),
  fk_preview.py, screenshot_deck.py, episode_player_gl.js (source of the
  injected player; block-replace between the 'recorded cramera episodes'
  marker and '/* -- theme switching' in defense.html, injecting EPISODES
  + MESHLIB json). Regenerate: clone cram2/cram-scenes, run compilers.

Videos (committed): s18 real-robot 2x2 video wall (cutting bread/cucumber/
zucchini + pouring, designator captions), s11 FT/robot-view video
(max-height capped), s12 PiP real-execution panel per action (cut/pour/
wipe->spreading sim). defense/videos/ gitignored; copy from
sunava/sunava.github.io files/ (README has the command). Play/pause via
onSlide; muted+playsinline for autoplay. Note: headless test Chromium has
no H.264 -> verified mechanics with a VP9 test clip; real browsers fine.

Narrative (frames 1-7 now follow the five research-talk questions):
02 problem + explicit problem statement, 03 NEW why it matters (coverage /
representational gap / guarantee-not-average), 04 why hard + why the obvious
routes fail (trajectory / action name / e2e policy), 05 RQ+hypotheses,
06 proposed solution (difficulty->solution-element mapping, gamma grounding
function), 07 NEW claims C1-C4 up front, paid off on frame 24 (tagged C1-C4,
after the reviewer-questions frames — see below).
OAAT frame states B_g = gamma(D, W_SDT, K_KG) and executable(O_A,B_g).
25 frames total (was 23); slide element IDs unchanged (s1..s20 +
sWhy/sContrib/sEp/sQ1/sQ2), so JS hooks (s8 minis, s10 gprows, s12 demo,
s20 closing) still match.
NOTE: the shareable Artifact is now BEHIND the local deck (it still has the
old 21-frame narrative + embedded v3 meshes); regenerate only if asked.

Since defense review (2026-08-18): read both Gutachten (Downloads/
"Hassouna - Lima-2.pdf" = Pedro Lima, summa cum laude; Downloads/
"PhD Gutachten Vanessa Hassouna.pdf" = Michael Beetz, magna cum laude) and
addressed their critique points in the deck rather than just noting them:
- Added two slides sQ1/sQ2 with prepared answers to the reviewers' toughest
  points: Lima's pre-execution diagnosability/platform-switch question,
  Lima's "noise-free SDT" (Ch 7.5) real-world-validity point, Lima's
  implementation-complexity/parameter-count point, and Beetz's
  representational-depth/"same template" identity point (the one he
  flagged as the summa-vs-magna line). Framed as honest, thesis-consistent
  answers (what's already built vs named future work), not deflection.
  First built as backup/appendix slides after s20 (only open if asked) —
  user then decided (after discussing the tradeoff) to move them INTO the
  main sequence instead, as frames 22-23 right after "hypotheses revisited"
  (s18, frame 21) and before "contributions revisited" (s19, now frame 24,
  was 22). Renumbered all downstream eyebrows (s19 22->24, s20 23->25) and
  fixed every "frame NN"/"appendix" cross-reference in slide bodies and in
  NOTES (sContrib, s8, s11, sQ1, sQ2). User's own words: "i think you can
  move it into the main sequence and then i can sharpen the whole thing
  myself" — so the sQ1/sQ2 wording is intentionally left as my draft;
  expect the user to rewrite/tighten it directly, don't re-touch that
  copy unprompted.
- More-visual pass: added a reusable phase-timeline diagram (reuses
  existing .flow/.fnode CSS from s5, zero new CSS) showing
  approach->contact->technique->withdrawal; inserted full-size into s7
  (chapter 3 observation, replacing a redundant bullet) and a compact
  Phi_1..Phi_n version above the code block in s9 (OAAT chapter 4) so the
  T_A=<Phi1...Phin> formalism has a picture before the math.
- Built a presenter-notes companion window per user request ("so when I
  screenshare and extend the display... I can have my notes"):
  defense/notes.html (new, standalone page, dark presenter theme, big
  readable note text, next-slide preview, live clock + resettable elapsed
  timer). index.html has a NOTES={} object (one entry per slide id, 25
  entries) and calls broadcastNotes(i) from go(i); sync is
  BroadcastChannel('cram-defense-notes') + a localStorage fallback/initial-
  state key (cram-defense-notes-state) so notes.html shows the right slide
  even if opened after the deck already moved on. Open via the 'n' key or
  clicking "n" in the navhint bar -> window.open('notes.html', ...) sized
  480x780. Verified: node --check on both extracted <script> blocks (syntax
  clean), section/div tag counts balanced (25 sections), all 25 slide ids
  have a NOTES entry, local http.server smoke-served both pages 200 OK.
  Could NOT do a real browser/visual check — no chromium/puppeteer/
  playwright available in this sandbox; user should open it for real
  (their actual multi-window/extended-display use case can't be simulated
  headlessly anyway) and confirm the notes window syncs live.
- notes.html upgrade (user request): view/edit switch (segmented toggle)
  for the notes area. Edit mode is a <textarea>; view mode renders it
  through a small dependency-free markdown renderer written inline
  (renderMarkdown() in notes.html — headers #/##/###, **bold**, *italic*,
  `code`, -/* bullet lists, 1. numbered lists, paragraphs; no external
  libs). Per-slide edits autosave to localStorage
  (cram-defense-notes-user:<slideId>) on every keystroke and take priority
  over the deck's built-in NOTES default for that slide; a "revert to
  auto-note" link clears the override. Mode (view/edit) itself also
  persists across reloads (cram-defense-notes-mode). Reset-timer button
  unchanged. Verified renderMarkdown() output via a Node vm sandbox
  (headers/bold/italic/code/lists all rendered correctly) — again no real
  browser check possible here.

Next (if asked):
- User is sharpening the sQ1/sQ2 wording themselves now that it's in the
  main sequence — don't rewrite that content unless asked again.
- User should test the notes window for real: open index.html + press 'n'
  (or click it), confirm the second window shows live-synced notes when
  advancing slides, and check it looks right on an actual extended display
  during a screenshare.
- Optionally extend the "more visual" pass to other text-heavy slides
  (s2/s3/s4/s18/s19) if the phase-timeline treatment of s7/s9 isn't enough.
- Optionally add Unitree_warehouse / tracy_lab / garmi episodes (compilers
  handle any bundle; garmi/tracy have no apartment env).
- Ghosted-walls toggle (data already flagged w:1); background image toggle
  like the cramera viewer.
