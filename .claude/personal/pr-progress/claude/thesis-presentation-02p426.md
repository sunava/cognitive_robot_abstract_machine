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

Next (if asked):
- Optionally add Unitree_warehouse / tracy_lab / garmi episodes (compilers
  handle any bundle; garmi/tracy have no apartment env).
- Ghosted-walls toggle (data already flagged w:1); background image toggle
  like the cramera viewer.
