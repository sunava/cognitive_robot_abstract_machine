# Branch: claude/thesis-presentation-02p426 — defense deck (artifact work, no PR)

Deliverable is the defense-deck Artifact, not repo code:
https://claude.ai/code/artifact/f96ca126-240d-4cc6-8f91-f9b14cb20de6

Done:
- 20-slide "cram_viewer" deck (earlier sessions).
- Slide 14 "recorded episodes": real cramera scene bundles
  (cram2/cram-scenes PR2/HSR/TIAGO_Apartment), trajectories downsampled
  ~10 fps as sparse joint tracks (~210 KB) and replayed with a JS FK
  player: robot switcher, plan-step chips, scrub/play/speed, orbit camera,
  base-path ribbon, light/dark themes. Later slides renumbered 15-21.
- v2 after user feedback ("not as good as my cramera viewer"): rendering
  upgraded from box proxies to REAL URDF visual meshes in a raw-WebGL
  renderer inside the deck (no libs): meshes decimated to <=1300 tris via
  fast-simplification, uint16-quantized, base64 meshlib ~3.9 MB / 240k
  tris shared across scenes; flat shading via derivatives, planar contact
  shadows, checkerboard-shader floor, wheel zoom; env colors from URDF
  materials + DAE texture means, robots keep shell/joint scheme (URDF
  robot colors are segmentation debug colors); apartment FK cached
  (static); SwiftShader detected -> lower dpr. Deck now ~4.4 MB.
  Verified all robots + dark mode headless; software-render fps is slow
  in headless but any real GPU handles 240k tris easily.
- Scripts in session scratchpad: compile_scene.py (tracks), compile_meshes.py
  (meshlib), fk_preview.py, screenshot_deck.py, episode_player_gl.js
  (source of the injected player). Regenerate: clone cram2/cram-scenes,
  run both compilers, re-inject via the python snippets in the session.

Next (if asked):
- Optionally add Unitree_warehouse / tracy_lab / garmi episodes (compilers
  handle any bundle; garmi/tracy have no apartment env).
- Walls/windows/ceiling are skipped for the dollhouse view; could add a
  ghosted-walls toggle.
