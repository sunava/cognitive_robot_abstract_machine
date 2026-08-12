## Branch cramera-features — chemical laboratory demo

**Done (this session):**
- New demo `coraplex/demos/coraplex_chemical_laboratory_demo/`: HSR runs a
  sample-analysis round in a chemical lab converted from the downloaded
  `~/Downloads/hemical-laboratory` FBX asset (Blender conversion script,
  recolored, floor at z=0, ceiling removed; scale 15 → 0.9 m benches).
- Demo runs green end to end: 3 pick/place transports (flask, reagent bottle,
  test-tube rack) with final-pose assertions.
- Workaround in demo: commit 8cc3cf690 (montessori) commented out
  AttachNode/DetachNode in PickUpAction/PlaceAction, breaking kinematic
  transport demos (wind_turbine_hall demo currently fails at CloseGripper).
  Demo uses local ModelAttachingPickUpAction/ModelDetachingPlaceAction
  subclasses instead of touching shared code.

**Rewrite branch `cramera-world-visualization`** (worktree ~/cram-worktrees/cramera-rewrite,
3 commits, not pushed):
- Rebuilt cramera on PR #36's WorldVisualization: CRAMERA backend
  (CORAPLEX_VISUALIZATION=cramera), bridge bound to the World object via
  state/model callbacks + PlanCallback (new on_motion_tick event); all six
  monkey-patches, model-source tracking and live onboarding deleted.
- Live scene now serialized from the world (UrdfDocument.of_bodies:
  environment + robot models); overlay OBJs keep their MTL materials
  (side-asset serving); tameMat no longer flattens authored finishes.
- 413 tests green; bullet demo verified end to end over HTTP.
- Pushed; draft PR #37 (stacked on PR #36 / rerun-demo-visualization):
  https://github.com/sunava/cognitive_robot_abstract_machine/pull/37
- Fixed reload flapping: content-based bundle signature (overlay
  re-parenting no longer reloads the scene); gimbal-lock warning
  silenced in the URDF origin writer.
- Next: check panels in a real browser; land #36 first, then mark #37
  ready when told.

**Next:**
- Ask the team whether attach/detach should be restored in shared actions
  (physics vs kinematic worlds) and whether wind_turbine_hall demo should be
  fixed the same way.
- Demo not yet committed (whole worktree has many uncommitted changes).
