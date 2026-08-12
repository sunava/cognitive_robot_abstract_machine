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

**Next:**
- Ask the team whether attach/detach should be restored in shared actions
  (physics vs kinematic worlds) and whether wind_turbine_hall demo should be
  fixed the same way.
- Demo not yet committed (whole worktree has many uncommitted changes).
