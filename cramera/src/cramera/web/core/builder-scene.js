/* ============================================================================
 * core/builder-scene.js — the Plan Builder's live scene.
 *
 * Two things the builder has to know about the scene it starts: which changes to the
 * plan under construction the running scene can be told about, and which are part of
 * the world it was started with and so only take effect in a new one; and how long to
 * keep looking for a scene it just asked for.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const CHANGE = {
    ROBOT: 'robot',
    ENVIRONMENT: 'environment',
    OBJECT_SET: 'objectSet',
    OBJECT_POSITION: 'objectPosition',
    OBJECT_ROTATION: 'objectRotation',
    PLAN_STEPS: 'planSteps',
    CONSTRAINTS: 'constraints',
    OUTPUT_STYLE: 'outputStyle',
  };
  /* Everything the builder lets a user change about the demo being built. */

  const BUILT_INTO_THE_WORLD = [
    CHANGE.ROBOT,
    CHANGE.ENVIRONMENT,
    CHANGE.OBJECT_SET,
    CHANGE.OBJECT_ROTATION,
  ];
  /* The changes a running scene cannot be told about, because they are the world it
     was started with: its robot, its environment, and which objects were spawned into
     it -- at the orientation they were spawned with. Everything else reaches the scene
     as it stands: a moved object travels as a pose, the plan is only read when it is
     run, and a constraint goes to the bridge on its own. */

  const SLOWEST_START_MS = 120000;
  /* How long a scene may take to come up before the builder stops waiting for it.
     Parsing a robot description and an environment is seconds' work, so this is a
     generous ceiling for a machine under load, not an expected wait. */

  const POLL_INTERVAL_MS = 500;
  /* How long to wait between two looks for a scene that was asked for. Short enough
     that the view is not left blank after the scene is already answering. */

  const WATCH_INTERVAL_MS = 2000;
  /* How long to wait between two looks for a scene the builder did not ask for -- one
     started before this page was opened, or by hand. Less urgent than waiting for a
     scene that was just asked for, and quiet enough not to fill a console with refused
     connections while none runs. */

  const TARGET = {
    SEMANTIC: 'semantic',
    POSE: 'pose',
  };
  /* How a transport step says where the object goes: a surface to sample a free spot
     on, or a pose pointed at in the scene. */

  const RUN = {
    RUNNING: 'running',
    FINISHED: 'finished',
    CRASHED: 'crashed',
  };
  /* What became of a demo the builder started. A finished demo leaves no live scene
     behind -- the generated demonstration stops its own visualization once the plan is
     performed -- so the scene has to be started again before anything can be dragged. */

  const SELECTION_SETTLE_MS = 400;
  /* How long to let a selection settle before starting the scene it asks for, so
     clicking through a dropdown starts one scene rather than one per entry. */

  global.BuilderScene = {
    CHANGE: CHANGE,
    SLOWEST_START_MS: SLOWEST_START_MS,
    POLL_INTERVAL_MS: POLL_INTERVAL_MS,
    WATCH_INTERVAL_MS: WATCH_INTERVAL_MS,
    POLL_ATTEMPTS: Math.ceil(SLOWEST_START_MS / POLL_INTERVAL_MS),
    SELECTION_SETTLE_MS: SELECTION_SETTLE_MS,
    RUN: RUN,
    TARGET: TARGET,

    /* Whether a change only takes effect in a newly started scene. */
    needsRestart: function (change) {
      return BUILT_INTO_THE_WORLD.indexOf(change) >= 0;
    },

    /* The placement surfaces a step can be asked for: the kinds the running scene
       actually holds, or everything the builder knows while no scene has said. A
       default the world has not got can only fail, and only once the plan runs. */
    surfaceTypesToOffer: function (liveSurfaces, offered) {
      const present = [];
      (liveSurfaces || []).forEach(function (surface) {
        if (present.indexOf(surface.type) < 0) present.push(surface.type);
      });
      return present.length ? present : offered;
    },

    /* Where a transport step starts off aiming. Sampling a free spot needs a surface
       to put the object *on*; a scene offering only containers to put it *in* cannot
       satisfy that -- placing into a closed drawer has no solution -- so such a step
       starts at a pose instead, which any scene can satisfy. */
    targetModeFor: function (liveSurfaces, placeOnTypes) {
      if (!liveSurfaces || !liveSurfaces.length) return TARGET.SEMANTIC;
      const canPlaceOn = liveSurfaces.some(function (surface) {
        return placeOnTypes.indexOf(surface.type) >= 0;
      });
      return canPlaceOn ? TARGET.SEMANTIC : TARGET.POSE;
    },

    /* The surface a step places on: the one it asks for while the scene has it. */
    surfaceTypeFor: function (chosen, available) {
      return available.indexOf(chosen) >= 0 ? chosen : available[0];
    },

    /* What became of the demo, from the exit code of its process -- none yet while it
       is still running. */
    outcomeOf: function (returncode) {
      if (returncode === null || returncode === undefined) return RUN.RUNNING;
      return returncode === 0 ? RUN.FINISHED : RUN.CRASHED;
    },
  };
})(window);
