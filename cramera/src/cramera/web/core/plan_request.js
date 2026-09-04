/* ============================================================================
 * core/plan_request.js — the plan as the running scene reads it.
 *
 * A plan used to reach the robot as generated Python in a file a new process was
 * started on. It now reaches the scene that is already up, as its steps: each one's own
 * parameters plus whatever its constraints switch on. The generated file is still what
 * Download and Save produce, and says the same thing.
 * ==========================================================================*/
(function (global) {
  'use strict';

  global.PlanRequest = {
    /* The body of the request that asks a running scene to perform `steps`. */
    of: function (steps) {
      return {
        steps: (steps || []).map(function (step) {
          const parameters = Object.assign({}, step.params);
          const switches = global.PlanConstraints.stepSwitches(step.constraints || []);
          Object.keys(switches).forEach(function (name) { parameters[name] = switches[name]; });
          return { type: step.type, params: parameters };
        }),
      };
    },
  };
})(window);
