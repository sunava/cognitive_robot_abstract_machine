/* ============================================================================
 * core/runnable_plan.js — what stops a built plan from running.
 *
 * A plan is handed to a demo process as generated Python, and a step that names no
 * object becomes a lookup of a body that does not exist. That failure happens seconds
 * later, inside a process the builder does not watch, so it is worth finding here --
 * before anything is started -- and saying which step is at fault.
 * ==========================================================================*/
(function (global) {
  'use strict';

  const NOTHING_TO_RUN = 'the plan has no steps';
  /* Said of a plan with no steps at all, which belongs to no step in particular. */

  function problemsOfStep(step, placedObjects) {
    if (step.type !== 'transport') return null;
    const named = step.params.object;
    if (!named) return 'has no object to transport';
    if (placedObjects.indexOf(named) < 0) {
      return 'transports "' + named + '", which is not placed in the scene';
    }
    return null;
  }

  global.RunnablePlan = {
    /* Everything that stops `steps` from running, each naming the 1-based step it is
       about, or null for the plan as a whole. An empty list means the plan can be run.

       `placedObjects` are the mesh names the builder has placed in the scene, which is
       what a transport step's object has to be one of. */
    problems: function (steps, placedObjects) {
      if (!steps || !steps.length) return [{ step: null, problem: NOTHING_TO_RUN }];
      const found = [];
      steps.forEach(function (step, index) {
        const problem = problemsOfStep(step, placedObjects || []);
        if (problem) found.push({ step: index + 1, problem: problem });
      });
      return found;
    },

    /* The problems as one line, for a status field that has room for one. */
    describe: function (problems) {
      return problems.map(function (found) {
        return found.step === null ? found.problem : 'step ' + found.step + ' ' + found.problem;
      }).join('; ');
    },
  };
})(window);
