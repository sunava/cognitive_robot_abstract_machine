// Unit tests for web/core/runnable_plan.js (node:test): what stops a built plan from
// being runnable at all, found before a demo process is started for it -- a plan that
// names no object reaches the robot as a body that does not exist.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/runnable_plan.js'), 'utf8'))(scope);
  return scope.RunnablePlan;
}

const PLACED = ['milk.stl', 'bowl.stl'];

// %% a plan that can run
test('a transport of a placed object has nothing wrong with it', function () {
  const plan = load();
  const steps = [
    { type: 'park_arms', params: { arm: 'BOTH' } },
    { type: 'transport', params: { object: 'milk.stl', arm: 'LEFT' } },
  ];
  assert.deepStrictEqual(plan.problems(steps, PLACED), []);
});

test('steps that carry no object are never a problem', function () {
  const plan = load();
  const steps = [
    { type: 'park_arms', params: { arm: 'BOTH' } },
    { type: 'move_torso', params: { torso: 'HIGH' } },
    { type: 'navigate', params: { x: 1, y: 2 } },
  ];
  assert.deepStrictEqual(plan.problems(steps, []), []);
});

// %% a plan that cannot
test('a transport with no object chosen is reported against its step', function () {
  const plan = load();
  const steps = [
    { type: 'park_arms', params: { arm: 'BOTH' } },
    { type: 'transport', params: { object: '', arm: 'LEFT' } },
  ];
  const [problem] = plan.problems(steps, PLACED);
  assert.strictEqual(problem.step, 2);
  assert.match(problem.problem, /object/);
  assert.strictEqual(plan.problems(steps, PLACED).length, 1);
});

test('a transport of an object no longer placed is reported too', function () {
  const plan = load();
  const steps = [{ type: 'transport', params: { object: 'spoon.stl', arm: 'LEFT' } }];
  const [problem] = plan.problems(steps, PLACED);
  assert.strictEqual(problem.step, 1);
  assert.match(problem.problem, /spoon\.stl/);
});

test('an empty plan has nothing to run', function () {
  const plan = load();
  const [problem] = plan.problems([], PLACED);
  assert.strictEqual(problem.step, null);
  assert.strictEqual(plan.problems([], PLACED).length, 1);
});

// %% saying it out loud
test('the problems are worded as one line naming every step at fault', function () {
  const plan = load();
  const steps = [
    { type: 'transport', params: { object: '', arm: 'LEFT' } },
    { type: 'transport', params: { object: 'spoon.stl', arm: 'LEFT' } },
  ];
  const said = plan.describe(plan.problems(steps, PLACED));
  assert.match(said, /step 1/);
  assert.match(said, /step 2/);
});
