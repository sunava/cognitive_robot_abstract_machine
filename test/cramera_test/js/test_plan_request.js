// Unit tests for web/core/plan_request.js (node:test): what the builder sends when it
// asks a running scene to perform its plan.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/plan_constraints.js'), 'utf8'))(scope);
  new Function('window', fs.readFileSync(path.join(WEB, 'core/plan_request.js'), 'utf8'))(scope);
  return scope;
}

function transportStep(constraints) {
  return {
    id: 's3',
    type: 'transport',
    params: { object: 'milk.stl', arm: 'LEFT', targetMode: 'pose', x: 3, y: 2, z: 1, yaw: 0 },
    constraints: constraints || [],
  };
}

// %% what a scene is asked to perform
test('a step is asked for by its type and its own parameters', function () {
  const scope = load();
  const request = scope.PlanRequest.of([{ type: 'park_arms', params: { arm: 'BOTH' } }]);
  assert.deepStrictEqual(request, { steps: [{ type: 'park_arms', params: { arm: 'BOTH' } }] });
});

test('the steps are asked for in the order they were built in', function () {
  const scope = load();
  const request = scope.PlanRequest.of([
    { type: 'park_arms', params: { arm: 'BOTH' } },
    { type: 'move_torso', params: { torso: 'HIGH' } },
  ]);
  assert.deepStrictEqual(request.steps.map(function (s) { return s.type; }), ['park_arms', 'move_torso']);
});

test('a constraint that the step enforces travels with it', function () {
  const scope = load();
  const compiled = scope.PlanConstraints.compile('Robot must look where it operates',
    { type: scope.PlanConstraints.STEP.TRANSPORT, params: { object: 'milk.stl' } });
  const request = scope.PlanRequest.of([transportStep([compiled])]);
  assert.strictEqual(request.steps[0].params[scope.PlanConstraints.ARGUMENT.LOOK_AT_OPERATION_SITE], true);
});

test('a constraint no step enforces switches nothing on', function () {
  const scope = load();
  const compiled = scope.PlanConstraints.compile('Milk must always stay upright',
    { type: scope.PlanConstraints.STEP.TRANSPORT, params: { object: 'milk.stl' } });
  const request = scope.PlanRequest.of([transportStep([compiled])]);
  assert.deepStrictEqual(request.steps[0].params, transportStep().params);
});

test('asking a scene to perform nothing asks for no steps', function () {
  const scope = load();
  assert.deepStrictEqual(scope.PlanRequest.of([]), { steps: [] });
});
