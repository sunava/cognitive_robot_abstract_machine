// Unit tests for web/core/builder-scene.js (node:test): which changes to the plan under
// construction reach the running scene, and which are baked into the world it was
// started with and so need it started again.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/builder-scene.js'), 'utf8'))(scope);
  return scope.BuilderScene;
}

// %% what a change means for a scene that is already running
test('the robot and the environment are the world a scene was started with', function () {
  const builder = load();
  assert.strictEqual(builder.needsRestart(builder.CHANGE.ROBOT), true);
  assert.strictEqual(builder.needsRestart(builder.CHANGE.ENVIRONMENT), true);
});

test('objects appear and rotate only in a freshly started scene', function () {
  const builder = load();
  assert.strictEqual(builder.needsRestart(builder.CHANGE.OBJECT_SET), true);
  assert.strictEqual(builder.needsRestart(builder.CHANGE.OBJECT_ROTATION), true);
});

test('a moved object travels into the running scene as a pose', function () {
  const builder = load();
  assert.strictEqual(builder.needsRestart(builder.CHANGE.OBJECT_POSITION), false);
});

test('the plan, its constraints and the generated code leave the scene alone', function () {
  const builder = load();
  assert.strictEqual(builder.needsRestart(builder.CHANGE.PLAN_STEPS), false);
  assert.strictEqual(builder.needsRestart(builder.CHANGE.CONSTRAINTS), false);
  assert.strictEqual(builder.needsRestart(builder.CHANGE.OUTPUT_STYLE), false);
});

// %% waiting for a scene the builder just started
test('the builder waits out the slowest start it allows for', function () {
  const builder = load();
  assert.ok(builder.POLL_INTERVAL_MS * builder.POLL_ATTEMPTS >= builder.SLOWEST_START_MS);
});

test('it looks often enough not to stand idle after the scene is up', function () {
  const builder = load();
  // the scene comes up in seconds, so the look-again gap has to stay well under one
  assert.ok(builder.POLL_INTERVAL_MS <= 1000);
});

test('a scene nobody asked for is noticed less urgently than one that was', function () {
  const builder = load();
  assert.ok(builder.WATCH_INTERVAL_MS >= builder.POLL_INTERVAL_MS);
  assert.ok(builder.WATCH_INTERVAL_MS <= 5000);
});

// %% what became of a demo the builder started
test('a demo that has not exited is still running', function () {
  const builder = load();
  assert.strictEqual(builder.outcomeOf(null), builder.RUN.RUNNING);
  assert.strictEqual(builder.outcomeOf(undefined), builder.RUN.RUNNING);
});

test('a demo that exited cleanly has finished, and took its scene with it', function () {
  const builder = load();
  assert.strictEqual(builder.outcomeOf(0), builder.RUN.FINISHED);
});

test('any other exit code is a crash, not a finished plan', function () {
  const builder = load();
  assert.strictEqual(builder.outcomeOf(1), builder.RUN.CRASHED);
  assert.strictEqual(builder.outcomeOf(-15), builder.RUN.CRASHED);
});

// %% which placement surfaces a step can be asked for
const OFFERED = ['CounterTop', 'Table', 'Drawer'];
/* What the builder knows how to place on, before any scene has said what it has. */

test('a scene that has said what it holds is what can be placed on', function () {
  const builder = load();
  const live = [{ type: 'Drawer', name: 'a' }, { type: 'Cabinet', name: 'b' }, { type: 'Drawer', name: 'c' }];
  assert.deepStrictEqual(builder.surfaceTypesToOffer(live, OFFERED), ['Drawer', 'Cabinet']);
});

test('with no scene to ask, everything the builder knows is offered', function () {
  const builder = load();
  assert.deepStrictEqual(builder.surfaceTypesToOffer([], OFFERED), OFFERED);
});

test('a step keeps the surface it was set to when the scene has one', function () {
  const builder = load();
  assert.strictEqual(builder.surfaceTypeFor('Drawer', ['Cabinet', 'Drawer']), 'Drawer');
});

test('a step asking for a surface the scene has not got is moved to one it has', function () {
  const builder = load();
  assert.strictEqual(builder.surfaceTypeFor('CounterTop', ['Cabinet', 'Drawer']), 'Cabinet');
});

// %% which target mode a transport step starts in
const PLACE_ON = ['CounterTop', 'Table'];
/* The surfaces an object can be put down on, as opposed to put inside. */

test('a scene with a surface to place on starts a transport off semantically', function () {
  const builder = load();
  const live = [{ type: 'Table', name: 'a' }, { type: 'Drawer', name: 'b' }];
  assert.strictEqual(builder.targetModeFor(live, PLACE_ON), builder.TARGET.SEMANTIC);
});

test('a scene offering only containers starts a transport off at a pose', function () {
  const builder = load();
  const live = [{ type: 'Drawer', name: 'a' }, { type: 'Cabinet', name: 'b' }];
  assert.strictEqual(builder.targetModeFor(live, PLACE_ON), builder.TARGET.POSE);
});

test('with no scene to ask, a transport starts off semantically as before', function () {
  const builder = load();
  assert.strictEqual(builder.targetModeFor([], PLACE_ON), builder.TARGET.SEMANTIC);
});
