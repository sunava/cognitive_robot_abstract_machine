// Unit tests for web/core/scene_picker.js (node:test).
// There is no independent mixing across axes: each onboarded scene is a fixed (robot,
// environment, task, version) recording. ScenePicker only looks up, among the scenes
// actually onboarded, which one matches a chosen combination, so the header's four
// dropdowns can look independent while only ever landing on a recorded combination.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/scene_picker.js'), 'utf8'))();
}

const SCENES = [
  { name: 'pr2/apartment/kitchen/v1', robot: 'pr2', environment: 'apartment', task: 'kitchen', version: 'v1' },
  { name: 'pr2/apartment/kitchen/v2', robot: 'pr2', environment: 'apartment', task: 'kitchen', version: 'v2' },
  { name: 'pr2/lab/pouring/v1', robot: 'pr2', environment: 'lab', task: 'pouring', version: 'v1' },
  { name: 'garmi/apartment/pick_place/v1', robot: 'garmi', environment: 'apartment', task: 'pick_place', version: 'v1' },
  { name: 'tracy/_none/bench_test/v1', robot: 'tracy', environment: null, task: 'bench_test', version: 'v1' },
];

test('robots() lists every robot with an onboarded scene, once each', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.robots(SCENES), ['pr2', 'garmi', 'tracy']);
});

test('environments() lists the environments onboarded with one robot', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.environments(SCENES, 'pr2'), ['apartment', 'lab']);
});

test('environments() includes null for a bench-only scene', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.environments(SCENES, 'tracy'), [null]);
});

test('environments() is empty for a robot with no onboarded scene', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.environments(SCENES, 'unitreeg1'), []);
});

test('tasks() lists the tasks onboarded with one (robot, environment) pair', function () {
  load();
  assert.deepStrictEqual(
    window.ScenePicker.tasks(SCENES, 'pr2', 'apartment'), ['kitchen']);
});

test('versions() lists the versions onboarded for one (robot, environment, task)', function () {
  load();
  assert.deepStrictEqual(
    window.ScenePicker.versions(SCENES, 'pr2', 'apartment', 'kitchen'), ['v1', 'v2']);
});

test('sceneFor() resolves a full (robot, environment, task, version) combination', function () {
  load();
  assert.strictEqual(
    window.ScenePicker.sceneFor(SCENES, 'pr2', 'apartment', 'kitchen', 'v2'),
    'pr2/apartment/kitchen/v2');
});

test('sceneFor() resolves a bench-only robot with a null environment', function () {
  load();
  assert.strictEqual(
    window.ScenePicker.sceneFor(SCENES, 'tracy', null, 'bench_test', 'v1'),
    'tracy/_none/bench_test/v1');
});

test('sceneFor() treats a missing environment argument as null', function () {
  load();
  assert.strictEqual(
    window.ScenePicker.sceneFor(SCENES, 'tracy', undefined, 'bench_test', 'v1'),
    'tracy/_none/bench_test/v1');
});

test('sceneFor() is null for a combination that was never onboarded', function () {
  load();
  assert.strictEqual(
    window.ScenePicker.sceneFor(SCENES, 'garmi', 'apartment', 'kitchen', 'v1'), null);
});

test('describe() reports the identity a named scene was recorded with', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.describe(SCENES, 'garmi/apartment/pick_place/v1'), {
    robot: 'garmi',
    environment: 'apartment',
    task: 'pick_place',
    version: 'v1',
  });
});

test('describe() reports a null environment for a bench-only scene', function () {
  load();
  assert.deepStrictEqual(window.ScenePicker.describe(SCENES, 'tracy/_none/bench_test/v1'), {
    robot: 'tracy',
    environment: null,
    task: 'bench_test',
    version: 'v1',
  });
});

test('describe() is null for a name not in the index', function () {
  load();
  assert.strictEqual(window.ScenePicker.describe(SCENES, 'unknown/scene/here/v1'), null);
});
