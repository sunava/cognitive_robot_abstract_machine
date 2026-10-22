// Unit tests for web/core/scene-rendering.js (node:test).
// scene.json's optional `rendering` block: every field a bundle leaves out keeps the
// viewer's own default, so a bundle written before a field existed renders as it did.
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const DATASET = path.join(__dirname, '..', 'dataset', 'usd_scene');

// %% the module, loaded into this realm the way a page script runs
function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/scene-rendering.js'), 'utf8'))();
  return window;
}

function fixture(name) {
  return JSON.parse(fs.readFileSync(path.join(DATASET, name), 'utf8'));
}

const { SceneRendering, SceneBounds } = load();
const SCENE = fixture('viewer_scene.json');
const LIGHTING = fixture('viewer_lighting.json');

// %% defaults of a bundle without the block
test('a scene without a rendering block keeps the floor', function () {
  assert.equal(SceneRendering.of({ name: 'plain' }).groundPlane, true);
});

test('a scene without a rendering block has no shadow bounds and no environment', function () {
  const rendering = SceneRendering.of(null);
  assert.deepEqual([rendering.shadowBounds, rendering.environment], [null, null]);
});

// %% fields a bundle sets
test('a bundle can turn the procedural floor off', function () {
  assert.equal(SceneRendering.of(SCENE).groundPlane, false);
});

test('shadow bounds are read as a minimum and a maximum corner', function () {
  const bounds = SceneRendering.of(SCENE).shadowBounds;
  assert.deepEqual([bounds.min, bounds.max], SCENE.rendering.shadowBounds);
});

// %% malformed bounds are ignored rather than half-applied
test('bounds with a corner of two coordinates are ignored', function () {
  assert.equal(SceneBounds.of([[0, 0], [1, 1, 1]]), null);
});

test('bounds whose minimum exceeds the maximum are ignored', function () {
  assert.equal(SceneBounds.of([[0, 2, 0], [1, 1, 1]]), null);
});

test('bounds with a coordinate that is not a number are ignored', function () {
  assert.equal(SceneBounds.of([[0, 0, 0], [1, 'far', 1]]), null);
});

// %% derived geometry of the bounds
test('the centre of the bounds lies halfway between the corners', function () {
  const [min, max] = SCENE.rendering.shadowBounds;
  assert.deepEqual(SceneBounds.of([min, max]).center(),
    [0, 1, 2].map(function (axis) { return (min[axis] + max[axis]) / 2; }));
});

test('the size of the bounds is the extent along every axis', function () {
  const [min, max] = SCENE.rendering.shadowBounds;
  assert.deepEqual(SceneBounds.of([min, max]).size(),
    [0, 1, 2].map(function (axis) { return max[axis] - min[axis]; }));
});

test('the bounds have eight distinct corners, each made of minimum or maximum coordinates', function () {
  const [min, max] = SCENE.rendering.shadowBounds;
  const corners = SceneBounds.of([min, max]).corners();
  assert.equal(new Set(corners.map(String)).size, 8);
  corners.forEach(function (corner) {
    corner.forEach(function (value, axis) { assert.ok(value === min[axis] || value === max[axis]); });
  });
});

// %% the lighting description
test('an environment path is fetched next to the scene file', async function () {
  const requested = [];
  const lighting = await SceneRendering.of(SCENE).lighting(function (url) {
    requested.push(url);
    return Promise.resolve(LIGHTING);
  }, 'scenes/viewer_lab/');
  assert.deepEqual([requested, lighting], [['scenes/viewer_lab/' + SCENE.rendering.environment], LIGHTING]);
});

test('an inline environment is used as given, without a request', async function () {
  const requested = [];
  const lighting = await SceneRendering.of({ rendering: { environment: LIGHTING } }).lighting(function (url) {
    requested.push(url);
    return Promise.resolve(null);
  }, 'scenes/viewer_lab/');
  assert.deepEqual([requested, lighting], [[], LIGHTING]);
});

test('a scene without an environment has no lighting description', async function () {
  const lighting = await SceneRendering.of({}).lighting(function () {
    throw new Error('nothing should be requested');
  }, 'scenes/plain/');
  assert.equal(lighting, null);
});

test('an environment file that cannot be fetched leaves the scene without a description', async function () {
  const lighting = await SceneRendering.of(SCENE).lighting(function () {
    return Promise.reject(new Error('404'));
  }, 'scenes/viewer_lab/');
  assert.equal(lighting, null);
});
