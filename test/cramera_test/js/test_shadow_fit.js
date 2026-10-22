// Unit tests for web/core/shadow-fit.js (node:test).
// The key light's shadow map is a fixed square of texels: fitted to the region a
// bundle names, every part of that region receives shadows and none of the map is
// spent outside it.
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const DATASET = path.join(__dirname, '..', 'dataset', 'usd_scene');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));
//: rounding tolerance of the matrix arithmetic, in metres
const TOLERANCE = 1e-9;

function load() {
  global.window = {};
  ['core/scene-rendering.js', 'core/shadow-fit.js'].forEach(function (file) {
    new Function(fs.readFileSync(path.join(WEB, file), 'utf8'))();
  });
  return window;
}

const { ShadowFit, SceneBounds, SceneRendering } = load();
const SCENE = JSON.parse(fs.readFileSync(path.join(DATASET, 'viewer_scene.json'), 'utf8'));
const LIGHTING = JSON.parse(fs.readFileSync(path.join(DATASET, 'viewer_lighting.json'), 'utf8'));
const BOUNDS = SceneRendering.of(SCENE).shadowBounds;
const DOWN = [0, 0, -1];

function near(actual, expected) {
  assert.ok(Math.abs(actual - expected) < TOLERANCE, actual + ' is not ' + expected);
}

// the shadow camera exactly as the renderer places it before drawing the shadow map
function fittedShadowCamera(direction) {
  const light = new THREE.DirectionalLight(0xffffff, 1);
  new ShadowFit(BOUNDS, direction).applyTo(THREE, light);
  light.updateMatrixWorld();
  light.shadow.updateMatrices(light);
  return light;
}

function cameraSpaceCorners(camera) {
  return BOUNDS.corners().map(function (corner) {
    return ShadowFit.toThree(THREE, corner).applyMatrix4(camera.matrixWorldInverse);
  });
}

// %% bundle frame to three's frame
test('a bundle point is turned the way the viewer turns its Z-up world root', function () {
  const point = [1.5, -2, 3.25];
  const turned = new THREE.Vector3().fromArray(point).applyAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
  const converted = ShadowFit.toThree(THREE, point);
  [0, 1, 2].forEach(function (axis) { near(converted.getComponent(axis), turned.getComponent(axis)); });
});

test('a three direction converts back to the bundle frame', function () {
  const direction = [0.35, -0.45, -0.82];
  assert.deepEqual(ShadowFit.fromThree(ShadowFit.toThree(THREE, direction)), direction);
});

// %% a light straight from above
test('a light from straight above sees exactly the footprint of the bounds plus the margin', function () {
  const frustum = new ShadowFit(BOUNDS, DOWN).frustum(THREE);
  const margin = ShadowFit.MARGIN;
  near(frustum.left, BOUNDS.min[0] - margin);
  near(frustum.right, BOUNDS.max[0] + margin);
  near(frustum.bottom, BOUNDS.min[1] - margin);
  near(frustum.top, BOUNDS.max[1] + margin);
});

test('a light from straight above sees the full height of the bounds plus the margin', function () {
  const frustum = new ShadowFit(BOUNDS, DOWN).frustum(THREE);
  near(frustum.far - frustum.near, BOUNDS.size()[2] + 2 * ShadowFit.MARGIN);
});

test('the shadow camera reaches from the margin before the nearest corner to the margin past the farthest', function () {
  [DOWN, LIGHTING.sun.direction].forEach(function (direction) {
    const camera = fittedShadowCamera(direction).shadow.camera;
    // the camera looks along its -z axis, so a corner's depth is its negated z
    const depths = cameraSpaceCorners(camera).map(function (corner) { return -corner.z; });
    near(camera.near, Math.min.apply(null, depths) - ShadowFit.MARGIN);
    near(camera.far, Math.max.apply(null, depths) + ShadowFit.MARGIN);
  });
});

// %% an oblique sun, through the renderer's own shadow camera placement
test('every corner of the bounds lies inside the shadow camera of an oblique light', function () {
  const camera = fittedShadowCamera(LIGHTING.sun.direction).shadow.camera;
  BOUNDS.corners().forEach(function (corner) {
    const projected = ShadowFit.toThree(THREE, corner).project(camera);
    [projected.x, projected.y, projected.z].forEach(function (value) {
      assert.ok(Math.abs(value) <= 1, 'corner ' + corner + ' falls outside the shadow map');
    });
  });
});

test('the shadow camera of an oblique light is no wider than the bounds plus the margin', function () {
  const camera = fittedShadowCamera(LIGHTING.sun.direction).shadow.camera;
  const corners = cameraSpaceCorners(camera);
  const width = Math.max.apply(null, corners.map(function (c) { return c.x; })) -
    Math.min.apply(null, corners.map(function (c) { return c.x; }));
  const height = Math.max.apply(null, corners.map(function (c) { return c.y; })) -
    Math.min.apply(null, corners.map(function (c) { return c.y; }));
  near(camera.right - camera.left, width + 2 * ShadowFit.MARGIN);
  near(camera.top - camera.bottom, height + 2 * ShadowFit.MARGIN);
});

test('the fitted light shines along the given direction onto the centre of the bounds', function () {
  const light = fittedShadowCamera(LIGHTING.sun.direction);
  const travel = light.target.position.clone().sub(light.position).normalize();
  const expected = ShadowFit.toThree(THREE, LIGHTING.sun.direction).normalize();
  const center = ShadowFit.toThree(THREE, BOUNDS.center());
  [0, 1, 2].forEach(function (axis) {
    near(travel.getComponent(axis), expected.getComponent(axis));
    near(light.target.position.getComponent(axis), center.getComponent(axis));
  });
});

test('a degenerate direction is refused rather than aiming the light nowhere', function () {
  assert.throws(function () { new ShadowFit(SceneBounds.of(SCENE.rendering.shadowBounds), [0, 0, 0]); },
    ShadowFit.DirectionError);
});
