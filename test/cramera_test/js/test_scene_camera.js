// Unit tests for web/core/scene-camera.js (node:test).
// scene.json's `camera` block may set the lens and how far the view reaches; a field
// a bundle does not set leaves the viewer's own camera as it is.
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const DATASET = path.join(__dirname, '..', 'dataset', 'usd_scene');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/scene-camera.js'), 'utf8'))();
  return window.SceneCamera;
}

const SceneCamera = load();
const SCENE = JSON.parse(fs.readFileSync(path.join(DATASET, 'viewer_scene.json'), 'utf8'));

// %% the viewer's own camera, as the scene panel builds it
const VIEWER_FOV = 45;
const VIEWER_FAR = 200;
const VIEWER_MAX_DISTANCE = 25;
const ASPECT = 1.6;
const NEAR = 0.01;

function viewerCamera() {
  return {
    camera: new THREE.PerspectiveCamera(VIEWER_FOV, ASPECT, NEAR, VIEWER_FAR),
    controls: { maxDistance: VIEWER_MAX_DISTANCE },
  };
}

function state(view) {
  return [view.camera.fov, view.camera.far, view.camera.projectionMatrix.elements.slice(), view.controls.maxDistance];
}

// %% fields a bundle does not set
test('a scene without a camera leaves the viewer\'s camera as it is', function () {
  const view = viewerCamera();
  const before = state(view);
  SceneCamera.of({}).applyTo(view.camera, view.controls);
  assert.deepEqual(state(view), before);
});

test('a camera with only a position and a target leaves the lens, far plane and zoom range', function () {
  const view = viewerCamera();
  const before = state(view);
  SceneCamera.of({ camera: { position: SCENE.camera.position, target: SCENE.camera.target } })
    .applyTo(view.camera, view.controls);
  assert.deepEqual(state(view), before);
});

test('a field of view outside the open range of a lens is not used', function () {
  [0, 180, -5, 'wide'].forEach(function (fov) {
    assert.equal(SceneCamera.of({ camera: { fov: fov } }).fov, null);
  });
});

test('a far plane that is not a positive number is not used', function () {
  [0, -1, 'far'].forEach(function (far) {
    assert.equal(SceneCamera.of({ camera: { far: far } }).far, null);
  });
});

// %% fields a bundle sets
test('the vertical field of view and the far plane come from the scene', function () {
  const view = SceneCamera.of(SCENE);
  assert.deepEqual([view.fov, view.far], [SCENE.camera.fov, SCENE.camera.far]);
});

test('the scene\'s lens and far plane reach the camera and its projection', function () {
  const view = viewerCamera();
  SceneCamera.of(SCENE).applyTo(view.camera, view.controls);
  const expected = new THREE.PerspectiveCamera(SCENE.camera.fov, ASPECT, NEAR, SCENE.camera.far);
  assert.deepEqual(state(view).slice(0, 3), [SCENE.camera.fov, SCENE.camera.far, expected.projectionMatrix.elements]);
});

test('the orbit zooms out to half the far plane when that is beyond its range', function () {
  const view = viewerCamera();
  SceneCamera.of(SCENE).applyTo(view.camera, view.controls);
  assert.equal(view.controls.maxDistance, SCENE.camera.far / 2);
});

test('a near far plane never shrinks the zoom range', function () {
  const view = viewerCamera();
  SceneCamera.of({ camera: { far: VIEWER_MAX_DISTANCE } }).applyTo(view.camera, view.controls);
  assert.deepEqual([view.camera.far, view.controls.maxDistance], [VIEWER_MAX_DISTANCE, VIEWER_MAX_DISTANCE]);
});

test('a lens alone leaves the far plane and the zoom range', function () {
  const view = viewerCamera();
  SceneCamera.of({ camera: { fov: SCENE.camera.fov } }).applyTo(view.camera, view.controls);
  assert.deepEqual([view.camera.fov, view.camera.far, view.controls.maxDistance],
    [SCENE.camera.fov, VIEWER_FAR, VIEWER_MAX_DISTANCE]);
});
