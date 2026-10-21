// Tests of the scene panel's rendering settings (panels/robot_scene/panel.js, node:test).
// The panel's own functions run against the panel's studio rig — the procedural floor,
// the fill lights and the shadow-casting key light — built as the panel builds them.
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const DATASET = path.join(__dirname, '..', 'dataset', 'usd_scene');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));
const PANEL = fs.readFileSync(path.join(WEB, 'panels/robot_scene/panel.js'), 'utf8');
const MODULES = ['core/scene-rendering.js', 'core/scene-camera.js', 'core/shadow-fit.js', 'core/lab-environment.js'];
//: the slice of the panel that applies a bundle's rendering settings
const SECTION = ['  // %% rendering settings the bundle asks for', '  // %% materials (identical treatment'];
//: rounding tolerance of the matrix arithmetic
const TOLERANCE = 1e-9;

const SCENE = JSON.parse(fs.readFileSync(path.join(DATASET, 'viewer_scene.json'), 'utf8'));
const LIGHTING = JSON.parse(fs.readFileSync(path.join(DATASET, 'viewer_lighting.json'), 'utf8'));
//: the lights the converter writes for a stage lit by a sun alone
const SUN_ONLY = JSON.parse(fs.readFileSync(path.join(DATASET, 'annotated_bundle', 'lighting.json'), 'utf8'));

// %% the panel's studio rig and the section under test
class EnvironmentRenders {
  constructor() { this.calls = []; }
  record(scene, sigma, near, far) {
    const texture = { disposed: false, dispose() { this.disposed = true; } };
    this.calls.push({ scene: scene, sigma: sigma, near: near, far: far, texture: texture });
    return texture;
  }
}

class StudioPanel {
  constructor() {
    const renders = this.renders = new EnvironmentRenders();
    const three = Object.assign({}, THREE, {
      PMREMGenerator: class {
        fromScene(scene, sigma, near, far) { return { texture: renders.record(scene, sigma, near, far) }; }
        dispose() {}
      },
    });
    this.floorLayer = { parentElement: { style: { display: '' } } };
    const window = {};
    const context = this.context = vm.createContext({ THREE: three, window: window, Math: Math, Promise: Promise });
    MODULES.forEach(function (file) { vm.runInContext(fs.readFileSync(path.join(WEB, file), 'utf8'), context); });
    ['SceneRendering', 'SceneBounds', 'SceneCamera', 'ShadowFit', 'LabEnvironment', 'LightingDescription']
      .forEach(function (name) { context[name] = window[name]; });
    const key = new THREE.DirectionalLight(0xfff2df, 1.05);
    key.position.set(4, 7, 3);
    const roomEnvironment = { dispose() { this.disposed = true; } };
    const floorLayer = this.floorLayer;
    Object.assign(context, {
      scene3: { environment: roomEnvironment }, renderer: {}, needsRender: false, sceneBase: 'scenes/viewer_lab/',
      camera: new THREE.PerspectiveCamera(45, 1.6, 0.01, 200), controls: { maxDistance: 25 },
      ground: { visible: true }, key: key,
      hemisphere: new THREE.HemisphereLight(0xf4efe6, 0x2a2d33, 0.32),
      ambient: new THREE.AmbientLight(0xfff4e6, 0.05),
      back: new THREE.DirectionalLight(0xdfe6f2, 0.3),
      KEY_DISTANCE: key.position.length(), ENVIRONMENT_BLUR: 0.04,
      $: function (id) { return id === 'lyr-floor' ? floorLayer : null; },
      fetch: function (url) {
        return Promise.resolve({ ok: url === 'scenes/viewer_lab/' + SCENE.rendering.environment, json: function () { return LIGHTING; } });
      },
    });
    const start = PANEL.indexOf(SECTION[0]);
    // `let`/`const` of the section stay script-scoped; the functions read and write them there
    vm.runInContext(PANEL.slice(start, PANEL.indexOf(SECTION[1], start)), context);
    this.roomEnvironment = roomEnvironment;
  }

  apply(scene) { return vm.runInContext('applySceneRendering', this.context)(scene); }
  get key() { return this.context.key; }
}

function travel(light) {
  return light.target.position.clone().sub(light.position).normalize();
}

function near(actual, expected) {
  assert.ok(Math.abs(actual - expected) < TOLERANCE, actual + ' is not ' + expected);
}

// %% a bundle without the fields
test('a scene without rendering fields leaves the floor, lights and environment as they are', async function () {
  const panel = new StudioPanel();
  const before = [panel.key.position.toArray(), panel.key.intensity, panel.key.shadow.camera.left];
  await panel.apply({ name: 'plain' });
  assert.deepEqual(
    [panel.context.ground.visible, panel.floorLayer.parentElement.style.display, panel.renders.calls.length,
      panel.context.hemisphere.intensity, [panel.key.position.toArray(), panel.key.intensity, panel.key.shadow.camera.left]],
    [true, '', 0, 0.32, before]);
});

// %% the floor
test('a scene that turns the floor off hides the floor and its layer switch', async function () {
  const panel = new StudioPanel();
  await panel.apply({ rendering: { groundPlane: false } });
  assert.deepEqual([panel.context.ground.visible, panel.floorLayer.parentElement.style.display], [false, 'none']);
});

test('the floor switch cannot bring back a floor the scene turned off', async function () {
  const panel = new StudioPanel();
  await panel.apply({ rendering: { groundPlane: false } });
  vm.runInContext('setFloorVisible', panel.context)(true);
  assert.equal(panel.context.ground.visible, false);
});

test('the floor switch shows and hides the floor of a scene that keeps it', async function () {
  const panel = new StudioPanel();
  await panel.apply({ name: 'plain' });
  const setFloorVisible = vm.runInContext('setFloorVisible', panel.context);
  setFloorVisible(false);
  const hidden = panel.context.ground.visible;
  setFloorVisible(true);
  assert.deepEqual([hidden, panel.context.ground.visible], [false, true]);
});

// %% the key light's shadow
test('shadow bounds fit the shadow camera of the default key light without turning it', async function () {
  const panel = new StudioPanel();
  const before = travel(panel.key);
  await panel.apply({ rendering: { shadowBounds: SCENE.rendering.shadowBounds } });
  const bounds = panel.context.SceneBounds.of(SCENE.rendering.shadowBounds);
  const frustum = new panel.context.ShadowFit(bounds, panel.context.ShadowFit.fromThree(before)).frustum(THREE);
  const camera = panel.key.shadow.camera;
  assert.deepEqual([camera.left, camera.right, camera.top, camera.bottom, camera.near, camera.far],
    [frustum.left, frustum.right, frustum.top, frustum.bottom, frustum.near, frustum.far]);
  [0, 1, 2].forEach(function (axis) { near(travel(panel.key).getComponent(axis), before.getComponent(axis)); });
});

// %% the bundle's own lights
test('a lighting description replaces the environment and switches the studio fill lights off', async function () {
  const panel = new StudioPanel();
  await panel.apply(SCENE);
  const [render] = panel.renders.calls;
  assert.deepEqual(
    [panel.renders.calls.length, panel.context.scene3.environment, panel.roomEnvironment.disposed,
      panel.context.hemisphere.intensity, panel.context.ambient.intensity, panel.context.back.intensity],
    [1, render.texture, true, 0, 0, 0]);
});

test('the environment is rendered from the described room, out to its farthest corner', async function () {
  const panel = new StudioPanel();
  await panel.apply(SCENE);
  const lab = new panel.context.LabEnvironment(panel.context.LightingDescription.of(LIGHTING),
    panel.context.SceneBounds.of(SCENE.rendering.shadowBounds));
  const [render] = panel.renders.calls;
  assert.deepEqual([render.sigma, render.far], [panel.context.ENVIRONMENT_BLUR, lab.reach()]);
});

test('the key light takes the colour, strength and direction of the strongest direct light', async function () {
  const panel = new StudioPanel();
  await panel.apply(SCENE);
  const keyLight = new panel.context.LabEnvironment(panel.context.LightingDescription.of(LIGHTING),
    panel.context.SceneBounds.of(SCENE.rendering.shadowBounds)).keyLight();
  const expected = panel.context.ShadowFit.toThree(THREE, keyLight.direction).normalize();
  assert.deepEqual([panel.key.color.toArray(), panel.key.intensity], [Array.from(keyLight.color), keyLight.intensity]);
  [0, 1, 2].forEach(function (axis) { near(travel(panel.key).getComponent(axis), expected.getComponent(axis)); });
});

test('the key light\'s shadows cover the region the key light asks for', async function () {
  const panel = new StudioPanel();
  await panel.apply(SCENE);
  const keyLight = new panel.context.LabEnvironment(panel.context.LightingDescription.of(LIGHTING),
    panel.context.SceneBounds.of(SCENE.rendering.shadowBounds)).keyLight();
  const frustum = new panel.context.ShadowFit(keyLight.shadowBounds, keyLight.direction).frustum(THREE);
  const camera = panel.key.shadow.camera;
  ['left', 'right', 'top', 'bottom', 'near', 'far'].forEach(function (side) { near(camera[side], frustum[side]); });
});

test('a sun alone keeps the studio environment and fill lights', async function () {
  const panel = new StudioPanel();
  await panel.apply({ rendering: { environment: SUN_ONLY } });
  assert.deepEqual(
    [panel.renders.calls.length, panel.context.scene3.environment, panel.roomEnvironment.disposed,
      panel.context.hemisphere.intensity, panel.context.ambient.intensity, panel.context.back.intensity],
    [0, panel.roomEnvironment, undefined, 0.32, 0.05, 0.3]);
});

test('a sun alone becomes the key light, aimed at the origin from the key light\'s distance', async function () {
  const panel = new StudioPanel();
  await panel.apply({ rendering: { environment: SUN_ONLY } });
  const keyLight = new panel.context.LabEnvironment(panel.context.LightingDescription.of(SUN_ONLY), null).keyLight();
  const expected = panel.context.ShadowFit.toThree(THREE, keyLight.direction).normalize()
    .multiplyScalar(-panel.context.KEY_DISTANCE);
  assert.equal(panel.key.intensity, keyLight.intensity);
  [0, 1, 2].forEach(function (axis) {
    near(panel.key.position.getComponent(axis), expected.getComponent(axis));
    near(panel.key.target.position.getComponent(axis), 0);
  });
});

test('a lighting file that cannot be fetched keeps the studio lighting', async function () {
  const panel = new StudioPanel();
  await panel.apply({ rendering: { environment: 'missing.json' } });
  assert.deepEqual([panel.renders.calls.length, panel.context.hemisphere.intensity], [0, 0.32]);
});

test('a description without usable lights keeps the studio lighting', async function () {
  const panel = new StudioPanel();
  await panel.apply({ rendering: { environment: { panels: [] } } });
  assert.deepEqual([panel.renders.calls.length, panel.context.hemisphere.intensity, panel.key.intensity], [0, 0.32, 1.05]);
});

// %% the camera
test('the scene\'s lens and far plane reach the panel camera and its orbit', async function () {
  const panel = new StudioPanel();
  await panel.apply(SCENE);
  assert.deepEqual([panel.context.camera.fov, panel.context.camera.far, panel.context.controls.maxDistance],
    [SCENE.camera.fov, SCENE.camera.far, SCENE.camera.far / 2]);
});

// %% the panel's wiring
test('the rendering modules load before the scene panel', function () {
  const html = fs.readFileSync(path.join(WEB, 'index.html'), 'utf8');
  const panel = html.indexOf('src="panels/robot_scene/panel.js"');
  MODULES.forEach(function (file) {
    const module = html.indexOf('src="' + file + '"');
    assert.ok(module >= 0 && module < panel, file + ' loads before the scene panel');
  });
});

test('a fetched scene gets its rendering settings right after it is loaded', function () {
  const loaded = /\.then\(function \(sc\) \{\s*loadScene\(sc\);\s*applySceneRendering\(sc\);/;
  assert.match(PANEL.slice(PANEL.indexOf("fetch(sceneBase + 'scene.json')")), loaded);
});
