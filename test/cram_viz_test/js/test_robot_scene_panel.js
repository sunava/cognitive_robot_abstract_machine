// Unit tests for panels/robot_scene/{playback,model-loader,drag-controls,
// live-bridge}.js (node:test). The panel itself (panel.js) is a thin
// composition of these modules plus a full three.js/WebGL bootstrap that
// isn't practical to stub faithfully here; these modules hold the actual
// bug-prone logic (trajectory blending, material theming, drag/teardown,
// live-bridge polling) and are what previously shipped with zero coverage.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cram_viz', 'src', 'cram_viz', 'web');

function load(file) {
  new Function(fs.readFileSync(path.join(WEB, file), 'utf8'))();
}

function resetGlobals() {
  global.window = { location: { search: '' } };
  delete global.THREE;
  delete global.document;
  delete global.fetch;
}

// ============================================================== playback ====
test('smooth clamps to [0,1] and eases like a smoothstep curve', function () {
  resetGlobals();
  load('panels/robot_scene/playback.js');
  const smooth = window.RobotScenePlayback.smooth;
  assert.strictEqual(smooth(-1), 0);
  assert.strictEqual(smooth(0), 0);
  assert.strictEqual(smooth(1), 1);
  assert.strictEqual(smooth(2), 1);
  assert.strictEqual(smooth(0.5), 0.5);
  assert.ok(smooth(0.25) < 0.25);
});

test('resolveJoint finds the joint on whichever model owns the key\'s prefix', function () {
  resetGlobals();
  load('panels/robot_scene/playback.js');
  const resolveJoint = window.RobotScenePlayback.resolveJoint;
  const pr2Joint = {};
  const models = [
    { prefix: 'pr2', obj: { joints: { l_shoulder_pan_joint: pr2Joint } } },
    { prefix: '', obj: { joints: { hinge: {} } } },
  ];
  assert.strictEqual(resolveJoint(models, 'pr2/l_shoulder_pan_joint'), pr2Joint);
  assert.strictEqual(resolveJoint(models, 'hinge'), models[1].obj.joints.hinge);
  assert.strictEqual(resolveJoint(models, 'pr2/missing_joint'), null);
  assert.strictEqual(resolveJoint(models, 'nope/x'), null);
});

test('setPose lerps position and slerps orientation between two poses', function () {
  resetGlobals();
  function Vector3(x, y, z) { this.x = x || 0; this.y = y || 0; this.z = z || 0; }
  Vector3.prototype.set = function (x, y, z) { this.x = x; this.y = y; this.z = z; return this; };
  Vector3.prototype.copy = function (v) { this.x = v.x; this.y = v.y; this.z = v.z; return this; };
  Vector3.prototype.lerp = function (v, t) {
    this.x += (v.x - this.x) * t; this.y += (v.y - this.y) * t; this.z += (v.z - this.z) * t;
    return this;
  };
  function Quaternion() { this.x = 0; this.y = 0; this.z = 0; this.w = 1; }
  Quaternion.prototype.set = function (x, y, z, w) { this.x = x; this.y = y; this.z = z; this.w = w; return this; };
  Quaternion.prototype.copy = function (q) { this.x = q.x; this.y = q.y; this.z = q.z; this.w = q.w; return this; };
  Quaternion.prototype.slerp = function (q, t) {
    this.x += (q.x - this.x) * t; this.y += (q.y - this.y) * t; this.z += (q.z - this.z) * t; this.w += (q.w - this.w) * t;
    return this;
  };
  global.THREE = { Vector3: Vector3, Quaternion: Quaternion };
  load('panels/robot_scene/playback.js');
  const obj = { position: new global.THREE.Vector3(), quaternion: new global.THREE.Quaternion() };
  window.RobotScenePlayback.setPose(obj, [0, 0, 0, 0, 0, 0, 1], [2, 4, 6, 0, 0, 0, 1], 0.5);
  assert.strictEqual(obj.position.x, 1);
  assert.strictEqual(obj.position.y, 2);
  assert.strictEqual(obj.position.z, 3);
});

test('TransportBlender blends an object from its drag offset to the place offset across the pick segment', function () {
  resetGlobals();
  load('panels/robot_scene/playback.js');
  const blender = new window.RobotScenePlayback.TransportBlender();
  const sceneObjects = [{ key: 'milk', spawn: [1, 1, 0] }];
  const segments = [{ picks: 'milk_id', attach: 10, detach: 20, start: 5, end: 25 }];
  const objectKeyById = { milk_id: 'milk' };
  blender.configure(sceneObjects, segments, objectKeyById);

  // before any drag: resting at zero offset, and not yet picked
  assert.deepStrictEqual(blender.objOffsetAt('milk', 0), { x: 0, y: 0 });
  assert.strictEqual(blender.restingBeforePick('milk', 0), true);
  assert.strictEqual(blender.restingBeforePick('milk', 15), false);

  // drag the object away from its recorded spawn before pickup
  blender.recordDrag('milk', 1.5, 1.25, 0.05);
  assert.deepStrictEqual(blender.pickDelta('milk'), { x: 0.5, y: 0.25, zAbs: 0.05 });
  assert.deepStrictEqual(blender.objOffsetAt('milk', 5), { x: 0.5, y: 0.25 });
  // at attach, offset is exactly the drag delta
  assert.deepStrictEqual(blender.objOffsetAt('milk', 10), { x: 0.5, y: 0.25 });

  blender.setPlaceDelta(2, -1);
  // mid-carry (attach..detach): the object and the carrying base blend
  // identically — this is the logic that used to be duplicated with
  // slightly different formulas in each caller.
  const objMid = blender.objOffsetAt('milk', 15);
  const baseMid = blender.baseOffsetAt(15);
  assert.deepStrictEqual(objMid, baseMid);
  assert.ok(objMid.x > 0.5 && objMid.x < 2);
  assert.ok(objMid.y < 0.25 && objMid.y > -1);

  // after detach: the object is fully at the place delta...
  assert.deepStrictEqual(blender.objOffsetAt('milk', 20), { x: 2, y: -1 });
  assert.deepStrictEqual(blender.objOffsetAt('milk', 24), { x: 2, y: -1 });
  // ...while the carrying base ramps back down toward zero by segment end
  const baseNearEnd = blender.baseOffsetAt(24);
  assert.ok(baseNearEnd.x > 0 && baseNearEnd.x < 2);
  // outside the segment entirely, the base offset is zero
  assert.deepStrictEqual(blender.baseOffsetAt(0), { x: 0, y: 0 });
});

// ============================================================ model-loader ====
function fakeCanvasDocument() {
  function fakeContext() {
    return new Proxy({}, {
      get(target, prop) {
        if (prop === 'measureText') return function () { return { width: 10 }; };
        if (prop === 'createLinearGradient') return function () { return { addColorStop: function () {} }; };
        if (prop in target) return target[prop];
        return function () {};
      },
      set(target, prop, value) { target[prop] = value; return true; },
    });
  }
  return { createElement: function () { return { width: 0, height: 0, getContext: function () { return fakeContext(); } }; } };
}

test('ModelTamer strips imported lights but keeps the scene\'s own, and dims washed-out robot materials', function () {
  resetGlobals();
  global.document = fakeCanvasDocument();
  function CanvasTexture() { this.wrapS = null; this.wrapT = null; this.anisotropy = 0; }
  global.THREE = { CanvasTexture: CanvasTexture, RepeatWrapping: 'repeat' };
  load('panels/robot_scene/model-loader.js');

  const ownLight = { isLight: true };
  const scene3 = { traverse: function (cb) { cb(ownLight); } };
  const tamer = new window.RobotSceneModelLoader.ModelTamer(scene3);

  const importedLight = { isLight: true, parent: { remove: function (l) { this.removed = l; } } };
  const brightMesh = {
    isMesh: true, userData: {}, castShadow: false, receiveShadow: false,
    material: { color: { r: 1, g: 1, b: 1, setRGB: function (r, g, b) { this.r = r; this.g = g; this.b = b; } } },
  };
  const robotObj = { traverse: function (cb) { cb(importedLight); cb(brightMesh); } };

  tamer.upgrade([{ robot: true, obj: robotObj }]);

  assert.strictEqual(importedLight.parent.removed, importedLight, 'imported light stripped');
  assert.strictEqual(brightMesh.userData._tamed, true);
  assert.strictEqual(brightMesh.castShadow, true);
  assert.ok(brightMesh.material.color.r < 0.92, 'washed-out white got dimmed');
});

test('ModelTamer themes environment materials by their URDF link name', function () {
  resetGlobals();
  global.document = fakeCanvasDocument();
  function CanvasTexture() { this.wrapS = null; this.wrapT = null; this.anisotropy = 0; }
  global.THREE = { CanvasTexture: CanvasTexture, RepeatWrapping: 'repeat' };
  load('panels/robot_scene/model-loader.js');

  const scene3 = { traverse: function () {} };
  const tamer = new window.RobotSceneModelLoader.ModelTamer(scene3);

  const cooktopLink = { isURDFLink: true, name: 'iai_kitchen/cooktop', parent: null };
  const cooktopMesh = {
    isMesh: true, userData: {}, parent: cooktopLink,
    material: { color: { setHex: function (h) { this.hex = h; } } },
  };
  const envObj = { traverse: function (cb) { cb(cooktopMesh); } };

  tamer.upgrade([{ robot: false, obj: envObj }]);

  assert.strictEqual(cooktopMesh.material.color.hex, 0x0a0b0d);
});

// ============================================================ drag-controls ====
function makeThreeForDrag() {
  function Vector2(x, y) { this.x = x || 0; this.y = y || 0; }
  Vector2.prototype.set = function (x, y) { this.x = x; this.y = y; return this; };
  Vector2.prototype.copy = function (v) { this.x = v.x; this.y = v.y; return this; };
  function Vector3(x, y, z) { this.x = x || 0; this.y = y || 0; this.z = z || 0; }
  Vector3.prototype.set = function () { return this; };
  Vector3.prototype.copy = function (v) { this.x = v.x; this.y = v.y; this.z = v.z; return this; };
  Vector3.prototype.clone = function () { return new Vector3(this.x, this.y, this.z); };
  Vector3.prototype.addScaledVector = function () { return this; };
  Vector3.prototype.applyQuaternion = function () { return this; };
  Vector3.prototype.normalize = function () { return this; };
  Vector3.prototype.add = function () { return this; };
  Vector3.prototype.distanceTo = function () { return 1; };
  function Plane() {}
  Plane.prototype.setFromNormalAndCoplanarPoint = function () { return this; };
  function Box3() {}
  Box3.prototype.makeEmpty = function () { return this; };
  Box3.prototype.expandByObject = function () { return this; };
  Box3.prototype.getCenter = function (target) { return target || new Vector3(); };
  function Raycaster() { this.far = Infinity; this.ray = { intersectPlane: function (plane, target) { return target; } }; }
  Raycaster.prototype.setFromCamera = function () {};
  Raycaster.prototype.set = function () {};
  Raycaster.prototype.intersectObjects = function (list) { return list.length ? [{ object: list[0], point: new Vector3() }] : []; };
  return { Vector2: Vector2, Vector3: Vector3, Plane: Plane, Box3: Box3, Raycaster: Raycaster };
}

function fakeDomElement() {
  const listeners = {};
  return {
    listeners: listeners,
    addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
    removeEventListener(type, fn) {
      const arr = listeners[type] || [];
      const i = arr.indexOf(fn);
      if (i >= 0) arr.splice(i, 1);
    },
    fire(type, evt) { (listeners[type] || []).slice().forEach(function (fn) { fn(evt); }); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; },
    setPointerCapture() {},
    style: {},
  };
}
function fakeGroup() {
  return {
    position: { x: 0, y: 0, z: 0 },
    traverse(cb) { cb({ isMesh: true, userData: {} }); },
    getWorldPosition(target) { return target; },
  };
}
function fakeCamera() {
  return { fov: 45, aspect: 1, position: { distanceTo: function () { return 1; } }, quaternion: {}, getWorldDirection: function () {} };
}

test('DragControls registers 4 pointer listeners and destroy() removes exactly those', function () {
  resetGlobals();
  global.THREE = makeThreeForDrag();
  load('panels/robot_scene/drag-controls.js');
  const domElement = fakeDomElement();
  const dc = new window.RobotSceneDragControls.DragControls({
    camera: fakeCamera(), renderer: { domElement: domElement }, worldRoot: { worldToLocal: function (v) { return v; } },
    getObjectMeshes: function () { return {}; },
    getObjectGroup: function () { return undefined; },
    getMarker: function () { return { group: fakeGroup(), visible: false }; },
    getEnvMeshes: function () { return []; },
    getDragBounds: function () { return { minX: -1, maxX: 1, minY: -1, maxY: 1 }; },
    getMarkerBounds: function () { return null; },
    isLive: function () { return false; },
    isPlaying: function () { return false; },
    objectIdFor: function () { return null; },
    classifyMiss: function () { return null; },
    controlsEnabled: function () {},
    onClick: function () {},
  });
  const types = ['pointerdown', 'pointermove', 'pointerup', 'pointercancel'];
  types.forEach(function (t) { assert.strictEqual(domElement.listeners[t].length, 1, t); });
  dc.destroy();
  types.forEach(function (t) { assert.strictEqual(domElement.listeners[t].length, 0, t + ' after destroy'); });
});

test('a plain click on a draggable object resolves to its entity id via onClick', function () {
  resetGlobals();
  global.THREE = makeThreeForDrag();
  load('panels/robot_scene/drag-controls.js');
  const domElement = fakeDomElement();
  const meshes = { milk: fakeGroup() };
  const clicked = [];
  const dc = new window.RobotSceneDragControls.DragControls({
    camera: fakeCamera(), renderer: { domElement: domElement }, worldRoot: { worldToLocal: function (v) { return v; } },
    getObjectMeshes: function () { return meshes; },
    getObjectGroup: function (key) { return meshes[key]; },
    getMarker: function () { return { group: fakeGroup(), visible: false }; },
    getEnvMeshes: function () { return []; },
    getDragBounds: function () { return { minX: -1, maxX: 1, minY: -1, maxY: 1 }; },
    getMarkerBounds: function () { return null; },
    isLive: function () { return false; },
    isPlaying: function () { return false; },
    objectIdFor: function (key) { return 'entity:' + key; },
    classifyMiss: function () { return null; },
    controlsEnabled: function () {},
    onClick: function (id) { clicked.push(id); },
  });
  domElement.fire('pointerdown', { button: 0, clientX: 10, clientY: 10, preventDefault: function () {} });
  domElement.fire('pointerup', { clientX: 10, clientY: 10 });
  assert.deepStrictEqual(clicked, ['entity:milk']);
  dc.destroy();
});

// ============================================================== live-bridge ====
test('postMove throttles rapid non-final drags but a final move always goes through', function () {
  resetGlobals();
  const calls = [];
  global.fetch = function (url, opts) { calls.push(opts); return Promise.resolve({ json: function () { return Promise.resolve({}); } }); };
  const realPerformance = global.performance;
  let fakeNow = 10000; // start well past the throttle window so the very first call isn't spuriously throttled
  global.performance = { now: function () { return fakeNow; } };
  load('panels/robot_scene/live-bridge.js');
  const bridge = new window.RobotSceneLiveBridge.LiveBridge({});
  bridge.postMove('milk', 1, 2, 3, false);
  bridge.postMove('milk', 1, 2, 3, false); // immediately after -> throttled
  assert.strictEqual(calls.length, 1);
  bridge.postMove('milk', 1, 2, 3, true); // final always goes through, even mid-throttle-window
  assert.strictEqual(calls.length, 2);
  fakeNow += 200; // past the throttle window
  bridge.postMove('milk', 1, 2, 3, false);
  assert.strictEqual(calls.length, 3);
  global.performance = realPerformance;
});

test('destroy() clears both the poll and probe intervals', function () {
  resetGlobals();
  global.fetch = function () { return Promise.resolve({ json: function () { return Promise.resolve({}); } }); };
  load('panels/robot_scene/live-bridge.js');
  const realClearInterval = global.clearInterval;
  let clearedCount = 0;
  global.clearInterval = function (id) { clearedCount++; return realClearInterval(id); };
  const bridge = new window.RobotSceneLiveBridge.LiveBridge({});
  bridge.startProbing();
  bridge.attach();
  bridge.destroy();
  global.clearInterval = realClearInterval;
  assert.strictEqual(clearedCount, 2);
});

test('a bridge that never attached still has nothing to clear on destroy', function () {
  resetGlobals();
  global.fetch = function () { return Promise.resolve({ json: function () { return Promise.resolve({}); } }); };
  load('panels/robot_scene/live-bridge.js');
  const bridge = new window.RobotSceneLiveBridge.LiveBridge({});
  assert.doesNotThrow(function () { bridge.destroy(); });
  assert.strictEqual(bridge.isOn(), false);
});
