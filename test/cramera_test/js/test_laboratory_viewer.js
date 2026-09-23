const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const panel = fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/panels/robot_scene/panel.js'), 'utf8');

// %% actual viewer functions
function functionSource(name) {
  const start = panel.indexOf('  function ' + name + '(');
  assert.notEqual(start, -1, 'viewer function ' + name + ' exists');
  const body = panel.indexOf('{', start);
  let depth = 1;
  let end = body + 1;
  for (; depth && end < panel.length; end += 1) {
    if (panel[end] === '{') depth += 1;
    if (panel[end] === '}') depth -= 1;
  }
  return panel.slice(start, end);
}

class Vector {
  constructor(x = 0, y = 0, z = 0) { Object.assign(this, {x, y, z}); }
  copy(other) { Object.assign(this, other); return this; }
  toArray() { return [this.x, this.y, this.z]; }
}

test('a robotless scene camera converts from laboratory Z-up to renderer Y-up', () => {
  const config = {position: [2, -3, 1.8], target: [.2, -.04, .98]};
  const context = {
    THREE: {Vector3: Vector}, SCENE: {camera: config}, robotModel: null,
    worldRoot: {updateMatrixWorld() {}, localToWorld(point) {
      [point.y, point.z] = [point.z, -point.y]; return point;
    }},
    camera: {position: new Vector()}, controls: {target: new Vector(), update() {}},
    needsRender: false, follow: true,
  };
  vm.createContext(context);
  vm.runInContext(functionSource('focusSceneCamera') + '\n' + functionSource('frameCamera'), context);
  context.frameCamera();
  assert.deepEqual(context.camera.position.toArray(), [2, 1.8, 3]);
  assert.deepEqual(context.controls.target.toArray(), [.2, .98, .04]);
  assert.equal(context.follow, false);
  assert.equal(context.needsRender, true);
});

test('manual controls wait for every asset and preserve independent object poses', () => {
  const clear = {position: new Vector(.16, -.065, .91), quaternion: {toArray: () => [0, 0, 0, 1]}};
  const amber = {position: new Vector(.22, -.065, .91), quaternion: {toArray: () => [0, 0, 0, 1]}};
  const context = {
    SCENE: {objects: [{key: 'clear'}, {key: 'amber'}]}, objectMeshes: {clear}, liveOn: false,
    models: [], needsRender: false,
    setPose(object, first) { object.position = new Vector(...first.slice(0, 3)); },
    JointRouting: {jointFor() { return null; }}, syncJointControls() {},
    stopTrajectory() {}, focusSceneCamera() {},
  };
  vm.createContext(context);
  vm.runInContext(functionSource('laboratoryAdapter'), context);
  const adapter = context.laboratoryAdapter();
  assert.equal(adapter.isReady(), false);
  context.objectMeshes.amber = amber;
  assert.equal(adapter.isReady(), true);
  assert.equal(adapter.setPose('missing', [0, 0, 0, 0, 0, 0, 1]), false);
  adapter.setPose('clear', [.16, -.065, 1.14, 0, 0, 0, 1]);
  assert.deepEqual(Array.from(adapter.getPose('clear')), [.16, -.065, 1.14, 0, 0, 0, 1]);
  assert.deepEqual(Array.from(adapter.getPose('amber')), [.22, -.065, .91, 0, 0, 0, 1]);
  context.liveOn = true;
  assert.equal(adapter.isReady(), false);
});

test('transmissive scenes can render without opaque ambient-occlusion buffers', () => {
  const calls = [];
  const context = {SCENE: {rendering: {ambientOcclusion: false, exposure: .65}}, scene3: {}, camera: {},
    composer: {render() { calls.push('occlusion'); }},
    renderer: {render() { calls.push('direct'); }}};
  vm.createContext(context);
  vm.runInContext(functionSource('renderFrame'), context);
  context.renderFrame();
  assert.equal(context.renderer.toneMappingExposure, context.SCENE.rendering.exposure);
  context.SCENE = {};
  context.renderFrame();
  assert.equal(context.renderer.toneMappingExposure, .95);
  assert.deepEqual(calls, ['direct', 'occlusion']);
});
