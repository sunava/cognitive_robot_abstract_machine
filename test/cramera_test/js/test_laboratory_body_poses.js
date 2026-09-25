'use strict';

// %% transformed authored body hierarchy
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const web = path.join(__dirname, '../../../cramera/src/cramera/web');
const THREE = require(path.join(web, 'vendor/three.min.js'));
const Physics = require(path.join(web, 'core/laboratory-physics.js'));
const scope = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(web, 'core/model_poses.js'), 'utf8'), scope);
const ModelPoses = scope.window.ModelPoses;

function interpolate(object, current, next, fraction) {
  object.position.fromArray(current).lerp(new THREE.Vector3().fromArray(next), fraction);
  object.quaternion.fromArray(current, 3).slerp(new THREE.Quaternion().fromArray(next, 3), fraction);
}

class MeasuredRackScene {
  constructor(prefix = '') {
    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2;
    this.model = new THREE.Group();
    this.model.position.set(.4, -.2, .1);
    this.model.rotation.z = .3;
    this.joint = new THREE.Group();
    this.joint.position.set(.22, -.04, .9);
    this.rack = new THREE.Group();
    this.handle = new THREE.Group();
    this.handle.position.set(0, -.08, .055);
    this.rack.add(this.handle);
    this.joint.add(this.rack);
    this.model.add(this.joint);
    this.model.links = {laboratory_rack: this.rack};
    this.models = [{name: 'laboratory', prefix, obj: this.model}];
    this.root.add(this.model);
  }
  pose(position) { return [...position, 0, 0, 0, 1]; }
  observed(object) {
    this.root.updateMatrixWorld(true);
    return this.root.worldToLocal(object.getWorldPosition(new THREE.Vector3()));
  }
}

test('measured rack poses replace the authored bench offset under a rotated scene root', () => {
  const scene = new MeasuredRackScene();
  const position = [-.52, -.2, .632];
  ModelPoses.applyBodies(scene.models, scene.root, {laboratory_rack: scene.pose(position)}, null, 0, interpolate);
  assert.ok(scene.observed(scene.rack).distanceTo(new THREE.Vector3(...position)) < 1e-12);
  assert.ok(scene.observed(scene.handle).distanceTo(new THREE.Vector3(-.52, -.28, .687)) < 1e-12);
  assert.equal(scene.rack.parent, scene.joint);
  assert.deepEqual(scene.joint.position.toArray(), [.22, -.04, .9]);
});

test('recorded body poses interpolate independently and retain an absent following observation', () => {
  const scene = new MeasuredRackScene();
  const start = scene.pose([0, 0, .632]), end = scene.pose([.4, -.2, .9]);
  ModelPoses.applyBodies(scene.models, scene.root, {laboratory_rack: start}, {laboratory_rack: end}, .5, interpolate);
  assert.ok(scene.observed(scene.rack).distanceTo(new THREE.Vector3(.2, -.1, .766)) < 1e-12);
  ModelPoses.applyBodies(scene.models, scene.root, {laboratory_rack: end}, {}, .5, interpolate);
  assert.ok(scene.observed(scene.rack).distanceTo(new THREE.Vector3(.4, -.2, .9)) < 1e-12);
});

test('body names select the requested model namespace and leave unrelated joints unchanged', () => {
  const first = new MeasuredRackScene('first');
  const second = new MeasuredRackScene('second');
  first.root.add(second.model);
  const original = first.rack.position.clone();
  const position = [-.52, -.3, .8];
  ModelPoses.applyBodies([...first.models, ...second.models], first.root, {'second/laboratory_rack': second.pose(position)}, null, 0, interpolate);
  assert.ok(first.observed(second.rack).distanceTo(new THREE.Vector3(...position)) < 1e-12);
  assert.deepEqual(first.rack.position.toArray(), original.toArray());
});

test('physics polling passes measured body poses to the viewer independently of tube objects', async () => {
  const bodies = {laboratory_rack: [-.52, -.3, .632, 0, 0, 0, 1]};
  const applied = [];
  const service = {state: async () => ({state: 'running', objects: {}, bodyPoses: bodies})};
  const controller = new Physics.Controller({objects: []}, service, {
    setObjectPose: () => assert.fail('Rack is an authored model link, not a loose tube'),
    setBodyPoses: value => applied.push(value),
  });
  await controller.refresh();
  assert.deepEqual(applied, [bodies]);
});

test('physics polling applies the measured drawer slide alongside independent rack motion', async () => {
  const joints = {laboratory_drawer_joint: .27};
  const applied = [];
  const controller = new Physics.Controller({objects: []}, {
    state: async () => ({state: 'running', objects: {}, drawer: {joints}}),
  }, {setObjectPose() {}, setJointFrames: value => applied.push(value)});
  await controller.refresh();
  assert.deepEqual(applied, [joints]);
});

test('recorded playback routes body tracks after updating model bases and joints', () => {
  const source = fs.readFileSync(path.join(web, 'panels/robot_scene/panel.js'), 'utf8');
  const scene = new MeasuredRackScene();
  const start = {laboratory_rack: scene.pose([0, 0, .632])};
  const end = {laboratory_rack: scene.pose([.4, -.2, .9])};
  const playback = {traj: {frames: [{}, {}], bodyPoses: [start, end]},
    models: scene.models, worldRoot: scene.root, robotModel: null, JointRouting: {}, ModelPoses,
    syncJointControls() {}, setPose: interpolate};
  const begin = source.indexOf('  function applyFrame(f)');
  const finish = source.indexOf('    if (traj.objects)', begin);
  vm.runInNewContext(source.slice(begin, finish) + '\n}', playback);
  playback.applyFrame(.5);
  assert.ok(scene.observed(scene.rack).distanceTo(new THREE.Vector3(.2, -.1, .766)) < 1e-12);
});

test('live scene updates preserve the measured rack orientation and moving handle frame', () => {
  const source = fs.readFileSync(path.join(web, 'panels/robot_scene/panel.js'), 'utf8');
  const scene = new MeasuredRackScene();
  const orientation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
  const position = [-.52, -.3, .8];
  const live = {SCENE: {}, models: scene.models, worldRoot: scene.root, robotModel: null, JointRouting: {}, ModelPoses,
    syncJointControls() {}, setPose: interpolate};
  const begin = source.indexOf('  function applyLive(st)');
  const finish = source.indexOf("    if (typeof st.markersVersion", begin);
  vm.runInNewContext(source.slice(begin, finish) + '\n}', live);
  live.applyLive({frames: {}, bodyPoses: {laboratory_rack: [...position, ...orientation.toArray()]}});
  assert.ok(scene.observed(scene.rack).distanceTo(new THREE.Vector3(...position)) < 1e-12);
  const expectedHandle = new THREE.Vector3(...position).add(scene.handle.position.clone().applyQuaternion(orientation));
  assert.ok(scene.observed(scene.handle).distanceTo(expectedHandle) < 1e-12);
});

// %% authored bodies in native streams
class NativeModelStream {
  constructor() {
    this.scene = new MeasuredRackScene('laboratory');
    this.added = [];
    this.catalogRequests = 0;
    this.catalog = [];
    this.context = {
      SCENE: {modelBodies: {laboratory_rack: 'laboratory/laboratory_rack'}},
      models: this.scene.models, worldRoot: this.scene.root, robotModel: null,
      JointRouting: {}, ModelPoses, syncJointControls() {}, setPose: interpolate,
      liveStateKeys: {}, liveDraggedKey: null, objectMeshes: {}, follow: false,
      needsRender: false, liveOn: true, liveSyncing: false, liveSpawned: {},
      liveUrl: () => '/native',
      fetch: async () => {
        this.catalogRequests += 1;
        return {json: async () => ({objects: this.catalog})};
      },
      addObject: spec => this.added.push(spec),
    };
    const source = fs.readFileSync(path.join(web, 'panels/robot_scene/panel.js'), 'utf8');
    const begin = source.indexOf('  function applyLive(st)');
    const finish = source.indexOf('  let livePolls', begin);
    vm.runInNewContext(source.slice(begin, finish), this.context);
    const catalogBegin = source.indexOf('  function syncLiveObjects()');
    const catalogFinish = source.indexOf('  // watching a running demo', catalogBegin);
    vm.runInNewContext(source.slice(catalogBegin, catalogFinish), this.context);
  }
  async reconcile() {
    this.context.syncLiveObjects();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(this.context.liveSyncing, false);
  }
}

test('native object observations move their source-owned rack and handles without requesting a loose duplicate', () => {
  const live = new NativeModelStream();
  const position = [-.52, -.3, .8];
  live.context.applyLive({frames: {}, objects: {laboratory_rack: live.scene.pose(position)},
    modelBases: {laboratory: live.scene.pose([.6, .1, .2])}});
  assert.ok(live.scene.observed(live.scene.rack).distanceTo(new THREE.Vector3(...position)) < 1e-12);
  const expectedHandle = new THREE.Vector3(...position).add(live.scene.handle.position);
  assert.ok(live.scene.observed(live.scene.handle).distanceTo(expectedHandle) < 1e-12);
  assert.equal(live.catalogRequests, 0);
});

test('measured body poses override semantic object observations for the same authored link', () => {
  const live = new NativeModelStream();
  const measured = [-.5, -.2, .9];
  live.context.applyLive({frames: {}, objects: {laboratory_rack: live.scene.pose([0, 0, 0])},
    bodyPoses: {'laboratory/laboratory_rack': live.scene.pose(measured)}});
  assert.ok(live.scene.observed(live.scene.rack).distanceTo(new THREE.Vector3(...measured)) < 1e-12);
  assert.equal(live.catalogRequests, 0);
});

test('live catalog keeps the authored rack subtree and still spawns unrelated loose samples', async () => {
  const live = new NativeModelStream();
  live.catalog = [{id: 'rack', key: 'laboratory_rack'}, {id: 'sample', key: 'new_sample'}];
  await live.reconcile();
  assert.deepEqual(live.added.map(spec => spec.key), ['new_sample']);
  assert.deepEqual(Object.keys(live.context.liveSpawned), ['new_sample']);
});

test('replacing the scene drops its old body mapping and preserves ordinary live overlays', async () => {
  const live = new NativeModelStream();
  live.context.SCENE = {};
  live.catalog = [{id: 'rack', key: 'laboratory_rack'}];
  await live.reconcile();
  assert.deepEqual(live.added.map(spec => spec.key), ['laboratory_rack']);
  const loose = new THREE.Group();
  live.context.objectMeshes.laboratory_rack = loose;
  const original = live.scene.rack.position.clone();
  const position = [1, 2, 3];
  live.context.applyLive({frames: {}, objects: {laboratory_rack: live.scene.pose(position)}});
  assert.deepEqual(loose.position.toArray(), position);
  assert.deepEqual(live.scene.rack.position.toArray(), original.toArray());
});
