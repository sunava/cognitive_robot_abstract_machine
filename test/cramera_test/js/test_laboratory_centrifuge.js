'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const THREE = require('../../../cramera/src/cramera/web/vendor/three.min.js');
const Centrifuge = require('../../../cramera/src/cramera/web/core/laboratory-centrifuge.js');

// %% authored display and measured-state fixture
class InstrumentDisplayScene {
  constructor() {
    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2;
    this.writes = [];
    this.canvas = {getContext: () => ({fillRect() {}, fillText: text => this.writes.push(text)})};
    this.document = {createElement: () => this.canvas};
    this.display = Centrifuge.create(THREE, this.root, this.document);
    this.snapshot = {available: true, running: false, locked: false, lidOpen: false, lidClosed: true, balanced: true, rpm: 0, targetRpm: 0, remainingSeconds: 0};
  }
  addInstrument() {
    this.instrument = new THREE.Group();
    this.instrument.position.set(-.53, -.07, .9);
    this.screen = new THREE.Mesh(new THREE.BoxGeometry(.123, .003, .034), new THREE.MeshBasicMaterial());
    this.screen.name = 'Centrifuge_display';
    this.screen.position.set(-.025, -.161, .081);
    this.instrument.add(this.screen);
    this.letters = ['Centrifuge_status_lettering', 'Centrifuge_capacity_lettering', 'Centrifuge_engraved_identity'].map(name => {
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(.03, .01), new THREE.MeshBasicMaterial());
      mesh.name = name;
      this.instrument.add(mesh);
      return mesh;
    });
    this.root.add(this.instrument);
  }
  update(changes = {}) { Object.assign(this.snapshot, changes); this.display.update(this.snapshot); }
}

test('late authored assets replace only static screen lettering and keep the panel in the actual instrument frame', () => {
  const scene = new InstrumentDisplayScene();
  scene.update();
  assert.equal(scene.display.mesh.visible, false);
  scene.addInstrument();
  scene.update();
  assert.equal(scene.display.mesh.visible, true);
  assert.deepEqual(scene.letters.map(node => node.visible), [false, false, true]);
  scene.root.updateMatrixWorld(true);
  const expected = new THREE.Vector3();
  scene.screen.getWorldPosition(expected);
  scene.root.worldToLocal(expected);
  const overlay = scene.display.mesh;
  assert.ok(Math.abs(overlay.position.x - expected.x) < 1e-8);
  assert.ok(Math.abs(overlay.position.z - expected.z) < 1e-8);
  assert.ok(overlay.position.y < expected.y - scene.screen.geometry.parameters.height / 2);
  assert.ok(overlay.geometry.parameters.width < scene.screen.geometry.parameters.width);
  assert.ok(overlay.geometry.parameters.height < scene.screen.geometry.parameters.depth);
  const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(overlay.quaternion);
  assert.ok(normal.distanceTo(new THREE.Vector3(0, -1, 0)) < 1e-12);
  scene.display.destroy();
  assert.deepEqual(scene.letters.map(node => node.visible), [true, true, true]);
});

test('the in-scene countdown and speed use server measurements without a browser clock', () => {
  const scene = new InstrumentDisplayScene();
  scene.addInstrument();
  scene.update({running: true, locked: true, targetRpm: 120, rpm: 37.4, remainingSeconds: 20});
  assert.ok(scene.writes.includes('00:20'));
  assert.ok(scene.writes.includes('37 RPM'));
  const texture = scene.display.mesh.material.map;
  const writes = scene.writes.length;
  scene.update();
  assert.equal(scene.writes.length, writes);
  scene.update({rpm: 120, remainingSeconds: 12.4});
  assert.ok(scene.writes.includes('00:13'));
  assert.ok(scene.writes.includes('120 RPM'));
  scene.update({remainingSeconds: .01});
  assert.equal(scene.writes.at(-2), '00:01');
  assert.equal(scene.display.mesh.material.map, texture);
  scene.display.destroy();
});

test('physical status distinguishes an open lid, unmet rules, motion, braking and unavailable observations', () => {
  const scene = new InstrumentDisplayScene();
  scene.addInstrument();
  for (const [changes, status] of [
    [{lidOpen: true, lidClosed: false}, 'OPEN'],
    [{lidOpen: false, lidClosed: true, knowledge: {evaluation: {allowed: false}}}, 'CHECK'],
    [{knowledge: {evaluation: {allowed: true}}}, 'READY'],
    [{running: true, targetRpm: 120, rpm: 80, remainingSeconds: 20}, 'RUNNING'],
    [{targetRpm: 0, remainingSeconds: 0}, 'BRAKING'],
    [{running: false, rpm: 0}, 'READY'],
  ]) {
    scene.update(changes);
    assert.equal(scene.writes.at(-3), status);
  }
  scene.display.update(null);
  assert.deepEqual(scene.writes.slice(-3), ['NO DATA', '--:--', '-- RPM']);
  scene.display.destroy();
});

test('teardown disposes owned resources and restores even previously hidden authored lettering', () => {
  const scene = new InstrumentDisplayScene();
  scene.addInstrument();
  scene.letters[0].visible = false;
  scene.update();
  const disposed = [];
  for (const [name, resource] of [['geometry', scene.display.mesh.geometry], ['material', scene.display.mesh.material], ['texture', scene.display.texture]]) {
    resource.addEventListener('dispose', () => disposed.push(name));
  }
  scene.display.destroy();
  assert.deepEqual(disposed.sort(), ['geometry', 'material', 'texture']);
  assert.deepEqual(scene.letters.map(node => node.visible), [false, true, true]);
  assert.deepEqual(scene.root.children, [scene.instrument]);
});

test('the real scene adapter forwards centrifuge snapshots and destroys the display', () => {
  const scene = new InstrumentDisplayScene();
  scene.display.destroy();
  scene.addInstrument();
  const source = fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/panels/robot_scene/panel.js'), 'utf8');
  const context = vm.createContext({
    THREE, LaboratoryCentrifuge: Centrifuge, SCENE: {physics: {centrifuge: {}}},
    worldRoot: scene.root, objectMeshes: {}, root: {ownerDocument: scene.document},
    focusSceneCamera() {}, needsRender: false,
  });
  const start = source.indexOf('  function laboratoryPhysicsAdapter()');
  vm.runInContext(source.slice(start, source.indexOf('  RobotView.onReady(function () {', start)), context);
  const adapter = vm.runInContext('laboratoryPhysicsAdapter()', context);
  adapter.setCentrifugeState({...scene.snapshot, running: true, targetRpm: 120, rpm: 120, remainingSeconds: 17.5});
  assert.ok(scene.writes.includes('00:18'));
  assert.equal(context.needsRender, true);
  adapter.destroy();
  assert.deepEqual(scene.root.children, [scene.instrument]);
});
