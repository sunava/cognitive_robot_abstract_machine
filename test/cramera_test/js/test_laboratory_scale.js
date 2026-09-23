'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const THREE = require('../../../cramera/src/cramera/web/vendor/three.min.js');
const Scale = require('../../../cramera/src/cramera/web/core/laboratory-scale.js');

// %% display scene and canvas recording
class ScaleScene {
  constructor(config = {}) {
    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2;
    this.writes = [];
    this.canvas = {getContext: () => ({fillRect() {}, fillText: text => this.writes.push(text)})};
    this.config = {displayPosition: [.65, .076, .94], displaySize: [.12, .021], ...config};
    this.display = Scale.create(THREE, this.root, this.config, {createElement: () => this.canvas});
  }
  addDigits(name = 'Balance_digits') {
    const digits = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());
    digits.name = name;
    this.root.add(digits);
    return digits;
  }
  update(grams, stable = true) { this.display.update({available: true, grams, stable}); }
}

test('authoritative weight replaces only the authored digits, including late scene loads', () => {
  const scene = new ScaleScene();
  scene.update(12.345);
  const unrelated = scene.addDigits('Other display');
  const digits = scene.addDigits();
  scene.update(12.345);
  assert.equal(digits.visible, false);
  assert.equal(unrelated.visible, true);
  assert.ok(scene.writes.includes('12.345 g'));
  scene.display.destroy();
  assert.equal(digits.visible, true);
});

test('physical display preserves the laboratory frame and faces the front of the balance', () => {
  const scene = new ScaleScene();
  scene.addDigits();
  scene.update(4.5);
  const display = scene.root.children.find(child => child.name === Scale.DISPLAY_NAME);
  assert.deepEqual(display.position.toArray(), scene.config.displayPosition);
  const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(display.quaternion);
  assert.ok(normal.distanceTo(new THREE.Vector3(0, -1, 0)) < 1e-12);
  assert.equal(display.geometry.parameters.width, scene.config.displaySize[0]);
  assert.equal(display.geometry.parameters.height, scene.config.displaySize[1]);
  scene.root.updateMatrixWorld(true);
  const expected = new THREE.Vector3().fromArray(scene.config.displayPosition).applyMatrix4(scene.root.matrixWorld);
  assert.ok(display.getWorldPosition(new THREE.Vector3()).distanceTo(expected) < 1e-12);
  scene.display.destroy();
});

test('negative tared readings remain negative and missing observations never masquerade as zero', () => {
  const scene = new ScaleScene();
  scene.addDigits();
  scene.update(-15.125, false);
  assert.ok(scene.writes.includes('-15.125 g'));
  scene.display.update(null);
  assert.equal(scene.writes.at(-1), '— g');
  scene.display.destroy();
});

test('display refresh reuses its texture and teardown releases owned resources', () => {
  const scene = new ScaleScene();
  const digits = scene.addDigits('authored');
  digits.userData.name = 'Balance digits';
  scene.update(1);
  const display = scene.root.children.find(child => child.name === Scale.DISPLAY_NAME);
  const disposed = [];
  for (const [name, resource] of [['geometry', display.geometry], ['material', display.material], ['texture', display.material.map]]) {
    resource.addEventListener('dispose', () => disposed.push(name));
  }
  const texture = display.material.map;
  scene.update(2);
  assert.equal(display.material.map, texture);
  scene.display.destroy();
  assert.deepEqual(disposed.sort(), ['geometry', 'material', 'texture']);
  assert.deepEqual(scene.root.children, [digits]);
  assert.equal(digits.visible, true);
});

test('the actual robot scene adapter updates and destroys the authored balance display', () => {
  const scene = new ScaleScene();
  scene.display.destroy();
  const digits = scene.addDigits();
  const source = fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/panels/robot_scene/panel.js'), 'utf8');
  const context = vm.createContext({
    THREE, LaboratoryScale: Scale, SCENE: {physics: {scale: scene.config}},
    worldRoot: scene.root, objectMeshes: {},
    root: {ownerDocument: {createElement: () => scene.canvas}},
    focusSceneCamera() {}, needsRender: false,
  });
  const start = source.indexOf('  function laboratoryPhysicsAdapter()');
  vm.runInContext(source.slice(start, source.indexOf('  RobotView.onReady(function () {', start)), context);
  const adapter = vm.runInContext('laboratoryPhysicsAdapter()', context);
  adapter.setScaleState({available: true, grams: 18.75, stable: true});
  assert.equal(digits.visible, false);
  assert.ok(scene.writes.includes('18.750 g'));
  assert.equal(context.needsRender, true);
  adapter.destroy();
  assert.deepEqual(scene.root.children, [digits]);
  assert.equal(digits.visible, true);
});

test('the visible holder uses the same radial box segments as the weighing contacts', () => {
  const holder = {innerRadius: .0115, outerRadius: .016, height: .04, segments: 24};
  const scene = new ScaleScene({panPosition: [.65, .21, .974], holder});
  const collar = scene.root.children.find(child => child.name === Scale.HOLDER_NAME);
  assert.ok(collar, 'a physical tube holder must be visible');
  assert.deepEqual(collar.position.toArray(), scene.config.panPosition);
  assert.equal(collar.children.length, holder.segments);
  for (const [index, wall] of collar.children.entries()) {
    const angle = 2 * Math.PI * index / holder.segments;
    const radius = (holder.innerRadius + holder.outerRadius) / 2;
    assert.deepEqual(wall.position.toArray(), [radius * Math.cos(angle), radius * Math.sin(angle), holder.height / 2]);
    assert.equal(wall.rotation.z, angle);
    assert.equal(wall.geometry.parameters.width, holder.outerRadius - holder.innerRadius);
    assert.equal(wall.geometry.parameters.height, 2 * holder.outerRadius * Math.tan(Math.PI / holder.segments));
    assert.equal(wall.geometry.parameters.depth, holder.height);
  }
  const resources = new Set(collar.children.flatMap(wall => [wall.geometry, wall.material]));
  const disposed = [];
  for (const resource of resources) resource.addEventListener('dispose', () => disposed.push(resource));
  scene.display.destroy();
  assert.deepEqual(scene.root.children, []);
  assert.equal(disposed.length, resources.size);
});
