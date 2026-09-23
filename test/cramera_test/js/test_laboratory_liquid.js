'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

const WEB = path.join(__dirname, '../../../cramera/src/cramera/web');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));
const Liquid = require(path.join(WEB, 'core/laboratory-liquid.js'));
const TUBE = Object.freeze({
  volumeMl: 13, capacityMl: 28, color: [.82, .32, .055], normal: [0, 0, 1],
  offset: .074, radius: .0077, bottom: .0013, roundingRadius: .0077, rim: .149,
});

// %% watertight contained surfaces
function pointKey(point) { return point.map(value => value.toFixed(9)).join(','); }

function inspectSurface(surface, parameters) {
  const positions = surface.positions;
  assert.ok(positions.length > 0);
  assert.equal(positions.length % 9, 0);
  assert.equal(surface.normals.length, positions.length);
  assert.ok([...positions, ...surface.normals].every(Number.isFinite));
  const edges = new Map();
  let volume = 0;
  for (let index = 0; index < positions.length; index += 9) {
    const first = positions.slice(index, index + 3);
    const second = positions.slice(index + 3, index + 6);
    const third = positions.slice(index + 6, index + 9);
    const cross = new THREE.Vector3().fromArray(second).cross(new THREE.Vector3().fromArray(third));
    volume += new THREE.Vector3().fromArray(first).dot(cross) / 6;
    for (const point of [first, second, third]) {
      assert.ok(Math.hypot(point[0], point[1]) <= parameters.radius + 1e-9);
      assert.ok(point[2] >= parameters.bottom - 1e-9 && point[2] <= parameters.rim + 1e-9);
      assert.ok(point.reduce((sum, value, axis) => sum + value * parameters.normal[axis], 0) <= parameters.offset + 1e-9);
    }
    for (const [start, end] of [[first, second], [second, third], [third, first]]) {
      const names = [pointKey(start), pointKey(end)].sort();
      const key = names.join('|');
      edges.set(key, (edges.get(key) || 0) + 1);
    }
  }
  assert.ok(volume > 0, 'triangles face outward around positive volume');
  assert.ok([...edges.values()].every(count => count === 2), 'every geometric edge belongs to exactly two triangles');
  return volume;
}

for (const [name, normal, offset] of [
  ['upright', [0, 0, 1], .074],
  ['horizontal', [1, 0, 0], .001],
  ['horizontal through the axis', [1, 0, 0], 0],
  ['hemisphere boundary', [0, 0, 1], TUBE.bottom + TUBE.roundingRadius],
  ['inverted', [0, 0, -1], -.06],
  ['oblique', [.6, 0, .8], .043],
  ['nearly horizontal', [.99999999995, 0, .00001], .001],
  ['fully filled', [0, 0, 1], .2],
]) {
  test(name + ' liquid is finite, contained and watertight', () => {
    const parameters = {...TUBE, normal, offset};
    inspectSurface(Liquid.buildSurface(parameters), parameters);
  });
}

test('upright geometry follows the hemisphere plus cylinder volume', () => {
  const volume = inspectSurface(Liquid.buildSurface(TUBE), TUBE);
  const centre = TUBE.bottom + TUBE.roundingRadius;
  const expected = 2 * Math.PI * TUBE.radius ** 2 * TUBE.roundingRadius / 3
    + Math.PI * TUBE.radius ** 2 * (TUBE.offset - centre);
  assert.ok(Math.abs(volume - expected) / expected < .004);
});

test('empty and entirely clipped liquid has no visible triangles', () => {
  assert.equal(Liquid.buildSurface({...TUBE, volumeMl: 0}).positions.length, 0);
  assert.equal(Liquid.buildSurface({...TUBE, offset: -1}).positions.length, 0);
});

// %% renderer lifecycle using the vendored Three.js
class LiquidScene {
  constructor() {
    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2;
    this.objects = {};
    this.renderer = Liquid.create(THREE, this.root, key => this.objects[key]);
  }

  addTube() {
    const object = this.objects.tube = new THREE.Group();
    const glass = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshPhysicalMaterial({transmission: 1}));
    glass.material.name = 'Borosilicate glass';
    const fill = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshPhysicalMaterial({transmission: .92}));
    fill.material.name = 'Amber solution';
    object.add(glass, fill);
    this.root.add(object);
    return {object, glass, fill};
  }

  update(overrides = {}) {
    this.renderer.update({tubes: {tube: TUBE}, droplets: [], puddles: [], ...overrides});
  }
}

test('late objects replace only authored contents and restore them when disposed', () => {
  const scene = new LiquidScene();
  scene.update();
  const {object, glass, fill} = scene.addTube();
  const originalGlass = glass.material.toJSON();
  scene.update();
  assert.equal(glass.visible, true);
  assert.deepEqual(glass.material.toJSON(), originalGlass);
  assert.equal(fill.visible, false);
  assert.equal(object.children.length, 3);
  const contents = object.children[2];
  assert.ok(contents.isMesh);
  assert.ok(contents.geometry.drawRange.count > 0);
  assert.equal(contents.material.transparent, true);
  assert.equal(contents.material.depthWrite, false);
  assert.equal(contents.castShadow, false);
  scene.renderer.destroy();
  assert.equal(fill.visible, true);
  assert.equal(object.children.length, 2);
});

test('updates reuse tube geometry and pooled droplets without changing rigid body poses', () => {
  const scene = new LiquidScene();
  const {object} = scene.addTube();
  object.position.set(.2, -.1, 1);
  object.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), 1.1);
  const pose = object.position.toArray().concat(object.quaternion.toArray());
  const droplets = [{position: [.3, -.12, .92], radius: .001, color: TUBE.color}];
  scene.update({droplets});
  const contents = object.children[2], geometry = contents.geometry, material = contents.material;
  const meshes = [];
  scene.root.traverse(mesh => { if (mesh.isMesh) meshes.push(mesh); });
  const drop = meshes.find(mesh => mesh !== contents && mesh.position.x === droplets[0].position[0]);
  assert.ok(drop);
  assert.deepEqual(drop.position.toArray(), droplets[0].position);
  for (let index = 0; index < 15; index += 1) scene.update({droplets});
  assert.equal(contents.geometry, geometry);
  assert.equal(contents.material, material);
  assert.deepEqual(object.position.toArray().concat(object.quaternion.toArray()), pose);
  scene.update();
  assert.equal(drop.visible, false);
  assert.equal(contents.geometry, geometry);
  scene.renderer.destroy();
});

test('puddles lie in the laboratory XY plane and retain their volume-derived size', () => {
  const scene = new LiquidScene();
  const puddle = {position: [.3, -.12, .901], volumeMl: 2, color: TUBE.color};
  scene.update({tubes: {}, puddles: [puddle]});
  const meshes = [];
  scene.root.traverse(mesh => { if (mesh.isMesh && mesh.visible) meshes.push(mesh); });
  assert.equal(meshes.length, 1);
  const mesh = meshes[0];
  assert.deepEqual(mesh.position.toArray(), puddle.position);
  assert.ok(mesh.scale.x > mesh.scale.z * 5);
  assert.equal(mesh.scale.x, mesh.scale.y);
  scene.renderer.destroy();
});

// %% continuous discharge reconstructed from conserved packets
function pouringPackets(source = 'tube_amber') {
  return Array.from({length: 9}, (_, index) => ({
    source, position: [.31 + index * .0004, -.1, 1.04 + index * .002],
    radius: .0011, color: TUBE.color,
  }));
}

function visibleLiquidMeshes(scene) {
  return scene.root.children.filter(object => object.isMesh && object.visible);
}

function streamVolume(mesh) {
  const positions = mesh.geometry.getAttribute('position');
  let volume = 0;
  const edges = new Map();
  for (let index = 0; index < mesh.geometry.drawRange.count; index += 3) {
    const triangle = [0, 1, 2].map(offset => new THREE.Vector3().fromBufferAttribute(positions, index + offset));
    volume += triangle[0].dot(triangle[1].clone().cross(triangle[2])) / 6;
    for (const [first, second] of [[0, 1], [1, 2], [2, 0]]) {
      const key = [pointKey(triangle[first].toArray()), pointKey(triangle[second].toArray())].sort().join('|');
      edges.set(key, (edges.get(key) || 0) + 1);
    }
  }
  assert.ok([...edges.values()].every(count => count === 2), 'the stream is a closed connected surface');
  assert.ok(volume > 0, 'the stream faces outward');
  return volume;
}

test('continuous pouring draws one smooth volume-preserving surface instead of individual packet beads', () => {
  const scene = new LiquidScene();
  const droplets = pouringPackets();
  const original = JSON.stringify(droplets);
  scene.update({tubes: {}, droplets});
  const meshes = visibleLiquidMeshes(scene);
  assert.equal(meshes.length, 1, 'one coherent stream replaces all adjacent packet spheres');
  const mesh = meshes[0];
  const expectedVolume = droplets.reduce((sum, packet) => sum + 4 * Math.PI * packet.radius ** 3 / 3, 0);
  assert.ok(Math.abs(streamVolume(mesh) - expectedVolume) / expectedVolume < .0005);
  const normals = mesh.geometry.getAttribute('normal');
  for (let index = 0; index < mesh.geometry.drawRange.count; index += 1) {
    assert.ok(Math.abs(new THREE.Vector3().fromBufferAttribute(normals, index).length() - 1) < 1e-5);
  }
  assert.equal(JSON.stringify(droplets), original, 'rendering cannot alter the simulation ledger');
  scene.renderer.destroy();
});

test('separate sources and detached drops never gain a fictitious connecting stream', () => {
  const scene = new LiquidScene();
  const first = pouringPackets();
  const second = pouringPackets('tube_teal').map(packet => ({...packet, position: packet.position.map((value, axis) => value + (axis === 0 ? .04 : 0))}));
  const detached = {...first[0], position: [.31, -.1, .97]};
  scene.update({tubes: {}, droplets: [...first, ...second, detached]});
  const meshes = visibleLiquidMeshes(scene);
  assert.equal(meshes.length, 3, 'two streams and a separated physical drop stay distinct');
  const drop = meshes.find(mesh => mesh.position.equals(new THREE.Vector3().fromArray(detached.position)));
  assert.ok(drop);
  assert.equal(drop.scale.x, detached.radius);
  scene.renderer.destroy();
});

test('stream geometry is reused across snapshots and disappears as soon as all packets have landed', () => {
  const scene = new LiquidScene();
  const droplets = pouringPackets();
  scene.update({tubes: {}, droplets});
  const [mesh] = visibleLiquidMeshes(scene), geometry = mesh.geometry;
  scene.update({tubes: {}, droplets: droplets.map(packet => ({...packet, position: packet.position.map((value, axis) => value - (axis === 2 ? .003 : 0))}))});
  assert.equal(visibleLiquidMeshes(scene)[0], mesh);
  assert.equal(mesh.geometry, geometry);
  scene.update({tubes: {}, droplets: []});
  assert.equal(visibleLiquidMeshes(scene).length, 0);
  let disposed = false;
  geometry.addEventListener('dispose', () => { disposed = true; });
  scene.renderer.destroy();
  assert.equal(disposed, true);
});

test('water-like optics remain transparent while preserving the simulated tint', () => {
  const scene = new LiquidScene();
  scene.update({tubes: {}, droplets: pouringPackets()});
  const [mesh] = visibleLiquidMeshes(scene);
  assert.equal(mesh.material.transparent, true);
  assert.equal(mesh.material.depthWrite, false);
  assert.ok(mesh.material.transmission >= .7 && mesh.material.transmission < 1);
  assert.ok(mesh.material.reflectivity >= .3 && mesh.material.reflectivity < .4);
  assert.deepEqual(mesh.material.color.toArray(), TUBE.color);
  scene.renderer.destroy();
});

test('an interrupted emission remains two separate streams even when the flight paths are close', () => {
  const scene = new LiquidScene();
  const droplets = pouringPackets().map((packet, index) => ({
    ...packet, emittedAt: index * .002 + (index >= 4 ? .08 : 0), velocity: [0, 0, -.2],
  }));
  scene.update({tubes: {}, droplets, time: .1});
  assert.equal(visibleLiquidMeshes(scene).length, 2);
  const expectedVolume = droplets.reduce((sum, packet) => sum + 4 * Math.PI * packet.radius ** 3 / 3, 0);
  const volume = visibleLiquidMeshes(scene).reduce((sum, mesh) => sum + streamVolume(mesh), 0);
  assert.ok(Math.abs(volume - expectedVolume) / expectedVolume < .0005);
  scene.renderer.destroy();
});

test('a live stream reaches its measured source lip and disconnects when emission has ended', () => {
  const scene = new LiquidScene();
  const {object} = scene.addTube();
  scene.objects.tube_amber = object;
  object.position.set(.3, 0, .9);
  const parameters = {...TUBE, normal: [1, 0, 0], offset: .004};
  const droplets = pouringPackets().map((packet, index) => ({
    ...packet, emittedAt: index * .002, velocity: [0, 0, -.3],
    position: [.3 - TUBE.radius, 0, 1.029 + index * .002],
  }));
  const snapshot = {tubes: {tube_amber: parameters}, droplets, time: .018};
  scene.update(snapshot);
  const [mesh] = visibleLiquidMeshes(scene);
  const highestPoint = () => {
    const positions = mesh.geometry.getAttribute('position');
    return Math.max(...Array.from({length: mesh.geometry.drawRange.count}, (_, index) => positions.getZ(index) + mesh.position.z));
  };
  assert.ok(Math.abs(highestPoint() - (object.position.z + TUBE.rim)) < 1e-8);
  const expectedVolume = droplets.reduce((sum, packet) => sum + 4 * Math.PI * packet.radius ** 3 / 3, 0);
  assert.ok(Math.abs(streamVolume(mesh) - expectedVolume) / expectedVolume < .0005);
  scene.update({...snapshot, time: .2});
  assert.ok(Math.abs(highestPoint() - droplets[droplets.length - 1].position[2]) < 1e-8);
  scene.renderer.destroy();
});

test('a curved thinning stream preserves packet volume and its volume-weighted liquid tint', () => {
  const scene = new LiquidScene();
  const droplets = Array.from({length: 32}, (_, index) => {
    const age = (32 - index) * .003;
    return {
      source: 'tube_amber', emittedAt: index * .003,
      position: [.3 + age * .08, 0, 1.06 - age * .1 - 9.81 * age ** 2 / 2],
      velocity: [.08, 0, -.1 - 9.81 * age], radius: .0005 + index * .00002,
      color: [index / 32, .3, .1],
    };
  });
  scene.update({tubes: {}, droplets, time: .1});
  const meshes = visibleLiquidMeshes(scene);
  assert.equal(meshes.length, 1);
  const volume = droplets.reduce((sum, packet) => sum + 4 * Math.PI * packet.radius ** 3 / 3, 0);
  assert.ok(Math.abs(streamVolume(meshes[0]) - volume) / volume < .0005);
  const expectedColor = droplets.reduce((sum, packet) => sum + packet.color[0] * 4 * Math.PI * packet.radius ** 3 / 3, 0) / volume;
  assert.ok(Math.abs(meshes[0].material.color.r - expectedColor) < 1e-12);
  const normals = meshes[0].geometry.getAttribute('normal');
  for (let index = 0; index < meshes[0].geometry.drawRange.count; index += 1) {
    assert.ok(Math.abs(new THREE.Vector3().fromBufferAttribute(normals, index).length() - 1) < 1e-5);
  }
  scene.renderer.destroy();
});

// %% actual viewer adapter
test('the panel forwards fluid state and disposes its liquid renderer', () => {
  const source = fs.readFileSync(path.join(WEB, 'panels/robot_scene/panel.js'), 'utf8');
  const start = source.indexOf('  function laboratoryPhysicsAdapter()');
  const end = source.indexOf('  RobotView.onReady', start);
  const calls = [];
  const object = new THREE.Group();
  const worldRoot = new THREE.Group();
  const context = vm.createContext({
    THREE, worldRoot, objectMeshes: {tube: object}, needsRender: false,
    models: [], JointRouting: {}, syncJointControls() {}, focusSceneCamera() {},
    LaboratoryLiquid: {create(three, root, getObject) {
      assert.equal(three, THREE);
      assert.equal(root, worldRoot);
      assert.equal(getObject('tube'), object);
      return {update(state) { calls.push(state); }, destroy() { calls.push('destroy'); }};
    }},
  });
  vm.runInContext(source.slice(start, end), context);
  const adapter = context.laboratoryPhysicsAdapter();
  const snapshot = {tubes: {tube: TUBE}};
  adapter.setLiquidState(snapshot);
  adapter.destroy();
  assert.deepEqual(calls, [snapshot, 'destroy']);
  assert.equal(context.needsRender, true);
  const html = fs.readFileSync(path.join(WEB, 'index.html'), 'utf8');
  const script = html.indexOf('src="core/laboratory-liquid.js"');
  assert.ok(script >= 0 && script < html.indexOf('src="panels/robot_scene/panel.js"'));
});
