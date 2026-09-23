'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const THREE = require('../../../cramera/src/cramera/web/vendor/three.min.js');
const Pipetting = require('../../../cramera/src/cramera/web/core/laboratory-pipetting.js');

// %% measured instrument rendering
class DosingDisplay {
  constructor() {
    this.root = new THREE.Group();
    this.words = [];
    this.document = {createElement: () => ({getContext: () => ({fillRect() {}, fillText: text => this.words.push(text)})})};
    this.metadata = {displayPosition: [-.15,.0387,1.056], displaySize: [.078,.028], nozzlePosition: [-.1,-.11,1.08], channels: []};
    this.view = Pipetting.create(THREE, this.root, this.metadata, this.document);
    this.snapshot = {available: true, state: 'dispensing', key: 'tube', source: 'amber', deliveredMl: .4, requestedMl: 1,
      nozzlePosition: this.metadata.nozzlePosition, reservoirs: [{id: 'amber', volumeMl: 49.6, color: [.8,.4,.1]}]};
    this.liquid = {tubes: {tube: {normal: [0,0,1], offset: .04, volumeMl: .4}}};
    this.objects = {tube: [-.1,-.11,.916,0,0,0,1]};
  }
  update() { this.view.update(this.snapshot, this.liquid, this.objects); }
}

test('display and continuous stream reflect measured dosing without a local clock', () => {
  const scene = new DosingDisplay();
  scene.update();
  assert.ok(scene.words.includes('DISPENSING'));
  assert.ok(scene.words.includes('0.40 / 1.00 ml'));
  assert.equal(scene.view.stream.visible, true);
  assert.ok(Math.abs(scene.view.stream.scale.y - (1.08 - .956)) < 1e-10);
  const count = scene.words.length;
  scene.update();
  assert.equal(scene.words.length, count);
  scene.snapshot.state = 'complete';
  scene.update();
  assert.equal(scene.view.stream.visible, false);
  scene.view.destroy();
  assert.equal(scene.root.children.length, 0);
});

test('missing observations hide dosing instead of leaving a stale stream', () => {
  const scene = new DosingDisplay();
  scene.update();
  scene.view.update(null);
  assert.equal(scene.view.stream.visible, false);
  assert.ok(scene.words.includes('NO DATA'));
  scene.view.destroy();
});

test('reservoir liquid shrinks from its fixed bottom as authoritative reagent is consumed', () => {
  const scene = new DosingDisplay();
  const reservoir = new THREE.Mesh(new THREE.CylinderGeometry(.02,.02,.1), new THREE.MeshBasicMaterial());
  reservoir.geometry.rotateX(Math.PI / 2);
  reservoir.position.set(.2,.2,1.2);
  reservoir.name = 'Pipetting_amber_liquid';
  scene.root.add(reservoir);
  scene.metadata.channels.push({id:'amber', liquidMesh:'Pipetting amber liquid'});
  scene.snapshot.reservoirs[0].volumeMl = 25;
  scene.snapshot.reservoirs[0].capacityMl = 50;
  scene.update();
  assert.equal(reservoir.visible, false);
  const live = scene.view.reservoirs.get('amber').mesh;
  live.geometry.computeBoundingBox();
  const bounds = live.geometry.boundingBox.clone().applyMatrix4(live.matrix);
  assert.ok(Math.abs(bounds.min.z - 1.15) < 1e-7);
  assert.ok(Math.abs(bounds.max.z - 1.20) < 1e-7);
  scene.view.destroy();
  assert.equal(reservoir.visible, true);
});
