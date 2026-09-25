'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const Parts = require('../../../cramera/src/cramera/web/core/laboratory-manual-parts.js');

// %% authoritative authored fixture
const resources = path.resolve(__dirname, '../../../cramera/src/cramera/resources/aicor_medical_lab');
const metadata = JSON.parse(fs.readFileSync(path.join(resources, 'scene.json'))).laboratory.centrifuge;

/** An observed instrument with an editable manual, retaining the packaged slot topology. */
class AnnotatedInstrument {
  constructor() {
    this.parts = Parts.centrifuge(metadata);
    this.snapshot = {
      available: true, lidOpen: true, lidClosed: false, locked: false,
      running: false, rpm: 0, targetRpm: 0, remainingSeconds: 0, balanced: true,
      slots: metadata.slots.map((slot, index) => ({...slot, secured: index % 2 === 0})),
      knowledge: {
        manual: {version: '1.0', minimumSamples: 2},
        evaluation: {allowed: false, requirements: [{id: 'lidClosed', label: 'Lid fully closed', satisfied: false}]},
      },
    };
  }
  part(id) { return this.parts.find(part => part.id === id); }
  describe(id) { return this.part(id).describe(this.snapshot); }
}

test('every numbered part anchors to actual Blender mesh geometry in the shipped assets', () => {
  const scene = new AnnotatedInstrument();
  const meshNames = new Set();
  for (const asset of ['centrifuge_base', 'centrifuge_rotor', 'centrifuge_lid']) {
    const buffer = fs.readFileSync(path.join(resources, 'assets', asset + '.glb'));
    const length = buffer.readUInt32LE(12);
    const content = JSON.parse(buffer.subarray(20, 20 + length).toString('utf8'));
    for (const node of content.nodes) if (node.mesh !== undefined) meshNames.add(node.name);
  }
  assert.equal(new Set(scene.parts.map(part => part.id)).size, scene.parts.length);
  assert.deepEqual(scene.parts.map(part => part.number), scene.parts.map((part, index) => index + 1));
  assert.ok(scene.parts.length >= 6 && scene.parts.length <= 8);
  for (const part of scene.parts) {
    assert.ok(part.title && part.description);
    assert.ok(['left', 'right'].includes(part.side));
    for (const name of part.meshNames) assert.ok(meshNames.has(name), name);
  }
});

test('interior callouts require an observed fully open lid', () => {
  const scene = new AnnotatedInstrument();
  for (const id of ['rotor', 'holders']) {
    const part = scene.part(id);
    assert.equal(part.visible(scene.snapshot), true);
    assert.equal(part.visible({...scene.snapshot, lidOpen: false}), false);
    assert.equal(part.visible({...scene.snapshot, available: false}), false);
    assert.equal(part.visible(null), false);
  }
});

test('holder explanation changes with the supplied manual instead of hardcoding a revision', () => {
  const scene = new AnnotatedInstrument();
  assert.match(scene.describe('holders'), /2 secured/);
  assert.match(scene.describe('holders'), /Manual 1\.0: at least 2/);
  assert.match(scene.describe('holders'), /balanced/);
  scene.snapshot.knowledge.manual = {version: 'Customer-3', minimumSamples: 4};
  scene.snapshot.balanced = false;
  assert.match(scene.describe('holders'), /Manual Customer-3: at least 4/);
  assert.match(scene.describe('holders'), /not balanced/);
  assert.match(scene.describe('holders'), /2 secured/);
});

test('lid and display report physical locking, speed and simulation countdown', () => {
  const scene = new AnnotatedInstrument();
  Object.assign(scene.snapshot, {lidOpen: false, lidClosed: true, locked: true, running: true, rpm: 119.6, targetRpm: 120, remainingSeconds: 12.2});
  assert.match(scene.describe('lid'), /Closed.*locked/);
  assert.equal(scene.describe('display'), '120 RPM · 00:13 remaining');
  scene.snapshot.targetRpm = 0;
  assert.match(scene.describe('rotor'), /Braking/);
  scene.snapshot.remainingSeconds = 0;
  assert.equal(scene.describe('display'), '120 RPM · 00:00 remaining');
});

test('the start annotation explains backend checks without claiming the mesh is a clickable switch', () => {
  const scene = new AnnotatedInstrument();
  assert.match(scene.part('start').description, /Centrifuge panel/);
  assert.match(scene.describe('start'), /Lid fully closed/);
  scene.snapshot.knowledge.evaluation = {allowed: true, requirements: []};
  assert.equal(scene.describe('start'), 'Start checks passed.');
  scene.snapshot.knowledge.evaluation = null;
  assert.equal(scene.describe('start'), 'Start readiness unknown.');
});

test('missing or unavailable measurements stay unknown rather than becoming safe or zero', () => {
  const scene = new AnnotatedInstrument();
  scene.snapshot = {available: true, slots: [{secured: true}, {}], rpm: null, remainingSeconds: null};
  assert.equal(scene.describe('display'), 'Speed and remaining time unknown.');
  assert.match(scene.describe('holders'), /Secured count unknown/);
  assert.match(scene.describe('holders'), /balance unknown/);
  assert.match(scene.describe('holders'), /Manual loading rule unknown/);
  assert.match(scene.describe('lid'), /Lid position unknown.*lock state unknown/);
  assert.equal(scene.describe('start'), 'Start readiness unknown.');
  scene.snapshot = {...new AnnotatedInstrument().snapshot, available: false, rpm: 120, remainingSeconds: 12};
  assert.equal(scene.describe('display'), 'Speed and remaining time unknown.');
  assert.match(scene.describe('holders'), /Secured count unknown/);
  assert.match(scene.describe('holders'), /balance unknown/);
  assert.equal(scene.describe('start'), 'Start readiness unknown.');
});

// %% independently selectable laboratory equipment
test('each instrument provides its own numbered mesh-backed manual without borrowing centrifuge parts', () => {
  const names = new Set();
  for (const asset of fs.readdirSync(path.join(resources, 'assets')).filter(name => name.endsWith('.glb'))) {
    const buffer = fs.readFileSync(path.join(resources, 'assets', asset));
    const content = JSON.parse(buffer.subarray(20, 20 + buffer.readUInt32LE(12)).toString('utf8'));
    for (const node of content.nodes) if (node.mesh !== undefined) names.add(node.name);
  }
  const scene = JSON.parse(fs.readFileSync(path.join(resources, 'scene.json'))).laboratory;
  for (const [factory, configuration] of [[Parts.dispenser, scene.pipetting], [Parts.scale, scene.scale], [Parts.rack, {slots: scene.rackSlots}]]) {
    const parts = factory(configuration);
    assert.ok(parts.length >= 3);
    assert.deepEqual(parts.map(part => part.number), parts.map((part, index) => index + 1));
    assert.equal(new Set(parts.map(part => part.id)).size, parts.length);
    for (const part of parts) {
      assert.ok(part.title && part.description);
      for (const name of part.meshNames) assert.ok(names.has(name), name);
    }
  }
});

test('dispenser annotations report only observed reservoir volumes and dose progress', () => {
  const parts = Parts.dispenser({});
  const describe = (id, snapshot) => parts.find(part => part.id === id).describe(snapshot);
  const snapshot = {available: true, state: 'dispensing', deliveredMl: 0.37, requestedMl: 1, reservoirs: [{id: 'amber', volumeMl: 49.63, capacityMl: 50}, {id: 'teal', volumeMl: 50, capacityMl: 50}]};
  assert.match(describe('amber', snapshot), /49.63.*50.*ml/);
  assert.match(describe('display', snapshot), /Dispensing.*0.37.*1.00.*ml/);
  assert.match(describe('amber', null), /unknown/i);
  assert.match(describe('display', {...snapshot, available: false}), /unknown/i);
});

test('scale annotations distinguish net load, settling and deliberate tare', () => {
  const parts = Parts.scale({});
  const describe = (id, snapshot) => parts.find(part => part.id === id).describe(snapshot);
  const snapshot = {available: true, grams: 1.257, stable: false, tareGrams: 8.13, forceNewtons: 0.092};
  assert.match(describe('display', snapshot), /1.257 g.*settling/i);
  assert.match(describe('housing', snapshot), /8.130 g/);
  snapshot.stable = true;
  assert.match(describe('display', snapshot), /stable/i);
  assert.match(describe('display', null), /unknown/i);
  assert.match(describe('display', {...snapshot, available: false}), /unknown/i);
});

test('rack manual lists the authored socket identifiers without inventing occupancy', () => {
  const slots = [{id: 'A1'}, {id: 'Customer-slot'}];
  const part = Parts.rack({slots}).find(part => part.id === 'sockets');
  assert.match(part.description, /A1.*Customer-slot/);
  assert.ok(!/occupied|free|empty/i.test(part.description));
});
