'use strict';

// %% laboratory fixture
const assert = require('node:assert/strict');
const test = require('node:test');
const LaboratoryWorkbench = require('../../../cramera/src/cramera/web/core/laboratory-workbench.js');

class ManualLaboratory {
  constructor() {
    this.config = {
      rackSlots: [
        {id: 'A1', pose: [.16, -.065, .910, 0, 0, 0, 1]},
        {id: 'A2', pose: [.22, -.065, .910, 0, 0, 0, 1]},
        {id: 'A3', pose: [.28, -.065, .910, 0, 0, 0, 1]},
        {id: 'B1', pose: [.16, -.015, .910, 0, 0, 0, 1]},
        {id: 'B2', pose: [.22, -.015, .910, 0, 0, 0, 1]},
        {id: 'B3', pose: [.28, -.015, .910, 0, 0, 0, 1]},
      ],
      tubes: [
        {key: 'tube_clear', slot: 'A1'},
        {key: 'tube_amber', slot: 'A2'},
        {key: 'tube_teal', slot: 'B1'},
      ],
      stopper: {key: 'stopper', home: [.50, -.12, .902, 0, 0, 0, 1], tubeHeight: .15, insertionDepth: .009},
      drawer: {joint: 'laboratory_drawer_joint', open: .36},
      cameras: {
        overview: {position: [2, -3, 2], target: [0, 0, 1]},
        closeup: {position: [1, -1, 1.5], target: [.2, 0, 1]},
      },
    };
    this.poses = new Map(this.config.tubes.map(tube => [tube.key, this.config.rackSlots.find(slot => slot.id === tube.slot).pose.slice()]));
    this.poses.set(this.config.stopper.key, this.config.stopper.home.slice());
    this.joints = new Map([[this.config.drawer.joint, 0]]);
    this.ready = true;
    this.pauses = 0;
    this.focused = null;
    this.writes = [];
    this.adapter = {
      isReady: () => this.ready,
      getPose: key => this.poses.has(key) ? this.poses.get(key).slice() : null,
      setPose: (key, pose) => {
        if (!this.ready || !this.poses.has(key)) return false;
        this.poses.set(key, pose.slice());
        this.writes.push(key);
        return true;
      },
      setJoint: (name, value) => {
        if (!this.ready) return false;
        this.joints.set(name, value);
        return true;
      },
      pause: () => { this.pauses += 1; },
      focus: (position, target) => { this.focused = {position, target}; },
    };
    this.controller = new LaboratoryWorkbench.Controller(this.config, this.adapter);
  }
}

// %% occupancy and independent poses
test('lifting one tube frees its slot without moving its neighbours', () => {
  const laboratory = new ManualLaboratory();
  const tube = laboratory.config.tubes[0];
  const expected = laboratory.poses.get(tube.key).slice();
  expected[2] = LaboratoryWorkbench.LIFT_HEIGHT;
  assert.equal(laboratory.controller.lift(tube.key).ok, true);
  assert.deepEqual(laboratory.poses.get(tube.key), expected);
  assert.deepEqual(laboratory.writes, [tube.key]);
  assert.equal(laboratory.controller.heldTube, tube.key);
  assert.equal(laboratory.controller.occupant(tube.slot), null);
  assert.equal(laboratory.pauses, 1);
});

test('an occupied destination preserves both tubes and the held state', () => {
  const laboratory = new ManualLaboratory();
  const [first, second] = laboratory.config.tubes;
  laboratory.controller.lift(first.key);
  const poses = structuredClone(laboratory.poses);
  assert.equal(laboratory.controller.insert(second.slot).code, LaboratoryWorkbench.Codes.OCCUPIED);
  assert.deepEqual(laboratory.poses, poses);
  assert.equal(laboratory.controller.heldTube, first.key);
  assert.equal(laboratory.controller.occupant(second.slot), second.key);
});

test('insertion occupies a free slot and allows the next tube to be lifted', () => {
  const laboratory = new ManualLaboratory();
  const [first, second] = laboratory.config.tubes;
  const destination = laboratory.config.rackSlots[2];
  laboratory.controller.lift(first.key);
  assert.equal(laboratory.controller.lift(second.key).code, LaboratoryWorkbench.Codes.HANDS_FULL);
  assert.equal(laboratory.controller.insert(destination.id).ok, true);
  assert.deepEqual(laboratory.poses.get(first.key), destination.pose);
  assert.equal(laboratory.controller.occupant(destination.id), first.key);
  assert.equal(laboratory.controller.heldTube, null);
  assert.equal(laboratory.controller.lift(second.key).ok, true);
});

test('an unloaded scene rejects commands without changing occupancy', () => {
  const laboratory = new ManualLaboratory();
  const tube = laboratory.config.tubes[0];
  laboratory.ready = false;
  assert.equal(laboratory.controller.lift(tube.key).code, LaboratoryWorkbench.Codes.NOT_READY);
  assert.equal(laboratory.controller.occupant(tube.slot), tube.key);
  assert.deepEqual(laboratory.writes, []);
  assert.equal(laboratory.pauses, 0);
});

test('a missing scene object cannot vacate its recorded slot', () => {
  const laboratory = new ManualLaboratory();
  const tube = laboratory.config.tubes[0];
  laboratory.poses.delete(tube.key);
  assert.equal(laboratory.controller.lift(tube.key).code, LaboratoryWorkbench.Codes.NOT_READY);
  assert.equal(laboratory.controller.occupant(tube.slot), tube.key);
});

// %% stopper attachment and reset
test('the attached stopper follows a tube from the rack through lifting and insertion', () => {
  const laboratory = new ManualLaboratory();
  const tube = laboratory.config.tubes[0];
  const destination = laboratory.config.rackSlots[4];
  const stopper = laboratory.config.stopper;
  assert.equal(laboratory.controller.attachStopper(tube.key).ok, true);
  for (const move of [() => laboratory.controller.lift(tube.key), () => laboratory.controller.insert(destination.id)]) {
    assert.equal(move().ok, true);
    const expected = laboratory.poses.get(tube.key).slice();
    expected[2] += stopper.tubeHeight - stopper.insertionDepth;
    assert.deepEqual(laboratory.poses.get(stopper.key), expected);
    assert.equal(laboratory.controller.cappedTube, tube.key);
  }
  assert.equal(laboratory.controller.removeStopper().ok, true);
  assert.deepEqual(laboratory.poses.get(stopper.key), stopper.home);
  assert.equal(laboratory.controller.cappedTube, null);
});

test('a stopper cannot be transferred directly from another tube', () => {
  const laboratory = new ManualLaboratory();
  const [first, second] = laboratory.config.tubes;
  laboratory.controller.attachStopper(first.key);
  const pose = laboratory.poses.get(laboratory.config.stopper.key).slice();
  assert.equal(laboratory.controller.attachStopper(second.key).code, LaboratoryWorkbench.Codes.STOPPER_IN_USE);
  assert.deepEqual(laboratory.poses.get(laboratory.config.stopper.key), pose);
});

test('a rejected stopper write rolls back its attached tube move', () => {
  const laboratory = new ManualLaboratory();
  const tube = laboratory.config.tubes[0];
  laboratory.controller.attachStopper(tube.key);
  const poses = structuredClone(laboratory.poses);
  const setPose = laboratory.adapter.setPose;
  laboratory.adapter.setPose = (key, pose) => key === laboratory.config.stopper.key ? false : setPose(key, pose);
  assert.equal(laboratory.controller.lift(tube.key).code, LaboratoryWorkbench.Codes.NOT_READY);
  assert.deepEqual(laboratory.poses, poses);
  assert.equal(laboratory.controller.occupant(tube.slot), tube.key);
});

test('reset restores every tube, the stopper, and the closed drawer', () => {
  const laboratory = new ManualLaboratory();
  const original = structuredClone(laboratory.poses);
  laboratory.controller.setDrawer(true);
  laboratory.controller.attachStopper(laboratory.config.tubes[0].key);
  laboratory.controller.lift(laboratory.config.tubes[0].key);
  laboratory.controller.insert(laboratory.config.rackSlots[4].id);
  laboratory.controller.lift(laboratory.config.tubes[1].key);
  assert.equal(laboratory.controller.reset().ok, true);
  assert.deepEqual(laboratory.poses, original);
  assert.equal(laboratory.joints.get(laboratory.config.drawer.joint), 0);
  assert.equal(laboratory.controller.drawerOpen, false);
  assert.equal(laboratory.controller.heldTube, null);
  assert.equal(laboratory.controller.cappedTube, null);
  for (const tube of laboratory.config.tubes) assert.equal(laboratory.controller.occupant(tube.slot), tube.key);
});

test('drawer and camera commands forward configured world coordinates', () => {
  const laboratory = new ManualLaboratory();
  assert.equal(laboratory.controller.setDrawer(true).ok, true);
  assert.equal(laboratory.joints.get(laboratory.config.drawer.joint), laboratory.config.drawer.open);
  assert.equal(laboratory.controller.focus('closeup').ok, true);
  assert.deepEqual(laboratory.focused, laboratory.config.cameras.closeup);
});

test('scenes without laboratory metadata mount no controls', () => {
  assert.equal(LaboratoryWorkbench.mount(null, null, null), null);
});

// %% mounted controls and asset readiness
class ManualControl {
  constructor(tag, ownerDocument) {
    this.tagName = tag;
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.listeners = new Map();
    this.dataset = {};
    this.attributes = new Map();
    this.selectedValue = null;
    this.parentNode = null;
    this.textContent = '';
  }

  setAttribute(name, value) { this.attributes.set(name, value); }
  appendChild(child) { this.children.push(child); child.parentNode = this; }
  addEventListener(event, callback) { this.listeners.set(event, callback); }
  removeEventListener(event, callback) {
    if (this.listeners.get(event) === callback) this.listeners.delete(event);
  }
  remove() { this.parentNode.children = this.parentNode.children.filter(child => child !== this); }
  get value() {
    if (this.selectedValue !== null) return this.selectedValue;
    return this.tagName === 'select' && this.children.length ? this.children[0].value : '';
  }
  set value(value) { this.selectedValue = value; }
  flatten() { return [this, ...this.children.flatMap(child => child.flatten())]; }
}

class ManualControls {
  constructor() {
    this.host = new ManualControl('div', this);
  }
  createElement(tag) { return new ManualControl(tag, this); }
  button(text) { return this.host.flatten().find(item => item.tagName === 'button' && item.textContent === text); }
  status() { return this.host.flatten().find(item => item.attributes.get('role') === 'status'); }
}

test('asset readiness enables mounted controls and clears the loading message', () => {
  const laboratory = new ManualLaboratory();
  const surface = new ManualControls();
  laboratory.ready = false;
  const mounted = LaboratoryWorkbench.mount(surface.host, laboratory.config, laboratory.adapter);
  assert.equal(surface.button('Glas aufnehmen').disabled, true);
  const loading = surface.status().textContent;
  laboratory.ready = true;
  mounted.refresh();
  assert.equal(surface.button('Glas aufnehmen').disabled, false);
  assert.notEqual(surface.status().textContent, loading);
});

test('mounted actions update free slots and destroy removes controls and listeners', () => {
  const laboratory = new ManualLaboratory();
  const surface = new ManualControls();
  const mounted = LaboratoryWorkbench.mount(surface.host, laboratory.config, laboratory.adapter);
  const controls = surface.host.flatten();
  surface.button('Glas aufnehmen').listeners.get('click')();
  assert.equal(mounted.controller.heldTube, laboratory.config.tubes[0].key);
  assert.equal(surface.button('In Platz einsetzen').disabled, false);
  surface.button('In Platz einsetzen').listeners.get('click')();
  assert.equal(mounted.controller.heldTube, null);
  assert.equal(mounted.controller.occupant(laboratory.config.rackSlots[2].id), laboratory.config.tubes[0].key);
  mounted.destroy();
  assert.deepEqual(surface.host.children, []);
  assert.equal(controls.reduce((count, control) => count + control.listeners.size, 0), 0);
});

// %% independent native robot controls
test('robot controls mount independently and never receive the manual pose adapter', () => {
  const laboratory = new ManualLaboratory();
  const surface = new ManualControls();
  let parent = null;
  let destroyed = false;
  const mounted = LaboratoryWorkbench.mount(surface.host, laboratory.config, laboratory.adapter, host => {
    parent = host;
    return {destroy() { destroyed = true; }};
  });
  assert.equal(parent.className, 'laboratory-body');
  assert.deepEqual(laboratory.writes, []);
  mounted.destroy();
  assert.equal(destroyed, true);
});
