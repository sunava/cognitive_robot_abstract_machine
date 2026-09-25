'use strict';

// %% controllable browser animation and scene adapter
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const Workshop = require('../../../cramera/src/cramera/web/core/workshop.js');

class AnimationClock {
  constructor() { this.callbacks = new Map(); this.time = 0; this.counter = 0; }
  now() { return this.time; }
  request(callback) { this.counter += 1; this.callbacks.set(this.counter, callback); return this.counter; }
  cancel(identifier) { this.callbacks.delete(identifier); }
  async advance(milliseconds = 100) {
    this.time += milliseconds;
    const pending = [...this.callbacks.values()];
    this.callbacks.clear();
    pending.forEach(callback => callback(this.time));
    await Promise.resolve();
  }
  async finish(promise) {
    let finished = false;
    promise.then(() => { finished = true; });
    for (let index = 0; index < 80 && !finished; index += 1) await this.advance();
    assert.equal(finished, true, 'A bounded command must settle.');
    return promise;
  }
}

class MovableWorkshop {
  constructor() {
    this.config = {
      drawers: [1, 2, 3].map(number => ({id: 'drawer_' + number, label: 'Drawer ' + number, joint: 'workshop_drawer_' + number + '_joint', open: .4})),
      parts: [
        {key: 'screw_1', label: 'Screw 1', kind: 'screw', home: [-.5, -.1, .96, 0, Math.SQRT1_2, 0, Math.SQRT1_2]},
        {key: 'screw_2', label: 'Screw 2', kind: 'screw', home: [-.5, 0, .96, 0, Math.SQRT1_2, 0, Math.SQRT1_2]},
        {key: 'nail_1', label: 'Nail 1', kind: 'nail', home: [-.2, 0, .96, 0, Math.SQRT1_2, 0, Math.SQRT1_2]},
      ],
      bins: [],
      liftHeight: 1.3,
      bounds: {min: [-.72, -.35], max: [.72, .32]},
      assembly: {socketPose: [.28, -.1, .962, 0, 0, 0, 1]},
      cameras: {overview: {position: [2, -2, 2], target: [0, 0, .9]}, bench: {position: [1, -1, 1.8], target: [0, 0, 1]}, drawers: {position: [-1, -2, 1], target: [-.7, 0, .5]}},
    };
    this.ready = true;
    this.poses = new Map(this.config.parts.map(part => [part.key, part.home.slice()]));
    this.joints = new Map(this.config.drawers.map(drawer => [drawer.joint, 0]));
    this.writes = [];
    this.jointWrites = [];
    this.clock = new AnimationClock();
    this.adapter = {
      isReady: () => this.ready,
      getPose: key => this.poses.has(key) ? this.poses.get(key).slice() : null,
      setPose: (key, pose) => { this.poses.set(key, pose.slice()); this.writes.push([key, pose.slice()]); return true; },
      getJoint: name => this.joints.get(name),
      setJoint: (name, value) => { this.joints.set(name, value); this.jointWrites.push([name, value]); return true; },
      pause() {},
      focus: (position, target) => { this.focused = {position, target}; },
    };
    this.controller = new Workshop.Controller(this.config, this.adapter, this.clock);
  }
  command(promise) { return this.clock.finish(promise); }
}

// %% individual parts and safe transit
test('lifting one part preserves its neighbours and interpolates vertical motion', async () => {
  const workshop = new MovableWorkshop();
  const part = workshop.config.parts[0];
  const neighbour = workshop.poses.get('screw_2').slice();
  const result = await workshop.command(workshop.controller.lift(part.key));
  assert.equal(result.ok, true);
  assert.equal(workshop.controller.heldPart, part.key);
  assert.deepEqual(workshop.poses.get(part.key), [part.home[0], part.home[1], workshop.config.liftHeight, ...part.home.slice(3)]);
  assert.deepEqual(workshop.poses.get('screw_2'), neighbour);
  assert.equal(workshop.writes.every(([key, pose]) => key === part.key && pose[0] === part.home[0] && pose[1] === part.home[1]), true);
  assert.equal(workshop.writes.some(([, pose]) => pose[2] > part.home[2] && pose[2] < workshop.config.liftHeight), true);
});

test('commands are rejected while a drawer or part is moving', async () => {
  const workshop = new MovableWorkshop();
  const opening = workshop.controller.setDrawer('drawer_1', true);
  assert.equal((await workshop.controller.lift('screw_1')).code, Workshop.Codes.BUSY);
  assert.equal((await workshop.command(opening)).ok, true);
  assert.equal(workshop.controller.heldPart, null);
});

test('a second part cannot be lifted while the first is held', async () => {
  const workshop = new MovableWorkshop();
  await workshop.command(workshop.controller.lift('screw_1'));
  assert.equal((await workshop.controller.lift('nail_1')).code, Workshop.Codes.HANDS_FULL);
  assert.deepEqual(workshop.poses.get('nail_1'), workshop.config.parts[2].home);
});

test('horizontal moves stay within the worktop boundary and above all fixtures', async () => {
  const workshop = new MovableWorkshop();
  await workshop.command(workshop.controller.lift('screw_1'));
  const before = workshop.poses.get('screw_1').slice();
  assert.equal((await workshop.controller.move(20, 0)).code, Workshop.Codes.OUTSIDE);
  assert.deepEqual(workshop.poses.get('screw_1'), before);
  workshop.writes = [];
  assert.equal((await workshop.command(workshop.controller.move(.1, 0))).ok, true);
  assert.equal(workshop.writes.every(([, pose]) => pose[2] === workshop.config.liftHeight), true);
  assert.equal(workshop.poses.get('screw_1')[0], before[0] + .1);
});

test('socket insertion aligns above the socket before descending and reserves it', async () => {
  const workshop = new MovableWorkshop();
  await workshop.command(workshop.controller.lift('screw_1'));
  workshop.writes = [];
  assert.equal((await workshop.command(workshop.controller.insert())).ok, true);
  const destination = workshop.config.assembly.socketPose;
  assert.deepEqual(workshop.poses.get('screw_1'), destination);
  assert.equal(workshop.controller.socketOccupant, 'screw_1');
  assert.equal(workshop.controller.heldPart, null);
  for (const [, pose] of workshop.writes) {
    if (pose[2] >= workshop.config.liftHeight) continue;
    assert.deepEqual([pose[0], pose[1], ...pose.slice(3)], [destination[0], destination[1], ...destination.slice(3)]);
  }
  await workshop.command(workshop.controller.lift('screw_2'));
  assert.equal((await workshop.controller.insert()).code, Workshop.Codes.OCCUPIED);
  assert.equal(workshop.controller.heldPart, 'screw_2');
});

test('nails cannot be inserted into the screw pilot hole', async () => {
  const workshop = new MovableWorkshop();
  await workshop.command(workshop.controller.lift('nail_1'));
  assert.equal((await workshop.controller.insert()).code, Workshop.Codes.WRONG_PART);
  assert.equal(workshop.controller.socketOccupant, null);
});

test('returning an inserted screw restores its exact position in its box', async () => {
  const workshop = new MovableWorkshop();
  await workshop.command(workshop.controller.lift('screw_1'));
  await workshop.command(workshop.controller.insert());
  await workshop.command(workshop.controller.lift('screw_1'));
  assert.equal(workshop.controller.socketOccupant, null);
  assert.equal((await workshop.command(workshop.controller.returnPart())).ok, true);
  assert.deepEqual(workshop.poses.get('screw_1'), workshop.config.parts[0].home);
  assert.equal(workshop.controller.heldPart, null);
});

test('a rejected animation restores the object pose without assigning a held part', async () => {
  const workshop = new MovableWorkshop();
  const write = workshop.adapter.setPose;
  let calls = 0;
  workshop.adapter.setPose = (key, pose) => { calls += 1; return calls === 3 ? false : write(key, pose); };
  assert.equal((await workshop.command(workshop.controller.lift('screw_1'))).code, Workshop.Codes.NOT_READY);
  assert.deepEqual(workshop.poses.get('screw_1'), workshop.config.parts[0].home);
  assert.equal(workshop.controller.heldPart, null);
});

test('destroy cancels pending animation without leaving a scheduled write', async () => {
  const workshop = new MovableWorkshop();
  const lifting = workshop.controller.lift('screw_1');
  await workshop.clock.advance();
  workshop.controller.destroy();
  assert.equal((await workshop.command(lifting)).code, Workshop.Codes.CANCELLED);
  assert.equal(workshop.clock.callbacks.size, 0);
});

// %% independent articulated drawers and reset
test('opening one drawer preserves the other drawers and uses intermediate joint values', async () => {
  const workshop = new MovableWorkshop();
  const drawer = workshop.config.drawers[1];
  assert.equal((await workshop.command(workshop.controller.setDrawer(drawer.id, true))).ok, true);
  assert.equal(workshop.joints.get(drawer.joint), drawer.open);
  assert.equal(workshop.joints.get(workshop.config.drawers[0].joint), 0);
  assert.equal(workshop.jointWrites.some(([, value]) => value > 0 && value < drawer.open), true);
  assert.equal((await workshop.command(workshop.controller.setDrawer(drawer.id, false))).ok, true);
  assert.equal(workshop.joints.get(drawer.joint), 0);
});

test('reset returns loose parts and every drawer to the scene initial state', async () => {
  const workshop = new MovableWorkshop();
  await workshop.command(workshop.controller.setDrawer('drawer_1', true));
  await workshop.command(workshop.controller.setDrawer('drawer_3', true));
  await workshop.command(workshop.controller.lift('screw_1'));
  await workshop.command(workshop.controller.insert());
  assert.equal((await workshop.command(workshop.controller.reset())).ok, true);
  for (const part of workshop.config.parts) assert.deepEqual(workshop.poses.get(part.key), part.home);
  for (const drawer of workshop.config.drawers) assert.equal(workshop.joints.get(drawer.joint), 0);
  assert.equal(workshop.controller.socketOccupant, null);
});

test('unloaded and live scenes do not accept direct positioning commands', async () => {
  const workshop = new MovableWorkshop();
  workshop.ready = false;
  assert.equal((await workshop.controller.lift('screw_1')).code, Workshop.Codes.NOT_READY);
  assert.equal((await workshop.controller.setDrawer('drawer_1', true)).code, Workshop.Codes.NOT_READY);
  assert.deepEqual(workshop.writes, []);
  assert.deepEqual(workshop.jointWrites, []);
});

test('camera choices forward only declared world positions', () => {
  const workshop = new MovableWorkshop();
  assert.equal(workshop.controller.focus('drawers').ok, true);
  assert.deepEqual(workshop.focused, workshop.config.cameras.drawers);
  assert.equal(workshop.controller.focus('unknown').code, Workshop.Codes.INVALID);
});

// %% mounted controls and scene lifetime
class ControlElement {
  constructor(tag, ownerDocument) {
    this.tagName = tag;
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.listeners = new Map();
    this.attributes = new Map();
    this.dataset = {};
    this.textContent = '';
    this.selectedValue = null;
  }
  setAttribute(key, value) { this.attributes.set(key, value); }
  appendChild(child) { this.children.push(child); child.parent = this; }
  addEventListener(event, handler) { this.listeners.set(event, handler); }
  removeEventListener(event) { this.listeners.delete(event); }
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  flatten() { return [this, ...this.children.flatMap(child => child.flatten())]; }
  get value() { return this.selectedValue === null && this.tagName === 'select' ? this.children[0].value : this.selectedValue; }
  set value(value) { this.selectedValue = value; }
}

class WorkshopControls {
  constructor() { this.host = new ControlElement('div', this); }
  createElement(tag) { return new ControlElement(tag, this); }
  button(text) { return this.host.flatten().find(item => item.tagName === 'button' && item.textContent === text); }
  status() { return this.host.flatten().find(item => item.attributes.get('role') === 'status'); }
}

test('mounted controls select independent parts and disable conflicting commands during motion', async () => {
  const workshop = new MovableWorkshop();
  const document = new WorkshopControls();
  const mounted = Workshop.mount(document.host, workshop.config, workshop.adapter, workshop.clock);
  assert.equal(mounted.select('nail_1'), true);
  assert.equal(mounted.select('screw_box'), false);
  const moving = document.button('Lift selected part').listeners.get('click')();
  assert.equal(document.button('Lift selected part').disabled, true);
  assert.equal(document.button('Reset workshop').disabled, true);
  await workshop.clock.finish(moving);
  assert.equal(mounted.controller.heldPart, 'nail_1');
  assert.equal(document.button('Insert in pilot hole').disabled, true);
  assert.equal(document.button('Return to box').disabled, false);
  assert.equal(document.status().textContent.includes('Nail 1'), true);
  const controls = document.host.flatten();
  mounted.destroy();
  assert.deepEqual(document.host.children, []);
  assert.equal(controls.every(control => control.listeners.size === 0), true);
});

test('mounted controls become available only after all scene objects are ready', () => {
  const workshop = new MovableWorkshop();
  const document = new WorkshopControls();
  workshop.ready = false;
  const mounted = Workshop.mount(document.host, workshop.config, workshop.adapter, workshop.clock);
  assert.equal(document.button('Lift selected part').disabled, true);
  workshop.ready = true;
  mounted.refresh();
  assert.equal(document.button('Lift selected part').disabled, false);
  assert.equal(document.status().textContent, 'Workshop ready. Choose a drawer or lift a part.');
  mounted.destroy();
});

test('the workshop scene-ready hook mounts authoritative physics once with its independent metadata and author credit', () => {
  const source = fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/panels/robot_scene/panel.js'), 'utf8');
  const start = source.indexOf('  // %% independent workshop controls');
  const end = source.indexOf('  // %% bus: outbound', start);
  assert.ok(start >= 0 && end > start, 'The standard scene viewer needs the workshop hook.');
  const workshop = new MovableWorkshop();
  const nodes = new Map(['.stage', '.panel-head h2', '.stage-hint'].map(key => [key, {}]));
  const rights = {principalAuthor: 'Vanessa Hassouna', institution: 'AICOR Institute'};
  let received;
  let mounts = 0;
  let folded = 0;
  let ready;
  const context = {
    workshopControls: null,
    SCENE: {workshop: workshop.config, objects: workshop.config.parts, assetRights: rights},
    Workshop: {PhysicsClient: class { constructor(fetch, url) { this.url = url; } }, mountPhysics(host, config, adapter, client) { mounts += 1; received = {host, config, adapter, client}; return {refresh() {}}; }},
    window: {fetch() {}, location: {href: 'https://example.test/workshop/'}},
    workshopContactLayer: null,
    bindLaboratoryContactLayer: () => ({}),
    stopTrajectory() {},
    liveBtn: {}, playBtn: {}, scrubber: {}, speedSelect: {},
    RobotView: {onReady(callback) { ready = callback; }},
    root: {querySelector: selector => nodes.get(selector), classList: {add() {}}},
    workshopPhysicsAdapter: () => workshop.adapter,
    layersFolded: false,
    showFold() { folded += 1; },
  };
  vm.runInNewContext(source.slice(start, end), context);
  ready();
  ready();
  assert.equal(mounts, 1);
  assert.equal(received.host, nodes.get('.stage'));
  assert.equal(received.config.parts, workshop.config.parts);
  assert.equal(received.config.assetRights, rights);
  assert.equal(received.config.objects, workshop.config.parts);
  assert.equal(received.client.url, context.window.location.href);
  assert.equal(context.playBtn.disabled, true);
  assert.equal(received.adapter, workshop.adapter);
  assert.equal(nodes.get('.panel-head h2').textContent, 'AICOR Wood Workshop');
  assert.equal(context.layersFolded, true);
  assert.equal(folded, 1);
});

test('a validated local robot recording is linked only when declared in the scene', () => {
  const workshop = new MovableWorkshop();
  const document = new WorkshopControls();
  const first = Workshop.mount(document.host, workshop.config, workshop.adapter, workshop.clock);
  assert.equal(document.host.flatten().some(item => item.tagName === 'a'), false);
  first.destroy();
  workshop.config.demoUrl = 'index.html?scene=aicor_workshop_pr2&layout=scene&offline=1';
  const second = Workshop.mount(document.host, workshop.config, workshop.adapter, workshop.clock);
  const link = document.host.flatten().find(item => item.tagName === 'a');
  assert.equal(link.href, workshop.config.demoUrl);
  assert.equal(link.textContent, 'Watch PR2 assembly demo');
  second.destroy();
  workshop.config.demoUrl = 'javascript:alert(1)';
  const third = Workshop.mount(document.host, workshop.config, workshop.adapter, workshop.clock);
  assert.equal(document.host.flatten().some(item => item.tagName === 'a'), false);
  third.destroy();
});
