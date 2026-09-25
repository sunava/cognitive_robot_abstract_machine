'use strict';

// %% authoritative snapshots and explicitly observed commands
const assert = require('node:assert/strict');
const test = require('node:test');
const Workshop = require('../../../cramera/src/cramera/web/core/workshop.js');

class PhysicsWorkshop {
  constructor() {
    this.config = {
      drawers: [{id: 'drawer_1', label: 'Lower drawer', joint: 'workshop_drawer_1_joint', open: .4}],
      parts: [{key: 'screw_1', label: 'Screw 1', kind: 'screw'}],
      bins: [{key: 'screw_box', label: 'Screw box'}],
      assembly: {key: 'workshop_assembly'}, driver: {key: 'driver', label: 'Screwdriver'},
      objects: ['screw_1', 'screw_box', 'workshop_assembly', 'driver'].map(key => ({key, id: key})),
      cameras: {overview: {position: [2, -3, 2], target: [0, 0, 1]}},
    };
    this.state = {
      objects: {screw_1: [-.3, -.1, .97, 0, 0, 0, 1], screw_box: [-.3, -.1, .92, 0, 0, 0, 1], workshop_assembly: [.28, -.1, .92, 0, 0, 0, 1], driver: [.52, -.26, .946, 0, 0, 0, 1]},
      joints: {workshop_drawer_1_joint: 0}, held: null, contacts: [], thread: {engaged: false}, time: 0, running: false,
    };
    this.commands = [];
    this.renders = [];
    this.ready = true;
    this.client = {
      state: async () => structuredClone(this.state),
      command: async body => {
        this.commands.push(structuredClone(body));
        if (body.action === 'start') this.state.running = true;
        if (body.action === 'pause') this.state.running = false;
        if (body.action === 'grab') this.state.held = body.object;
        if (body.action === 'release') this.state.held = null;
        return structuredClone(this.state);
      },
    };
    this.adapter = {isReady: () => this.ready, applySnapshot: state => this.renders.push(structuredClone(state)), pause() {}, focus() {}};
    this.controller = new Workshop.PhysicsController(this.config, this.adapter, this.client);
  }
}

function deferred() {
  let resolve;
  const promise = new Promise(complete => { resolve = complete; });
  return {promise, resolve};
}

// %% mounted authentication and transport
test('physics URLs retain the protected workshop mount and same-origin cookies', async () => {
  const requests = [];
  const client = new Workshop.PhysicsClient(async (url, options) => {
    requests.push({url, options});
    return {ok: true, json: async () => ({objects: {}, joints: {}, running: false})};
  }, 'https://example.test/workshop/index.html?scene=aicor_workshop');
  await client.state();
  await client.command({action: 'grab', object: 'screw_box'});
  assert.equal(requests[0].url, 'https://example.test/workshop/api/workshop/state');
  assert.equal(requests[1].url, 'https://example.test/workshop/api/workshop/command');
  assert.equal(requests[0].options.credentials, 'same-origin');
  assert.equal(requests[1].options.credentials, 'same-origin');
  assert.equal(requests[1].options.headers['Content-Type'], 'application/json');
  assert.deepEqual(JSON.parse(requests[1].options.body), {action: 'grab', object: 'screw_box'});
});

test('backend errors retain their visible explanation and never render invented poses', async () => {
  const workshop = new PhysicsWorkshop();
  workshop.controller.client = new Workshop.PhysicsClient(async () => ({ok: false, status: 503, json: async () => ({error: 'Physics is unavailable.'})}), 'https://example.test/workshop/');
  const result = await workshop.controller.start();
  assert.equal(result.ok, false);
  assert.equal(result.message, 'Physics is unavailable.');
  assert.deepEqual(workshop.renders, []);
  assert.equal(workshop.controller.isReady(), false);
});

// %% physics remains authoritative
test('box and contents are rendered from their independent measured poses', async () => {
  const workshop = new PhysicsWorkshop();
  await workshop.controller.start();
  workshop.state.objects.screw_box = [-.2, -.1, 1.1, 0, 0, .2, Math.sqrt(.96)];
  workshop.state.objects.screw_1 = [-.5, -.1, .93, .5, .5, .5, .5];
  await workshop.controller.poll();
  assert.deepEqual(workshop.renders.at(-1).objects, workshop.state.objects);
  assert.equal(workshop.controller.select('screw_box'), true);
  assert.equal(workshop.controller.select('workshop_assembly'), true);
  assert.equal(workshop.controller.select('driver'), true);
  assert.equal(workshop.controller.select('unknown'), false);
});

test('moving and tilting a held box change only its spring target', async () => {
  const workshop = new PhysicsWorkshop();
  await workshop.controller.start();
  await workshop.controller.grab('screw_box');
  const observed = structuredClone(workshop.state.objects);
  await workshop.controller.moveBy([0, 0, .05]);
  const move = workshop.commands.at(-1);
  assert.equal(move.action, 'move');
  assert.equal(move.object, 'screw_box');
  assert.deepEqual(move.pose.slice(0, 3), [-.3, -.1, .97]);
  await workshop.controller.tilt(0, Math.PI / 2);
  const tilt = workshop.commands.at(-1);
  assert.equal(tilt.object, 'screw_box');
  assert.deepEqual(tilt.pose.slice(0, 3), move.pose.slice(0, 3));
  assert.ok(Math.abs(tilt.pose[3] - Math.SQRT1_2) < 1e-12);
  assert.ok(Math.abs(tilt.pose[6] - Math.SQRT1_2) < 1e-12);
  assert.deepEqual(workshop.renders.at(-1).objects, observed);
  assert.equal(workshop.commands.some(command => command.object === 'screw_1'), false);
});

test('drawers and driver commands are physics operations without local joint writes', async () => {
  const workshop = new PhysicsWorkshop();
  await workshop.controller.start();
  await workshop.controller.setDrawer('drawer_1', true);
  assert.deepEqual(workshop.commands.at(-1), {action: 'drawer', joint: workshop.config.drawers[0].joint, position: workshop.config.drawers[0].open});
  assert.deepEqual(workshop.renders.at(-1).joints, workshop.state.joints);
  await workshop.controller.drive(1);
  assert.deepEqual(workshop.commands.at(-1), {action: 'drive', direction: 1});
  await workshop.controller.drive(0);
  assert.deepEqual(workshop.commands.at(-1), {action: 'drive', direction: 0});
});

test('a delayed state response cannot undo a newer successful grab', async () => {
  const workshop = new PhysicsWorkshop();
  await workshop.controller.start();
  const old = structuredClone(workshop.state);
  const request = deferred();
  workshop.client.state = () => request.promise;
  const polling = workshop.controller.poll();
  await workshop.controller.grab('screw_box');
  request.resolve(old);
  await polling;
  assert.equal(workshop.controller.snapshot.held, 'screw_box');
  assert.equal(workshop.renders.at(-1).held, 'screw_box');
});

test('pointer release waits for its queued grab and latest target', async () => {
  const workshop = new PhysicsWorkshop();
  await workshop.controller.start();
  const pendingGrab = deferred();
  const command = workshop.client.command;
  workshop.client.command = async body => {
    if (body.action === 'grab') await pendingGrab.promise;
    return command(body);
  };
  assert.equal(workshop.controller.beginDrag('screw_box'), true);
  workshop.controller.dragTo([-.2, -.1, .92]);
  workshop.controller.dragTo([-.1, -.1, .92]);
  const releasing = workshop.controller.endDrag();
  pendingGrab.resolve();
  await releasing;
  assert.deepEqual(workshop.commands.map(body => body.action), ['start', 'grab', 'move', 'release']);
  assert.deepEqual(workshop.commands[2].pose.slice(0, 3), [-.1, -.1, .92]);
  assert.equal(workshop.controller.snapshot.held, null);
});

test('destroy prevents late responses from updating the detached viewer', async () => {
  const workshop = new PhysicsWorkshop();
  await workshop.controller.start();
  const pending = deferred();
  workshop.client.state = () => pending.promise;
  const polling = workshop.controller.poll();
  const renders = workshop.renders.length;
  workshop.controller.destroy();
  pending.resolve(structuredClone(workshop.state));
  await polling;
  assert.equal(workshop.renders.length, renders);
});

// %% measured alignment paths
test('aligning the selected screw lifts clear before crossing and seats through physics targets', async () => {
  const workshop = new PhysicsWorkshop();
  const goal = [.28, -.1, .9952, 0, 0, 0, 1];
  workshop.state.thread = {screw: 'a_different_screw', screwTargetPose: goal};
  const command = workshop.client.command;
  workshop.client.command = async body => {
    if (body.action === 'move') workshop.state.objects[body.object] = body.pose.slice();
    if (body.action === 'release') Object.assign(workshop.state.thread, {screw: 'screw_1', state: 'seated'});
    return command(body);
  };
  await workshop.controller.start();
  workshop.controller.select('screw_1');
  const result = await workshop.controller.align('screw');
  assert.equal(result.ok, true);
  const moves = workshop.commands.filter(body => body.action === 'move');
  assert.equal(moves.length, 3);
  assert.deepEqual(moves[0].pose.slice(0, 2), [-.3, -.1]);
  assert.ok(moves[0].pose[2] >= 1.25);
  assert.deepEqual(moves[1].pose.slice(0, 2), goal.slice(0, 2));
  assert.equal(moves[1].pose[2], moves[0].pose[2]);
  assert.deepEqual(moves[2].pose, goal);
  assert.equal(workshop.commands.at(-1).action, 'release');
});

test('blocked alignment stops at the observed obstruction without sending later waypoints', async () => {
  const workshop = new PhysicsWorkshop();
  workshop.state.thread = {screwTargetPose: [.28, -.1, .9952, 0, 0, 0, 1]};
  let now = 0;
  workshop.controller.clock = {now: () => now, wait: async milliseconds => { now += milliseconds; }};
  await workshop.controller.start();
  const result = await workshop.controller.align('screw');
  assert.equal(result.ok, false);
  assert.match(result.message, /blocked|reach/i);
  assert.equal(workshop.commands.filter(body => body.action === 'move').length, 1);
  assert.deepEqual(workshop.renders.at(-1).objects, workshop.state.objects);
});

// %% mounted controls and cleanup
class PhysicsControlElement {
  constructor(tag, document) {
    this.tagName = tag; this.ownerDocument = document; this.children = [];
    this.listeners = new Map(); this.attributes = new Map(); this.dataset = {}; this.textContent = '';
  }
  setAttribute(key, value) { this.attributes.set(key, value); }
  appendChild(child) { this.children.push(child); child.parent = this; }
  addEventListener(event, handler) { this.listeners.set(event, handler); }
  removeEventListener(event) { this.listeners.delete(event); }
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  flatten() { return [this, ...this.children.flatMap(child => child.flatten())]; }
}
class PhysicalControlsDocument {
  constructor() { this.host = new PhysicsControlElement('div', this); }
  createElement(tag) { return new PhysicsControlElement(tag, this); }
  button(label) { return this.host.flatten().find(item => item.tagName === 'button' && item.textContent === label); }
}

test('mounted physics exposes box, driver and workpiece, and sends lift and release commands', async () => {
  const workshop = new PhysicsWorkshop();
  const document = new PhysicalControlsDocument();
  const mounted = Workshop.mountPhysics(document.host, workshop.config, workshop.adapter, workshop.client);
  try {
    await mounted.controller.queue;
    const options = document.host.flatten().filter(item => item.tagName === 'option').map(item => item.value);
    assert.deepEqual(options, workshop.config.objects.map(object => object.key));
    mounted.select('screw_box');
    await document.button('Grab selected').listeners.get('click')();
    assert.equal(document.button('Lift').disabled, false);
    await document.button('Lift').listeners.get('click')();
    assert.equal(workshop.commands.at(-1).action, 'move');
    await document.button('Release').listeners.get('click')();
    assert.equal(workshop.commands.at(-1).action, 'release');
    assert.equal(document.button('Lift').disabled, true);
  } finally { mounted.destroy(); }
  assert.deepEqual(document.host.children, []);
});

test('pausing during alignment prevents later crossing and descent targets', async () => {
  const workshop = new PhysicsWorkshop();
  workshop.state.thread = {screwTargetPose: [.28, -.1, .9952, 0, 0, 0, 1]};
  const waiting = deferred();
  workshop.controller.clock = {now: () => 0, wait: () => waiting.promise};
  await workshop.controller.start();
  const aligning = workshop.controller.align('screw');
  for (let index = 0; index < 20 && !workshop.commands.some(command => command.action === 'move'); index += 1) await Promise.resolve();
  await workshop.controller.pause();
  waiting.resolve();
  const result = await aligning;
  assert.equal(result.ok, false);
  assert.equal(workshop.controller.aligning, false);
  assert.equal(workshop.commands.filter(command => command.action === 'move').length, 1);
});

// %% viewer integration cannot bypass physics
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const panelSource = fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/panels/robot_scene/panel.js'), 'utf8');

test('viewer dragging sends a spring target without editing the rendered object', () => {
  const start = panelSource.indexOf("  renderer.domElement.addEventListener('pointermove', function (e)");
  const end = panelSource.indexOf('  function endDrag()', start);
  let move;
  const targets = [];
  const position = {x: -.2, y: -.1, z: .93};
  const context = {
    renderer: {domElement: {addEventListener(type, handler) { move = handler; }}},
    dragging: true, dragTarget: {name: 'screw_box', physicsOffset: {x: -.1, y: .03}},
    objectMeshes: {screw_box: {position}},
    surfacePointAt: () => ({x: .2, y: .1, z: 1.1}), worldRoot: {worldToLocal() {}},
    workshopControls: {controller: {dragTo(target) { targets.push(Array.from(target)); }}},
  };
  vm.runInNewContext(panelSource.slice(start, end), context);
  move({});
  assert.deepEqual(targets, [[.1, .13, 1.1]]);
  assert.deepEqual(position, {x: -.2, y: -.1, z: .93});
});

test('physical workshop never exposes direct joint-position sliders', () => {
  const start = panelSource.indexOf('  function refreshJointControls()');
  const end = panelSource.indexOf('  // the sliders follow', start);
  const host = {innerHTML: 'stale manual controls'};
  const context = {SCENE: {workshop: {}}, $: () => host, jointControls: [{}]};
  vm.runInNewContext(panelSource.slice(start, end), context);
  context.refreshJointControls();
  assert.equal(host.innerHTML, '');
  assert.deepEqual(context.jointControls, []);
});

test('closing controls releases an in-flight grasp after its response arrives', async () => {
  const workshop = new PhysicsWorkshop();
  await workshop.controller.start();
  const pending = deferred();
  const command = workshop.client.command;
  workshop.client.command = async body => {
    if (body.action === 'grab') await pending.promise;
    return command(body);
  };
  const grabbing = workshop.controller.grab('screw_box');
  const closing = workshop.controller.destroy();
  pending.resolve();
  await grabbing;
  await closing;
  assert.equal(workshop.state.held, null);
  assert.equal(workshop.commands.at(-1).action, 'release');
});

test('a solver failure in a successful snapshot stays visible and Reset remains available', async () => {
  const workshop = new PhysicsWorkshop();
  workshop.state.available = false;
  workshop.state.error = 'Solver stopped. Reset the workshop.';
  const command = workshop.client.command;
  workshop.client.command = async body => {
    const state = await command(body);
    if (body.action === 'start') state.running = false;
    if (body.action === 'reset') {
      delete workshop.state.error;
      workshop.state.available = true;
      workshop.state.running = false;
      return structuredClone(workshop.state);
    }
    return state;
  };
  const document = new PhysicalControlsDocument();
  const mounted = Workshop.mountPhysics(document.host, workshop.config, workshop.adapter, workshop.client);
  try {
    await mounted.controller.queue;
    const status = document.host.flatten().find(element => element.attributes.get('role') === 'status');
    assert.equal(status.textContent, workshop.state.error);
    assert.equal(status.dataset.error, 'true');
    assert.equal(mounted.controller.isReady(), false);
    assert.equal(document.button('Reset').disabled, false);
    await mounted.controller.poll();
    assert.equal(status.textContent, workshop.state.error);
    await document.button('Reset').listeners.get('click')();
    assert.equal(mounted.controller.error, null);
    assert.equal(status.dataset.error, 'false');
    assert.equal(document.button('Resume physics').disabled, false);
  } finally { mounted.destroy(); }
});

test('final screw positioning waits inside the pilot capture radius before release', async () => {
  const workshop = new PhysicsWorkshop();
  const goal = [.28, -.1, 1.021, 0, 0, 0, 1];
  workshop.state.thread = {screw: 'screw_1', state: 'free', screwTargetPose: goal};
  const command = workshop.client.command;
  let now = 0, finalPending = false, waited = false;
  workshop.controller.clock = {now: () => now, wait: async milliseconds => { now += milliseconds; }};
  workshop.client.command = async body => {
    if (body.action === 'move') {
      workshop.state.objects[body.object] = body.pose.slice();
      if (body.pose[2] === goal[2]) {
        workshop.state.objects[body.object][0] += .003;
        finalPending = true;
      }
    }
    if (body.action === 'release') {
      assert.equal(waited, true, 'The release must wait for the tighter measured tolerance.');
      workshop.state.thread.state = 'seated';
    }
    return command(body);
  };
  workshop.client.state = async () => {
    if (finalPending) {
      workshop.state.objects.screw_1 = goal.slice();
      workshop.state.objects.screw_1[0] += .001;
      waited = true;
    }
    return structuredClone(workshop.state);
  };
  await workshop.controller.start();
  const result = await workshop.controller.align('screw');
  assert.equal(result.ok, true);
  assert.equal(workshop.controller.snapshot.thread.state, 'seated');
});

test('driver alignment cannot claim success from pose proximity without contact', async () => {
  const workshop = new PhysicsWorkshop();
  workshop.state.thread = {screw: 'screw_1', state: 'seated', driverAligned: false, driverTargetPose: [.28, -.1, 1.13, 0, 0, 0, 1]};
  const command = workshop.client.command;
  let now = 0;
  workshop.controller.clock = {now: () => now, wait: async milliseconds => { now += milliseconds; }};
  workshop.client.command = async body => {
    if (body.action === 'move') workshop.state.objects[body.object] = body.pose.slice();
    return command(body);
  };
  await workshop.controller.start();
  const result = await workshop.controller.align('driver');
  assert.equal(result.ok, false);
  assert.match(result.message, /contact|align/i);
});
