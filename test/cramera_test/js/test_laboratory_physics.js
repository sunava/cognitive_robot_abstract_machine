'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const Physics = require('../../../cramera/src/cramera/web/core/laboratory-physics.js');

// %% authoritative simulation fixture
class ContactSimulation {
  constructor() {
    this.config = {
      objects: [{key: 'tube', label: 'Glas'}],
      bounds: {min: [-.8, -.65, .8], max: [.8, .45, 1.6]},
    };
    this.answer = {ok: true, state: 'running', objects: {tube: [.16, -.065, .906, 0, 0, 0, 1]}, target: null, contacts: []};
    this.commands = [];
    this.poses = [];
    this.service = {
      state: async () => this.answer,
      target: async target => { this.commands.push({type: 'target', ...target}); return {ok: true}; },
      release: async () => { this.commands.push({type: 'release'}); return {ok: true}; },
      reset: async () => { this.commands.push({type: 'reset'}); return {ok: true}; },
    };
    this.adapter = {setObjectPose: (key, pose) => this.poses.push({key, pose: pose.slice()})};
    this.controller = new Physics.Controller(this.config, this.service, this.adapter);
  }
}

test('drag targets are bounded and never replace the visible simulated pose', async () => {
  const scene = new ContactSimulation();
  await scene.controller.refresh();
  const observed = structuredClone(scene.poses);
  assert.equal(scene.controller.beginDrag('tube'), true);
  await scene.controller.dragTo([2, -2, 3]);
  assert.deepEqual(scene.commands.at(-1), {type: 'target', key: 'tube', position: [.8, -.65, .906]});
  assert.deepEqual(scene.poses, observed);
  scene.answer.objects.tube = [.18, -.06, .906, 0, 0, 0, 1];
  await scene.controller.refresh();
  assert.deepEqual(scene.poses.at(-1).pose, scene.answer.objects.tube);
});

test('drag requests coalesce and release runs after the in-flight target', async () => {
  const scene = new ContactSimulation();
  await scene.controller.refresh();
  const finish = [];
  scene.service.target = target => {
    scene.commands.push({type: 'target', ...target});
    return new Promise(resolve => finish.push(resolve));
  };
  scene.controller.beginDrag('tube');
  scene.controller.dragTo([.2, 0, 1]);
  scene.controller.dragTo([.3, 0, 1]);
  const moving = scene.controller.dragTo([.4, 0, 1]);
  assert.equal(scene.commands.length, 1);
  finish.shift()({ok: true});
  await new Promise(setImmediate);
  assert.equal(scene.commands.length, 2);
  assert.deepEqual(scene.commands[1].position, [.4, 0, .906]);
  const released = scene.controller.release();
  scene.controller.dragTo([.5, 0, 1]);
  finish.shift()({ok: true});
  await Promise.all([moving, released]);
  assert.deepEqual(scene.commands.map(command => command.type), ['target', 'target', 'release']);
  assert.equal(scene.controller.target, null);
});

test('a state response started before a drag cannot restore an old target', async () => {
  const scene = new ContactSimulation();
  await scene.controller.refresh();
  let finish;
  scene.service.state = () => new Promise(resolve => { finish = resolve; });
  const polling = scene.controller.refresh();
  await scene.controller.hold('tube');
  await scene.controller.nudge([0, 0, .02]);
  const desired = structuredClone(scene.controller.target);
  finish(scene.answer);
  await polling;
  assert.deepEqual(scene.controller.target, desired);
});

test('a state read during a pending command cannot undo its completed target', async () => {
  const scene = new ContactSimulation();
  await scene.controller.refresh();
  let completeCommand;
  let completePoll;
  scene.service.target = () => new Promise(resolve => { completeCommand = resolve; });
  const holding = scene.controller.hold('tube');
  scene.service.state = () => new Promise(resolve => { completePoll = resolve; });
  const polling = scene.controller.refresh();
  const desired = structuredClone(scene.controller.target);
  completeCommand({ok: true});
  await holding;
  completePoll(scene.answer);
  await polling;
  assert.deepEqual(scene.controller.target, desired);
});

test('release and reset send no browser object poses', async () => {
  const requests = [];
  const client = new Physics.Client(async (url, options) => {
    requests.push({url, options});
    return {ok: true, json: async () => ({ok: true})};
  });
  await client.release();
  await client.reset();
  assert.deepEqual(requests.map(request => request.url), [Physics.Routes.RELEASE, Physics.Routes.RESET]);
  assert.deepEqual(requests.map(request => JSON.parse(request.options.body)), [{}, {}]);
});

test('contact feedback describes server forces and target distance', async () => {
  const scene = new ContactSimulation();
  scene.answer.contacts = [{position: [0, 0, .9], normal: [0, 0, 1], force: 1.25, bodyA: 'tube', bodyB: 'bench'}];
  scene.answer.target = {key: 'tube', position: [.16, -.065, 1.006]};
  await scene.controller.refresh();
  const feedback = scene.controller.feedback();
  assert.equal(feedback.contactCount, scene.answer.contacts.length);
  assert.equal(feedback.force, scene.answer.contacts[0].force);
  assert.ok(Math.abs(feedback.distance - .1) < 1e-10);
});

// %% robot contact program
test('robot physics requests use isolated routes and send no recorded poses', async () => {
  const requests = [];
  const client = new Physics.Client(async (url, options) => {
    requests.push({url, options});
    return {ok: true, json: async () => ({ok: true})};
  }, {robot: true});
  await client.start();
  await client.state();
  await client.run();
  await client.pause();
  await client.stop();
  assert.deepEqual(requests.map(request => request.url), [
    '/api/laboratory/physics/robot/start', '/api/laboratory/physics/robot/state',
    '/api/laboratory/physics/robot/run', '/api/laboratory/physics/robot/pause',
    '/api/laboratory/physics/robot/stop',
  ]);
  assert.deepEqual(requests.filter(request => request.options.method === 'POST').map(request => JSON.parse(request.options.body)), [{}, {}, {}, {}]);
});

test('robot joints follow only observed simulation frames during manual interaction', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  const frames = [];
  scene.adapter.setJointFrames = observed => frames.push(structuredClone(observed));
  scene.answer.frames = {'pr2/l_shoulder_pan_joint': .27};
  scene.answer.robot = {state: 'idle', phase: 'ready', progress: 0};
  await scene.controller.refresh();
  await scene.controller.hold('tube');
  assert.deepEqual(frames, [scene.answer.frames]);
  scene.answer.frames = {'pr2/l_shoulder_pan_joint': .29};
  await scene.controller.refresh();
  assert.deepEqual(frames.at(-1), scene.answer.frames);
});

test('run and pause serialize behind targets and reject competing manual manipulation', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  scene.answer.robot = {state: 'idle', phase: 'ready', progress: 0};
  await scene.controller.refresh();
  let finishTarget;
  let finishRun;
  scene.service.target = target => {
    scene.commands.push({type: 'target', ...target});
    return new Promise(resolve => { finishTarget = resolve; });
  };
  scene.service.run = () => {
    scene.commands.push({type: 'run'});
    return new Promise(resolve => { finishRun = resolve; });
  };
  scene.service.pause = async () => { scene.commands.push({type: 'pause'}); return {ok: true}; };
  scene.controller.hold('tube');
  const running = scene.controller.run();
  assert.equal(scene.controller.beginDrag('tube'), false);
  assert.equal(await scene.controller.hold('tube'), false);
  finishTarget({ok: true});
  await new Promise(setImmediate);
  const paused = scene.controller.pause();
  finishRun({ok: true});
  await Promise.all([running, paused]);
  assert.deepEqual(scene.commands.map(command => command.type), ['target', 'run', 'pause']);
  assert.equal(scene.controller.robot.state, 'paused');
  assert.equal(scene.controller.target, null);
});

test('an old idle poll cannot unlock manual movement after a robot run command', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  scene.answer.robot = {state: 'idle', phase: 'approach', progress: 0};
  scene.service.run = async () => ({ok: true});
  await scene.controller.refresh();
  let finish;
  scene.service.state = () => new Promise(resolve => { finish = resolve; });
  const polling = scene.controller.refresh();
  await scene.controller.run();
  finish(scene.answer);
  await polling;
  assert.equal(scene.controller.robot.state, 'running');
  assert.equal(scene.controller.beginDrag('tube'), false);
});

// %% mounted controls and service lifecycle
class PhysicsControl {
  constructor(tag, ownerDocument) {
    this.tagName = tag;
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.listeners = new Map();
    this.dataset = {};
    this.attributes = new Map();
  }
  appendChild(child) { child.parent = this; this.children.push(child); }
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  addEventListener(event, handler) { this.listeners.set(event, handler); }
  removeEventListener(event) { this.listeners.delete(event); }
  setAttribute(name, value) { this.attributes.set(name, value); }
  flatten() { return [this, ...this.children.flatMap(child => child.flatten())]; }
}

class PhysicsControls {
  constructor() { this.host = new PhysicsControl('div', this); }
  createElement(tag) { return new PhysicsControl(tag, this); }
  find(className) { return this.host.flatten().find(item => item.className === className); }
}

class PhysicsPolling {
  constructor() { this.callbacks = new Map(); this.count = 0; }
  setInterval(callback, interval) {
    this.count += 1;
    this.callbacks.set(this.count, {callback, interval});
    return this.count;
  }
  clearInterval(timer) { this.callbacks.delete(timer); }
}

// %% liquid manipulation
class LiquidSimulation extends ContactSimulation {
  constructor() {
    super();
    this.answer.liquid = {
      model: 'reduced_free_surface',
      tubes: {tube: {volumeMl: 12, capacityMl: 22}},
      spilledMl: .75,
    };
    this.service.fillLiquid = async (key, volumeMl) => {
      this.commands.push({type: 'liquid', key, volumeMl});
      return {ok: true};
    };
  }
}

// %% two-liquid robot demonstration
test('mixing requests start a server-owned recipe without browser object poses', async () => {
  const requests = [];
  const client = new Physics.Client(async (url, options) => {
    requests.push({url, method: options.method, body: JSON.parse(options.body)});
    return {ok: true, json: async () => ({ok: true})};
  }, {robot: true});
  await client.runMixing();
  assert.deepEqual(requests, [{url: Physics.RobotRoutes.MIX, method: 'POST', body: {}}]);
});

test('mixing waits for movement and locks manual commands until a queued pause completes', async () => {
  const scene = new LiquidSimulation();
  scene.config.robot = true;
  await scene.controller.refresh();
  let finishTarget, finishMixing;
  scene.service.target = target => {
    scene.commands.push({type: 'target', ...target});
    return new Promise(resolve => { finishTarget = resolve; });
  };
  scene.service.runMixing = () => {
    scene.commands.push({type: 'runMixing'});
    return new Promise(resolve => { finishMixing = resolve; });
  };
  scene.service.pause = async () => { scene.commands.push({type: 'pause'}); return {ok: true}; };
  scene.controller.hold('tube');
  const running = scene.controller.runMixing();
  assert.equal(scene.controller.beginDrag('tube'), false);
  assert.equal(await scene.controller.fillLiquid(5), false);
  assert.equal(await scene.controller.run(), false);
  assert.equal(await scene.controller.runMixing(), false);
  finishTarget({ok: true});
  await new Promise(setImmediate);
  const paused = scene.controller.pause();
  finishMixing({ok: true});
  await Promise.all([running, paused]);
  assert.deepEqual(scene.commands.map(command => command.type), ['target', 'runMixing', 'pause']);
  assert.equal(scene.controller.robot.state, Physics.RobotStates.PAUSED);
  assert.equal(scene.controller.robot.source, Physics.RobotSources.MIXING);
});

test('mixing discards a pre-recipe poll instead of restoring its previous liquid', async () => {
  const scene = new LiquidSimulation();
  scene.config.robot = true;
  await scene.controller.refresh();
  let finish;
  scene.service.state = () => new Promise(resolve => { finish = resolve; });
  scene.service.runMixing = async () => ({ok: true});
  const polling = scene.controller.refresh();
  await scene.controller.runMixing();
  scene.controller.liquid = null;
  finish(scene.answer);
  await polling;
  assert.equal(scene.controller.liquid, null);
  assert.equal(scene.controller.robot.state, Physics.RobotStates.RUNNING);
});

test('mixing control displays the recipe, stage feedback and completion in A2', async () => {
  const scene = new LiquidSimulation();
  scene.config.robot = true;
  scene.answer.robot = {state: 'idle'};
  scene.service.runMixing = async () => { scene.commands.push({type: 'runMixing'}); return {ok: true}; };
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  const mixing = surface.find('laboratory-physics-robot-mix');
  assert.equal(mixing.textContent, 'Misch-Demo starten');
  assert.match(surface.find('laboratory-physics-robot-recipe').textContent, /5 ml/);
  assert.match(surface.find('laboratory-physics-robot-recipe').textContent, /A2/);
  await mixing.listeners.get('click')();
  assert.deepEqual(scene.commands, [{type: 'runMixing'}]);
  scene.answer.robot = {source: Physics.RobotSources.MIXING, state: 'running', phase: 'pour', progress: .4, message: 'Türkise Flüssigkeit eingießen'};
  await mounted.refresh();
  assert.equal(mixing.disabled, true);
  assert.equal(surface.find('laboratory-physics-robot-run').disabled, true);
  assert.equal(surface.find('laboratory-physics-fill').disabled, true);
  assert.equal(surface.find('laboratory-physics-robot-status').textContent, 'PR2 mischt die Flüssigkeiten · ' + scene.answer.robot.message);
  scene.answer.robot = {source: Physics.RobotSources.MIXING, state: 'succeeded', phase: 'complete', progress: 1};
  await mounted.refresh();
  assert.equal(surface.find('laboratory-physics-robot-status').textContent, 'Mischung in A2 abgelegt');
  assert.equal(mixing.disabled, false);
  mounted.destroy();
});

test('paused mixing offers its own resume while the original transfer remains a fresh start', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  scene.answer.robot = {source: Physics.RobotSources.MIXING, state: 'paused', phase: 'swirl', progress: .8};
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  assert.equal(surface.find('laboratory-physics-robot-mix').textContent, 'Misch-Demo fortsetzen');
  assert.equal(surface.find('laboratory-physics-robot-run').textContent, 'PR2: A1 → A3 starten');
  mounted.destroy();
});

test('mixing phase fallback describes swirling and reports server failure without claiming completion', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  scene.answer.robot = {source: Physics.RobotSources.MIXING, state: 'running', phase: 'swirl', progress: 1};
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  assert.equal(surface.find('laboratory-physics-robot-status').textContent, 'PR2 mischt die Flüssigkeiten · Mischung sanft schwenken');
  scene.answer.robot = {source: Physics.RobotSources.MIXING, state: 'failed', phase: 'place', progress: 1, error: 'Stand in A2 nicht bestätigt'};
  await mounted.refresh();
  assert.equal(surface.find('laboratory-physics-robot-status').textContent, scene.answer.robot.error);
  mounted.destroy();
});

test('robot motion notes distinguish a recorded transfer from the local mixing controller', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  scene.answer.robot = {source: Physics.RobotSources.TRANSFER, state: 'running'};
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  const note = surface.find('laboratory-physics-robot-motion-note');
  assert.match(note.textContent, /gespeicherter CRAM-Plan/);
  scene.answer.robot = {source: Physics.RobotSources.MIXING, state: 'running'};
  await mounted.refresh();
  assert.match(note.textContent, /lokaler Kontaktregler/);
  assert.doesNotMatch(note.textContent, /gespeicherter CRAM-Plan/);
  mounted.destroy();
});

test('placement phase labels describe donor returns without claiming the mixture is in A2', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  scene.answer.robot = {source: Physics.RobotSources.MIXING, state: 'running', phase: 'place', object: 'tube_amber'};
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  assert.equal(surface.find('laboratory-physics-robot-status').textContent, 'PR2 mischt die Flüssigkeiten · Glas im Ständer abstellen');
  mounted.destroy();
});

test('liquid fill requests use the selected simulation route and a volume payload', async () => {
  for (const robot of [false, true]) {
    const requests = [];
    const client = new Physics.Client(async (url, options) => {
      requests.push({url, body: JSON.parse(options.body)});
      return {ok: true, json: async () => ({ok: true})};
    }, {robot});
    await client.fillLiquid('tube', 15);
    assert.deepEqual(requests, [{url: (robot ? Physics.RobotRoutes : Physics.Routes).LIQUID, body: {key: 'tube', volumeMl: 15}}]);
  }
});

test('liquid rendering receives the authoritative snapshot after simulated object poses', async () => {
  const scene = new LiquidSimulation();
  const events = [];
  scene.adapter.setObjectPose = (key, pose) => events.push({key, pose});
  scene.adapter.setLiquidState = liquid => events.push({liquid});
  await scene.controller.refresh();
  assert.deepEqual(events, [{key: 'tube', pose: scene.answer.objects.tube}, {liquid: scene.answer.liquid}]);
  assert.deepEqual(scene.controller.liquid, scene.answer.liquid);
});

test('liquid observations continue rendering while an ordinary movement command is pending', async () => {
  const scene = new LiquidSimulation();
  const rendered = [];
  scene.adapter.setLiquidState = liquid => rendered.push(structuredClone(liquid));
  await scene.controller.refresh();
  let finish;
  scene.service.target = () => new Promise(resolve => { finish = resolve; });
  const moving = scene.controller.nudge([0, 0, .01]);
  scene.answer.liquid.tubes.tube.volumeMl = 11;
  await scene.controller.refresh();
  assert.equal(rendered.at(-1).tubes.tube.volumeMl, scene.answer.liquid.tubes.tube.volumeMl);
  finish({ok: true});
  await moving;
});

test('tilting retains orientation through translations and pointer movement without writing poses', async () => {
  const scene = new LiquidSimulation();
  await scene.controller.refresh();
  const observed = structuredClone(scene.poses);
  await scene.controller.tilt(Physics.TILT_STEP);
  const orientation = scene.controller.target.orientation.slice();
  const angle = Physics.TILT_STEP * Math.PI / 180;
  assert.deepEqual(orientation, [0, Math.sin(angle / 2), 0, Math.cos(angle / 2)]);
  await scene.controller.nudge([0, 0, Physics.LIFT_CLEARANCE]);
  scene.controller.beginDrag('tube');
  await scene.controller.dragTo([.3, 0, 1.2]);
  for (const command of scene.commands) assert.deepEqual(command.orientation, orientation);
  assert.deepEqual(scene.poses, observed);
  await scene.controller.upright();
  assert.deepEqual(scene.commands.at(-1).orientation, [0, 0, 0, 1]);
});

test('fill commands validate the measured capacity and serialize behind movement', async () => {
  const scene = new LiquidSimulation();
  await scene.controller.refresh();
  for (const volume of [-1, NaN, Infinity, scene.answer.liquid.tubes.tube.capacityMl + 1]) {
    assert.equal(await scene.controller.fillLiquid(volume), false);
  }
  let finish;
  scene.service.target = target => {
    scene.commands.push({type: 'target', ...target});
    return new Promise(resolve => { finish = resolve; });
  };
  const holding = scene.controller.hold('tube');
  const filling = scene.controller.fillLiquid(15);
  assert.equal(scene.commands.length, 1);
  finish({ok: true});
  await Promise.all([holding, filling]);
  assert.deepEqual(scene.commands.map(command => command.type), ['target', 'liquid']);
  assert.equal(scene.commands.at(-1).volumeMl, 15);
});

test('a liquid snapshot captured before filling cannot replace the confirmed new volume', async () => {
  const scene = new LiquidSimulation();
  await scene.controller.refresh();
  let finish;
  scene.service.state = () => new Promise(resolve => { finish = resolve; });
  const polling = scene.controller.refresh();
  const updated = structuredClone(scene.answer);
  updated.liquid.tubes.tube.volumeMl = 15;
  scene.service.fillLiquid = async () => updated;
  await scene.controller.fillLiquid(updated.liquid.tubes.tube.volumeMl);
  finish(scene.answer);
  await polling;
  assert.equal(scene.controller.liquid.tubes.tube.volumeMl, updated.liquid.tubes.tube.volumeMl);
});

test('a poll started while filling cannot restore liquid after its command completes', async () => {
  const scene = new LiquidSimulation();
  await scene.controller.refresh();
  let finishFill, finishPoll;
  scene.service.fillLiquid = () => new Promise(resolve => { finishFill = resolve; });
  const filling = scene.controller.fillLiquid(15);
  scene.service.state = () => new Promise(resolve => { finishPoll = resolve; });
  const polling = scene.controller.refresh();
  const updated = structuredClone(scene.answer);
  updated.liquid.tubes.tube.volumeMl = 15;
  finishFill(updated);
  await filling;
  finishPoll(scene.answer);
  await polling;
  assert.equal(scene.controller.liquid.tubes.tube.volumeMl, updated.liquid.tubes.tube.volumeMl);
});

test('reset prevents an in-flight fill response from painting discarded liquid', async () => {
  const scene = new LiquidSimulation();
  const rendered = [];
  scene.adapter.setLiquidState = liquid => rendered.push(liquid);
  await scene.controller.refresh();
  let finish;
  scene.service.fillLiquid = () => new Promise(resolve => { finish = resolve; });
  const filling = scene.controller.fillLiquid(15);
  const resetting = scene.controller.reset();
  const discarded = structuredClone(scene.answer);
  discarded.liquid.tubes.tube.volumeMl = 15;
  finish(discarded);
  await Promise.all([filling, resetting]);
  assert.deepEqual(rendered, [scene.answer.liquid]);
  await scene.controller.refresh();
  assert.equal(scene.controller.liquid.tubes.tube.volumeMl, scene.answer.liquid.tubes.tube.volumeMl);
});

test('liquid controls show measured volume and tilt, respect capacity and lock during robot motion', async () => {
  const scene = new LiquidSimulation();
  scene.config.robot = true;
  const angle = Math.PI / 6;
  scene.answer.objects.tube.splice(3, 4, 0, Math.sin(angle / 2), 0, Math.cos(angle / 2));
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  assert.equal(surface.find('laboratory-physics-liquid').hidden, false);
  assert.equal(Number(surface.find('laboratory-physics-volume').max), scene.answer.liquid.tubes.tube.capacityMl);
  assert.equal(surface.find('laboratory-physics-liquid-status').textContent, '12.0 / 22.0 ml · Ausgelaufen: 0.8 ml');
  assert.equal(surface.find('laboratory-physics-tilt-status').textContent, 'Neigung: 30°');
  surface.find('laboratory-physics-volume').value = '15';
  await surface.find('laboratory-physics-fill').listeners.get('click')();
  assert.deepEqual(scene.commands.at(-1), {type: 'liquid', key: 'tube', volumeMl: 15});
  scene.answer.robot = {state: 'running'};
  await mounted.refresh();
  for (const name of ['fill', 'volume', 'tilt-left', 'tilt-right', 'upright', 'lift-clear']) {
    assert.equal(surface.find('laboratory-physics-' + name).disabled, true);
  }
  assert.equal(await mounted.controller.fillLiquid(10), false);
  assert.equal(await mounted.controller.tilt(Physics.TILT_STEP), false);
  mounted.destroy();
});

test('liquid controls stay hidden for unsupported snapshots and non-tube objects', async () => {
  const scene = new ContactSimulation();
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  assert.equal(surface.find('laboratory-physics-liquid').hidden, true);
  scene.answer.liquid = {tubes: {another: {volumeMl: 2, capacityMl: 10}}, spilledMl: 0};
  await mounted.refresh();
  assert.equal(surface.find('laboratory-physics-liquid').hidden, true);
  assert.equal(await mounted.controller.fillLiquid(5), false);
  mounted.destroy();
});

test('the liquid lift button raises the force target before tilting at the same height', async () => {
  const scene = new LiquidSimulation();
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  await surface.find('laboratory-physics-lift-clear').listeners.get('click')();
  const expected = scene.answer.objects.tube.slice(0, 3);
  expected[2] += Physics.LIFT_CLEARANCE;
  assert.deepEqual(scene.commands.at(-1).position, expected);
  await surface.find('laboratory-physics-tilt-right').listeners.get('click')();
  assert.deepEqual(scene.commands.at(-1).position, expected);
  assert.equal(scene.commands.at(-1).orientation[1], Math.sin(Physics.TILT_STEP * Math.PI / 360));
  mounted.destroy();
});

test('the glass camera follows the measured lifted glass instead of its pending target', async () => {
  const scene = new LiquidSimulation();
  const focused = [];
  scene.adapter.focus = (position, target) => focused.push({position, target});
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  scene.answer.objects.tube = [.25, -.1, 1.19, 0, 0, 0, 1];
  await mounted.refresh();
  await mounted.controller.nudge([0, 0, .2]);
  await surface.find('laboratory-physics-view-glass').listeners.get('click')();
  const [x, y, z] = scene.answer.objects.tube;
  assert.deepEqual(focused, [{position: [x + .32, y - .42, z + .28], target: [x, y, z + .075]}]);
  mounted.destroy();
});

test('editing the fill amount survives polling and invalid amounts disable filling', async () => {
  const scene = new LiquidSimulation();
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  const volume = surface.find('laboratory-physics-volume');
  const fill = surface.find('laboratory-physics-fill');
  for (const value of ['', '-1', String(scene.answer.liquid.tubes.tube.capacityMl + 1)]) {
    volume.value = value;
    volume.listeners.get('input')();
    assert.equal(fill.disabled, true);
  }
  volume.value = '15';
  volume.listeners.get('input')();
  await mounted.refresh();
  assert.equal(volume.value, '15');
  assert.equal(fill.disabled, false);
  mounted.destroy();
});

test('mounted controls show contacts, issue physical commands and stop polling when removed', async () => {
  const scene = new ContactSimulation();
  const surface = new PhysicsControls();
  const clock = new PhysicsPolling();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, clock);
  await mounted.initialized;
  assert.equal(surface.find('laboratory-physics-hold').disabled, false);
  await surface.find('laboratory-physics-hold').listeners.get('click')();
  assert.equal(surface.find('laboratory-physics-release').disabled, false);
  scene.answer.contacts = [{position: [0, 0, .9], force: 1.25, bodyA: 'tube', bodyB: 'bench'}];
  scene.answer.target = structuredClone(mounted.controller.target);
  await mounted.refresh();
  assert.equal(surface.find('laboratory-physics-force').textContent, 'Kontakt: 1 · 1.25 N');
  await surface.find('laboratory-physics-release').listeners.get('click')();
  assert.equal(scene.commands.at(-1).type, 'release');
  assert.deepEqual(Array.from(clock.callbacks.values()).map(entry => entry.interval), [Physics.POLL_INTERVAL]);
  mounted.destroy();
  assert.equal(clock.callbacks.size, 0);
  assert.deepEqual(surface.host.children, []);
});

test('robot program controls stay idle on entry, show progress and disable conflicting glass controls', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  scene.answer.robot = {state: 'idle', phase: 'ready', progress: 0};
  scene.service.run = async () => { scene.commands.push({type: 'run'}); return {ok: true}; };
  scene.service.pause = async () => { scene.commands.push({type: 'pause'}); return {ok: true}; };
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  assert.deepEqual(scene.commands, []);
  const run = surface.find('laboratory-physics-robot-run');
  const pause = surface.find('laboratory-physics-robot-pause');
  assert.equal(run.disabled, false);
  assert.equal(pause.disabled, true);
  await run.listeners.get('click')();
  scene.answer.robot = {state: 'running', phase: 'lift', progress: .45};
  await mounted.refresh();
  assert.equal(surface.find('laboratory-physics-hold').disabled, true);
  assert.equal(surface.find('laboratory-physics-robot-progress').value, scene.answer.robot.progress);
  assert.equal(pause.disabled, false);
  await pause.listeners.get('click')();
  assert.deepEqual(scene.commands.map(command => command.type), ['run', 'pause']);
  assert.equal(surface.find('laboratory-physics-hold').disabled, false);
  mounted.destroy();
});

test('robot physics entry starts its own service before opening the resulting scene', async () => {
  const surface = new PhysicsControls();
  const navigation = [];
  const answer = {ok: true, viewerUrl: '/index.html?scene=precision_lab_pr2_physics&offline=1'};
  const mounted = Physics.mountStart(surface.host, {robot: true, start: async () => answer}, url => navigation.push(url));
  const button = surface.host.flatten().find(item => item.tagName === 'button');
  assert.equal(button.textContent, 'PR2 mit Kontaktphysik');
  await button.listeners.get('click')();
  assert.deepEqual(navigation, [answer.viewerUrl]);
  mounted.destroy();
});

test('robot feedback shows its physical phase without deriving success from completed joint targets', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  scene.answer.robot = {state: 'running', phase: 'settle', progress: 1};
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  assert.equal(surface.find('laboratory-physics-robot-status').textContent, 'PR2 bewegt das Glas · Stabilen Stand prüfen');
  assert.equal(surface.find('laboratory-physics-robot-run').disabled, true);
  scene.answer.robot = {state: 'failed', phase: 'settle', progress: 1, error: 'Glass did not settle'};
  await mounted.refresh();
  assert.equal(surface.find('laboratory-physics-robot-status').textContent, scene.answer.robot.error);
  mounted.destroy();
});

test('robot grip feedback displays measured contact force and maximum lift', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  scene.answer.robot = {state: 'running', phase: 'transport', progress: .7, gripperForce: .65, maximumLift: .14};
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  assert.equal(surface.find('laboratory-physics-robot-measurements').textContent, 'Greifkontakt: 0.65 N · Max. Hub: 14.0 cm');
  mounted.destroy();
});

test('robot holding feedback distinguishes physical grip from a released manual target', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  for (const state of ['running', 'paused']) {
    scene.answer.robot = {state, phase: 'transport', progress: .6, holding: true};
    await mounted.refresh();
    assert.equal(surface.find('laboratory-physics-distance').textContent, 'Vom PR2 gehalten · Kontaktphysik aktiv');
  }
  scene.answer.robot = {state: 'running', phase: 'approach', progress: .2, holding: false};
  await mounted.refresh();
  assert.equal(surface.find('laboratory-physics-distance').textContent, 'PR2-Auftrag aktiv · Kontaktphysik aktiv');
  scene.answer.robot = {state: 'idle', progress: 0, holding: false};
  await mounted.refresh();
  assert.equal(surface.find('laboratory-physics-distance').textContent, 'Losgelassen · Schwerkraft aktiv');
  mounted.destroy();
});

test('robot start explains the initial reset while a paused program offers resume', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  scene.answer.robot = {state: 'idle', phase: 'approach', progress: 0};
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  assert.equal(surface.find('laboratory-physics-robot-run').textContent, 'PR2: A1 → A3 starten');
  assert.match(surface.find('laboratory-physics-robot-reset-note').textContent, /Füllmengen bleiben erhalten/);
  scene.answer.robot = {state: 'paused', phase: 'transport', progress: .6};
  await mounted.refresh();
  assert.equal(surface.find('laboratory-physics-robot-run').textContent, 'PR2 fortsetzen');
  mounted.destroy();
});

for (const state of ['idle', 'failed']) {
  test('an inactive bookmarked physics page can start from ' + state, async () => {
    const scene = new ContactSimulation();
    scene.answer.state = state;
    const surface = new PhysicsControls();
    const clock = new PhysicsPolling();
    let finish;
    let starts = 0;
    scene.service.start = () => {
      starts += 1;
      return new Promise(resolve => { finish = resolve; });
    };
    const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, clock);
    await mounted.initialized;
    const start = surface.find('laboratory-physics-start');
    assert.ok(start, 'inactive scene needs an explicit start control');
    assert.equal(start.hidden, false);
    assert.equal(surface.find('laboratory-physics-hold').disabled, true);
    const starting = start.listeners.get('click')();
    assert.equal(start.disabled, true);
    await start.listeners.get('click')();
    assert.equal(starts, 1);
    scene.answer.state = 'running';
    finish(scene.answer);
    await starting;
    assert.equal(start.hidden, true);
    assert.equal(surface.find('laboratory-physics-hold').disabled, false);
    assert.deepEqual(scene.poses.at(-1).pose, scene.answer.objects.tube);
    mounted.destroy();
  });
}

test('a stopped or unreachable simulation exposes recovery instead of enabled manipulation', async () => {
  const scene = new ContactSimulation();
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  scene.service.state = async () => { throw new Error('Simulation stopped'); };
  await mounted.refresh();
  assert.equal(surface.find('laboratory-physics-hold').disabled, true);
  const start = surface.find('laboratory-physics-start');
  assert.equal(start.hidden, false);
  scene.service.start = async () => { throw new Error('Start failed'); };
  await start.listeners.get('click')();
  assert.equal(start.disabled, false);
  assert.equal(surface.find('laboratory-physics-status').textContent, 'Start failed');
  mounted.destroy();
});

test('a rejected target is visible and clears queued drag targets', async () => {
  const scene = new ContactSimulation();
  await scene.controller.refresh();
  let reject;
  scene.service.target = () => new Promise((resolve, failed) => { reject = failed; });
  scene.controller.beginDrag('tube');
  const pending = scene.controller.dragTo([.3, .2, .906]);
  reject(new Error('Simulation unavailable'));
  assert.equal(await pending, false);
  assert.equal(scene.controller.error, 'Simulation unavailable');
  assert.equal(scene.controller.target, null);
  assert.equal(scene.controller.dragged, null);
  assert.deepEqual(scene.controller.queue, []);
});

test('physics entry prevents duplicate starts and navigates to the confirmed viewer', async () => {
  const surface = new PhysicsControls();
  const navigation = [];
  let finish;
  let starts = 0;
  const mounted = Physics.mountStart(surface.host, {start() {
    starts += 1;
    return new Promise(resolve => { finish = resolve; });
  }}, url => navigation.push(url));
  const button = surface.host.flatten().find(item => item.tagName === 'button');
  const started = button.listeners.get('click')();
  await button.listeners.get('click')();
  assert.equal(starts, 1);
  const answer = {ok: true, viewerUrl: '/index.html?scene=physics&offline=1'};
  finish(answer);
  await started;
  assert.deepEqual(navigation, [answer.viewerUrl]);
  mounted.destroy();
  assert.deepEqual(surface.host.children, []);
});

// %% actual scene drag path
test('scale readings reach the scene and tare requests use the selected simulation route', async () => {
  for (const robot of [false, true]) {
    const requests = [];
    const client = new Physics.Client(async (url, options) => {
      requests.push({url, body: JSON.parse(options.body)});
      return {ok: true, json: async () => ({ok: true})};
    }, {robot});
    await client.tareScale();
    assert.deepEqual(requests, [{url: (robot ? Physics.RobotRoutes : Physics.Routes).TARE, body: {}}]);
  }
  const scene = new ContactSimulation();
  scene.answer.scale = {available: true, grams: 15.125, grossGrams: 15.125, tareGrams: 0, stable: true};
  const observed = [];
  scene.adapter.setScaleState = snapshot => observed.push(snapshot);
  await scene.controller.refresh();
  assert.deepEqual(observed, [scene.answer.scale]);
  assert.deepEqual(scene.controller.scale, scene.answer.scale);
});

test('a pre-tare poll cannot restore the previous scale reading', async () => {
  const scene = new ContactSimulation();
  scene.answer.scale = {available: true, grams: 15, grossGrams: 15, tareGrams: 0, stable: true};
  await scene.controller.refresh();
  let finish;
  scene.service.state = () => new Promise(resolve => { finish = resolve; });
  const polling = scene.controller.refresh();
  const tared = {...scene.answer.scale, grams: 0, tareGrams: 15};
  scene.service.tareScale = async () => ({ok: true, scale: tared});
  await scene.controller.tareScale();
  finish(scene.answer);
  await polling;
  assert.deepEqual(scene.controller.scale, tared);
});

test('scale controls display measured grams and permit tare only for settled readings', async () => {
  const scene = new ContactSimulation();
  scene.config.scale = {};
  scene.answer.scale = {available: true, grams: 15.125, grossGrams: 15.125, tareGrams: 0, stable: false};
  scene.service.tareScale = async () => {
    scene.commands.push({type: 'tareScale'});
    return {ok: true, scale: {...scene.answer.scale, grams: 0, tareGrams: 15.125}};
  };
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  assert.equal(surface.find('laboratory-physics-scale-reading').textContent, '15.125 g');
  const tare = surface.find('laboratory-physics-scale-tare');
  assert.equal(tare.disabled, true);
  assert.equal(await mounted.controller.tareScale(), false);
  scene.answer.scale.stable = true;
  await mounted.refresh();
  assert.equal(tare.disabled, false);
  await tare.listeners.get('click')();
  assert.deepEqual(scene.commands, [{type: 'tareScale'}]);
  assert.equal(surface.find('laboratory-physics-scale-reading').textContent, '0.000 g');
  mounted.destroy();
});

test('moving to the scale waits for observed waypoints before lowering and releasing', async () => {
  const scene = new ContactSimulation();
  scene.config.scale = {panPosition: [.65, .21, .974], panRadius: .085};
  await scene.controller.refresh();
  const initial = scene.answer.objects.tube.slice();
  await scene.controller.moveToScale();
  assert.equal(scene.commands.length, 1);
  assert.deepEqual(scene.commands[0].position.slice(0, 2), initial.slice(0, 2));
  assert.ok(scene.commands[0].position[2] >= initial[2] + Physics.LIFT_CLEARANCE);
  await scene.controller.refresh();
  assert.equal(scene.commands.length, 1, 'an unachieved lift must not advance horizontally');
  for (let stage = 0; stage < 4; stage += 1) {
    const desired = scene.commands.at(-1);
    assert.equal(desired.type, 'target');
    scene.answer.objects.tube = [...desired.position, ...desired.orientation];
    scene.answer.target = structuredClone(scene.controller.target);
    await scene.controller.refresh();
  }
  assert.equal(scene.commands.at(-1).type, 'release');
  assert.equal(scene.controller.scaleTransfer, null);
  assert.ok(scene.commands.every(command => command.type === 'target' || command.type === 'release'));
  assert.deepEqual(scene.commands.at(-2).position.slice(0, 2), scene.config.scale.panPosition.slice(0, 2));
});

test('an obstructed scale transfer stops without advancing or releasing into the obstacle', async () => {
  const scene = new ContactSimulation();
  scene.config.scale = {panPosition: [.65, .21, .974], panRadius: .085};
  await scene.controller.refresh();
  await scene.controller.moveToScale();
  scene.controller.scaleTransfer.startedAt -= Physics.SCALE_MOVE_TIMEOUT + 1;
  await scene.controller.refresh();
  assert.equal(scene.controller.scaleTransfer, null);
  assert.equal(scene.commands.length, 1);
  assert.ok(scene.controller.error);
});

test('manual input cancels a scale transfer and a running robot cannot start one', async () => {
  const scene = new ContactSimulation();
  scene.config.scale = {panPosition: [.65, .21, .974], panRadius: .085};
  await scene.controller.refresh();
  await scene.controller.moveToScale();
  await scene.controller.nudge([.01, 0, 0]);
  assert.equal(scene.controller.scaleTransfer, null);
  scene.config.robot = true;
  scene.controller.robot.state = Physics.RobotStates.RUNNING;
  assert.equal(await scene.controller.moveToScale(), false);
});

test('a blocked scale transfer stays explained until the user resumes manual movement', async () => {
  const scene = new ContactSimulation();
  scene.config.scale = {panPosition: [.65, .21, .974], panRadius: .085};
  await scene.controller.refresh();
  await scene.controller.moveToScale();
  scene.controller.scaleTransfer.startedAt -= Physics.SCALE_MOVE_TIMEOUT + 1;
  await scene.controller.refresh();
  const explanation = scene.controller.error;
  await scene.controller.refresh();
  assert.equal(scene.controller.error, explanation);
  await scene.controller.nudge([0, 0, .01]);
  assert.equal(scene.controller.error, null);
});

test('unreachable physics invalidates the displayed scale reading', async () => {
  const scene = new ContactSimulation();
  scene.answer.scale = {available: true, grams: 15, stable: true};
  const observed = [];
  scene.adapter.setScaleState = snapshot => observed.push(snapshot);
  await scene.controller.refresh();
  scene.service.state = async () => { throw new Error('offline'); };
  await scene.controller.refresh();
  assert.equal(scene.controller.scale, null);
  assert.equal(observed.at(-1), null);
});

test('a running robot prevents scale tare even if its current reading is stable', async () => {
  const scene = new ContactSimulation();
  scene.config.robot = true;
  scene.config.scale = {};
  scene.answer.robot = {state: Physics.RobotStates.RUNNING};
  scene.answer.scale = {available: true, grams: 0, tareGrams: 0, stable: true};
  scene.service.tareScale = async () => { scene.commands.push({type: 'tareScale'}); return {ok: true}; };
  const surface = new PhysicsControls();
  const mounted = Physics.mount(surface.host, scene.config, scene.service, scene.adapter, new PhysicsPolling());
  await mounted.initialized;
  assert.equal(surface.find('laboratory-physics-scale-tare').disabled, true);
  assert.equal(await mounted.controller.tareScale(), false);
  assert.deepEqual(scene.commands, []);
  mounted.destroy();
});

test('scale transport waits for measured motion to stop after reaching a waypoint', async () => {
  const scene = new ContactSimulation();
  scene.config.scale = {panPosition: [.65, .21, .974], panRadius: .085};
  scene.answer.time = 0;
  await scene.controller.refresh();
  await scene.controller.moveToScale();
  const desired = scene.commands[0];
  scene.answer.objects.tube = [...desired.position, ...desired.orientation];
  scene.answer.time = .033;
  await scene.controller.refresh();
  assert.equal(scene.commands.length, 1, 'arrival while still moving must not start the next waypoint');
  scene.answer.time = .066;
  await scene.controller.refresh();
  assert.equal(scene.commands.length, 2);
});

test('a rejected scale transport command cancels the remaining waypoint sequence', async () => {
  const scene = new ContactSimulation();
  scene.config.scale = {panPosition: [.65, .21, .974], panRadius: .085};
  await scene.controller.refresh();
  scene.service.target = async () => { throw new Error('Movement rejected'); };
  assert.equal(await scene.controller.moveToScale(), false);
  assert.equal(scene.controller.scaleTransfer, null);
});

test('a scale with a holder receives the object through measured contact before release', async () => {
  const scene = new ContactSimulation();
  scene.config.scale = {panPosition: [.65, .21, .974], panRadius: .085, holder: {height: .04}};
  await scene.controller.refresh();
  await scene.controller.moveToScale();
  for (let stage = 0; stage < 4; stage += 1) {
    const desired = scene.commands.at(-1);
    scene.answer.objects.tube = [...desired.position, ...desired.orientation];
    await scene.controller.refresh();
  }
  assert.equal(scene.commands.length, 4, 'arrival alone must not release above the support');
  assert.ok(scene.commands.at(-1).position[2] < scene.config.scale.panPosition[2]);
  scene.answer.scale = {available: true, objects: ['tube'], forceNewtons: .03};
  await scene.controller.refresh();
  assert.equal(scene.commands.at(-1).type, 'release');
});

test('a solid stopper is placed on the open pan beside the narrower vial holder', async () => {
  const scene = new ContactSimulation();
  scene.config.scale = {panPosition: [.65, .21, .974], panRadius: .085, holder: {innerRadius: .0095, outerRadius: .016, height: .04, segments: 24}};
  await scene.controller.refresh();
  await scene.controller.moveToScale();
  for (let stage = 0; stage < 2; stage += 1) {
    const desired = scene.commands.at(-1);
    scene.answer.objects.tube = [...desired.position, ...desired.orientation];
    await scene.controller.refresh();
  }
  const destination = scene.commands.at(-1).position;
  const offset = Math.hypot(...destination.slice(0, 2).map((value, axis) => value - scene.config.scale.panPosition[axis]));
  assert.ok(offset > scene.config.scale.holder.outerRadius);
  assert.ok(offset < scene.config.scale.panRadius);
});

test('the physics scene routes actual joints into the loaded robot', () => {
  const web = path.join(__dirname, '../../../cramera/src/cramera/web');
  const THREE = require(path.join(web, 'vendor/three.min.js'));
  const source = fs.readFileSync(path.join(web, 'panels/robot_scene/panel.js'), 'utf8');
  const values = [];
  const models = [{prefix: 'pr2', obj: {joints: {l_shoulder_pan_joint: {setJointValue: value => values.push(value)}}}}];
  const window = {};
  vm.runInNewContext(fs.readFileSync(path.join(web, 'core/joint-routing.js'), 'utf8'), {window});
  const context = vm.createContext({THREE, worldRoot: new THREE.Group(), objectMeshes: {}, models,
    JointRouting: window.JointRouting, syncJointControls() {}, focusSceneCamera() {}, needsRender: false});
  vm.runInContext(source.slice(source.indexOf('  function laboratoryPhysicsAdapter()'), source.indexOf('  RobotView.onReady(function () {', source.indexOf('  function laboratoryPhysicsAdapter()'))), context);
  const adapter = vm.runInContext('laboratoryPhysicsAdapter()', context);
  adapter.setJointFrames({'pr2/l_shoulder_pan_joint': .19, 'missing/joint': 0});
  assert.deepEqual(values, [.19]);
  adapter.destroy();
});

test('physics pointer moves send a desired target without editing or snapping the mesh', () => {
  const source = fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/panels/robot_scene/panel.js'), 'utf8');
  const moved = [];
  let handler;
  const object = {position: {x: 1, y: 2, z: 3}};
  const context = vm.createContext({
    renderer: {domElement: {addEventListener(name, callback) { handler = callback; }}},
    dragging: true, dragTarget: {name: 'tube'}, objectMeshes: {tube: object},
    laboratoryPhysics: {controller: {dragTo(position) { moved.push(Array.from(position)); }}},
    surfacePointAt: () => ({x: .4, y: .2, z: 1}), worldRoot: {worldToLocal() {}},
    snapToSurface() { throw new Error('physics must not snap'); }, liveOn: false,
    needsRender: false,
  });
  vm.runInContext(source.slice(source.indexOf("  renderer.domElement.addEventListener('pointermove'"), source.indexOf('  function endDrag()')), context);
  handler({});
  assert.deepEqual(moved, [[.4, .2, 1]]);
  assert.deepEqual(object.position, {x: 1, y: 2, z: 3});
});

test('grabbing a glass above its base does not move the physics target before the pointer moves', () => {
  const web = path.join(__dirname, '../../../cramera/src/cramera/web');
  const THREE = require(path.join(web, 'vendor/three.min.js'));
  const source = fs.readFileSync(path.join(web, 'panels/robot_scene/panel.js'), 'utf8');
  const worldRoot = new THREE.Group();
  worldRoot.rotation.x = -Math.PI / 2;
  const object = new THREE.Group();
  object.position.set(.16, -.065, .906);
  worldRoot.add(object);
  worldRoot.updateMatrixWorld(true);
  const camera = new THREE.PerspectiveCamera(45, 4 / 3, .01, 10);
  camera.position.set(1, 1.5, 1);
  camera.lookAt(object.getWorldPosition(new THREE.Vector3()));
  camera.updateMatrixWorld(true);
  const pointOnGlass = object.localToWorld(new THREE.Vector3(0, 0, .1)).project(camera);
  const dragNdc = new THREE.Vector2(pointOnGlass.x, pointOnGlass.y);
  const handlers = new Map();
  const targets = [];
  const context = vm.createContext({
    THREE, worldRoot, camera, dragNdc,
    renderer: {domElement: {
      style: {}, setPointerCapture() {}, addEventListener(name, callback) { handlers.set(name, callback); },
    }},
    pointerNdc() {}, pickDraggable: () => ({name: 'tube', group: object}),
    playing: false, SCENE: {physics: {}}, liveOn: false, guidedKeys: new Set(), controls: {},
    dragStartNdc: new THREE.Vector2(), dragStartWorld: new THREE.Vector3(),
    _ray: new THREE.Raycaster(), _dragPlane: new THREE.Plane(), _hitPt: new THREE.Vector3(),
    objectMeshes: {tube: object},
    laboratoryPhysics: {controller: {beginDrag: () => true, dragTo(position) { targets.push(Array.from(position)); }}},
  });
  vm.runInContext(source.slice(source.indexOf('  function surfacePointAt(e)'), source.indexOf('  function snapToSurface(')), context);
  vm.runInContext(source.slice(source.indexOf("  renderer.domElement.addEventListener('pointerdown'"), source.indexOf('  function endDrag()')), context);
  const event = {button: 0, clientX: 400, clientY: 300, pointerId: 1, preventDefault() {}};
  handlers.get('pointerdown')(event);
  handlers.get('pointermove')(event);
  assert.equal(targets.length, 1);
  for (let axis = 0; axis < 2; axis += 1) {
    assert.ok(Math.abs(targets[0][axis] - object.position.toArray()[axis]) < 1e-12);
  }
});
