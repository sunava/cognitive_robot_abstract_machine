'use strict';

// %% independent instrument controls
const assert = require('node:assert/strict');
const test = require('node:test');
const Physics = require('../../../cramera/src/cramera/web/core/laboratory-physics.js');
const Workflow = require('../../../cramera/src/cramera/web/core/laboratory-workflow.js');

class InstrumentControl {
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

class InstrumentControls {
  constructor() { this.host = new InstrumentControl('div', this); }
  createElement(tag) { return new InstrumentControl(tag, this); }
  find(className) { return this.host.flatten().find(item => item.className === className); }
  control(instrument, action) { return this.find('laboratory-physics-' + instrument + '-robot-' + action); }
}

class InstrumentClock {
  setInterval() { return 1; }
  clearInterval() {}
}

class IndependentInstrumentSimulation {
  constructor() {
    this.commands = [];
    this.config = {robot: true, objects: [{key: 'tube_clear'}], bounds: {min: [-2, -2, 0], max: [2, 2, 2]},
      centrifuge: {}, refrigerator: {slots: [{id: 'R1'}]}};
    this.answer = {state: 'running', objects: {tube_clear: [0, 0, 1, 0, 0, 0, 1]}, contacts: [],
      robot: {state: 'idle'},
      refrigerator: {available: true, temperatureC: 4, doorClosed: true, slots: [{id: 'R1', key: 'tube_clear'}]},
      centrifuge: {available: true, slots: [{id: 'C1', key: 'tube_amber', secured: true, grams: 14}], rpm: 0, rcf: 0,
        remainingSeconds: 0, running: false, locked: false, maximumRpm: 300, lidClosed: true, balanced: true},
      pipetting: null, scale: null,
    };
    this.service = {state: async () => this.answer,
      runInstrument: async command => {
        this.commands.push(command);
        this.answer.robot = {source: Workflow.Source, state: 'running', instrument: command, progress: 0,
          message: 'Approaching the handle', workflow: {stage: 'open', samples: []}};
        return {robot: this.answer.robot};
      },
      runWorkflow: async prepare => {
        assert.equal(prepare, false);
        return this.service.runInstrument(this.answer.robot.instrument);
      },
    };
    this.controls = new InstrumentControls();
    this.mounted = Physics.mount(this.controls.host, this.config, this.service, {setObjectPose() {}}, new InstrumentClock());
    this.controller = this.mounted.controller;
  }
}

test('client sends only the selected device and handle operation to the dedicated route', async () => {
  const requests = [];
  const client = new Physics.Client(async (url, options) => {
    requests.push({url, body: JSON.parse(options.body)});
    return {ok: true, json: async () => ({ok: true})};
  }, {robot: true});
  for (const instrument of Object.values(Physics.RobotInstruments)) {
    for (const action of Object.values(Physics.InstrumentActions)) {
      await client.runInstrument({instrument, action});
    }
  }
  assert.deepEqual(requests, Object.values(Physics.RobotInstruments).flatMap(instrument =>
    Object.values(Physics.InstrumentActions).map(action => ({url: Physics.RobotRoutes.INSTRUMENT, body: {instrument, action}}))));
});

test('each device exposes its three standalone robot commands without preparing or emptying samples', async () => {
  for (const instrument of ['centrifuge', 'refrigerator']) {
    for (const action of ['open', 'close', 'open_close']) {
      const scene = new IndependentInstrumentSimulation();
      await scene.mounted.initialized;
      const before = structuredClone(scene.controller.objects);
      const control = scene.controls.control(instrument, action);
      assert.equal(control.disabled, false);
      assert.equal(control.textContent, Physics.InstrumentButtons[action]);
      await control.listeners.get('click')();
      assert.deepEqual(scene.commands, [{instrument, action}]);
      assert.deepEqual(scene.controller.objects, before);
      assert.equal(scene.controller.refrigerator.slots[0].key, 'tube_clear');
      assert.equal(scene.controller.centrifuge.slots[0].secured, true);
      assert.equal(scene.controller.manualReady(), false);
      scene.mounted.destroy();
    }
  }
});

test('pending handle requests immediately disable both devices and preserve the accepted busy state', async () => {
  const scene = new IndependentInstrumentSimulation();
  await scene.mounted.initialized;
  let finish;
  scene.service.runInstrument = command => new Promise(resolve => {finish = () => resolve({robot: {
    source: Workflow.Source, state: 'running', instrument: command, progress: 0}});});
  const starting = scene.controller.runInstrument('centrifuge', 'open_close');
  for (const instrument of ['centrifuge', 'refrigerator']) {
    assert.equal(scene.controls.control(instrument, 'open_close').disabled, true);
    assert.equal(await scene.controller.runInstrument(instrument, 'open'), false);
  }
  assert.match(scene.controls.control('centrifuge', 'status').textContent, /Starting/);
  finish();
  assert.equal(await starting, true);
  assert.equal(scene.controls.control('refrigerator', 'open').disabled, true);
  scene.mounted.destroy();
});

test('paused full workflows and manual transfers cannot be replaced by a handle test', async () => {
  const scene = new IndependentInstrumentSimulation();
  await scene.mounted.initialized;
  scene.answer.robot = {source: Workflow.Source, state: 'paused'};
  await scene.mounted.refresh();
  assert.equal(await scene.controller.runInstrument('centrifuge', 'open'), false);
  scene.answer.robot = {state: 'idle'};
  await scene.mounted.refresh();
  scene.controller.scaleTransfer = {key: 'tube_clear'};
  assert.equal(await scene.controller.runInstrument('refrigerator', 'open'), false);
  scene.controller.scaleTransfer = null;
  scene.controller.target = {key: 'tube_clear'};
  assert.equal(await scene.controller.runInstrument('refrigerator', 'open'), false);
  assert.deepEqual(scene.commands, []);
  scene.mounted.destroy();
});

test('robot centrifuge controls respect a running or locked rotor', async () => {
  const scene = new IndependentInstrumentSimulation();
  await scene.mounted.initialized;
  for (const rotor of [{rpm: 25}, {rpm: 0, locked: true}, {locked: false, running: true}]) {
    Object.assign(scene.answer.centrifuge, rotor);
    await scene.mounted.refresh();
    assert.equal(scene.controls.control('centrifuge', 'open').disabled, true);
    assert.equal(await scene.controller.runInstrument('centrifuge', 'open_close'), false);
  }
  assert.deepEqual(scene.commands, []);
  scene.mounted.destroy();
});

test('standalone completion never claims that the full sample workflow finished', async () => {
  const scene = new IndependentInstrumentSimulation();
  await scene.mounted.initialized;
  const command = {instrument: 'refrigerator', action: 'open_close'};
  scene.answer.robot = {source: Workflow.Source, instrument: command, state: 'succeeded', progress: 1,
    workflow: {stage: 'complete', samples: []}};
  await scene.mounted.refresh();
  assert.equal(scene.controls.find('laboratory-workflow-status').textContent, Workflow.Labels.idle);
  assert.equal(scene.controls.find('laboratory-workflow-progress').value, 0);
  assert.equal(scene.controls.find('laboratory-physics-robot-status').textContent, Physics.InstrumentLabels.succeeded);
  assert.equal(scene.controls.control('refrigerator', 'status').textContent, Physics.InstrumentLabels.succeeded);
  scene.mounted.destroy();
});

test('only the matching paused handle operation can resume and it keeps standalone labels', async () => {
  const scene = new IndependentInstrumentSimulation();
  await scene.mounted.initialized;
  const command = {instrument: 'centrifuge', action: 'open_close'};
  scene.answer.robot = {source: Workflow.Source, instrument: command, state: 'paused', progress: .5};
  await scene.mounted.refresh();
  assert.equal(scene.controls.find('laboratory-workflow-start').disabled, true);
  assert.equal(scene.controls.find('laboratory-workflow-start').textContent, Workflow.Buttons.START);
  assert.equal(scene.controls.control('centrifuge', 'open_close').disabled, false);
  assert.equal(scene.controls.control('centrifuge', 'open').disabled, true);
  assert.equal(scene.controls.control('refrigerator', 'open_close').disabled, true);
  await scene.controls.control('centrifuge', 'open_close').listeners.get('click')();
  assert.deepEqual(scene.commands, [command]);
  scene.mounted.destroy();
});

test('request errors stay visible beside the selected instrument after polling', async () => {
  const scene = new IndependentInstrumentSimulation();
  await scene.mounted.initialized;
  const message = 'The handle path is blocked';
  scene.service.runInstrument = async () => {throw new Error(message);};
  assert.equal(await scene.controller.runInstrument('refrigerator', 'open'), false);
  assert.equal(scene.controls.control('refrigerator', 'status').textContent, message);
  await scene.mounted.refresh();
  assert.equal(scene.controls.control('refrigerator', 'status').textContent, message);
  assert.equal(scene.controls.control('refrigerator', 'status').dataset.error, 'true');
  scene.mounted.destroy();
});

test('a full recipe opening its centrifuge remains a full workflow despite reporting the hinge name', async () => {
  const scene = new IndependentInstrumentSimulation();
  await scene.mounted.initialized;
  scene.answer.robot = {source: Workflow.Source, instrument: 'centrifuge', state: 'running', progress: .5,
    workflow: {stage: 'open', stageLabel: 'Open centrifuge lid', samples: []}};
  await scene.mounted.refresh();
  assert.equal(scene.controls.find('laboratory-workflow-status').textContent, Workflow.Labels.running);
  assert.equal(scene.controls.find('laboratory-workflow-progress').value, scene.answer.robot.progress);
  assert.equal(scene.controls.find('laboratory-physics-robot-status').textContent, Workflow.Labels.running);
  scene.mounted.destroy();
});

test('other device controls explain that the robot is occupied by the current handle test', async () => {
  const scene = new IndependentInstrumentSimulation();
  await scene.mounted.initialized;
  await scene.controller.runInstrument('centrifuge', 'open_close');
  assert.match(scene.controls.control('refrigerator', 'status').textContent, /another task/);
  scene.mounted.destroy();
});

test('a response without authoritative handle status keeps both device commands locked until a fresh observation', async () => {
  const scene = new IndependentInstrumentSimulation();
  await scene.mounted.initialized;
  scene.service.runInstrument = async () => ({ok: true});
  assert.equal(await scene.controller.runInstrument('refrigerator', 'open'), false);
  assert.equal(scene.controller.manualReady(), false);
  assert.equal(await scene.controller.runInstrument('centrifuge', 'open'), false);
  await scene.mounted.refresh();
  assert.equal(scene.controller.manualReady(), true);
  assert.equal(scene.controls.control('centrifuge', 'open').disabled, false);
  scene.mounted.destroy();
});

test('a stale idle poll cannot replace an accepted standalone robot operation', async () => {
  const scene = new IndependentInstrumentSimulation();
  await scene.mounted.initialized;
  const idle = structuredClone(scene.answer);
  let observe;
  scene.service.state = () => new Promise(resolve => {observe = resolve;});
  const polling = scene.mounted.refresh();
  await scene.controller.runInstrument('centrifuge', 'open_close');
  const running = structuredClone(scene.controller.robot);
  observe(idle);
  await polling;
  assert.deepEqual(scene.controller.robot, running);
  assert.equal(scene.controller.manualReady(), false);
  scene.mounted.destroy();
});
