'use strict';

// %% observed workflow fixture
const assert = require('node:assert/strict');
const test = require('node:test');
const Workflow = require('../../../cramera/src/cramera/web/core/laboratory-workflow.js');
const Physics = require('../../../cramera/src/cramera/web/core/laboratory-physics.js');

class WorkflowControl {
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

class WorkflowControls {
  constructor() { this.host = new WorkflowControl('div', this); }
  createElement(tag) { return new WorkflowControl(tag, this); }
  find(className) { return this.host.flatten().find(item => item.className === className); }
}

class ObservedWorkflow {
  constructor() {
    this.robot = {source: Workflow.Source, state: 'running', progress: .5, workflow: {
      stage: 'weigh', samples: [{key: 'tube_clear', label: 'Sample A1', doseAMl: 1, doseBMl: .98, weightGrams: null, location: 'scale'}],
    }};
    this.surface = new WorkflowControls();
    this.commands = [];
    this.view = new Workflow.View(this.surface.host, {
      start: () => this.commands.push('start'), pause: () => this.commands.push('pause'), reset: () => this.commands.push('reset'),
    });
    this.controls = {ready: true, canStart: false, canPause: true, pending: false};
  }
  render() { this.view.render(this.robot, this.controls); }
  row() { return this.surface.find('laboratory-workflow-samples').children[0]; }
}

test('sample metrics display only observed values and preserve unmeasured weight', () => {
  const scene = new ObservedWorkflow();
  scene.render();
  const row = scene.row();
  assert.equal(row.children[0].textContent, scene.robot.workflow.samples[0].label);
  assert.equal(row.children[1].textContent, 'A: 1.00 ml · B: 0.98 ml');
  assert.equal(row.children[2].textContent, 'Mass: — g');
  assert.equal(row.children[3].textContent, Workflow.Locations.scale);
  scene.robot.workflow.samples[0].weightGrams = 25.983;
  scene.render();
  assert.equal(scene.row().children[2].textContent, 'Mass: 25.983 g');
});

test('starting a transfer does not label an ungrasped sample as held', () => {
  const scene = new ObservedWorkflow();
  scene.robot.workflow.samples[0].location = 'robot';
  scene.robot.holding = false;
  scene.render();
  assert.equal(scene.row().children[3].textContent, 'PR2 transfer in progress');
});

test('supported grasp changes show their observed rack slot', () => {
  const scene = new ObservedWorkflow();
  scene.robot.workflow.samples[0].location = 'regrasp';
  scene.robot.workflow.samples[0].rackSlot = 'A1';
  scene.render();
  assert.equal(scene.row().children[3].textContent, 'At the grasp-change position · A1');
});

test('full progress does not invent success or completed stages', () => {
  const scene = new ObservedWorkflow();
  scene.robot.progress = 1;
  scene.render();
  assert.equal(scene.surface.find('laboratory-workflow-status').textContent, Workflow.Labels.running);
  assert.equal(scene.surface.find('laboratory-workflow-stage').textContent, Workflow.Stages.weigh);
  assert.equal(scene.surface.find('laboratory-workflow-progress').value, 1);
  assert.equal(scene.surface.find('laboratory-workflow').dataset.state, 'running');
  scene.robot.state = 'failed';
  scene.robot.error = 'Scale did not settle';
  scene.render();
  assert.equal(scene.surface.find('laboratory-workflow-status').textContent, scene.robot.error);
});

test('pause and reset remain explicit actions and resume preserves measured rows', () => {
  const scene = new ObservedWorkflow();
  scene.robot.state = 'paused';
  scene.controls.canStart = true;
  scene.controls.canPause = false;
  scene.render();
  const start = scene.surface.find('laboratory-workflow-start');
  assert.equal(start.textContent, Workflow.Buttons.RESUME);
  assert.equal(start.disabled, false);
  start.listeners.get('click')();
  scene.surface.find('laboratory-workflow-reset').listeners.get('click')();
  assert.deepEqual(scene.commands, ['start', 'reset']);
  assert.equal(scene.row().children[1].textContent, 'A: 1.00 ml · B: 0.98 ml');
});

test('new unrelated robot snapshots clear old workflow rows instead of carrying them forward', () => {
  const scene = new ObservedWorkflow();
  scene.render();
  scene.robot = {source: Physics.RobotSources.MIXING, state: 'running'};
  scene.render();
  assert.equal(scene.surface.find('laboratory-workflow-samples').children.length, 0);
  assert.equal(scene.surface.find('laboratory-workflow-progress').value, 0);
  assert.equal(scene.surface.find('laboratory-workflow-status').textContent, Workflow.Labels.idle);
});

test('offline or pending controls cannot claim a sample result and remove listeners on disposal', () => {
  const scene = new ObservedWorkflow();
  scene.controls = {ready: false, canStart: false, canPause: false, pending: true};
  scene.render();
  for (const name of ['start', 'pause', 'reset']) assert.equal(scene.surface.find('laboratory-workflow-' + name).disabled, true);
  const start = scene.surface.find('laboratory-workflow-start');
  scene.view.destroy();
  assert.equal(start.listeners.size, 0);
  assert.deepEqual(scene.surface.host.children, []);
});

// %% server commands and ownership
class WorkflowSimulation {
  constructor() {
    this.config = {robot: true, objects: [{key: 'tube_clear'}], bounds: {min: [-2, -2, 0], max: [2, 2, 2]}, refrigerator: {slots: [{id: 'R1'}]}};
    this.commands = [];
    this.answer = {state: 'running', objects: {tube_clear: [0, 0, 1, 0, 0, 0, 1]}, contacts: [],
      robot: {source: Workflow.Source, state: 'paused', progress: .8, workflow: {stage: 'store', samples: []}},
      refrigerator: {available: true, slots: [{id: 'R1', key: 'tube_clear'}]},
      centrifuge: {available: true, slots: [{id: 'C1', key: 'tube_amber', secured: true}]},
      pipetting: {available: true, state: 'idle'}, scale: {available: true},
    };
    this.service = {state: async () => this.answer,
      runWorkflow: async prepare => {this.commands.push({type: 'runWorkflow', prepare}); return {robot: {...this.answer.robot, state: 'running'}};},
      runMixing: async () => {this.commands.push('runMixing'); return {ok: true};},
      target: async () => {this.commands.push('target'); return {ok: true};},
    };
    this.controller = new Physics.Controller(this.config, this.service, {setObjectPose() {}});
  }
}

test('workflow start sends no browser object poses or pretend measurements', async () => {
  const requests = [];
  const client = new Physics.Client(async (url, options) => {
    requests.push({url, body: JSON.parse(options.body)});
    return {ok: true, json: async () => ({ok: true})};
  }, {robot: true});
  await client.runWorkflow();
  await client.runWorkflow(false);
  assert.deepEqual(requests, [{url: Physics.RobotRoutes.WORKFLOW, body: {prepare: true}}, {url: Physics.RobotRoutes.WORKFLOW, body: {prepare: false}}]);
});

test('paused workflow resumes through its own occupied instruments while keeping manual edits blocked', async () => {
  const scene = new WorkflowSimulation();
  await scene.controller.refresh();
  assert.equal(scene.controller.manualReady(), false);
  assert.equal(await scene.controller.runMixing(), false);
  assert.equal(await scene.controller.targetCommand('tube_clear', [0, 0, 1]), false);
  assert.equal(await scene.controller.runWorkflow(), true);
  assert.deepEqual(scene.commands, [{type: 'runWorkflow', prepare: false}]);
  assert.equal(scene.controller.robot.state, 'running');
});

test('a rejected resume keeps paused authoritative progress instead of inventing running state', async () => {
  const scene = new WorkflowSimulation();
  await scene.controller.refresh();
  scene.service.runWorkflow = async () => {throw new Error('Storage path is blocked');};
  const observed = structuredClone(scene.controller.robot);
  assert.equal(await scene.controller.runWorkflow(), false);
  assert.deepEqual(scene.controller.robot, observed);
  assert.equal(scene.controller.error, 'Storage path is blocked');
});

test('a rejected resume remains explained after ordinary paused polling until a successful retry', async () => {
  const scene = new WorkflowSimulation();
  await scene.controller.refresh();
  scene.service.runWorkflow = async () => {throw new Error('The paused dose changed; reset the workflow');};
  await scene.controller.runWorkflow();
  const rejected = scene.controller.error;
  await scene.controller.refresh();
  assert.equal(scene.controller.workflowError, rejected);
  scene.service.runWorkflow = async () => ({robot: {...scene.answer.robot, state: 'running'}});
  await scene.controller.runWorkflow();
  assert.equal(scene.controller.workflowError, null);
});

test('preparing a new workflow explicitly resets existing samples instead of invoking resume', async () => {
  const scene = new WorkflowSimulation();
  scene.answer.robot.state = 'succeeded';
  await scene.controller.refresh();
  await scene.controller.runWorkflow();
  assert.deepEqual(scene.commands, [{type: 'runWorkflow', prepare: true}]);
});

test('a paused workflow may resume its own active dose but a new workflow cannot interrupt manual dosing', async () => {
  const scene = new WorkflowSimulation();
  scene.answer.pipetting.state = 'dispensing';
  await scene.controller.refresh();
  assert.equal(scene.controller.workflowReady(), true);
  await scene.controller.runWorkflow();
  assert.deepEqual(scene.commands, [{type: 'runWorkflow', prepare: false}]);
  scene.answer.robot = {state: 'idle'};
  await scene.controller.refresh();
  assert.equal(scene.controller.workflowReady(), false);
});

test('an uncertain start reserves the samples until a fresh authoritative robot snapshot arrives', async () => {
  const scene = new WorkflowSimulation();
  scene.answer.robot = {state: 'idle'};
  await scene.controller.refresh();
  scene.service.runWorkflow = async () => ({ok: true});
  await scene.controller.runWorkflow();
  assert.equal(scene.controller.manualReady(), false);
  assert.equal(scene.controller.workflowReady(), false);
  assert.equal(scene.controller.robot.state, 'idle');
  await scene.controller.refresh();
  assert.equal(scene.controller.manualReady(), true);
});

// %% complete panel integration
class WorkflowClock {
  constructor() { this.callbacks = new Map(); }
  setInterval(callback) { const key = this.callbacks.size + 1;this.callbacks.set(key, callback);return key; }
  clearInterval(key) { this.callbacks.delete(key); }
}

test('the mounted panel exposes resume with occupied devices and disables incompatible demonstrations', async () => {
  const scene = new WorkflowSimulation();
  scene.answer.centrifuge = null;
  scene.answer.pipetting = null;
  scene.answer.refrigerator = null;
  scene.answer.scale = null;
  const controls = new WorkflowControls();
  const clock = new WorkflowClock();
  const mounted = Physics.mount(controls.host, scene.config, scene.service, {setObjectPose() {}}, clock);
  await mounted.initialized;
  assert.equal(controls.find('laboratory-workflow-start').textContent, Workflow.Buttons.RESUME);
  assert.equal(controls.find('laboratory-physics-robot-run').disabled, true);
  assert.equal(controls.find('laboratory-physics-robot-mix').disabled, true);
  assert.equal(controls.find('laboratory-physics-robot-pipette').disabled, true);
  assert.equal(controls.find('laboratory-physics-status').textContent, Workflow.PausedNote);
  mounted.destroy();
  assert.deepEqual(controls.host.children, []);
  assert.equal(clock.callbacks.size, 0);
});

test('instrument stop controls remain usable while a running workflow reserves all other device operations', async () => {
  const scene = new WorkflowSimulation();
  scene.answer.robot.state = 'running';
  scene.answer.refrigerator = {available: true, temperatureC: 4, doorClosed: true, slots: []};
  scene.answer.pipetting = {available: true, state: 'dispensing', reservoirs: [], requestedMl: 1, deliveredMl: .4};
  scene.answer.scale = {available: true, grams: 0, stable: true, tareGrams: 0};
  scene.answer.centrifuge = {available: true, slots: [], rpm: 120, rcf: 1, remainingSeconds: 5, running: true, maximumRpm: 300, lidClosed: true, balanced: true};
  scene.service.pipetting = async payload => {scene.commands.push({type: 'pipetting', ...payload});return {ok: true};};
  scene.service.centrifuge = async payload => {scene.commands.push({type: 'centrifuge', ...payload});return {ok: true};};
  const controls = new WorkflowControls();
  const mounted = Physics.mount(controls.host, scene.config, scene.service, {setObjectPose() {}}, new WorkflowClock());
  await mounted.initialized;
  const pumpStop = controls.find('laboratory-physics-pipetting-stop');
  const rotorStop = controls.find('laboratory-physics-centrifuge-stop');
  assert.equal(pumpStop.disabled, false);
  assert.equal(rotorStop.disabled, false);
  assert.equal(controls.find('laboratory-physics-refrigerator-open').disabled, true);
  assert.equal(controls.find('laboratory-physics-centrifuge-open').disabled, true);
  assert.equal(controls.find('laboratory-physics-pipetting-refill').disabled, true);
  await pumpStop.listeners.get('click')();
  await rotorStop.listeners.get('click')();
  assert.deepEqual(scene.commands, [{type: 'pipetting', action: 'stop'}, {type: 'centrifuge', action: 'stop'}]);
  mounted.destroy();
});
