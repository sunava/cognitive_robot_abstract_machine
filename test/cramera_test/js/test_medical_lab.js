'use strict';

// %% scene-scoped instrument controls
const assert = require('node:assert/strict');
const test = require('node:test');
const MedicalLab = require('../../../cramera/src/cramera/web/core/medical-lab.js');
const Physics = require('../../../cramera/src/cramera/web/core/laboratory-physics.js');

class MedicalControl {
  constructor(tag, document) {
    this.tagName = tag; this.ownerDocument = document; this.children = [];
    this.listeners = new Map(); this.attributes = new Map(); this.dataset = {};
  }
  appendChild(child) {
    if (child.parent) child.parent.children = child.parent.children.filter(item => item !== child);
    child.parent = this; this.children.push(child); return child;
  }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(item => item !== this); }
  setAttribute(name, value) { this.attributes.set(name, value); }
  removeAttribute(name) { this.attributes.delete(name); }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  removeEventListener(name) { this.listeners.delete(name); }
  flatten() { return [this, ...this.children.flatMap(child => child.flatten())]; }
}
class MedicalDocument {
  constructor() { this.host = new MedicalControl('div', this); }
  createElement(tag) { return new MedicalControl(tag, this); }
  find(className) { return this.host.flatten().find(item => item.className === className); }
}

function medicalScene() {
  return {stations: [{id: 'measurement', label: 'Measurement', camera: {position: [1, 2, 3], target: [0, 1, 1]}}],
    instruments: [{id: 'digital_microscope', label: 'Digital microscope', station: 'measurement',
      joints: [{name: 'microscope_focus', type: 'prismatic', rest: 0, inspection: .01}]},
    {id: 'ph_probe', label: 'pH probe', station: 'measurement', joints: []}]};
}

test('mechanism controls send bounded selected commands and focus declared station cameras', () => {
  const document = new MedicalDocument(), commands = [], focused = [], config = medicalScene();
  const view = new MedicalLab.View(document.host, config, {devices: command => commands.push(command),
    focus: (position, target) => focused.push({position, target}), robot() {}, pauseRobot() {}});
  document.find('medical-lab-device').value = 'digital_microscope';
  document.find('medical-lab-device').listeners.get('change')();
  document.find('medical-lab-inspect').listeners.get('click')();
  document.find('medical-lab-rest').listeners.get('click')();
  document.find('medical-lab-run').listeners.get('click')();
  document.find('medical-lab-pause').listeners.get('click')();
  document.find('medical-lab-reset').listeners.get('click')();
  document.find('medical-lab-focus').listeners.get('click')();
  assert.deepEqual(commands, [
    {action: MedicalLab.Actions.OPEN, device: 'digital_microscope'},
    {action: MedicalLab.Actions.CLOSE, device: 'digital_microscope'},
    {action: MedicalLab.Actions.RUN, device: 'digital_microscope'},
    {action: MedicalLab.Actions.PAUSE}, {action: MedicalLab.Actions.RESET}]);
  assert.deepEqual(focused, [config.stations[0].camera]);
  view.destroy(); assert.equal(document.host.children.length, 0);
});

test('displayed positions come only from observed joint values and static accessories have no motor controls', () => {
  const document = new MedicalDocument();
  const view = new MedicalLab.View(document.host, medicalScene(), {devices() {}, focus() {}, robot() {}, pauseRobot() {}});
  const controls = {ready: true, canCommand: () => true, robotReady: {}, robot: {state: 'idle'}};
  view.render({demo: {state: 'running', device: 'all', action: 'Inspect microscope', completed: 1, total: 4},
    joints: {microscope_focus: .003}, devices: []}, controls);
  assert.equal(document.find('medical-lab-joint-value').textContent, '3.0 mm');
  assert.equal(document.find('medical-lab-progress').value, .25);
  view.render({demo: {state: 'paused', device: 'all', completed: 1, total: 4}, joints: {}, devices: []}, controls);
  assert.equal(document.find('medical-lab-joint-value').textContent, 'Not reported');
  document.find('medical-lab-device').value = 'ph_probe';
  document.find('medical-lab-device').listeners.get('change')();
  assert.equal(document.find('medical-lab-inspect').disabled, true);
  assert.equal(document.find('medical-lab-run').disabled, true);
  assert.match(document.find('medical-lab-description').textContent, /no driven joints/);
  view.destroy();
});

test('client uses the protected laboratory route for instrument commands', async () => {
  const requests = [];
  const client = new Physics.Client(async (url, options) => {
    requests.push({url, command: JSON.parse(options.body)});
    return {ok: true, json: async () => ({ok: true})};
  }, {robot: true});
  const command = {action: MedicalLab.Actions.RUN, device: 'digital_microscope'};
  await client.devices(command);
  assert.deepEqual(requests, [{url: Physics.RobotRoutes.DEVICES, command}]);
});

test('in-flight and running mechanism sequences own the controls until observed completion', async () => {
  let finish;
  const config = {medicalLab: medicalScene(), robot: true, objects: [], bounds: {min: [-1, -1, 0], max: [1, 1, 2]}};
  const accepted = {demo: {state: 'running', device: 'all', completed: 0, total: 2}, joints: {}, devices: []};
  const service = {devices: () => new Promise(resolve => { finish = () => resolve({instruments: accepted}); })};
  const controller = new Physics.Controller(config, service, {}, () => {});
  controller.state = 'running'; controller.instruments = {demo: {state: 'idle'}, joints: {}, devices: []};
  const requested = controller.runDevices({action: 'run', device: 'all'});
  assert.equal(controller.devicesBusy(), true);
  assert.equal(controller.manualReady(), false);
  assert.equal(await controller.runDevices({action: 'open', device: 'digital_microscope'}), false);
  finish(); assert.equal(await requested, true);
  assert.equal(controller.instruments, accepted);
  assert.equal(await controller.runMixing(), false);
  assert.equal(controller.deviceReady('pause'), true);
  controller.instruments = {demo: {state: 'complete'}, joints: {}, devices: []};
  assert.equal(controller.manualReady(), true);
  controller.destroy();
});

test('a robot task or held manual target prevents new instrument movements', async () => {
  const controller = new Physics.Controller({medicalLab: medicalScene(), robot: true, objects: []}, {}, {}, () => {});
  controller.state = 'running'; controller.instruments = {demo: {state: 'idle'}, joints: {}, devices: []};
  for (const state of ['running', 'paused']) {
    controller.robot = {state};
    assert.equal(await controller.runDevices({action: 'run', device: 'all'}), false);
  }
  controller.robot = {state: 'idle'}; controller.target = {key: 'tube_clear'};
  assert.equal(await controller.runDevices({action: 'run', device: 'all'}), false);
  controller.destroy();
});

test('late polls cannot replace an accepted instrument command or its measured joint position', async () => {
  let resolvePoll;
  const frames = [];
  const config = {medicalLab: medicalScene(), robot: true, objects: []};
  const running = {demo: {state: 'running', device: 'all'}, joints: {microscope_focus: .004}};
  const idle = {demo: {state: 'idle'}, joints: {microscope_focus: 0}};
  const controller = new Physics.Controller(config, {
    state: () => new Promise(resolve => { resolvePoll = resolve; }),
    devices: async () => ({instruments: running}),
  }, {setJointFrames: measured => frames.push(measured)}, () => {});
  controller.state = 'running'; controller.instruments = idle;
  const stale = controller.refresh();
  await controller.runDevices({action: 'run', device: 'all'});
  resolvePoll({state: 'running', instruments: idle});
  await stale;
  assert.equal(controller.instruments, running);
  assert.deepEqual(frames, [running.joints]);
  assert.equal(controller.manualReady(), false);
  controller.destroy();
});

test('an uncertain command keeps other tasks blocked until fresh observed state arrives', async () => {
  const controller = new Physics.Controller({medicalLab: medicalScene(), robot: true, objects: []}, {
    devices: async () => { throw new Error('Connection interrupted'); },
    state: async () => ({state: 'running', instruments: {demo: {state: 'idle'}, joints: {}}}),
  }, {}, () => {});
  controller.state = 'running'; controller.instruments = {demo: {state: 'idle'}, joints: {}};
  assert.equal(await controller.runDevices({action: 'run', device: 'all'}), false);
  assert.equal(controller.manualReady(), false);
  assert.equal(await controller.runMixing(), false);
  assert.equal(controller.deviceReady('run', 'all'), false);
  await controller.refresh();
  assert.equal(controller.manualReady(), true);
  assert.equal(controller.deviceReady('run', 'all'), true);
  assert.equal(controller.deviceError, 'Connection interrupted');
  controller.destroy();
});

test('expanded scene mounts English demos and renders actual mechanism movement', async () => {
  const document = new MedicalDocument(), cameras = [], frames = [], commands = [];
  let observed = {demo: {state: 'idle'}, joints: {microscope_focus: .002}, devices: []};
  const mounted = Physics.mount(document.host, {robot: true, objects: [], medicalLab: medicalScene()}, {
    state: async () => ({state: 'running', instruments: observed}),
    devices: async command => {
      commands.push(command);
      observed = {demo: {state: 'running', device: command.device}, joints: {microscope_focus: .003}};
      return {instruments: observed};
    },
  }, {focus: (position, target) => cameras.push({position, target}), setJointFrames: value => frames.push(value)},
  {setInterval: () => 1, clearInterval() {}});
  await mounted.initialized;
  assert.ok(document.find('medical-lab-legacy'));
  assert.equal(document.find('medical-lab-run').disabled, false);
  assert.equal(document.find('medical-lab-joint-value').textContent, '2.0 mm');
  await document.find('medical-lab-run').listeners.get('click')();
  assert.deepEqual(commands, [{action: 'run', device: 'all'}]);
  assert.equal(document.find('medical-lab-joint-value').textContent, '3.0 mm');
  assert.deepEqual(frames, [{microscope_focus: .002}, {microscope_focus: .003}]);
  document.find('medical-lab-focus').listeners.get('click')();
  assert.deepEqual(cameras, [medicalScene().stations[0].camera]);
  assert.equal(document.find('laboratory-physics-robot-run').disabled, true);
  mounted.destroy();
  assert.equal(document.host.children.length, 0);
});
