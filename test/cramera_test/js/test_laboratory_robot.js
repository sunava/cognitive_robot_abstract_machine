'use strict';

// %% asynchronous robot execution
const assert = require('node:assert/strict');
const test = require('node:test');
const LaboratoryRobot = require('../../../cramera/src/cramera/web/core/laboratory-robot.js');

class TransferService {
  constructor() {
    this.starts = 0;
    this.statuses = 0;
    this.answer = {ok: true, state: LaboratoryRobot.States.RUNNING, liveUrl: '/index.html?layout=scene'};
  }
  async start() { this.starts += 1; return this.answer; }
  async status() { this.statuses += 1; return this.answer; }
}

test('a pending request prevents a repeated launch', async () => {
  const service = new TransferService();
  let finish;
  service.start = () => { service.starts += 1; return new Promise(resolve => { finish = resolve; }); };
  const controller = new LaboratoryRobot.Controller(service);
  const first = controller.start();
  assert.equal(controller.state, LaboratoryRobot.States.PENDING);
  await controller.start();
  assert.equal(service.starts, 1);
  finish(service.answer);
  await first;
  assert.equal(controller.state, LaboratoryRobot.States.RUNNING);
  assert.equal(controller.busy(), true);
});

test('polling cannot turn a running process into success without a success response', async () => {
  const service = new TransferService();
  const controller = new LaboratoryRobot.Controller(service);
  await controller.start();
  await controller.refresh();
  assert.equal(controller.state, LaboratoryRobot.States.RUNNING);
  assert.equal(controller.recordingUrl, null);
  service.answer = {ok: false, state: LaboratoryRobot.States.FAILED, error: 'IK failed', log: 'unreachable grasp'};
  await controller.refresh();
  assert.equal(controller.state, LaboratoryRobot.States.FAILED);
  assert.equal(controller.error, service.answer.error);
  assert.equal(controller.log, service.answer.log);
});

test('only a successful server outcome enables recorded replay', async () => {
  const service = new TransferService();
  const controller = new LaboratoryRobot.Controller(service);
  await controller.start();
  service.answer = {ok: true, state: LaboratoryRobot.States.SUCCEEDED, recordingUrl: '/index.html?scene=precision_lab_pr2'};
  await controller.refresh();
  assert.equal(controller.state, LaboratoryRobot.States.SUCCEEDED);
  assert.equal(controller.recordingUrl, service.answer.recordingUrl);
  assert.equal(controller.busy(), false);
});

test('a network failure reports uncertain execution and preserves the duplicate-start guard', async () => {
  const service = new TransferService();
  const controller = new LaboratoryRobot.Controller(service);
  service.start = async () => { service.starts += 1; throw new Error('connection closed'); };
  await controller.start();
  assert.equal(controller.state, LaboratoryRobot.States.UNKNOWN);
  assert.equal(controller.busy(), true);
  await controller.start();
  assert.equal(service.starts, 1);
  service.answer = {ok: true, state: LaboratoryRobot.States.IDLE};
  await controller.refresh();
  assert.equal(controller.state, LaboratoryRobot.States.IDLE);
  assert.equal(controller.busy(), false);
});

test('the client posts no object positions or executable text', async () => {
  const requests = [];
  const client = new LaboratoryRobot.Client(async (url, options) => {
    requests.push({url, options});
    return {ok: true, json: async () => ({ok: true, state: LaboratoryRobot.States.RUNNING})};
  });
  await client.start();
  assert.equal(requests[0].url, LaboratoryRobot.Routes.START);
  assert.equal(requests[0].options.body, '{}');
  assert.equal(requests[0].options.method, 'POST');
  await client.status();
  assert.equal(requests[1].url, LaboratoryRobot.Routes.STATUS);
  assert.equal(requests[1].options.method, 'GET');
});

// %% mounted operation status
class RobotControl {
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
class RobotControls {
  constructor() { this.host = new RobotControl('div', this); }
  createElement(tag) { return new RobotControl(tag, this); }
  find(className) { return this.host.flatten().find(item => item.className === className); }
}
class PollingClock {
  constructor() { this.callbacks = new Map(); this.count = 0; }
  setInterval(callback) { this.count += 1; this.callbacks.set(this.count, callback); return this.count; }
  clearInterval(timer) { this.callbacks.delete(timer); }
}

test('mounted operation shows failure details and cancels polling on destruction', async () => {
  const service = new TransferService();
  service.answer = {ok: true, state: LaboratoryRobot.States.IDLE};
  const surface = new RobotControls();
  const clock = new PollingClock();
  const mounted = LaboratoryRobot.mount(surface.host, service, clock);
  await mounted.initialized;
  const button = surface.find('laboratory-robot-start');
  assert.equal(button.disabled, false);
  service.answer = {ok: true, state: LaboratoryRobot.States.RUNNING};
  await button.listeners.get('click')();
  assert.equal(button.disabled, true);
  service.answer = {ok: false, state: LaboratoryRobot.States.FAILED, error: 'Plan failed', log: 'Traceback: test failure'};
  await mounted.refresh();
  assert.equal(surface.find('laboratory-robot-status').textContent, service.answer.error);
  assert.equal(surface.find('laboratory-robot-log').textContent, service.answer.log);
  assert.equal(surface.find('laboratory-robot-replay').hidden, true);
  assert.equal(button.disabled, false);
  mounted.destroy();
  assert.equal(clock.callbacks.size, 0);
  assert.deepEqual(surface.host.children, []);
});

// %% guarded live-view entry
class LaboratoryLiveService extends TransferService {
  constructor() {
    super();
    this.live = {running: true, robot: 'PR2', objects: ['tube_clear']};
    this.bundle = {scene: '__live__'};
    this.prepared = 0;
  }
  async info() { return this.live; }
  async prepareLive() { this.prepared += 1; return this.bundle; }
}

test('live entry bundles the identified laboratory before opening the viewer', async () => {
  const service = new LaboratoryLiveService();
  const navigation = [];
  const entry = new LaboratoryRobot.LiveEntry(service, url => navigation.push(url));
  await entry.refresh();
  assert.equal(service.prepared, 1);
  assert.deepEqual(navigation, [LaboratoryRobot.Live.VIEWER_URL]);
});

test('live entry never attaches to another robot or a world without the laboratory tube', async () => {
  for (const info of [
    {running: true, robot: 'PR2', objects: ['milk']},
    {running: true, robot: 'Stretch', objects: ['tube_clear']},
    {running: false, robot: 'PR2', objects: ['tube_clear']},
  ]) {
    const service = new LaboratoryLiveService();
    service.live = info;
    const navigation = [];
    const entry = new LaboratoryRobot.LiveEntry(service, url => navigation.push(url));
    await entry.refresh();
    assert.equal(service.prepared, 0);
    assert.deepEqual(navigation, []);
  }
});

test('a completed run opens its recorded result instead of attaching another live world', async () => {
  const service = new LaboratoryLiveService();
  service.answer = {ok: true, state: LaboratoryRobot.States.SUCCEEDED, recordingUrl: '/index.html?scene=precision_lab_pr2'};
  const navigation = [];
  const entry = new LaboratoryRobot.LiveEntry(service, url => navigation.push(url));
  await entry.refresh();
  assert.deepEqual(navigation, [service.answer.recordingUrl]);
  assert.equal(service.prepared, 0);
});

// %% standalone control page without Three.js
test('standalone controls launch through the shared client and release their polling timer', async () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const vm = require('node:vm');
  const surface = new RobotControls();
  const clock = new PollingClock();
  const requests = [];
  const handlers = new Map();
  const window = {
    fetch: async (url, options) => {
      requests.push({url, options});
      const state = options.method === 'POST' ? LaboratoryRobot.States.RUNNING : LaboratoryRobot.States.IDLE;
      return {ok: true, json: async () => ({ok: true, state})};
    },
    setInterval: callback => clock.setInterval(callback),
    clearInterval: timer => clock.clearInterval(timer),
    addEventListener: (event, callback) => handlers.set(event, callback),
  };
  const script = fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/laboratory-pr2.js'), 'utf8');
  vm.runInNewContext(script, {window, document: {getElementById: () => surface.host}, LaboratoryRobot});
  await new Promise(setImmediate);
  const button = surface.find('laboratory-robot-start');
  assert.equal(button.disabled, false);
  await button.listeners.get('click')();
  assert.equal(button.disabled, true);
  assert.equal(requests.filter(request => request.url === LaboratoryRobot.Routes.START).length, 1);
  handlers.get('pagehide')();
  assert.equal(clock.callbacks.size, 0);
  assert.deepEqual(surface.host.children, []);
});
