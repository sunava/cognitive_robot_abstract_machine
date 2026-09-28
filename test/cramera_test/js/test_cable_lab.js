'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const CableLab = require('../../../cramera/src/cramera/web/cable-lab.js');

// %% physical commands
class CableTransport {
  constructor() {
    this.requests = [];
    this.answer = {ok: true, state: 'running', physical: true, segments: []};
    this.client = new CableLab.Client(async (url, options) => {
      this.requests.push({url, options});
      return {ok: true, json: async () => this.answer};
    });
  }
}

test('start sends total mass and contact settings without obsolete strategies', async () => {
  const transport = new CableTransport();
  const config = CableLab.Configuration.read({
    segment_count: '12', segment_length: '.04', radius: '.006', mass: '.08',
    bend: '1000', twist: '500', friction: '.2', anchor_to_parent: false,
    anchor_offset_x: '.1', anchor_offset_y: '.2', anchor_offset_z: '.3',
    anchor_rpy_x: '0', anchor_rpy_y: '.5', anchor_rpy_z: '1',
  });
  await transport.client.start(config);
  assert.deepEqual(JSON.parse(transport.requests[0].options.body), {
    segment_count: 12, segment_length: .04, radius: .006, mass: .08,
    bend: 1000, twist: 500, friction: [.2, .005, .0001], anchor_to_parent: false,
    anchor_offset: [.1, .2, .3], anchor_rpy: [0, .5, 1],
  });
  assert.equal(transport.requests[0].url, CableLab.Routes.START);
});

test('pinch and lift send commands without modifying a cable pose', async () => {
  const transport = new CableTransport();
  const observed = [];
  const controller = new CableLab.Controller(transport.client, state => observed.push(state));
  await controller.command(CableLab.Actions.PINCH);
  await controller.command(CableLab.Actions.LIFT);
  assert.deepEqual(transport.requests.map(request => JSON.parse(request.options.body)), [
    {action: CableLab.Actions.PINCH}, {action: CableLab.Actions.LIFT},
  ]);
  assert.deepEqual(observed, [transport.answer, transport.answer]);
});

test('late polling cannot overwrite a completed reset', async () => {
  const observed = [];
  let finish;
  const reset = {ok: true, state: 'running', simulation_time: 0};
  const client = {state: () => new Promise(resolve => { finish = resolve; }), reset: async () => reset};
  const controller = new CableLab.Controller(client, state => observed.push(state));
  const pending = controller.refresh();
  await controller.reset();
  finish({ok: true, state: 'running', simulation_time: 20});
  await pending;
  assert.deepEqual(observed, [reset]);
});

test('rejected backend commands expose their error instead of successful feedback', async () => {
  const transport = new CableTransport();
  transport.answer = {ok: false, error: 'Simulator is unavailable'};
  await assert.rejects(transport.client.command(CableLab.Actions.PINCH), {message: transport.answer.error});
});

test('reset and stop carry no browser-authored positions', async () => {
  const transport = new CableTransport();
  await transport.client.reset();
  await transport.client.stop();
  assert.deepEqual(transport.requests.map(request => request.url), [CableLab.Routes.RESET, CableLab.Routes.STOP]);
  assert.deepEqual(transport.requests.map(request => JSON.parse(request.options.body)), [{}, {}]);
});

test('placeholder fallback is distinguishable from a physical cable', () => {
  const feedback = CableLab.Feedback.read({state: 'running', physical: false, warnings: ['Cable compilation failed']});
  assert.equal(feedback.canCommand, false);
  assert.equal(feedback.status, 'Placeholders only');
  assert.equal(feedback.warning, 'Cable compilation failed');
});

test('partial or missing contact measurements are never reported as a stable grasp', () => {
  const feedback = CableLab.Feedback.read({state: 'running', physical: true});
  assert.equal(feedback.grasp, 'No contact data');
  assert.equal(feedback.normalForce, null);
  assert.equal(feedback.frictionForce, null);
});

test('state reads preserve failed snapshots for authoritative status display', async () => {
  const transport = new CableTransport();
  transport.answer = {ok: false, state: 'failed', error: 'Worker stopped'};
  assert.deepEqual(await transport.client.state(), transport.answer);
});

test('a retained idle simulation is displayed as stopped', () => {
  const feedback = CableLab.Feedback.read({state: 'idle', physical: true, config: {}});
  assert.equal(feedback.status, 'Physics stopped');
});

// %% canvas projection of measured geometry
test('orthographic view preserves separation of opposing gripper pads', () => {
  const left = CableLab.Projection.point([0, -.02, 0]);
  const right = CableLab.Projection.point([0, .02, 0]);
  assert.ok(left[0] < right[0]);
  assert.ok(left[1] > right[1]);
});

test('world height projects upward without changing horizontal position', () => {
  const base = CableLab.Projection.point([.1, .2, 0]);
  const lifted = CableLab.Projection.point([.1, .2, .1]);
  assert.equal(base[0], lifted[0]);
  assert.ok(lifted[1] < base[1]);
});

test('rotated measured box corners honor the server XYZW quaternion', () => {
  const box = {position: [1, 2, 3], size: [2, 4, 6], quaternion: [0, 0, Math.SQRT1_2, Math.SQRT1_2]};
  const corners = CableLab.Projection.corners(box);
  assert.equal(corners.length, 8);
  const rounded = corners.map(position => position.map(value => Math.round(value * 1e10) / 1e10));
  assert.deepEqual(rounded[0], [3, 1, 0]);
  assert.deepEqual(rounded[7], [-1, 3, 6]);
});

// %% form defaults
test('every numeric form default satisfies browser bounds and step validation', () => {
  const filesystem = require('node:fs');
  const path = require('node:path');
  const html = filesystem.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/cable-lab.html'), 'utf8');
  const inputs = [...html.matchAll(/<input\b[^>]*\btype="number"[^>]*>/g)];
  assert.ok(inputs.length > 0);
  for (const [input] of inputs) {
    const attributes = Object.fromEntries([...input.matchAll(/([\w-]+)="([^"]*)"/g)].map(match => [match[1], match[2]]));
    const value = Number(attributes.value);
    assert.ok(Number.isFinite(value), attributes.name);
    if (attributes.min !== undefined) assert.ok(value >= Number(attributes.min), attributes.name);
    if (attributes.max !== undefined) assert.ok(value <= Number(attributes.max), attributes.name);
    const steps = (value - Number(attributes.min || 0)) / Number(attributes.step || 1);
    assert.ok(Math.abs(steps - Math.round(steps)) < 1e-7, attributes.name + ' has a step mismatch');
  }
});

test('reconnecting uses the running cable configuration in the editable form', () => {
  const values = {
    segment_count: '8', segment_length: '.05', radius: '.008', mass: '.2',
    bend: '20', twist: '10', friction: '.3', anchor_to_parent: false,
    anchor_offset_x: '.2', anchor_offset_y: '0', anchor_offset_z: '.1',
    anchor_rpy_x: '0', anchor_rpy_y: '.5', anchor_rpy_z: '1',
  };
  const config = CableLab.Configuration.read(values);
  const form = {elements: Object.fromEntries(Object.keys(values).map(name => [name, {}]))};
  CableLab.Configuration.apply(form, config);
  const observed = Object.fromEntries(Object.keys(values).map(name => [name, name === 'anchor_to_parent' ? form.elements[name].checked : form.elements[name].value]));
  assert.deepEqual(CableLab.Configuration.read(observed), config);
});
