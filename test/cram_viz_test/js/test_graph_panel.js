// Unit tests for panels/graph/panel.js (node:test) against a stubbed DOM, bus,
// fetch and window.Graph: tab loading/switching, the knowledge-tab reuse of
// kb:ready, view-route 404 handling, and specifically the two-overlapping-
// showTab()-calls race that setView()'s targetTab guard fixes.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cram_viz', 'src', 'cram_viz', 'web');

// ---- stubs -----------------------------------------------------------------
function makeElement() {
  const el = {
    _html: '',
    _children: {},
    _buttons: null,
    _listeners: {},
    style: {},
    children: [],
    dataset: {},
    get innerHTML() { return this._html; },
    set innerHTML(value) { this._html = value; this._children = {}; this._buttons = null; },
    get textContent() { return this._text || ''; },
    set textContent(value) { this._text = value; },
    querySelector(selector) {
      const match = /^#([\w-]+)$/.exec(selector);
      if (!match) return null;
      if (!this._children[match[1]]) {
        const child = makeElement();
        child._html = this._html;   // let a descendant's querySelectorAll('button') see the same markup
        this._children[match[1]] = child;
      }
      return this._children[match[1]];
    },
    // production code only ever calls this on the tabs container looking for
    // 'button' — enough to fake it by scanning the injected markup for the
    // data-view attributes the real template writes
    querySelectorAll(selector) {
      if (selector !== 'button') return [];
      if (!this._buttons) {
        this._buttons = [];
        const re = /data-view="([\w-]+)"/g;
        let match;
        while ((match = re.exec(this._html))) {
          const button = makeElement();
          button.dataset.view = match[1];
          this._buttons.push(button);
        }
      }
      return this._buttons;
    },
    addEventListener(event, cb) { (this._listeners[event] = this._listeners[event] || []).push(cb); },
    dispatch(event, payload) { (this._listeners[event] || []).forEach(function (cb) { cb(payload || {}); }); },
    classList: { add() {}, remove() {}, toggle() {} },
  };
  return el;
}

function makeBus() {
  const handlers = {};
  return {
    on(event, cb) { (handlers[event] = handlers[event] || []).push(cb); },
    emit(event, payload) { (handlers[event] || []).forEach(function (cb) { cb(payload); }); },
    fire(event, payload) { (handlers[event] || []).forEach(function (cb) { cb(payload); }); },
  };
}

function makeGraphStub() {
  const builds = [];
  let selectCb = function () {};
  let dblCb = function () {};
  return {
    builds: builds,
    mount() {},
    build(data) { builds.push(data); },
    setStatuses() { return true; },
    highlight() {},
    reset() {},
    onSelect(cb) { selectCb = cb; },
    onDoubleSelect(cb) { dblCb = cb; },
    fireSelect(id) { selectCb(id); },
    fireDoubleSelect(id) { dblCb(id); },
  };
}

function loadPanelFactory() {
  global.window = {};
  global.Panels = { _factories: {}, define(id, factory) { this._factories[id] = factory; } };
  global.Graph = makeGraphStub();
  new Function(fs.readFileSync(path.join(WEB, 'panels/graph/panel.js'), 'utf8'))();
  return global.Panels._factories.graph;
}

async function settle() {
  await new Promise(function (resolve) { setImmediate(resolve); });
  await new Promise(function (resolve) { setImmediate(resolve); });
}

function jsonResponse(body, status) {
  return Promise.resolve({ status: status || 200, json: function () { return Promise.resolve(body); } });
}

function knowledgePayload(extra) {
  return Object.assign({ ok: true, nodes: [{ id: 'a', label: 'A', group: 'robot' }], edges: [], details: { a: { label: 'A' } } }, extra || {});
}

function clickTab(root, name) {
  root.querySelector('#graph-tabs').querySelectorAll('button')
    .filter(function (b) { return b.dataset.view === name; })[0]
    .dispatch('click');
}

async function mount(fetchImpl) {
  const factory = loadPanelFactory();
  global.fetch = fetchImpl;
  const root = makeElement();
  const bus = makeBus();
  factory(root, bus);
  await settle();
  return { root: root, bus: bus, Graph: global.Graph };
}

// ---- boot / tab loading -------------------------------------------------------
test('boots into the knowledge tab and renders it via Graph.build', async function () {
  const { Graph } = await mount(function () { return jsonResponse(knowledgePayload()); });
  assert.strictEqual(Graph.builds.length, 1);
  assert.deepStrictEqual(Graph.builds[0].nodes, [{ id: 'a', label: 'A', group: 'robot' }]);
});

test('a second load of the knowledge tab reuses a kb:ready payload delivered in the meantime, skipping a second fetch', async function () {
  const pending = {};
  let fetchCount = 0;
  const factory = loadPanelFactory();
  global.fetch = function (url) {
    fetchCount++;
    return new Promise(function (resolve) {
      pending[url] = function () { resolve({ status: 200, json: function () { return Promise.resolve(knowledgePayload()); } }); };
    });
  };
  const root = makeElement();
  const bus = makeBus();
  factory(root, bus);   // boot's own showTab('knowledge') starts a /api/kb fetch, left pending
  await settle();
  // eql's independent fetch resolves first and broadcasts kb:ready
  bus.emit('kb:ready', { payload: knowledgePayload({ nodes: [{ id: 'z', label: 'Z', group: 'robot' }] }) });
  // the user re-selects Knowledge before the panel's own initial fetch has resolved
  clickTab(root, 'knowledge');
  await settle();
  assert.strictEqual(fetchCount, 1);   // no second /api/kb request was made
  assert.deepStrictEqual(global.Graph.builds[global.Graph.builds.length - 1].nodes, [{ id: 'z', label: 'Z', group: 'robot' }]);
});

test('a 404 on a view route surfaces the "restart the server" message on the current tab', async function () {
  const { root } = await mount(function (url) {
    if (url === '/api/kb') return jsonResponse(knowledgePayload());
    return jsonResponse({}, 404);
  });
  clickTab(root, 'kinematics');
  await settle();
  assert.ok(root.querySelector('#graph-empty').textContent.indexOf('/api/kb/view') >= 0);
});

test('switching tabs (click) drives Graph.build for the newly selected tab', async function () {
  const { root, Graph } = await mount(function (url) {
    if (url === '/api/kb') return jsonResponse(knowledgePayload());
    return jsonResponse({ ok: true, nodes: [{ id: 'kin', label: 'Kin', group: 'robot' }], edges: [] });
  });
  clickTab(root, 'kinematics');
  await settle();
  assert.deepStrictEqual(Graph.builds[Graph.builds.length - 1].nodes, [{ id: 'kin', label: 'Kin', group: 'robot' }]);
});

// ---- the tab-switch race (webcore-16) ------------------------------------------
test('switching tabs while an earlier tab load is still in flight does not cross-contaminate state', async function () {
  const pending = {};
  function urlPayload(url) {
    if (url === '/api/kb') return knowledgePayload();
    if (url === '/api/kb/view?name=kinematics') return { ok: true, nodes: [{ id: 'kin', label: 'Kin', group: 'robot' }], edges: [] };
    if (url === '/api/kb/view?name=plan') return { ok: true, nodes: [{ id: 'plan', label: 'Plan', group: 'event' }], edges: [] };
    throw new Error('unexpected url ' + url);
  }
  const factory = loadPanelFactory();
  global.fetch = function (url) {
    return new Promise(function (resolve) {
      pending[url] = function () { resolve({ status: 200, json: function () { return Promise.resolve(urlPayload(url)); } }); };
    });
  };
  const root = makeElement();
  const bus = makeBus();
  factory(root, bus);
  await settle();
  pending['/api/kb']();
  await settle();

  // fire both clicks before either fetch resolves — exactly the interleaving
  // that used to let the older (kinematics) response render over the newer
  // (plan) tab once its await resolved after the user had already moved on
  clickTab(root, 'kinematics');
  clickTab(root, 'plan');
  pending['/api/kb/view?name=kinematics']();
  await settle();
  pending['/api/kb/view?name=plan']();
  await settle();

  const lastBuild = global.Graph.builds[global.Graph.builds.length - 1];
  assert.deepStrictEqual(lastBuild.nodes, [{ id: 'plan', label: 'Plan', group: 'event' }]);
  assert.ok(!global.Graph.builds.some(function (b) { return b.nodes[0] && b.nodes[0].id === 'kin'; }),
    'the superseded kinematics fetch must not render over the plan tab');
});

test('the reverse interleaving (newer resolves first) also lands on the tab actually selected', async function () {
  const pending = {};
  function urlPayload(url) {
    if (url === '/api/kb') return knowledgePayload();
    if (url === '/api/kb/view?name=kinematics') return { ok: true, nodes: [{ id: 'kin', label: 'Kin', group: 'robot' }], edges: [] };
    if (url === '/api/kb/view?name=plan') return { ok: true, nodes: [{ id: 'plan', label: 'Plan', group: 'event' }], edges: [] };
    throw new Error('unexpected url ' + url);
  }
  const factory = loadPanelFactory();
  global.fetch = function (url) {
    return new Promise(function (resolve) {
      pending[url] = function () { resolve({ status: 200, json: function () { return Promise.resolve(urlPayload(url)); } }); };
    });
  };
  const root = makeElement();
  const bus = makeBus();
  factory(root, bus);
  await settle();
  pending['/api/kb']();
  await settle();

  clickTab(root, 'kinematics');
  clickTab(root, 'plan');
  pending['/api/kb/view?name=plan']();
  await settle();
  pending['/api/kb/view?name=kinematics']();
  await settle();

  const lastBuild = global.Graph.builds[global.Graph.builds.length - 1];
  assert.deepStrictEqual(lastBuild.nodes, [{ id: 'plan', label: 'Plan', group: 'event' }]);
});

// ---- node click / drill-down ----------------------------------------------------
test('clicking a node emits entity:select with its detail and relations', async function () {
  const { bus, Graph } = await mount(function () {
    return jsonResponse(knowledgePayload({
      nodes: [{ id: 'a', label: 'A', group: 'robot' }, { id: 'b', label: 'B', group: 'robot' }],
      edges: [{ from: 'a', to: 'b', kind: 'prop', label: 'has' }],
      details: { a: { label: 'A' }, b: { label: 'B' } },
    }));
  });
  const events = [];
  bus.on('entity:select', function (p) { events.push(p); });
  Graph.fireSelect('a');
  assert.strictEqual(events.length, 1);
  assert.strictEqual(events[0].id, 'a');
  assert.deepStrictEqual(events[0].relations, [{ s: 'A', p: 'has', o: 'B' }]);
});

test('double-clicking a node drills in, pushing the parent view onto the back stack', async function () {
  const { root, Graph } = await mount(function (url) {
    if (url === '/api/kb') {
      return jsonResponse(knowledgePayload({
        nodes: [{ id: 'a', label: 'A', group: 'robot' }],
        details: { a: { label: 'A' } },
      }));
    }
    return jsonResponse({ ok: true, nodes: [{ id: 'inner', label: 'Inner', group: 'robot' }], edges: [], crumb: 'A' });
  });
  Graph.fireDoubleSelect('a');
  await settle();
  assert.deepStrictEqual(Graph.builds[Graph.builds.length - 1].nodes, [{ id: 'inner', label: 'Inner', group: 'robot' }]);
  assert.notStrictEqual(root.querySelector('#graph-nav').style.display, 'none');
});

// ---- entity:highlight spotlighting ------------------------------------------------
test('entity:highlight spotlights matching ids and their neighbours via focus', async function () {
  const { bus, Graph } = await mount(function () {
    return jsonResponse(knowledgePayload({
      nodes: [{ id: 'a', label: 'A', group: 'robot' }, { id: 'b', label: 'B', group: 'robot' }, { id: 'c', label: 'C', group: 'robot' }],
      edges: [{ from: 'a', to: 'b', kind: 'prop' }],
      details: { a: {}, b: {}, c: {} },
    }));
  });
  let highlighted = null;
  Graph.highlight = function (ids) { highlighted = ids; };
  bus.fire('entity:highlight', { ids: ['a'], focus: 'a' });
  assert.deepStrictEqual(highlighted.sort(), ['a', 'b']);   // a itself + its neighbour b
});

test('entity:highlight with no matching ids resets instead of highlighting', async function () {
  const { bus, Graph } = await mount(function () { return jsonResponse(knowledgePayload()); });
  let resetCalled = false;
  Graph.reset = function () { resetCalled = true; };
  bus.fire('entity:highlight', { ids: ['ghost'] });
  assert.strictEqual(resetCalled, true);
});
