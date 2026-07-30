// Unit tests for panels/eql/panel.js (node:test) against a stubbed DOM, bus
// and fetch: boot (success/error/unreachable), describe(), render()'s HTML
// escaping, and the shared-palette colour lookup (core/palette.js).
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
    _listeners: {},
    children: [],
    dataset: {},
    value: '',
    title: '',
    get innerHTML() { return this._html; },
    set innerHTML(value) { this._html = value; this._children = {}; },
    get textContent() { return this._text || ''; },
    set textContent(value) { this._text = value; },
    querySelector(selector) {
      const match = /^#([\w-]+)$/.exec(selector);
      if (!match) return null;
      if (!this._children[match[1]]) this._children[match[1]] = makeElement();
      return this._children[match[1]];
    },
    appendChild(child) { this.children.push(child); },
    addEventListener(event, cb) { (this._listeners[event] = this._listeners[event] || []).push(cb); },
    dispatch(event, payload) { (this._listeners[event] || []).forEach(function (cb) { cb(payload || {}); }); },
    classList: { add() {}, remove() {}, toggle() {} },
  };
  return el;
}

function makeBus() {
  const handlers = {};
  const emitted = [];
  return {
    on(event, cb) { (handlers[event] = handlers[event] || []).push(cb); },
    emit(event, payload) { emitted.push([event, payload]); (handlers[event] || []).forEach(function (cb) { cb(payload); }); },
    fire(event, payload) { (handlers[event] || []).forEach(function (cb) { cb(payload); }); },
    emitted: emitted,
  };
}

function loadEqlFactory() {
  global.window = {};
  global.document = { createElement: makeElement };
  global.Panels = { _factories: {}, define(id, factory) { this._factories[id] = factory; } };
  new Function(fs.readFileSync(path.join(WEB, 'core/palette.js'), 'utf8'))();
  new Function(fs.readFileSync(path.join(WEB, 'panels/eql/panel.js'), 'utf8'))();
  return global.Panels._factories.eql;
}

async function settle() {
  await new Promise(function (resolve) { setImmediate(resolve); });
  await new Promise(function (resolve) { setImmediate(resolve); });
}

async function mount(fetchImpl) {
  const factory = loadEqlFactory();
  global.fetch = fetchImpl;
  const root = makeElement();
  const bus = makeBus();
  factory(root, bus);
  await settle();
  return { root: root, bus: bus };
}

function jsonResponse(body, status) {
  return Promise.resolve({ status: status || 200, json: function () { return Promise.resolve(body); } });
}

// ---- boot --------------------------------------------------------------------
test('boot() success: shows the KB status, builds presets, emits kb:ready', async function () {
  const { root, bus } = await mount(function () {
    return jsonResponse({ ok: true, status: 'ready (3 objects)', presets: [{ text: 'milk', code: 'the(...)' }], details: {} });
  });
  assert.strictEqual(root.querySelector('#kb-status').textContent, 'ready (3 objects)');
  assert.strictEqual(root.querySelector('#presets').children.length, 1);
  assert.deepStrictEqual(bus.emitted[0], ['kb:ready', { payload: { ok: true, status: 'ready (3 objects)', presets: [{ text: 'milk', code: 'the(...)' }], details: {} } }]);
});

test('boot() error payload: shows "EQL unavailable" and the escaped server error', async function () {
  const { root } = await mount(function () {
    return jsonResponse({ ok: false, error: '<bad>' });
  });
  assert.strictEqual(root.querySelector('#kb-status').textContent, 'EQL unavailable');
  assert.ok(root.querySelector('#answer').innerHTML.indexOf('&lt;bad&gt;') >= 0);
});

test('a network-level fetch failure shows "KB error" with the escaped error text', async function () {
  const { root } = await mount(function () {
    return Promise.reject(new Error('<offline>'));
  });
  assert.strictEqual(root.querySelector('#kb-status').textContent, 'KB error');
  assert.ok(root.querySelector('#answer').innerHTML.indexOf('&lt;offline&gt;') >= 0);
});

test('clicking a preset fills the query box and runs it', async function () {
  let eqlBody = null;
  const { root } = await mount(function (url, opts) {
    if (url === '/api/kb') {
      return jsonResponse({ ok: true, status: 'ready', presets: [{ text: 'milk preset', code: "the(entity(object).where(object.name=='milk'))" }], details: {} });
    }
    eqlBody = JSON.parse(opts.body);
    return jsonResponse({ ok: true, count: 0 });
  });
  const presetButton = root.querySelector('#presets').children[0];
  presetButton.dispatch('click');
  await settle();
  assert.strictEqual(root.querySelector('#query-input').value, "the(entity(object).where(object.name=='milk'))");
  assert.deepStrictEqual(eqlBody, { code: "the(entity(object).where(object.name=='milk'))" });
});

test('pressing Enter (without shift) in the query box runs the query', async function () {
  let ran = false;
  const { root } = await mount(function (url) {
    if (url === '/api/kb') return jsonResponse({ ok: true, status: 'ready', presets: [], details: {} });
    ran = true;
    return jsonResponse({ ok: true, count: 0 });
  });
  root.querySelector('#query-input').value = 'the(...)';
  root.querySelector('#query-input').dispatch('keydown', { key: 'Enter', shiftKey: false, preventDefault: function () {} });
  await settle();
  assert.strictEqual(ran, true);
});

test('Shift+Enter does not run the query', async function () {
  let ran = false;
  const { root } = await mount(function (url) {
    if (url === '/api/kb') return jsonResponse({ ok: true, status: 'ready', presets: [], details: {} });
    ran = true;
    return jsonResponse({ ok: true, count: 0 });
  });
  root.querySelector('#query-input').value = 'the(...)';
  root.querySelector('#query-input').dispatch('keydown', { key: 'Enter', shiftKey: true, preventDefault: function () {} });
  await settle();
  assert.strictEqual(ran, false);
});

// ---- describe() via bus events ------------------------------------------------
test('scene:part-clicked describes a known entity from the KB overview and highlights it', async function () {
  const { root, bus } = await mount(function () {
    return jsonResponse({ ok: true, status: 'ready', presets: [], details: { milk: { group: 'object', label: 'Milk', lines: ['a bench object'] } } });
  });
  bus.emitted.length = 0;
  bus.fire('scene:part-clicked', { id: 'milk' });
  assert.ok(root.querySelector('#answer').innerHTML.indexOf('Milk') >= 0);
  assert.deepStrictEqual(bus.emitted[0], ['entity:highlight', { ids: ['milk'], focus: 'milk' }]);
});

test('scene:part-clicked for an unknown id leaves the answer panel untouched', async function () {
  const { root, bus } = await mount(function () {
    return jsonResponse({ ok: true, status: 'ready', presets: [], details: {} });
  });
  const before = root.querySelector('#answer').innerHTML;
  bus.fire('scene:part-clicked', { id: 'ghost' });
  assert.strictEqual(root.querySelector('#answer').innerHTML, before);
});

test('entity:select describes using the detail/relations sent by the graph panel, with truncation past 40', async function () {
  const { root, bus } = await mount(function () {
    return jsonResponse({ ok: true, status: 'ready', presets: [], details: {} });
  });
  const relations = [];
  for (let i = 0; i < 45; i++) relations.push({ s: 'a' + i, p: 'rel', o: 'b' + i });
  bus.fire('entity:select', { id: 'x', detail: { group: 'object', label: 'X', lines: [] }, relations: relations });
  const html = root.querySelector('#answer').innerHTML;
  assert.ok(html.indexOf('Relations') >= 0);
  assert.ok(html.indexOf('… 5 more') >= 0);
});

test('scene:step describes the running episode when it is a known entity', async function () {
  const { root, bus } = await mount(function () {
    return jsonResponse({ ok: true, status: 'ready', presets: [], details: { ep1: { group: 'event', label: 'Episode 1', lines: [] } } });
  });
  bus.fire('scene:step', { step: 'ep1' });
  assert.ok(root.querySelector('#answer').innerHTML.indexOf('Episode 1') >= 0);
});

test('scene:step ignores the __done__ sentinel', async function () {
  const { root, bus } = await mount(function () {
    return jsonResponse({ ok: true, status: 'ready', presets: [], details: {} });
  });
  const before = root.querySelector('#answer').innerHTML;
  bus.fire('scene:step', { step: '__done__' });
  assert.strictEqual(root.querySelector('#answer').innerHTML, before);
});

// ---- render(): escaping, empty/error results, entity vs. value rows -----------
test('render() escapes entity-row fields so a malicious value cannot inject markup', async function () {
  const { root } = await mount(function (url) {
    if (url === '/api/kb') return jsonResponse({ ok: true, status: 'ready', presets: [], details: {} });
    return jsonResponse({ ok: true, count: 1, rows: [{ __entity__: '<img onerror=alert(1)>', __type__: 'BenchObject', name: '<script>' }] });
  });
  root.querySelector('#query-input').value = 'the(...)';
  root.querySelector('#query-run').dispatch('click');
  await settle();
  const html = root.querySelector('#answer').innerHTML;
  assert.strictEqual(html.indexOf('<img onerror'), -1);
  assert.strictEqual(html.indexOf('<script>'), -1);
  assert.ok(html.indexOf('&lt;img onerror=alert(1)&gt;') >= 0);
});

test('a PythonClass entity row uses the shared palette\'s "pyclass" colour (matches the graph panel)', async function () {
  const { root } = await mount(function (url) {
    if (url === '/api/kb') return jsonResponse({ ok: true, status: 'ready', presets: [], details: {} });
    return jsonResponse({ ok: true, count: 1, rows: [{ __entity__: 'Foo', __type__: 'PythonClass' }] });
  });
  root.querySelector('#query-input').value = 'the(...)';
  root.querySelector('#query-run').dispatch('click');
  await settle();
  assert.ok(root.querySelector('#answer').innerHTML.indexOf('background:#ffb648') >= 0);
});

test('a value row (no __entity__) renders each field as escaped key = value', async function () {
  const { root } = await mount(function (url) {
    if (url === '/api/kb') return jsonResponse({ ok: true, status: 'ready', presets: [], details: {} });
    return jsonResponse({ ok: true, count: 1, rows: [{ x: '<b>1</b>' }] });
  });
  root.querySelector('#query-input').value = 'the(...)';
  root.querySelector('#query-run').dispatch('click');
  await settle();
  const html = root.querySelector('#answer').innerHTML;
  assert.ok(html.indexOf('x = &lt;b&gt;1&lt;/b&gt;') >= 0);
});

test('zero results shows "No solutions" and clears the highlight', async function () {
  const { root, bus } = await mount(function (url) {
    if (url === '/api/kb') return jsonResponse({ ok: true, status: 'ready', presets: [], details: {} });
    return jsonResponse({ ok: true, count: 0 });
  });
  bus.emitted.length = 0;
  root.querySelector('#query-input').value = 'the(...)';
  root.querySelector('#query-run').dispatch('click');
  await settle();
  assert.ok(root.querySelector('#answer').innerHTML.indexOf('No solutions') >= 0);
  assert.deepStrictEqual(bus.emitted[bus.emitted.length - 1], ['entity:highlight', { ids: [] }]);
});

test('a query error shows the escaped server error and clears the highlight', async function () {
  const { root, bus } = await mount(function (url) {
    if (url === '/api/kb') return jsonResponse({ ok: true, status: 'ready', presets: [], details: {} });
    return jsonResponse({ ok: false, error: '<parse error>' });
  });
  bus.emitted.length = 0;
  root.querySelector('#query-input').value = 'the(...)';
  root.querySelector('#query-run').dispatch('click');
  await settle();
  assert.ok(root.querySelector('#answer').innerHTML.indexOf('&lt;parse error&gt;') >= 0);
  assert.deepStrictEqual(bus.emitted[bus.emitted.length - 1], ['entity:highlight', { ids: [] }]);
});

test('a truncated result set shows the "(truncated)" note and highlights the returned ids', async function () {
  const { root, bus } = await mount(function (url) {
    if (url === '/api/kb') return jsonResponse({ ok: true, status: 'ready', presets: [], details: {} });
    return jsonResponse({ ok: true, count: 50, more: true, highlight: ['a', 'b'], rows: [{ x: 1 }] });
  });
  bus.emitted.length = 0;
  root.querySelector('#query-input').value = 'the(...)';
  root.querySelector('#query-run').dispatch('click');
  await settle();
  assert.ok(root.querySelector('#answer').innerHTML.indexOf('(truncated)') >= 0);
  assert.deepStrictEqual(bus.emitted[bus.emitted.length - 1], ['entity:highlight', { ids: ['a', 'b'] }]);
});
