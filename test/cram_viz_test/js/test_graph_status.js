// Unit tests for panels/graph/graph.js (node:test) against stubbed vis + DOM:
// status rings, layouts per view kind, in-place status patching and the zoom
// floor that keeps rings readable on big graphs.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cram_viz', 'src', 'cram_viz', 'web');

// ---- stubs -----------------------------------------------------------------
class DataSet {
  constructor(items) {
    this.items = {};
    this.auto = 0;
    (items || []).forEach((item) => {
      const id = item.id !== undefined ? item.id : '_' + this.auto++;
      this.items[id] = item;
    });
  }
  update(list) {
    (Array.isArray(list) ? list : [list]).forEach((patch) => {
      this.items[patch.id] = Object.assign({}, this.items[patch.id], patch);
    });
  }
  get(id) { return this.items[id]; }
}

let lastOptions = null;
let lastData = null;
let lastNetwork = null;

class Network {
  constructor(container, data, options) {
    lastData = data;
    lastOptions = options;
    lastNetwork = this;
    this.handlers = {};
    this.destroyed = false;
  }
  once(event, cb) { (this.handlers[event] = this.handlers[event] || []).push(cb); }
  on(event, cb) { (this.handlers[event] = this.handlers[event] || []).push(cb); }
  fire(event, payload) { (this.handlers[event] || []).forEach((cb) => cb(payload)); }
  setOptions(options) { this.lastSetOptions = options; }
  fit() { this.fitted = (this.fitted || 0) + 1; }
  getScale() { return this.scale === undefined ? 0.3 : this.scale; }
  getViewPosition() { return { x: 0, y: 0 }; }
  moveTo(options) { this.moved = options; }
  getPositions() { return {}; }
  selectNodes(ids) { this.selected = ids; }
  unselectAll() { this.unselected = true; }
  focus(id, options) { this.focused = id; this.focusedOptions = options; }
  redraw() { this.redrawn = (this.redrawn || 0) + 1; }
  destroy() { this.destroyed = true; }
}

function makeElement() {
  return { className: '', innerHTML: '', children: [], appendChild(child) { this.children.push(child); } };
}

function loadGraphJs() {
  const el = makeElement();
  const legendEl = makeElement();
  const fakeRoot = {
    querySelector(sel) {
      if (sel === '#graph') return el;
      if (sel === '#legend') return legendEl;
      return null;
    },
  };
  global.document = {
    createElement: makeElement,
    createTextNode(text) { return { nodeType: 3, textContent: text }; },
  };
  global.window = {};
  global.vis = { DataSet, Network };
  new Function(fs.readFileSync(path.join(WEB, 'core/palette.js'), 'utf8'))();
  new Function(fs.readFileSync(path.join(WEB, 'panels/graph/graph.js'), 'utf8'))();
  global.window.Graph.mount(fakeRoot);
  return { Graph: global.window.Graph, legendEl: legendEl };
}

function planFixture(Graph) {
  Graph.build({
    key: 'plan',
    layout: 'hier',
    arrows: true,
    statusLegend: true,
    nodes: [
      { id: 'p0', label: 'Sequential', group: 'ind', status: 'SUCCEEDED' },
      { id: 'p1', label: 'Transport', group: 'event', status: 'RUNNING' },
      { id: 'p2', label: 'MoveTCP', group: 'robot', status: 'CREATED' },
      { id: 'p3', label: 'Place', group: 'event', status: 'FAILED' },
      { id: 'p4', label: 'plain', group: 'goal' },
    ],
    edges: [
      { from: 'p0', to: 'p1', kind: 'prop' },
      { from: 'p1', to: 'p2', kind: 'prop' },
      { from: 'p0', to: 'p3', kind: 'prop' },
    ],
  });
}

const node = (id) => lastData.nodes.get(id);

// ---- status rings ------------------------------------------------------------
test('status renders as a coloured ring + status word', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  assert.strictEqual(node('p1').color.border, '#ffb648');            // running: amber
  assert.strictEqual(node('p1').label, 'Transport\nrunning');
  assert.strictEqual(node('p0').color.border, '#4bd38a');            // succeeded: green
  assert.strictEqual(node('p3').color.border, '#ff6b8b');            // failed: red
  assert.ok(node('p1').borderWidth > node('p2').borderWidth);        // active > created
  assert.strictEqual(node('p1').borderWidthSelected, node('p1').borderWidth);
  assert.ok(Array.isArray(node('p2').shapeProperties.borderDashes)); // created: dashed
  assert.strictEqual(node('p4').color, undefined);                   // no status: group style
});

test('group fill survives the status ring patch', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  assert.strictEqual(node('p1').color.background, '#b98cff');        // event group fill
});

// ---- live patching -------------------------------------------------------------
test('setStatuses re-colours in place and rebuilds labels from the base', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  assert.strictEqual(Graph.setStatuses({ p2: 'RUNNING', p1: 'SUCCEEDED' }), true);
  assert.strictEqual(node('p2').label, 'MoveTCP\nrunning');
  assert.strictEqual(node('p1').label, 'Transport\nsucceeded');      // not appended twice
});

test('setStatuses reports unknown ids so the caller rebuilds', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  assert.strictEqual(Graph.setStatuses({ ghost: 'RUNNING' }), false);
});

// ---- layouts --------------------------------------------------------------------
test('trees are hierarchical with physics off; entity graphs keep the force layout', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  assert.strictEqual(lastOptions.layout.hierarchical.enabled, true);
  assert.strictEqual(lastOptions.physics, false);
  assert.strictEqual(lastOptions.edges.arrows.to.enabled, true);

  Graph.build({ key: 'knowledge', nodes: [{ id: 'k', label: 'k', group: 'robot' }], edges: [] });
  assert.strictEqual(lastOptions.layout.improvedLayout, true);
  assert.strictEqual(typeof lastOptions.physics, 'object');
  assert.strictEqual(lastOptions.edges.arrows.to.enabled, false);
});

test('status views scale nodes, labels and spacing up so rings stay readable', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  const statusScaling = lastOptions.nodes.scaling.min;
  const statusFont = lastOptions.groups.event.font.size;
  Graph.build({ key: 'knowledge', nodes: [{ id: 'k', label: 'k', group: 'robot' }], edges: [] });
  assert.ok(statusScaling > lastOptions.nodes.scaling.min);
  assert.ok(statusFont > lastOptions.groups.event.font.size);
});

// ---- zoom floor -------------------------------------------------------------------
test('a big status graph is never fitted below the zoom floor', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);                       // stub getScale() reports 0.3 after fit
  lastNetwork.fire('afterDrawing');
  assert.ok(lastNetwork.fitted > 0);
  assert.ok(lastNetwork.moved && lastNetwork.moved.scale >= 0.7);
});

test('plain entity graphs fit freely (no zoom floor)', function () {
  const { Graph } = loadGraphJs();
  Graph.build({ key: 'knowledge', nodes: [{ id: 'k', label: 'k', group: 'robot' }], edges: [] });
  lastNetwork.fire('stabilizationIterationsDone');
  assert.strictEqual(lastNetwork.moved, undefined);
});

// ---- statechart transitions ---------------------------------------------------------
test('statechart transition kinds get distinct edge styles', function () {
  const { Graph } = loadGraphJs();
  Graph.build({
    key: 'chart', layout: 'hier', arrows: true,
    nodes: [
      { id: 's0', label: 'Goal', group: 'klass', status: 'RUNNING' },
      { id: 's1', label: 'Move', group: 'robot', status: 'DONE' },
    ],
    edges: [
      { from: 's0', to: 's1', kind: 'START' },
      { from: 's1', to: 's0', kind: 'PAUSE' },
    ],
  });
  const edges = Object.values(lastData.edges.items);
  assert.strictEqual(edges[0].color.color, '#4bd38a');               // START: green
  assert.ok(Array.isArray(edges[1].dashes));                         // PAUSE: dashed
});

// ---- rebuild lifecycle -------------------------------------------------------------
test('build() destroys the previous network before creating a new one', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  const first = lastNetwork;
  planFixture(Graph);
  assert.strictEqual(first.destroyed, true);
  assert.notStrictEqual(lastNetwork, first);
});

// ---- duplicate edges ----------------------------------------------------------------
test('duplicate edges (same from/to/label) are de-duplicated', function () {
  const { Graph } = loadGraphJs();
  Graph.build({
    key: 'knowledge',
    nodes: [{ id: 'a', label: 'a', group: 'robot' }, { id: 'b', label: 'b', group: 'robot' }],
    edges: [{ from: 'a', to: 'b', kind: 'prop' }, { from: 'a', to: 'b', kind: 'prop' }],
  });
  assert.strictEqual(Object.keys(lastData.edges.items).length, 1);
});

// ---- legend --------------------------------------------------------------------------
test('a custom legend renders labels as text, never as parsed markup', function () {
  const { Graph, legendEl } = loadGraphJs();
  Graph.build({
    key: 'kinematics', nodes: [{ id: 'k', label: 'k', group: 'robot' }], edges: [],
    legend: [{ group: 'robot', label: '<b>evil</b> & <link>' }],
  });
  assert.strictEqual(legendEl.children.length, 1);
  const textNode = legendEl.children[0].children[0];
  assert.strictEqual(textNode.textContent, '<b>evil</b> & <link>');
});

test('a legend row for an unknown group is skipped', function () {
  const { Graph, legendEl } = loadGraphJs();
  Graph.build({
    key: 'kinematics', nodes: [{ id: 'k', label: 'k', group: 'robot' }], edges: [],
    legend: [{ group: 'no-such-group', label: 'ghost' }],
  });
  assert.strictEqual(legendEl.children.length, 0);
});

test('statusLegend rows are appended after a custom legend when requested', function () {
  const { Graph, legendEl } = loadGraphJs();
  Graph.build({
    key: 'plan', nodes: [{ id: 'k', label: 'k', group: 'robot' }], edges: [],
    legend: [{ group: 'robot', label: 'Task' }], statusLegend: true,
  });
  assert.ok(legendEl.children.length > 1);
});

// ---- highlight / reset / focus / resize ----------------------------------------------
test('highlight() dims everything but the given ids and selects+fits them', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  Graph.highlight(['p1']);
  assert.strictEqual(node('p1').opacity, 1);
  assert.strictEqual(node('p0').opacity, 0.16);
  assert.deepStrictEqual(lastNetwork.selected, ['p1']);
  assert.ok(lastNetwork.fitted > 0);
});

test('highlight() ignores ids that are not in the current graph', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  assert.doesNotThrow(function () { Graph.highlight(['ghost']); });
  assert.strictEqual(lastNetwork.selected, undefined);
});

test('reset() restores full opacity and unselects', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  Graph.highlight(['p1']);
  Graph.reset();
  assert.strictEqual(node('p0').opacity, 1);
  assert.strictEqual(node('p1').opacity, 1);
  assert.strictEqual(lastNetwork.unselected, true);
});

test('focus() focuses a known node and ignores an unknown one', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  Graph.focus('p2');
  assert.strictEqual(lastNetwork.focused, 'p2');
  lastNetwork.focused = undefined;
  Graph.focus('ghost');
  assert.strictEqual(lastNetwork.focused, undefined);
});

test('resize() redraws and re-fits the network', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  const fittedBefore = lastNetwork.fitted || 0;   // hier view: not yet fitted (awaits afterDrawing)
  Graph.resize();
  assert.strictEqual(lastNetwork.redrawn, 1);
  assert.ok(lastNetwork.fitted > fittedBefore);
});

// ---- drag / click wiring --------------------------------------------------------------
test('dragging a node in a hierarchical (tree) view does not touch physics', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);                                   // hier: true
  lastNetwork.fire('dragStart', { nodes: ['p1'] });
  assert.strictEqual(lastNetwork.lastSetOptions, undefined);
});

test('dragging a node re-enables physics for a force-directed (entity) view', function () {
  const { Graph } = loadGraphJs();
  Graph.build({ key: 'knowledge', nodes: [{ id: 'k', label: 'k', group: 'robot' }], edges: [] });
  lastNetwork.fire('dragStart', { nodes: ['k'] });
  assert.strictEqual(lastNetwork.lastSetOptions.physics, true);
  assert.doesNotThrow(function () { lastNetwork.fire('dragEnd', { nodes: ['k'] }); });
});

test('a pan (no nodes) does not toggle physics on drag start/end', function () {
  const { Graph } = loadGraphJs();
  Graph.build({ key: 'knowledge', nodes: [{ id: 'k', label: 'k', group: 'robot' }], edges: [] });
  lastNetwork.fire('dragStart', { nodes: [] });
  assert.strictEqual(lastNetwork.lastSetOptions, undefined);
});

test('click on a node invokes the onSelect callback with its id', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  const selected = [];
  Graph.onSelect(function (id) { selected.push(id); });
  lastNetwork.fire('click', { nodes: ['p2'] });
  assert.deepStrictEqual(selected, ['p2']);
  lastNetwork.fire('click', { nodes: [] });
  assert.deepStrictEqual(selected, ['p2']);               // empty-space click: no callback
});

test('double-click on a node invokes onDoubleSelect; on empty space it re-fits', function () {
  const { Graph } = loadGraphJs();
  planFixture(Graph);
  const doubled = [];
  Graph.onDoubleSelect(function (id) { doubled.push(id); });
  lastNetwork.fire('doubleClick', { nodes: ['p2'], edges: [] });
  assert.deepStrictEqual(doubled, ['p2']);
  const fittedBefore = lastNetwork.fitted || 0;   // hier view: not yet fitted (awaits afterDrawing)
  lastNetwork.fire('doubleClick', { nodes: [], edges: [] });
  assert.ok(lastNetwork.fitted > fittedBefore);
});
