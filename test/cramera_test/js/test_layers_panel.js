// Unit tests for web/core/layers-panel.js (node:test): folding the layers overlay away.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/layers-panel.js'), 'utf8'))();
  return window.LayersPanel;
}

function store(initial) {
  const data = Object.assign({}, initial);
  return {
    getItem: function (key) { return key in data ? data[key] : null; },
    setItem: function (key, value) { data[key] = value; },
    data: data,
  };
}

function refusingStore() {
  return {
    getItem: function () { throw new Error('storage is off'); },
    setItem: function () { throw new Error('storage is off'); },
  };
}

// %% the remembered choice
test('the overlay starts unfolded', function () {
  assert.strictEqual(load().collapsed(store({})), false);
});

test('a remembered fold is honoured on the next page', function () {
  const LayersPanel = load();
  const remembered = store({});

  LayersPanel.remember(remembered, true);

  assert.strictEqual(LayersPanel.collapsed(remembered), true);
});

test('unfolding is remembered as well', function () {
  const LayersPanel = load();
  const remembered = store({ layersCollapsed: 'yes' });

  LayersPanel.remember(remembered, false);

  assert.strictEqual(LayersPanel.collapsed(remembered), false);
});

test('a store that refuses to answer leaves the overlay open', function () {
  assert.strictEqual(load().collapsed(refusingStore()), false);
});

test('a store that refuses to remember does not break the fold', function () {
  load().remember(refusingStore(), true);
});

// %% what the button says
test('the button offers to fold while the overlay is open', function () {
  assert.deepStrictEqual(load().button(false), { glyph: '▾', title: 'Fold the layers away' });
});

test('the button offers to show them while it is folded', function () {
  assert.deepStrictEqual(load().button(true), { glyph: '▸', title: 'Show the layers' });
});
