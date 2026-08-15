// Unit tests for web/core/api.js (node:test).
// ServerApi is the single place that decides where the JSON API sits relative to the
// page, so the viewer answers the same whether it is served at a host's root or under
// a path prefix such as Binder's <base>/cramera/.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/api.js'), 'utf8'))();
  return window.ServerApi;
}

test('urlFor() keeps the url below the page', function () {
  assert.strictEqual(load().urlFor('knowledge'), 'api/knowledge');
});

test('urlFor() keeps a route with a query string intact', function () {
  assert.strictEqual(
    load().urlFor('knowledge/view?name=plan'),
    'api/knowledge/view?name=plan'
  );
});

test('a url built by urlFor() resolves under the page it was loaded from', function () {
  const resolved = new URL(load().urlFor('eql'), 'https://binder.example/user/abc/cramera/');
  assert.strictEqual(resolved.pathname, '/user/abc/cramera/api/eql');
});
