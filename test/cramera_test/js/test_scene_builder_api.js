'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const context = {window: {}, URL, document: {getElementById: () => null}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/scene_builder.js'), 'utf8'), context);
const Api = context.window.SceneBuilderApi;
(async () => {
  const calls = [];
  const api = new Api('https://example.test/workshop/scene_builder.html', async (url, options) => {
    calls.push({url: url.toString(), options});
    return {ok: true, headers: {get: () => 'application/json'}, json: async () => ({drafts: []})};
  });
  await api.request('drafts');
  assert.equal(calls[0].url, 'https://example.test/workshop/api/scene-builder/drafts');
  assert.equal(calls[0].options.credentials, 'same-origin');
  await api.request('drafts', {document: {version: 1}});
  assert.equal(calls[1].options.method, 'POST');
  assert.equal(calls[1].options.headers['Content-Type'], 'application/json');
  assert.deepEqual(JSON.parse(calls[1].options.body), {document: {version: 1}});
  const rejected = new Api('https://example.test/scene_builder.html', async () => ({ok: false, status: 400, headers: {get: () => 'application/json'}, json: async () => ({error: 'Invalid draft'})}));
  await assert.rejects(() => rejected.request('drafts'), /Invalid draft/);
  const login = new Api('https://example.test/workshop/scene_builder.html', async () => ({ok: true, status: 200, headers: {get: () => 'text/html'}, json: async () => { throw new Error('must not parse login'); }}));
  await assert.rejects(() => login.request('drafts'), /sign in/);
  console.log('Scene Builder API: mounted routes, same-origin credentials, validated responses and login expiry passed');
})().catch((error) => { console.error(error); process.exitCode = 1; });
