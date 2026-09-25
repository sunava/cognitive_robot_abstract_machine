'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// %% editor DOM and isolated asset API
class EditorElement {
  constructor() { this.value = ''; this.children = []; this.listeners = {}; this.dataset = {}; this.attributes = {}; this.hidden = false; this.disabled = false; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  replaceChildren(...children) { this.children = [...children]; }
  append(...children) { this.children.push(...children); }
  add(child) { this.children.push(child); }
  setAttribute(name, value) { this.attributes[name] = value; }
  get options() { return this.children; }
  closest() { return null; }
  checkValidity() { return true; }
}
class DraftEditingPage {
  constructor() {
    this.elements = new Map();
    this.requests = [];
    this.drafts = new Map();
    this.storage = new Map();
    this.assets = [{id: 'models/robot.urdf', name: 'PR2', kind: 'urdf', category: 'robot', format: 'urdf', notices: []}, {id: 'models/bench.urdf', name: 'Workbench', kind: 'urdf', category: 'environment', format: 'urdf', notices: []}];
    const page = this;
    this.context = vm.createContext({URL, Blob, setTimeout,
      Option: class extends EditorElement { constructor(name, value) { super(); this.textContent = name; this.value = value; } },
      document: {getElementById: (id) => id === 'scene-builder' ? null : this.element(id), createElement: () => new EditorElement(), addEventListener() {}, activeElement: null},
      window: {location: {href: 'https://example.test/workshop/scene_builder.html', pathname: '/workshop/scene_builder.html', origin: 'https://example.test'}, localStorage: {setItem: (key, value) => this.storage.set(key, value), getItem: (key) => this.storage.get(key)}, fetch: async (url, options) => {
        const route = url.pathname.split('/scene-builder/')[1];
        this.requests.push({route, options});
        let body;
        if (route === 'catalog') body = {assets: this.assets, scenes: [{id: 'workshop', name: 'Workshop'}]};
        else if (route === 'drafts' && options.method === 'POST') {
          const payload = JSON.parse(options.body);
          const id = payload.id || `draft_${this.drafts.size + 1}`;
          body = {id, document: payload.document};
          this.drafts.set(id, body);
        } else if (route === 'drafts') body = {drafts: [...this.drafts.values()].map((draft) => ({id: draft.id, name: draft.document.name}))};
        else if (route.startsWith('drafts/')) body = this.drafts.get(route.split('/')[1]);
        else throw new Error(`Unexpected route: ${route}`);
        return {ok: true, headers: {get: () => 'application/json'}, json: async () => body};
      }, SceneBuilderView: class {
        constructor(options) { this.options = options; this.available = false; page.viewport = this; }
        setDocument(document) { this.document = document; }
        setSelection() {}
        getStructure() { return {links: [{name: 'cabinet'}], joints: [{name: 'wrist_roll', type: 'continuous', lower: 0, upper: 0, value: 0}]}; }
        frameAll() {}
      }},
    });
    const web = path.join(__dirname, '../../../cramera/src/cramera/web');
    for (const filename of ['core/scene-builder-state.js', 'scene_builder.js']) vm.runInContext(fs.readFileSync(path.join(web, filename), 'utf8'), this.context, {filename});
    this.app = new this.context.window.SceneBuilderApp();
  }
  element(id) { if (!this.elements.has(id)) this.elements.set(id, new EditorElement()); return this.elements.get(id); }
  async change(id, value) { this.element(id).value = value; await this.element(id).listeners.change(); await new Promise(setImmediate); }
  async click(id) { await this.element(id).listeners.click(); await new Promise(setImmediate); }
}

(async () => {
  const page = new DraftEditingPage();
  await page.app.start();
  assert.equal(page.element('view-fallback').hidden, false);
  const robot = page.app.state.add(page.assets[0].id);
  page.app.select(robot.id);
  await page.change('position-x', '2.5');
  await page.change('rotation-z', '90');
  assert.equal(page.app.state.instance(robot.id).position[0], 2.5);
  assert.equal(page.app.state.instance(robot.id).rotation[2], Math.PI / 2);
  const jointInput = page.element('joint-fields').children[0].children[0];
  assert.equal(jointInput.min, undefined, 'Continuous joints must not be locked to URDF placeholder limits.');
  assert.equal(jointInput.max, undefined);
  page.app.select(robot.id, 'cabinet');
  await page.change('position-y', '0.4');
  assert.equal(page.app.state.instance(robot.id).linkOffsets.cabinet.position[1], 0.4);
  assert.equal(page.app.state.instance(robot.id).position[1], 0);
  await page.app.save(false);
  const firstDraft = page.app.draftId;
  const snapshot = page.app.state.export();
  page.app.state.rename('Changed draft');
  await page.app.save(false);
  assert.equal(page.drafts.size, 1);
  await page.app.save(true);
  assert.equal(page.drafts.size, 2);
  page.element('draft-list').value = firstDraft;
  await page.app.loadDraft();
  assert.equal(page.app.draftId, firstDraft);
  assert.equal(page.app.state.instance(robot.id).position[0], 2.5);
  assert.equal(page.app.state.instance(robot.id).rotation[2], Math.PI / 2);
  assert.equal(page.requests.some((request) => /physics|command|plan|run/.test(request.route)), false);
  await page.click('new-scene');
  assert.equal(page.app.state.document.instances.length, 0);
  await page.click('undo');
  assert.equal(page.app.state.instance(robot.id).position[0], 2.5);
  page.element('import-file').files = [{size: 64, text: async () => '{"version":99}'}];
  const before = page.app.state.export();
  await assert.rejects(() => page.app.import(), /Unsupported scene version/);
  assert.equal(page.app.state.export(), before);
  const recovered = new page.context.window.SceneBuilderApp();
  recovered.restoreRecovery();
  assert.equal(recovered.state.instance(robot.id).position[0], 2.5);
  assert(snapshot.includes('linkOffsets'));
  console.log('Scene Builder page: WebGL fallback, robot pose editing, URDF subtree, continuous joints, mounted save/load/copy, history and recovery passed');
})().catch((error) => { console.error(error); process.exitCode = 1; });
