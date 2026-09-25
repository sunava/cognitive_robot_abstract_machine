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
    this.scanError = null;
    this.assets = [{id: 'models/robot.urdf', name: 'PR2', kind: 'urdf', category: 'robot', format: 'urdf', notices: []}, {id: 'models/bench.urdf', name: 'Workbench', kind: 'urdf', category: 'environment', format: 'urdf', notices: []}];
    const page = this;
    this.context = vm.createContext({URL, Blob, setTimeout,
      Option: class extends EditorElement { constructor(name, value) { super(); this.textContent = name; this.value = value; } },
      document: {getElementById: (id) => id === 'scene-builder' ? null : this.element(id), createElement: () => new EditorElement(), addEventListener() {}, activeElement: null},
      window: {location: {href: 'https://example.test/workshop/scene_builder.html', pathname: '/workshop/scene_builder.html', origin: 'https://example.test'}, localStorage: {setItem: (key, value) => this.storage.set(key, value), getItem: (key) => this.storage.get(key)}, fetch: async (url, options) => {
        const route = url.pathname.split('/scene-builder/')[1];
        this.requests.push({route, options});
        let body;
        if (route.startsWith('point-clouds/')) body = {id: 'point-clouds/uploaded.ply', name: 'Uploaded scan', kind: 'point-cloud', category: 'point-cloud', format: 'ply', notices: []};
        else if (route === 'catalog') body = {assets: this.assets, scenes: [{id: 'workshop', name: 'Workshop'}]};
        else if (route === 'drafts' && options.method === 'POST') {
          const payload = JSON.parse(options.body);
          const id = payload.id || `draft_${this.drafts.size + 1}`;
          body = {id, document: payload.document};
          this.drafts.set(id, body);
        } else if (route === 'drafts') body = {drafts: [...this.drafts.values()].map((draft) => ({id: draft.id, name: draft.document.name}))};
        else if (route.startsWith('drafts/')) body = this.drafts.get(route.split('/')[1]);
        else throw new Error(`Unexpected route: ${route}`);
        return {ok: true, headers: {get: () => 'application/json'}, json: async () => body};
      }, PointCloudSupport: {inspect: () => { if (page.scanError) throw new Error(page.scanError); return {pointCount: 12, finitePointCount: 12, displayPointCount: 12, hasColor: true, format: 'ply'}; }}, SceneBuilderView: class {
        constructor(options) { this.options = options; this.available = false; page.viewport = this; }
        setDocument(document) { this.document = document; }
        setSelection() {}
        setMode(mode) { this.mode = mode; }
        setScaleLinked(value) { this.scaleLinked = value; }
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
  // %% linked scale edits, atomic gizmo commits, and stored room references
  page.app.select(robot.id);
  await page.change('scale-x', '2');
  assert.deepEqual(JSON.parse(JSON.stringify(page.app.state.instance(robot.id).scale)), [2, 2, 2]);
  page.element('scale-linked').checked = false;
  await page.element('scale-linked').listeners.change();
  await page.change('scale-y', '3');
  assert.deepEqual(JSON.parse(JSON.stringify(page.app.state.instance(robot.id).scale)), [2, 3, 2]);
  const beforeScaleGesture = page.app.state.export();
  page.viewport.options.onTransform({instanceId: robot.id, linkName: null, position: [8, 2, 1], rotation: [0, 0, 0], scale: [4, 6, 4]});
  assert.equal(page.app.state.instance(robot.id).scale[1], 6);
  page.app.state.undo();
  assert.equal(page.app.state.export(), beforeScaleGesture);
  await page.click('mode-scale');
  assert.equal(page.viewport.mode, 'scale');
  page.app.select(robot.id, 'cabinet');
  assert.equal(page.element('mode-scale').disabled, true);
  assert.equal(page.viewport.mode, 'translate');
  const scanFile = {name: 'Measured room.ply', size: 256, arrayBuffer: async () => new Uint8Array([112, 108, 121]).buffer};
  page.element('scan-file').files = [scanFile];
  page.element('scan-units').value = 'mm';
  await page.app.uploadScan();
  const scan = page.app.state.instance(page.app.selection);
  assert.equal(scan.name, 'Measured room');
  assert.deepEqual(JSON.parse(JSON.stringify(scan.scale)), [.001, .001, .001]);
  assert.deepEqual(JSON.parse(JSON.stringify(scan.position)), [0, 0, 0]);
  assert.equal(scan.pointCloud.locked, true);
  assert.equal(page.element('position-x').disabled, true);
  assert.equal(page.element('mode-scale').disabled, true);
  assert.equal(page.requests.find((request) => request.route === 'point-clouds/ply').options.body, scanFile);
  page.element('reference-locked').checked = false;
  await page.element('reference-locked').listeners.change();
  assert.equal(page.element('position-x').disabled, false);
  await page.change('position-z', '.7');
  await page.change('reference-opacity', '.4');
  await page.change('reference-point-size', '8');
  assert.equal(page.app.state.instance(scan.id).position[2], .7);
  assert.equal(page.app.state.instance(scan.id).pointCloud.opacity, .4);
  assert.equal(page.app.state.instance(scan.id).pointCloud.pointSize, .008);
  page.element('reference-visible').checked = false;
  await page.element('reference-visible').listeners.change();
  await page.app.save(true);
  const scanDraft = page.app.draftId;
  page.app.state.remove(scan.id);
  page.element('draft-list').value = scanDraft;
  await page.app.loadDraft();
  assert.equal(page.app.state.instance(scan.id).pointCloud.visible, false);
  assert.equal(page.app.state.instance(scan.id).pointCloud.opacity, .4);
  const beforeBadScan = page.app.state.export();
  const uploadCount = page.requests.filter((request) => request.route.startsWith('point-clouds/')).length;
  page.scanError = 'Malformed point cloud';
  page.element('scan-file').files = [scanFile];
  await assert.rejects(() => page.app.uploadScan(), /Malformed point cloud/);
  assert.equal(page.app.state.export(), beforeBadScan);
  assert.equal(page.requests.filter((request) => request.route.startsWith('point-clouds/')).length, uploadCount);
  page.element('scan-file').files = [{...scanFile, size: page.context.window.SceneBuilderApp.MAX_SCAN_BYTES + 1}];
  await assert.rejects(() => page.app.uploadScan(), /64 MiB/);
  const recovered = new page.context.window.SceneBuilderApp();
  recovered.restoreRecovery();
  assert.equal(recovered.state.instance(robot.id).position[0], 2.5);
  assert(snapshot.includes('linkOffsets'));
  // %% delayed uploads and saves cannot modify a different scene draft
  const delayedPage = new DraftEditingPage();
  await delayedPage.app.start();
  delayedPage.element('scan-file').files = [scanFile];
  delayedPage.element('scan-units').value = 'mm';
  let finishUpload;
  delayedPage.app.api.uploadPointCloud = () => new Promise((resolve) => { finishUpload = resolve; });
  const uploading = delayedPage.app.uploadScan();
  await new Promise(setImmediate);
  await delayedPage.click('new-scene');
  const newDocument = delayedPage.app.state.export();
  const delayedAsset = {id: 'point-clouds/delayed.ply', name: 'Delayed scan', kind: 'point-cloud', category: 'point-cloud', format: 'ply'};
  finishUpload(delayedAsset);
  await uploading;
  assert.equal(delayedPage.app.state.export(), newDocument, 'A finished upload must not insert into a different draft.');
  assert.equal(delayedPage.app.assets.has(delayedAsset.id), true, 'An already uploaded scan remains accessible in the catalog.');
  const originalRequest = delayedPage.app.api.request.bind(delayedPage.app.api);
  let finishSave;
  delayedPage.app.api.request = (route, payload) => payload ? new Promise((resolve) => { finishSave = () => resolve({id: 'old-draft', document: payload.document}); }) : originalRequest(route);
  delayedPage.app.state.add(delayedPage.assets[0].id);
  const saving = delayedPage.app.save(false);
  await delayedPage.click('new-scene');
  finishSave();
  await saving;
  assert.equal(delayedPage.app.draftId, null, 'Saving the previous document must not retarget the new draft.');
  assert.equal(delayedPage.app.state.document.instances.length, 0);
  let finishLoad;
  delayedPage.element('draft-list').value = 'older-scene';
  delayedPage.app.api.request = (route) => route === 'drafts/older-scene' ? new Promise((resolve) => { finishLoad = resolve; }) : originalRequest(route);
  const loading = delayedPage.app.loadDraft();
  await delayedPage.click('new-scene');
  finishLoad({id: 'older-scene', document: {version: 1, name: 'Older scene', instances: []}});
  await loading;
  assert.equal(delayedPage.app.state.document.name, 'Untitled scene', 'A stale load response must not replace a newer scene choice.');
  assert.equal(delayedPage.app.draftId, null);
  console.log('Scene Builder page: WebGL fallback, robot pose editing, URDF subtree, continuous joints, mounted save/load/copy, history and recovery passed');
})().catch((error) => { console.error(error); process.exitCode = 1; });
