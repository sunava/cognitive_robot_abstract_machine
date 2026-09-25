'use strict';

// %% literal-text rights disclosure in the actual laboratory controls
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const Physics = require('../../../cramera/src/cramera/web/core/laboratory-physics.js');

class RightsElement {
  constructor(tag, ownerDocument) {
    this.tagName = tag;
    this.ownerDocument = ownerDocument;
    this.children = [];
    this.listeners = new Map();
    this.dataset = {};
    this.attributes = new Map();
    this.content = '';
  }
  set innerHTML(value) { assert.fail('Rights metadata must never be interpreted as HTML: ' + value); }
  set textContent(value) { this.content = String(value); this.children = []; }
  get textContent() { return this.content + this.children.map(child => child.textContent).join(''); }
  appendChild(child) { child.parent = this; this.children.push(child); }
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  addEventListener(event, handler) { this.listeners.set(event, handler); }
  removeEventListener(event) { this.listeners.delete(event); }
  setAttribute(name, value) { this.attributes.set(name, value); }
  flatten() { return [this, ...this.children.flatMap(child => child.flatten())]; }
}

class RightsDocument {
  constructor() {
    this.host = new RightsElement('div', this);
    this.defaultView = {setInterval() { return 1; }, clearInterval() {}};
  }
  createElement(tag) { return new RightsElement(tag, this); }
  find(suffix) { return this.host.flatten().find(item => item.className === 'laboratory-asset-rights' + suffix); }
  mount(assetRights) {
    const config = {objects: [], bounds: {min: [-1, -1, 0], max: [1, 1, 2]}, assetRights};
    const mounted = Physics.mount(this.host, config, {state: async () => ({state: 'idle', objects: {}, contacts: []})}, {});
    return {config, mounted};
  }
}

function declaredRights() {
  return {
    principalAuthor: 'Vanessa Hassouna', institution: 'AICOR Institute',
    copyright: 'Copyright © 2026 Vanessa Hassouna.', license: 'Permission required',
    permissionNotice: 'Reuse requires written permission.\nContact the rights holder.',
    priorLicenseNotice: 'Previously granted valid GPL permissions remain unaffected.',
    noticeFile: 'ASSET-NOTICE.txt', licenseFile: 'ASSET-LICENSE.txt',
  };
}

test('the scene-ready hook passes top-level asset rights to the laboratory controls', () => {
  const source = fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/panels/robot_scene/panel.js'), 'utf8');
  const begin = source.indexOf('  RobotView.onReady(function () {', source.indexOf('  function bindLaboratoryContactLayer'));
  const finish = source.indexOf('      laboratoryPhysics.adapter = adapter;', begin);
  const rights = declaredRights();
  let received;
  const context = {
    SCENE: {physics: {objects: [], robot: true}, assetRights: rights},
    laboratoryManualMode: {}, laboratoryPhysics: null,
    RobotView: {onReady: callback => callback()},
    stopTrajectory() {}, laboratoryPhysicsAdapter: () => ({}),
    root: {querySelector: () => ({})}, window: {fetch() {}},
    LaboratoryPhysics: {Client: Physics.Client, mount(host, config) { received = config; return {}; }},
  };
  vm.runInNewContext(source.slice(begin, finish) + '\n    }\n  });', context);
  assert.equal(received.assetRights, rights);
  assert.equal(received.robot, context.SCENE.physics.robot);
  assert.equal(received.objects, context.SCENE.physics.objects);
});

test('laboratory controls show principal credit and expandable complete terms without fetching files', async () => {
  const document = new RightsDocument(), rights = declaredRights();
  const {mounted} = document.mount(rights);
  await mounted.initialized;
  assert.equal(document.find('').hidden, false);
  assert.equal(document.find('-credit').textContent, 'Principal author: ' + rights.principalAuthor + ' · ' + rights.institution);
  const details = document.find('-details');
  assert.equal(details.tagName, 'details');
  assert.equal(Boolean(details.open), false);
  assert.equal(details.children[0].tagName, 'summary');
  assert.equal(details.children[0].textContent, 'Asset rights & credits');
  for (const field of ['copyright', 'license', 'permissionNotice', 'priorLicenseNotice', 'noticeFile', 'licenseFile']) {
    const paragraph = document.find('-' + field);
    assert.equal(paragraph.hidden, false);
    assert.ok(paragraph.textContent.endsWith(rights[field]), field);
    assert.equal(paragraph.tagName, 'p');
  }
  assert.equal(document.find('-priorLicenseNotice').textContent, rights.priorLicenseNotice);
  mounted.destroy();
});

test('metadata markup and file names remain inert literal text', async () => {
  const document = new RightsDocument(), rights = declaredRights();
  rights.principalAuthor = '<img src=x onerror=alert(1)>';
  rights.permissionNotice = '<script>globalThis.unwanted = true</script>';
  rights.noticeFile = 'javascript:alert(1)';
  const {mounted} = document.mount(rights);
  await mounted.initialized;
  assert.ok(document.find('-credit').textContent.includes(rights.principalAuthor));
  assert.equal(document.find('-permissionNotice').textContent, rights.permissionNotice);
  assert.equal(document.find('-noticeFile').textContent, 'Notice file: ' + rights.noticeFile);
  assert.equal(document.find('').flatten().some(item => ['a', 'script', 'img'].includes(item.tagName)), false);
  mounted.destroy();
});

test('missing metadata clears all previous rights and scene remount does not retain the old credit', async () => {
  const document = new RightsDocument();
  const first = document.mount(declaredRights());
  await first.mounted.initialized;
  document.find('-details').open = true;
  delete first.config.assetRights;
  first.mounted.controller.changed();
  assert.equal(document.find('').hidden, true);
  assert.equal(document.find('-credit').textContent, '');
  assert.equal(document.find('-priorLicenseNotice').textContent, '');
  assert.equal(document.find('-details').open, false);
  first.mounted.destroy();
  assert.equal(document.host.children.length, 0);
  const next = document.mount(undefined);
  await next.mounted.initialized;
  assert.equal(document.find('').hidden, true);
  assert.equal(document.find('-credit').textContent, '');
  next.mounted.destroy();
});

test('partial metadata displays only declared values without inventing permission or ownership', async () => {
  const document = new RightsDocument();
  const {mounted} = document.mount({institution: 'Example laboratory'});
  await mounted.initialized;
  assert.equal(document.find('').hidden, false);
  assert.equal(document.find('-credit').textContent, 'Example laboratory');
  assert.equal(document.find('-details').hidden, true);
  assert.equal(document.find('-permissionNotice').textContent, '');
  mounted.destroy();
});
