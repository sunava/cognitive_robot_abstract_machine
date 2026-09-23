'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const THREE = require('../../../cramera/src/cramera/web/vendor/three.min.js');

// %% real scene adapter with a layers checkbox
class ContactLayerScene {
  constructor() {
    this.root = new THREE.Group();
    this.instrument = new THREE.Group();
    this.root.add(this.instrument);
    this.listeners = new Map();
    this.checkbox = {
      checked: true,
      addEventListener: (event, handler) => this.listeners.set(event, handler),
      removeEventListener: (event, handler) => {
        assert.equal(this.listeners.get(event), handler);
        this.listeners.delete(event);
      },
    };
    this.row = {style: {display: 'none'}};
    this.source = fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/panels/robot_scene/panel.js'), 'utf8');
    this.context = vm.createContext({
      THREE, worldRoot: this.root, SCENE: {physics: {}}, objectMeshes: {},
      root: {ownerDocument: {}}, focusSceneCamera() {}, needsRender: false,
      $: id => ({'lyr-contacts': this.checkbox, 'lyr-contacts-row': this.row})[id],
    });
    const start = this.source.indexOf('  function laboratoryPhysicsAdapter()');
    vm.runInContext(this.source.slice(start, this.source.indexOf('  RobotView.onReady(function () {', start)), this.context);
    this.adapter = vm.runInContext('laboratoryPhysicsAdapter()', this.context);
    this.context.adapter = this.adapter;
    this.layer = vm.runInContext('bindLaboratoryContactLayer(adapter)', this.context);
    this.feedback = this.root.children.find(child => child !== this.instrument);
    this.target = this.feedback.children[0];
  }
  toggle(visible) {
    this.checkbox.checked = visible;
    this.listeners.get('change')();
  }
  destroy() { this.layer.destroy(); this.adapter.destroy(); }
}

test('the Layers checkbox hides contacts and the orange target without hiding laboratory objects', () => {
  const scene = new ContactLayerScene();
  assert.match(scene.source, /id="lyr-contacts"[^>]*checked><span>Contact points &amp; targets<\/span>/);
  assert.equal(scene.row.style.display, '');
  const contacts = [{force: 2, position: [.1, .2, .9]}];
  const desired = {position: [.2, .3, 1]};
  scene.adapter.showContacts(contacts, desired);
  assert.equal(scene.feedback.visible, true);
  assert.equal(scene.target.visible, true);
  assert.equal(scene.feedback.children[1].visible, true);
  scene.context.needsRender = false;
  scene.toggle(false);
  assert.equal(scene.feedback.visible, false);
  assert.equal(scene.instrument.visible, true);
  assert.equal(scene.context.needsRender, true);
  assert.deepEqual(contacts, [{force: 2, position: [.1, .2, .9]}]);
  scene.destroy();
});

test('polling while hidden retains the latest markers for immediate redisplay', () => {
  const scene = new ContactLayerScene();
  scene.toggle(false);
  const latest = [1, 2, 3];
  scene.adapter.showContacts([{force: 3, position: latest}], {position: [2, 3, 4]});
  assert.equal(scene.feedback.visible, false);
  scene.toggle(true);
  assert.equal(scene.feedback.visible, true);
  assert.deepEqual(scene.feedback.children[1].position.toArray(), latest);
  assert.deepEqual(scene.target.position.toArray(), [2, 3, 4]);
  scene.toggle(false);
  scene.adapter.showContacts([], null);
  scene.toggle(true);
  assert.equal(scene.target.visible, false);
  assert.equal(scene.feedback.children[1].visible, false);
  scene.destroy();
});

test('layer initialization honors its checkbox and teardown releases its listener and markers', () => {
  const scene = new ContactLayerScene();
  scene.layer.destroy();
  scene.checkbox.checked = false;
  scene.layer = vm.runInContext('bindLaboratoryContactLayer(adapter)', scene.context);
  assert.equal(scene.feedback.visible, false);
  scene.destroy();
  assert.equal(scene.listeners.size, 0);
  assert.deepEqual(scene.root.children, [scene.instrument]);
});
