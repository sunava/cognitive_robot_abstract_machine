'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const WEB = path.join(__dirname, '../../../cramera/src/cramera/web');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));

// %% panel controls without a browser GPU
class ViewElement {
  constructor() {
    this.handlers = new Map();
    this.attributes = new Map();
    this.children = [];
    this.checked = false;
    this.hidden = false;
    this.value = '';
    this.classes = new Set();
    this.draws = [];
    this.context = {drawImage: (...arguments_) => this.draws.push(arguments_)};
    this.classList = {toggle: (name, enabled) => enabled ? this.classes.add(name) : this.classes.delete(name)};
    this.ownerDocument = {createElement: () => new ViewElement()};
  }
  addEventListener(name, handler) { this.handlers.set(name, handler); }
  removeEventListener(name) { this.handlers.delete(name); }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  appendChild(child) { this.children.push(child); }
  replaceChildren() { this.children = []; }
  focus() { this.focused = true; }
  getBoundingClientRect() { return this.bounds; }
  getContext() { return this.context; }
  dispatch(name, extra = {}) {
    const handler = this.handlers.get(name);
    if (handler) handler({stopPropagation() {}, preventDefault() {}, ...extra});
  }
}

class InsetRenderer {
  constructor() {
    this.viewport = new THREE.Vector4(0, 0, 800, 600);
    this.scissor = new THREE.Vector4(2, 3, 400, 300);
    this.scissorTest = false;
    this.renders = [];
    this.domElement = {width: 1600, height: 1200};
  }
  getViewport(target) { return target.copy(this.viewport); }
  getPixelRatio() { return 2; }
  getScissor(target) { return target.copy(this.scissor); }
  getScissorTest() { return this.scissorTest; }
  setViewport(...values) { this.viewport = values[0].isVector4 ? values[0].clone() : new THREE.Vector4(...values); }
  setScissor(...values) { this.scissor = values[0].isVector4 ? values[0].clone() : new THREE.Vector4(...values); }
  setScissorTest(value) { this.scissorTest = value; }
  clear() {}
  render(scene, camera) { this.renders.push({scene, camera, viewport: this.viewport.clone(), scissorTest: this.scissorTest}); }
}

function mount() {
  const scope = {};
  for (const file of ['robot-camera.js', 'robot-camera-panel.js']) {
    new Function('window', fs.readFileSync(path.join(WEB, 'core', file), 'utf8'))(scope);
  }
  const root = new ViewElement();
  const elements = new Map(['source', 'expand', 'close', 'viewport', 'status'].map(name => ['[data-robot-camera="' + name + '"]', new ViewElement()]));
  root.querySelector = selector => elements.get(selector);
  const viewport = elements.get('[data-robot-camera="viewport"]');
  viewport.bounds = {left: 500, bottom: 500, width: 280, height: 180};
  const layer = new ViewElement();
  const renderer = new InsetRenderer();
  const container = new ViewElement();
  container.bounds = {left: 100, bottom: 650, width: 800, height: 600};
  let changes = 0;
  class SizeObserver {
    constructor(callback) { this.callback = callback; }
    observe() {}
    disconnect() { this.disconnected = true; }
  }
  const panel = new scope.RobotCameraPanel({THREE, root, layer, renderer,
    scene: new THREE.Scene(), container, invalidate: () => changes++, ResizeObserver: SizeObserver});
  return {panel, root, layer, renderer, elements, changes: () => changes};
}

// %% visibility and expansion
test('layer toggle, enlarge, Escape and close stay synchronized', () => {
  const view = mount();
  assert.equal(view.root.hidden, true);
  view.layer.checked = true;
  view.layer.dispatch('change');
  assert.equal(view.root.hidden, false);
  view.elements.get('[data-robot-camera="expand"]').dispatch('click');
  assert.equal(view.root.classes.has('expanded'), true);
  view.root.dispatch('keydown', {key: 'Escape'});
  assert.equal(view.root.classes.has('expanded'), false);
  assert.equal(view.root.hidden, false);
  view.elements.get('[data-robot-camera="close"]').dispatch('click');
  assert.equal(view.layer.checked, false);
  assert.equal(view.root.hidden, true);
});

test('hidden and missing-camera panels skip the second render', () => {
  const view = mount();
  view.panel.render();
  view.layer.checked = true;
  view.layer.dispatch('change');
  view.panel.setModels([]);
  view.panel.render();
  assert.equal(view.renderer.renders.length, 0);
  assert.equal(view.elements.get('[data-robot-camera="status"]').hidden, false);
});

test('inset uses CSS coordinates and restores the main renderer state', () => {
  const view = mount();
  const originalViewport = view.renderer.viewport.clone();
  const originalScissor = view.renderer.scissor.clone();
  view.panel.setModels([{name: 'pr2', robot: true, obj: {links: {wide_stereo_optical_frame: new THREE.Object3D()}}}]);
  view.layer.checked = true;
  view.layer.dispatch('change');
  view.panel.render();
  assert.equal(view.renderer.renders.length, 1);
  assert.deepEqual(view.renderer.renders[0].viewport.toArray(), [400, 150, 280, 180]);
  assert.equal(view.renderer.renders[0].scissorTest, true);
  assert.deepEqual(view.renderer.viewport, originalViewport);
  assert.deepEqual(view.renderer.scissor, originalScissor);
  assert.equal(view.renderer.scissorTest, false);
});

test('destroy disconnects resize observation and removes control listeners', () => {
  const view = mount();
  const before = view.changes();
  view.panel.destroy();
  view.layer.checked = true;
  view.layer.dispatch('change');
  assert.equal(view.changes(), before);
  assert.equal(view.panel.observer.disconnected, true);
});

// %% asynchronously loaded robot instances
test('the scene primary robot supplies the default camera even when its name sorts last', () => {
  const view = mount();
  const first = {name: 'alpha', robot: true, obj: {links: {camera_optical_frame: new THREE.Object3D()}}};
  const primary = {name: 'zebra', robot: true, obj: {links: {camera_optical_frame: new THREE.Object3D()}}};
  view.panel.setModels([first]);
  view.panel.setModels([first, primary], primary);
  assert.equal(view.panel.selected.link, primary.obj.links.camera_optical_frame);
});

test('an explicit camera choice survives another model finishing loading', () => {
  const view = mount();
  const robot = {name: 'pr2', robot: true, obj: {links: {
    wide_stereo_optical_frame: new THREE.Object3D(),
    r_forearm_cam_optical_frame: new THREE.Object3D(),
  }}};
  view.panel.setModels([robot]);
  const hand = view.panel.sources.find(source => source.link === robot.obj.links.r_forearm_cam_optical_frame);
  const select = view.elements.get('[data-robot-camera="source"]');
  select.value = hand.id;
  select.dispatch('change');
  view.panel.setModels([robot, {name: 'room', robot: false, obj: {links: {}}}], robot);
  assert.equal(view.panel.selected.link, hand.link);
  assert.equal(select.value, hand.id);
});

test('the window copies the rendered viewport at device resolution to cover underlying menus', () => {
  const view = mount();
  view.panel.setModels([{name: 'pr2', robot: true, obj: {links: {wide_stereo_optical_frame: new THREE.Object3D()}}}]);
  view.layer.checked = true;
  view.layer.dispatch('change');
  view.panel.render();
  const viewport = view.elements.get('[data-robot-camera="viewport"]');
  const canvas = viewport.children[0];
  assert.ok(canvas);
  assert.equal(canvas.width, 560);
  assert.equal(canvas.height, 360);
  assert.deepEqual(canvas.draws, [[view.renderer.domElement, 800, 540, 560, 360, 0, 0, 560, 360]]);
});
