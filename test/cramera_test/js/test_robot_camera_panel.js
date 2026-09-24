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
    this.style = {left: '', top: '', right: '', bottom: ''};
    this.capturedPointers = new Set();
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
  closest(selector) { return selector === 'button' && this.button ? this : null; }
  setPointerCapture(pointerId) { this.capturedPointers.add(pointerId); }
  releasePointerCapture(pointerId) { this.capturedPointers.delete(pointerId); }
  hasPointerCapture(pointerId) { return this.capturedPointers.has(pointerId); }
  getBoundingClientRect() {
    const bounds = {...this.bounds};
    if (this.positionContainer) {
      if (this.style.left) bounds.left = this.positionContainer.bounds.left + parseFloat(this.style.left);
      if (this.style.top) bounds.top = this.positionContainer.bounds.top + parseFloat(this.style.top);
      bounds.right = bounds.left + bounds.width;
      bounds.bottom = bounds.top + bounds.height;
    }
    return bounds;
  }
  getContext() { return this.context; }
  dispatch(name, extra = {}) {
    const handler = this.handlers.get(name);
    if (handler) handler({stopPropagation() {}, preventDefault() {}, target: this,
      pointerId: 1, button: 0, isPrimary: true, ...extra});
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
  root.bounds = {left: 500, top: 250, width: 320, height: 248};
  const elements = new Map(['source', 'expand', 'close', 'viewport', 'status'].map(name => ['[data-robot-camera="' + name + '"]', new ViewElement()]));
  const header = new ViewElement();
  elements.set('.robot-camera-head', header);
  for (const name of ['expand', 'close']) elements.get('[data-robot-camera="' + name + '"]').button = true;
  root.querySelector = selector => elements.get(selector);
  const viewport = elements.get('[data-robot-camera="viewport"]');
  viewport.bounds = {left: 500, bottom: 500, width: 280, height: 180};
  const layer = new ViewElement();
  const renderer = new InsetRenderer();
  const container = new ViewElement();
  container.bounds = {left: 100, top: 50, bottom: 650, width: 800, height: 600};
  root.positionContainer = container;
  let changes = 0;
  const observers = [];
  class SizeObserver {
    constructor(callback) { this.callback = callback; this.observed = new Set(); observers.push(this); }
    observe(element) { this.observed.add(element); }
    disconnect() { this.disconnected = true; }
  }
  const panel = new scope.RobotCameraPanel({THREE, root, layer, renderer,
    scene: new THREE.Scene(), container, invalidate: () => changes++, ResizeObserver: SizeObserver});
  return {panel, root, layer, renderer, elements, header, container, changes: () => changes,
    resize: element => observers.filter(observer => observer.observed.has(element) && !observer.disconnected)
      .forEach(observer => observer.callback())};
}

function show(view) {
  view.layer.checked = true;
  view.layer.dispatch('change');
}

function drag(view, left, top) {
  const bounds = view.root.getBoundingClientRect();
  view.header.dispatch('pointerdown', {clientX: bounds.left + 40, clientY: bounds.top + 12});
  view.header.dispatch('pointermove', {clientX: bounds.left + 40 + left, clientY: bounds.top + 12 + top});
  view.header.dispatch('pointerup');
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

// %% dragging within the scene
test('the title bar moves the panel by the pointer delta and releases it on pointerup', () => {
  const view = mount();
  show(view);
  view.header.dispatch('pointerdown', {pointerId: 7, clientX: 540, clientY: 262});
  assert.equal(view.header.hasPointerCapture(7), true);
  view.header.dispatch('pointermove', {pointerId: 7, clientX: 590, clientY: 312});
  assert.deepEqual(view.root.style, {left: '450px', top: '250px', right: 'auto', bottom: 'auto'});
  view.header.dispatch('pointerup', {pointerId: 7});
  assert.equal(view.header.hasPointerCapture(7), false);
  view.header.dispatch('pointermove', {pointerId: 7, clientX: 640, clientY: 362});
  assert.equal(view.root.style.left, '450px');
  assert.equal(view.root.style.top, '250px');
});

test('dragging keeps all four panel edges inside the scene', () => {
  const view = mount();
  show(view);
  drag(view, -2000, -2000);
  assert.equal(view.root.style.left, '8px');
  assert.equal(view.root.style.top, '8px');
  drag(view, 2000, 2000);
  assert.equal(view.root.style.left, '472px');
  assert.equal(view.root.style.top, '344px');
});

test('buttons, nonprimary pointers, and right clicks do not start dragging', () => {
  for (const event of [{button: 2}, {isPrimary: false}, {buttonName: 'expand'}, {buttonName: 'close'}]) {
    const view = mount();
    show(view);
    const target = event.buttonName ? view.elements.get('[data-robot-camera="' + event.buttonName + '"]') : view.header;
    view.header.dispatch('pointerdown', {clientX: 540, clientY: 262, target, ...event});
    view.header.dispatch('pointermove', {clientX: 590, clientY: 312});
    assert.equal(view.header.hasPointerCapture(1), false);
    assert.equal(view.root.style.left, '');
    assert.equal(view.root.style.top, '');
  }
});

test('cancelled or lost pointers cannot continue moving the panel', () => {
  for (const ending of ['pointercancel', 'lostpointercapture']) {
    const view = mount();
    show(view);
    view.header.dispatch('pointerdown', {clientX: 540, clientY: 262});
    view.header.dispatch('pointermove', {clientX: 590, clientY: 312});
    view.header.dispatch(ending);
    view.header.dispatch('pointermove', {clientX: 640, clientY: 362});
    assert.equal(view.root.style.left, '450px');
    assert.equal(view.root.style.top, '250px');
  }
});

test('expansion suspends dragging and restores the saved position within the resized scene', () => {
  const view = mount();
  show(view);
  drag(view, 50, 50);
  view.elements.get('[data-robot-camera="expand"]').dispatch('click');
  assert.deepEqual(view.root.style, {left: '', top: '', right: '', bottom: ''});
  drag(view, -100, -100);
  assert.equal(view.root.style.left, '');
  assert.equal(view.root.style.top, '');
  view.container.bounds.width = 600;
  view.container.bounds.height = 400;
  view.resize(view.container);
  view.root.dispatch('keydown', {key: 'Escape'});
  assert.equal(view.root.style.left, '272px');
  assert.equal(view.root.style.top, '144px');
});

test('container and panel resizing keeps a dragged panel within reach', () => {
  const view = mount();
  show(view);
  drag(view, 50, 50);
  view.container.bounds.width = 600;
  view.container.bounds.height = 400;
  view.resize(view.container);
  assert.equal(view.root.style.left, '272px');
  assert.equal(view.root.style.top, '144px');
  view.root.bounds.width = 400;
  view.root.bounds.height = 300;
  view.resize(view.elements.get('[data-robot-camera="viewport"]'));
  assert.equal(view.root.style.left, '192px');
  assert.equal(view.root.style.top, '92px');
});

test('a hidden panel retains its dragged position and clamps it when shown again', () => {
  const view = mount();
  show(view);
  drag(view, 50, 50);
  view.elements.get('[data-robot-camera="close"]').dispatch('click');
  show(view);
  assert.equal(view.root.style.left, '450px');
  assert.equal(view.root.style.top, '250px');
  view.elements.get('[data-robot-camera="close"]').dispatch('click');
  view.container.bounds.width = 600;
  view.container.bounds.height = 400;
  show(view);
  assert.equal(view.root.style.left, '272px');
  assert.equal(view.root.style.top, '144px');
});

test('destroy releases an active drag and removes its input handlers', () => {
  const view = mount();
  show(view);
  view.header.dispatch('pointerdown', {clientX: 540, clientY: 262});
  view.header.dispatch('pointermove', {clientX: 590, clientY: 312});
  view.panel.destroy();
  assert.equal(view.header.hasPointerCapture(1), false);
  view.header.dispatch('pointermove', {clientX: 640, clientY: 362});
  assert.equal(view.root.style.left, '450px');
  assert.equal(view.root.style.top, '250px');
  assert.equal(view.header.handlers.size, 0);
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
