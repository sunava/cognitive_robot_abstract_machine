'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const THREE = require('../../../cramera/src/cramera/web/vendor/three.min.js');
const Manual = require('../../../cramera/src/cramera/web/core/laboratory-manual.js');

// %% a DOM boundary with measurable annotation cards
class AnnotationElement {
  constructor(tag, document) {
    this.tagName = tag;
    this.ownerDocument = document;
    this.children = [];
    this.attributes = {};
    this.style = {};
    this.dataset = {};
    this.listeners = new Map();
    this.hidden = false;
    this.textContent = '';
    this.offsetHeight = 86;
    this.clientWidth = 1500;
    this.clientHeight = 1000;
    this.className = '';
  }
  appendChild(child) { this.children.push(child); child.parentNode = this; return child; }
  removeChild(child) { this.children.splice(this.children.indexOf(child), 1); child.parentNode = null; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name]; }
  addEventListener(name, listener) { this.listeners.set(name, listener); }
  removeEventListener(name, listener) { assert.equal(this.listeners.get(name), listener); this.listeners.delete(name); }
  click() { this.listeners.get('click')({stopPropagation() {}}); }
  getBoundingClientRect() { return {width: this.clientWidth, height: this.clientHeight, left: 0, top: 0}; }
}

class ArticulatedManualScene {
  constructor(parts) {
    this.document = {createElement: name => new AnnotationElement(name, this.document), createElementNS: (namespace, name) => new AnnotationElement(name, this.document)};
    this.host = new AnnotationElement('section', this.document);
    this.root = new THREE.Group();
    this.camera = new THREE.PerspectiveCamera(45, 1.5, .01, 100);
    this.camera.position.set(0, 0, 8);
    this.camera.lookAt(0, 0, 0);
    this.mesh = new THREE.Mesh(new THREE.BoxGeometry(.8, .5, .2), new THREE.MeshBasicMaterial());
    this.mesh.name = 'Centrifuge_lid';
    this.mesh.position.set(1, 0, 0);
    this.hinge = new THREE.Group();
    this.hinge.add(this.mesh);
    this.root.add(this.hinge);
    this.parts = parts || [{id: 'lid', number: 1, title: 'Safety lid', description: 'Close before starting.', meshNames: ['Centrifuge lid'], side: 'left', describe: snapshot => snapshot ? snapshot.status : 'No observation'}];
    this.overlay = Manual.create({THREE, root: this.root, host: this.host, camera: () => this.camera, parts: this.parts, reservedRight: () => 300});
  }
  enable() { this.overlay.setVisible(true); }
  destroy() { this.overlay.destroy(); }
}

test('annotations start hidden and articulated anchors follow the authored mesh bounds', () => {
  const scene = new ArticulatedManualScene();
  assert.equal(scene.overlay.element.hidden, true);
  scene.enable();
  const entry = scene.overlay.entries[0];
  const first = Number.parseFloat(entry.hotspot.style.left);
  scene.hinge.rotation.z = Math.PI / 2;
  scene.overlay.render();
  assert.ok(first > scene.host.clientWidth / 2);
  assert.equal(Number.parseFloat(entry.hotspot.style.left), scene.host.clientWidth / 2);
  assert.ok(Number.parseFloat(entry.hotspot.style.top) < scene.host.clientHeight / 2);
  assert.equal(entry.hotspot.hidden, false);
  scene.destroy();
});

test('clipped and behind-camera points never produce floating labels or connecting lines', () => {
  const scene = new ArticulatedManualScene();
  scene.enable();
  const entry = scene.overlay.entries[0];
  for (const position of [[0, 0, 9], [100, 0, 0], [0, 0, -200]]) {
    scene.mesh.position.set(...position);
    scene.overlay.render();
    assert.equal(entry.hotspot.hidden, true);
    assert.equal(entry.card.hidden, true);
    assert.equal(entry.line.getAttribute('display'), 'none');
  }
  scene.destroy();
});

test('visible predicates hide interior parts and observations update a separate text field', () => {
  const parts = [{id: 'rotor', number: 2, title: 'Rotor', description: 'Carries the sample holders.', meshNames: ['Centrifuge lid'], side: 'right', visible: snapshot => Boolean(snapshot && snapshot.lidOpen), describe: snapshot => snapshot.rpm + ' RPM'}];
  const scene = new ArticulatedManualScene(parts);
  scene.enable();
  const entry = scene.overlay.entries[0];
  assert.equal(entry.hotspot.hidden, true);
  scene.overlay.update({lidOpen: true, rpm: 120});
  assert.equal(entry.hotspot.hidden, false);
  assert.equal(entry.description.textContent, parts[0].description);
  assert.equal(entry.status.textContent, '120 RPM');
  scene.overlay.update({lidOpen: false, rpm: 0});
  assert.equal(entry.hotspot.hidden, true);
  scene.destroy();
});

test('rails keep nearby descriptions apart and outside reserved controls', () => {
  const parts = [0, 1, 2, 3, 4, 5].map(number => ({id: 'part-' + number, number: number + 1, title: 'Part ' + number, description: 'An explanation.', meshNames: ['Centrifuge lid'], side: number < 3 ? 'left' : 'right'}));
  const scene = new ArticulatedManualScene(parts);
  scene.enable();
  assert.equal(scene.overlay.element.dataset.layout, 'rails');
  for (const side of ['left', 'right']) {
    const entries = scene.overlay.entries.filter(entry => entry.part.side === side).sort((left, right) => Number.parseFloat(left.card.style.top) - Number.parseFloat(right.card.style.top));
    for (let index = 1; index < entries.length; index += 1) assert.ok(Number.parseFloat(entries[index].card.style.top) >= Number.parseFloat(entries[index - 1].card.style.top) + entries[index - 1].card.offsetHeight);
    for (const entry of entries) {
      assert.ok(Number.parseFloat(entry.card.style.top) >= (side === 'left' ? 300 : 70));
      assert.ok(Number.parseFloat(entry.card.style.left) + Number.parseFloat(entry.card.style.width) <= scene.host.clientWidth - 300);
      assert.ok(Number.parseFloat(entry.card.style.top) + entry.card.offsetHeight <= scene.host.clientHeight);
    }
  }
  scene.destroy();
});

test('compact views select one accessible detail card without losing any visible part', () => {
  const parts = [1, 2].map(number => ({id: 'part-' + number, number, title: 'Part ' + number, description: 'Explanation ' + number, meshNames: ['Centrifuge lid'], side: 'left'}));
  const scene = new ArticulatedManualScene(parts);
  scene.host.clientWidth = 720;
  scene.enable();
  assert.equal(scene.overlay.element.dataset.layout, 'compact');
  assert.equal(scene.overlay.compact.hidden, false);
  scene.overlay.entries[1].chip.click();
  assert.equal(scene.overlay.compactTitle.textContent, '2 · ' + parts[1].title);
  assert.equal(scene.overlay.compactDescription.textContent, parts[1].description);
  assert.equal(scene.overlay.entries[1].hotspot.getAttribute('aria-pressed'), 'true');
  assert.equal(scene.overlay.entries[0].hotspot.getAttribute('aria-pressed'), 'false');
  assert.equal(scene.overlay.entries[1].hotspot.getAttribute('aria-label'), '2. ' + parts[1].title);
  scene.destroy();
});

test('compact detail retains one illustrated leader that follows the selected articulation', () => {
  const parts = [1, 2].map(number => ({id: 'part-' + number, number, title: 'Part ' + number, description: 'Explanation ' + number, meshNames: ['Centrifuge lid'], side: 'left'}));
  const scene = new ArticulatedManualScene(parts);
  scene.host.clientWidth = 720;
  scene.enable();
  const [first, second] = scene.overlay.entries;
  assert.equal(first.line.getAttribute('display'), '');
  assert.equal(second.line.getAttribute('display'), 'none');
  const points = first.line.getAttribute('points').split(' ').map(point => point.split(',').map(Number));
  assert.deepEqual(points[0], [Number.parseFloat(first.hotspot.style.left), Number.parseFloat(first.hotspot.style.top)]);
  const left = Number.parseFloat(scene.overlay.compact.style.left);
  const right = left + Number.parseFloat(scene.overlay.compact.style.width);
  const top = scene.host.clientHeight - Number.parseFloat(scene.overlay.compact.style.bottom) - scene.overlay.compact.offsetHeight;
  assert.equal(points.at(-1)[1], top);
  assert.ok(points.at(-1)[0] >= left && points.at(-1)[0] <= right);
  second.chip.click();
  assert.equal(first.line.getAttribute('display'), 'none');
  assert.equal(second.line.getAttribute('display'), '');
  scene.hinge.rotation.z = Math.PI / 2;
  scene.overlay.render();
  const movedStart = second.line.getAttribute('points').split(' ')[0].split(',').map(Number);
  assert.deepEqual(movedStart, [Number.parseFloat(second.hotspot.style.left), Number.parseFloat(second.hotspot.style.top)]);
  assert.notDeepEqual(movedStart, points[0]);
  scene.destroy();
});

test('narrow views keep a readable compact card or defer it while controls cover the space', () => {
  const scene = new ArticulatedManualScene();
  scene.host.clientWidth = 360;
  scene.overlay.options.reservedRight = () => 0;
  scene.enable();
  assert.equal(scene.overlay.compact.hidden, false);
  const cardWidth = Number.parseFloat(scene.overlay.compact.style.width);
  assert.ok(cardWidth >= 220);
  assert.ok(cardWidth + Number.parseFloat(scene.overlay.compact.style.left) <= scene.host.clientWidth);
  scene.overlay.options.reservedRight = () => 270;
  scene.overlay.render();
  assert.equal(scene.overlay.compact.hidden, true);
  assert.equal(scene.overlay.entries[0].hotspot.hidden, false);
  assert.equal(scene.overlay.entries[0].line.getAttribute('display'), 'none');
  assert.ok(scene.overlay.entries[0].hotspot.title.includes(scene.parts[0].description));
  scene.destroy();
});

test('late-loaded assets resolve by authored metadata and guide content remains literal text', () => {
  const scene = new ArticulatedManualScene([{id: 'screen', number: 1, title: '<img src=x>', description: '<script>unsafe()</script>', meshNames: ['Actual authored name'], side: 'right'}]);
  scene.enable();
  const entry = scene.overlay.entries[0];
  assert.equal(entry.hotspot.hidden, true);
  scene.mesh.userData.name = 'Actual_authored_name';
  scene.overlay.render();
  assert.equal(entry.hotspot.hidden, false);
  assert.equal(entry.title.textContent, scene.parts[0].title);
  assert.equal(entry.description.textContent, scene.parts[0].description);
  scene.destroy();
});

test('removed scene components cannot leave stale annotations behind', () => {
  const scene = new ArticulatedManualScene();
  scene.enable();
  const entry = scene.overlay.entries[0];
  scene.hinge.remove(scene.mesh);
  scene.overlay.render();
  assert.equal(entry.hotspot.hidden, true);
  const replacement = new THREE.Mesh(new THREE.BoxGeometry(.5, .5, .5), scene.mesh.material);
  replacement.name = scene.mesh.name;
  scene.hinge.add(replacement);
  scene.overlay.render();
  assert.equal(entry.hotspot.hidden, false);
  assert.equal(Number.parseFloat(entry.hotspot.style.left), scene.host.clientWidth / 2);
  scene.destroy();
});

test('an absent control panel permits labels to use the full available rail height', () => {
  const scene = new ArticulatedManualScene();
  scene.overlay.options.reservedTopLeft = () => 0;
  scene.overlay.options.reservedTopRight = () => 0;
  scene.host.clientHeight = 200;
  scene.enable();
  assert.equal(scene.overlay.element.dataset.layout, 'rails');
  assert.ok(Number.parseFloat(scene.overlay.entries[0].card.style.top) < scene.host.clientHeight);
  scene.destroy();
});

test('teardown removes event listeners and DOM without mutating physical objects', () => {
  const scene = new ArticulatedManualScene();
  const geometry = scene.mesh.geometry;
  const material = scene.mesh.material;
  const transform = scene.mesh.position.clone();
  scene.enable();
  const controls = scene.overlay.entries.flatMap(entry => [entry.hotspot, entry.card, entry.chip]);
  scene.destroy();
  assert.deepEqual(scene.host.children, []);
  assert.deepEqual(controls.map(control => control.listeners.size), [0, 0, 0]);
  assert.equal(scene.mesh.geometry, geometry);
  assert.equal(scene.mesh.material, material);
  assert.equal(scene.mesh.visible, true);
  assert.ok(scene.mesh.position.equals(transform));
  assert.deepEqual(scene.root.children, [scene.hinge]);
  scene.overlay.render();
});
