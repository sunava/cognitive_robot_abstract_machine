'use strict';

// %% actual browser loader and frame renderer
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const web = path.join(__dirname, '../../../cramera/src/cramera/web');
const THREE = require(path.join(web, 'vendor/three.min.js'));
const description = JSON.parse(fs.readFileSync(process.env.CRAMERA_HANDLE_DESCRIPTION, 'utf8'));
const handleNames = Object.freeze([
  'laboratory_drawer_handle',
  'laboratory_cabinet_lower_handle',
  'laboratory_cabinet_middle_handle',
  'laboratory_cabinet_right_handle',
  'laboratory_centrifuge_lid_handle',
  'laboratory_refrigerator_door_handle',
]);

/** Expose parsed XML as the DOM element interface accepted by the vendor loader. */
class ParsedDescriptionElement {
  constructor(element) {
    this.nodeName = element.nodeName;
    this.attributes = element.attributes;
    this.children = element.children.map(child => new ParsedDescriptionElement(child));
  }
  getAttribute(name) { return this.attributes[name] ?? null; }
  querySelector(selector) {
    const match = /^([^\[]+)\[([^=]+)="([^"]+)"\]$/.exec(selector);
    assert.ok(match, 'The loader uses an element and exact attribute selector');
    for (const child of this.children) {
      if (child.nodeName === match[1] && child.getAttribute(match[2]) === match[3]) return child;
      const descendant = child.querySelector(selector);
      if (descendant) return descendant;
    }
    return null;
  }
}

class HandleFrameDisplay {
  constructor() {
    this.scope = {THREE, Element: ParsedDescriptionElement, Document: class ParsedDocument {},
      frameDisplay: {visible: true, names: true, size: .1},
      makeLabel(name) {const label = new THREE.Group();label.userData.text = name;return label;},
    };
    vm.runInNewContext(fs.readFileSync(path.join(web, 'vendor/URDFLoader.js'), 'utf8'), this.scope);
    vm.runInNewContext(fs.readFileSync(path.join(web, 'core/frame-axes.js'), 'utf8'), this.scope);
    const panel = fs.readFileSync(path.join(web, 'panels/robot_scene/panel.js'), 'utf8');
    const begin = panel.indexOf('  function buildFrameTriad(frame) {');
    const end = panel.indexOf('  function disposeFrameTriad(triad) {', begin);
    assert.ok(begin >= 0 && end > begin, 'Use the scene panel frame renderer');
    vm.runInNewContext(panel.slice(begin, end), this.scope);
    const loader = new this.scope.URDFLoader();
    loader.parseVisual = false;
    loader.parseCollision = false;
    this.robot = loader.parse(new ParsedDescriptionElement(description));
    this.model = {name: 'laboratory', prefix: '', obj: this.robot};
    this.frames = this.scope.FrameAxes.framesOf([this.model], {});
    this.triads = new Map(this.frames.filter(frame => handleNames.includes(frame.name))
      .map(frame => [frame.name, this.scope.buildFrameTriad(frame)]));
    this.robot.updateMatrixWorld(true);
  }
  position(object) { return object.getWorldPosition(new THREE.Vector3()); }
}

test('all six empty handle links expose independently named and labeled TF axes', () => {
  const scene = new HandleFrameDisplay();
  assert.deepEqual([...scene.triads.keys()].sort(), [...handleNames].sort());
  for (const name of handleNames) {
    const link = scene.robot.links[name];
    const frame = scene.frames.find(item => item.name === name);
    const triad = scene.triads.get(name);
    assert.equal(link.urdfNode.children.length, 0);
    assert.equal(frame.object, link);
    assert.equal(triad.group.parent, link);
    assert.equal(triad.label.visible, true);
    assert.equal(triad.label.userData.text, name);
    assert.deepEqual(scene.position(triad.arms).toArray(), scene.position(link).toArray());
  }
});

test('hiding one handle frame preserves the other device and cabinet handles', () => {
  const scene = new HandleFrameDisplay();
  const target = scene.frames.find(frame => frame.name === 'laboratory_centrifuge_lid_handle');
  assert.ok(target);
  const shown = scene.scope.FrameAxes.visibleFrames(scene.frames, {frames: {[target.id]: true}});
  assert.deepEqual(Array.from(shown.filter(frame => handleNames.includes(frame.name)), frame => frame.name).sort(),
    handleNames.filter(name => name !== target.name).sort());
});

for (const [jointName, handleName, displacement] of [
  ['laboratory_drawer_joint', 'laboratory_drawer_handle', .1],
  ['laboratory_centrifuge_lid_joint', 'laboratory_centrifuge_lid_handle', .7],
  ['laboratory_refrigerator_door_joint', 'laboratory_refrigerator_door_handle', .7],
]) {
  test('the TF triad follows articulation of ' + jointName + ' and returns on closure', () => {
    const scene = new HandleFrameDisplay();
    const triad = scene.triads.get(handleName);
    assert.ok(triad, 'The authored moving handle has its own TF triad');
    const original = scene.position(triad.arms);
    const cabinet = scene.triads.get('laboratory_cabinet_lower_handle');
    const fixed = scene.position(cabinet.arms);
    scene.robot.joints[jointName].setJointValue(displacement);
    scene.robot.updateMatrixWorld(true);
    const moved = scene.position(triad.arms);
    assert.ok(moved.distanceTo(original) > .01, 'The handle axes follow the physical movable link');
    assert.deepEqual(moved.toArray(), scene.position(scene.robot.links[handleName]).toArray());
    assert.deepEqual(scene.position(cabinet.arms).toArray(), fixed.toArray());
    scene.robot.joints[jointName].setJointValue(0);
    scene.robot.updateMatrixWorld(true);
    assert.ok(scene.position(triad.arms).distanceTo(original) < 1e-12, 'Closing returns the handle frame to its original pose');
  });
}
