// Unit tests for web/core/lab-environment.js (node:test).
// An imported scene's lights (lighting.json) become the viewer's image-based lighting:
// a room the size of the scene with a glowing quad per ceiling panel, and a key light
// along the sun.
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
const DATASET = path.join(__dirname, '..', 'dataset', 'usd_scene');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));
//: rounding tolerance of the matrix and colour arithmetic
const TOLERANCE = 1e-9;
//: rounding tolerance of vertex positions, which three stores in single precision
const VERTEX_TOLERANCE = 1e-6;

function load() {
  global.window = {};
  ['core/scene-rendering.js', 'core/shadow-fit.js', 'core/lab-environment.js'].forEach(function (file) {
    new Function(fs.readFileSync(path.join(WEB, file), 'utf8'))();
  });
  return window;
}

function fixture(name) {
  return JSON.parse(fs.readFileSync(path.join(DATASET, name), 'utf8'));
}

const { LabEnvironment, LightingDescription, SceneBounds, SceneRendering, ShadowFit } = load();
const SCENE = fixture('viewer_scene.json');
const LIGHTING = fixture('viewer_lighting.json');
const BOUNDS = SceneRendering.of(SCENE).shadowBounds;

function environment(payload, bounds) {
  return new LabEnvironment(LightingDescription.of(payload || LIGHTING), bounds === undefined ? BOUNDS : bounds);
}

function near(actual, expected, tolerance) {
  assert.ok(Math.abs(actual - expected) < (tolerance || TOLERANCE), actual + ' is not ' + expected);
}

function nearAll(actual, expected, tolerance) {
  assert.equal(actual.length, expected.length);
  actual.forEach(function (value, index) { near(value, expected[index], tolerance); });
}

function meshesNamed(scene, name) {
  scene.updateMatrixWorld(true);
  const found = [];
  scene.traverse(function (object) { if (object.isMesh && object.name === name) found.push(object); });
  return found;
}

function colorOf(mesh, face) {
  const material = face === undefined ? mesh.material : mesh.material[face];
  return material.color.toArray();
}

// the material index of every face of the room box, told apart by where three draws it:
// the floor lowest, the ceiling highest, the walls in between
function roomFaces(room) {
  room.updateMatrixWorld(true);
  const positions = room.geometry.getAttribute('position');
  const index = room.geometry.getIndex();
  const heights = room.geometry.groups.map(function (group) {
    let sum = 0;
    for (let corner = group.start; corner < group.start + group.count; corner += 1) {
      sum += new THREE.Vector3().fromBufferAttribute(positions, index.getX(corner)).applyMatrix4(room.matrixWorld).y;
    }
    return { material: group.materialIndex, height: sum / group.count };
  }).sort(function (first, second) { return first.height - second.height; });
  return {
    floor: heights[0].material,
    ceiling: heights[heights.length - 1].material,
    walls: heights.slice(1, -1).map(function (face) { return face.material; }),
  };
}

function scaled(color, factor) {
  return color.map(function (channel) { return channel * factor; });
}

// %% colour temperature
//: Rec. 709 luminance weights of linear sRGB
const LUMINANCE = [0.2126, 0.7152, 0.0722];
//: rounding of the published colour-space matrices
const COLORIMETRY_TOLERANCE = 1e-3;

function luminance(color) {
  return color.reduce(function (sum, channel, index) { return sum + channel * LUMINANCE[index]; }, 0);
}

test('a black body has the luminance of white at every temperature of the locus', function () {
  [2700, 4000, 5200, 6500, 10000].forEach(function (kelvin) {
    near(luminance(LabEnvironment.blackBody(kelvin)), 1, COLORIMETRY_TOLERANCE);
  });
});

test('a lower colour temperature is redder', function () {
  const [red, green, blue] = LabEnvironment.blackBody(3000);
  assert.ok(red > green && green > blue);
});

test('a higher colour temperature is bluer', function () {
  const [red, , blue] = LabEnvironment.blackBody(10000);
  assert.ok(blue > red);
});

test('temperatures beyond the fitted locus keep the colour at its end', function () {
  assert.deepEqual([LabEnvironment.blackBody(1000), LabEnvironment.blackBody(40000)],
    [LabEnvironment.blackBody(1667), LabEnvironment.blackBody(25000)]);
});

test('no black-body channel is negative', function () {
  [1000, 1900, 2700, 40000].forEach(function (kelvin) {
    LabEnvironment.blackBody(kelvin).forEach(function (channel) { assert.ok(channel >= 0); });
  });
});

// %% reading the description
test('a panel without an area is left out', function () {
  const flat = Object.assign({}, LIGHTING.panels[0], { width: 0 });
  const description = LightingDescription.of({ panels: [flat, LIGHTING.panels[1]] });
  assert.deepEqual(description.panels.map(function (panel) { return panel.center; }), [LIGHTING.panels[1].center]);
});

test('a sun without a direction is left out', function () {
  const description = LightingDescription.of({ sun: Object.assign({}, LIGHTING.sun, { direction: [0, 0, 0] }) });
  assert.equal(description.sun, null);
});

test('a description without lights has no sun, no dome and no panels', function () {
  const description = LightingDescription.of({});
  assert.deepEqual([description.sun, description.dome, description.panels], [null, null, []]);
});

test('a light shows its black body when it states only a colour temperature', function () {
  const [panel] = LightingDescription.of(LIGHTING).panels;
  assert.deepEqual(panel.color, LabEnvironment.blackBody(LIGHTING.panels[0].colorTemperature));
});

test('a light\'s colour tints its black body', function () {
  const tint = [0.5, 1, 0.25];
  const [panel] = LightingDescription.of({ panels: [Object.assign({}, LIGHTING.panels[0], { color: tint })] }).panels;
  const body = LabEnvironment.blackBody(LIGHTING.panels[0].colorTemperature);
  assert.deepEqual(panel.color, tint.map(function (channel, index) { return channel * body[index]; }));
});

test('a light without a colour temperature shows its colour as given', function () {
  const color = [0.2, 0.4, 0.6];
  const sun = Object.assign({}, LIGHTING.sun, { colorTemperature: null, color: color });
  assert.deepEqual(LightingDescription.of({ sun: sun }).sun.color, color);
});

test('a light without colour and colour temperature is white', function () {
  const sun = { direction: LIGHTING.sun.direction, intensity: LIGHTING.sun.intensity };
  assert.deepEqual(LightingDescription.of({ sun: sun }).sun.color, [1, 1, 1]);
});

test('a description without usable lights is empty, so the viewer keeps its own', function () {
  assert.deepEqual([LightingDescription.of(null).empty, LightingDescription.of({ panels: [{ width: 1 }] }).empty],
    [true, true]);
});

test('panels and a dome light the room from all around; a sun alone does not', function () {
  assert.deepEqual([{ sun: LIGHTING.sun }, { dome: LIGHTING.dome }, { panels: LIGHTING.panels }].map(function (payload) {
    return LightingDescription.of(payload).surrounding;
  }), [false, true, true]);
});

test('any one usable light makes a description', function () {
  assert.deepEqual([{ sun: LIGHTING.sun }, { dome: LIGHTING.dome }, { panels: LIGHTING.panels }].map(function (payload) {
    return LightingDescription.of(payload).empty;
  }), [false, false, false]);
});

// %% brightness reference
test('the reference luminance is the panels\' light spread over the floor of the room', function () {
  const size = BOUNDS.size();
  const output = LIGHTING.panels.reduce(function (sum, panel) {
    return sum + panel.intensity * panel.width * panel.height;
  }, 0);
  near(environment().referenceLuminance, output / (size[0] * size[1]));
});

test('without panels the dome is the reference', function () {
  assert.equal(environment({ sun: LIGHTING.sun, dome: LIGHTING.dome }).referenceLuminance, LIGHTING.dome.intensity);
});

// %% key light: the strongest direct light
// the lighting fixture with its sun set to give the floor `factor` times the panels' irradiance
function withSunOutshiningPanels(factor) {
  const ceilingIrradiance = Math.PI * environment().ceilingLuminance;
  return Object.assign({}, LIGHTING, { sun: Object.assign({}, LIGHTING.sun, { intensity: ceilingIrradiance * factor }) });
}

test('a sun that outshines the panels is the key light, in its colour', function () {
  const lab = environment(withSunOutshiningPanels(2));
  const key = lab.keyLight();
  assert.deepEqual([key.direction, key.color],
    [LIGHTING.sun.direction, LabEnvironment.blackBody(LIGHTING.sun.colorTemperature)]);
  near(key.intensity, lab.lighting.sun.intensity / Math.PI / lab.referenceLuminance);
});

test('panels that outshine the sun light the room from above in the room\'s light', function () {
  const lab = environment(withSunOutshiningPanels(0.5));
  const key = lab.keyLight();
  assert.deepEqual([key.direction, key.color], [LIGHTING.panels[0].normal, lab.lightColor()]);
  near(key.intensity, LabEnvironment.KEY_SHARE * lab.ceilingLuminance / lab.referenceLuminance);
});

test('panels without a sun light the room from above', function () {
  assert.deepEqual(environment({ panels: LIGHTING.panels }).keyLight().direction, LIGHTING.panels[0].normal);
});

test('panels facing each other give no direction to shine from', function () {
  const up = Object.assign({}, LIGHTING.panels[0], { normal: [0, 0, 1] });
  const down = Object.assign({}, LIGHTING.panels[0], { normal: [0, 0, -1] });
  assert.equal(environment({ panels: [up, down] }).keyLight(), null);
});

test('the shadows of a key along the sun cover the scene\'s bounds', function () {
  assert.equal(environment(withSunOutshiningPanels(2)).keyLight().shadowBounds, BOUNDS);
});

test('the shadows of a key from the panels stop below the panels\' fixtures', function () {
  const high = SceneBounds.of([BOUNDS.min, [BOUNDS.max[0], BOUNDS.max[1], 4.0]]);
  const lowest = Math.min.apply(null, LIGHTING.panels.map(function (panel) { return panel.center[2]; }));
  const shadow = environment({ panels: LIGHTING.panels }, high).keyLight().shadowBounds;
  assert.deepEqual([shadow.min, shadow.max.slice(0, 2)], [high.min, high.max.slice(0, 2)]);
  near(shadow.max[2], lowest - LabEnvironment.FIXTURE_DEPTH);
});

test('bounds already below the panels\' fixtures bound the shadows of a key from the panels', function () {
  const shadow = environment({ panels: LIGHTING.panels }).keyLight().shadowBounds;
  assert.deepEqual([shadow.min, shadow.max], [BOUNDS.min, BOUNDS.max]);
});

test('without bounds the shadows of a key from the panels cover the room below them', function () {
  const lab = environment({ panels: LIGHTING.panels }, null);
  const shadow = lab.keyLight().shadowBounds;
  const lowest = Math.min.apply(null, LIGHTING.panels.map(function (panel) { return panel.center[2]; }));
  assert.deepEqual([shadow.min, shadow.max.slice(0, 2)], [lab.room.min, lab.room.max.slice(0, 2)]);
  near(shadow.max[2], lowest - LabEnvironment.FIXTURE_DEPTH);
});

test('a panel hanging low in the room leaves the shadows of the ceiling\'s key as they are', function () {
  const high = SceneBounds.of([BOUNDS.min, [BOUNDS.max[0], BOUNDS.max[1], 4.0]]);
  const ceiling = Math.min.apply(null, LIGHTING.panels.map(function (panel) { return panel.center[2]; }));
  const lamp = Object.assign({}, LIGHTING.panels[0], { center: [0, 0, 1.06], width: 0.176, height: 0.006 });
  const shadow = environment({ panels: LIGHTING.panels.concat([lamp]) }, high).keyLight().shadowBounds;
  near(shadow.max[2], ceiling - LabEnvironment.FIXTURE_DEPTH);
});

test('a description without sun and panels has no key light of its own', function () {
  assert.equal(environment({ dome: LIGHTING.dome }).keyLight(), null);
});

test('a key light from the panels carries its share of their light spread over the floor', function () {
  assert.equal(environment({ panels: LIGHTING.panels }).keyLight().intensity, LabEnvironment.KEY_SHARE);
});

// %% colour of the room's light
test('the room is filled with the panels\' light', function () {
  const lab = environment();
  assert.deepEqual(lab.lightColor(), lab.panelColor());
});

test('without panels the room is filled with the dome\'s light', function () {
  assert.deepEqual(environment({ sun: LIGHTING.sun, dome: LIGHTING.dome }).lightColor(), LIGHTING.dome.color);
});

// %% the room
test('the room spans the bounds and closes just above the highest panel', function () {
  const room = environment().room;
  const ceiling = Math.max.apply(null, LIGHTING.panels.map(function (panel) { return panel.center[2]; }));
  assert.deepEqual([room.min, room.max.slice(0, 2)], [BOUNDS.min, BOUNDS.max.slice(0, 2)]);
  near(room.max[2], ceiling + LabEnvironment.PANEL_GAP);
});

test('without bounds the room covers every panel and stands on the ground', function () {
  const room = environment(LIGHTING, null).room;
  LIGHTING.panels.forEach(function (panel) {
    const reach = Math.max(panel.width, panel.height) / 2;
    assert.ok(room.min[0] <= panel.center[0] - reach && room.max[0] >= panel.center[0] + reach);
    assert.ok(room.min[1] <= panel.center[1] - reach && room.max[1] >= panel.center[1] + reach);
  });
  assert.equal(room.min[2], 0);
});

test('the environment is seen from the centre of the room', function () {
  const lab = environment();
  assert.deepEqual(lab.probe(), lab.room.center());
});

test('the room box of the rendered scene has the size of the room around the probe', function () {
  const lab = environment();
  const [room] = meshesNamed(lab.scene(THREE), LabEnvironment.ROOM);
  const box = new THREE.Box3().setFromObject(room);
  const probe = lab.probe();
  const corners = [lab.room.min, lab.room.max].map(function (corner) {
    return ShadowFit.toThree(THREE, corner.map(function (value, axis) { return value - probe[axis]; }));
  });
  const expected = new THREE.Box3().setFromPoints(corners);
  nearAll(box.min.toArray(), expected.min.toArray(), VERTEX_TOLERANCE);
  nearAll(box.max.toArray(), expected.max.toArray(), VERTEX_TOLERANCE);
});

// %% panels
test('every panel glows as a quad at its place, facing along its normal', function () {
  const lab = environment();
  const quads = meshesNamed(lab.scene(THREE), LabEnvironment.PANEL);
  const probe = lab.probe();
  assert.equal(quads.length, LIGHTING.panels.length);
  quads.forEach(function (quad, index) {
    const panel = LIGHTING.panels[index];
    const place = ShadowFit.toThree(THREE, panel.center.map(function (value, axis) { return value - probe[axis]; }));
    nearAll(quad.getWorldPosition(new THREE.Vector3()).toArray(), place.toArray());
    nearAll(quad.getWorldDirection(new THREE.Vector3()).toArray(), ShadowFit.toThree(THREE, panel.normal).toArray());
  });
});

test('a panel quad has the panel\'s width and height', function () {
  const [quad] = meshesNamed(environment().scene(THREE), LabEnvironment.PANEL);
  quad.geometry.computeBoundingBox();
  const size = quad.geometry.boundingBox.getSize(new THREE.Vector3());
  nearAll([size.x, size.y], [LIGHTING.panels[0].width, LIGHTING.panels[0].height], VERTEX_TOLERANCE);
});

test('a panel glows in its colour at its luminance over the reference', function () {
  const lab = environment();
  const quads = meshesNamed(lab.scene(THREE), LabEnvironment.PANEL);
  quads.forEach(function (quad, index) {
    const panel = LIGHTING.panels[index];
    nearAll(colorOf(quad),
      scaled(LabEnvironment.blackBody(panel.colorTemperature), panel.intensity / lab.referenceLuminance));
  });
});

// %% the room's surfaces, lit once by the ceiling
test('the floor reflects the ceiling\'s light', function () {
  const lab = environment();
  const [room] = meshesNamed(lab.scene(THREE), LabEnvironment.ROOM);
  nearAll(colorOf(room, roomFaces(room).floor), scaled(lab.lightColor(), LabEnvironment.FLOOR_REFLECTANCE));
});

test('the ceiling between the panels reflects the floor\'s light', function () {
  const lab = environment();
  const [room] = meshesNamed(lab.scene(THREE), LabEnvironment.ROOM);
  nearAll(colorOf(room, roomFaces(room).ceiling),
    scaled(lab.lightColor(), LabEnvironment.FLOOR_REFLECTANCE * LabEnvironment.CEILING_REFLECTANCE));
});

test('the walls reflect half the ceiling\'s and half the floor\'s light', function () {
  const lab = environment();
  const [room] = meshesNamed(lab.scene(THREE), LabEnvironment.ROOM);
  const wall = LabEnvironment.WALL_REFLECTANCE * (1 + LabEnvironment.FLOOR_REFLECTANCE) / 2;
  const walls = roomFaces(room).walls;
  assert.equal(walls.length, 4);
  walls.forEach(function (face) {
    nearAll(colorOf(room, face), scaled(lab.lightColor(), wall));
  });
});

test('the colour of several panels mixes by their output', function () {
  const lab = environment();
  const outputs = LIGHTING.panels.map(function (panel) { return panel.intensity * panel.width * panel.height; });
  const total = outputs[0] + outputs[1];
  const expected = [0, 1, 2].map(function (channel) {
    return LIGHTING.panels.reduce(function (sum, panel, index) {
      return sum + LabEnvironment.blackBody(panel.colorTemperature)[channel] * outputs[index] / total;
    }, 0);
  });
  nearAll(lab.panelColor(), expected);
});

test('without panels the ceiling is the dome\'s sky', function () {
  const lab = environment({ dome: LIGHTING.dome });
  const [room] = meshesNamed(lab.scene(THREE), LabEnvironment.ROOM);
  nearAll(colorOf(room, roomFaces(room).ceiling), LIGHTING.dome.color);
});

test('the cube render reaches the farthest corner of the room and the panel gap beyond', function () {
  const lab = environment();
  const size = lab.room.size();
  near(lab.reach(), Math.hypot(size[0], size[1], size[2]) / 2 + LabEnvironment.PANEL_GAP);
});
