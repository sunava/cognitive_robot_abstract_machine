// Unit tests for web/core/obj-mesh-material.js (node:test).
// THREE.OBJLoader always wraps its result in a Group, even for an OBJ with exactly
// one mesh and no per-face materials of its own (e.g. one written by trimesh).
// URDFLoader only assigns a URDF <material> to an object that is itself a Mesh, so
// that single mesh must be found and handed back in place of its wrapping Group.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  global.window = {};
  new Function(fs.readFileSync(path.join(WEB, 'core/obj-mesh-material.js'), 'utf8'))();
}

function group(children) {
  return { isMesh: false, isGroup: true, children: children || [] };
}

function mesh(name) {
  return { isMesh: true, name: name, children: [] };
}

test('a group wrapping exactly one mesh resolves to that mesh', function () {
  load();
  const theMesh = mesh('board');
  assert.strictEqual(window.ObjMeshMaterial.singleMeshChild(group([theMesh])), theMesh);
});

test('a mesh found at any nesting depth still resolves', function () {
  load();
  const theMesh = mesh('shape');
  const nested = group([group([theMesh])]);
  assert.strictEqual(window.ObjMeshMaterial.singleMeshChild(nested), theMesh);
});

test('a group wrapping several meshes resolves to nothing', function () {
  load();
  const wrapped = group([mesh('a'), mesh('b')]);
  assert.strictEqual(window.ObjMeshMaterial.singleMeshChild(wrapped), null);
});

test('a group wrapping no mesh at all resolves to nothing', function () {
  load();
  assert.strictEqual(window.ObjMeshMaterial.singleMeshChild(group([])), null);
});

test('an object that is already a mesh resolves to itself', function () {
  load();
  const theMesh = mesh('primitive');
  assert.strictEqual(window.ObjMeshMaterial.singleMeshChild(theMesh), theMesh);
});
