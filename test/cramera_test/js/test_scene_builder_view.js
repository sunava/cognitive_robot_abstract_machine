'use strict';

// %% real scene graph fixtures
const test = require('node:test');
const assert = require('node:assert/strict');
const THREE = require('../../../cramera/src/cramera/web/vendor/three.min.js');
const {SceneBuilderInstance, SceneBuilderView, SceneBuilderAssetLoader} = require('../../../cramera/src/cramera/web/core/scene-builder-view.js');

function model() {
  const root = new THREE.Group();
  const cabinet = new THREE.Group();
  const drawer = new THREE.Group();
  cabinet.name = 'cabinet'; cabinet.position.set(2, 0, 0);
  drawer.name = 'drawer'; drawer.position.set(0, 0, 1);
  root.add(cabinet); cabinet.add(drawer);
  root.links = {cabinet, drawer}; root.joints = {};
  return root;
}

function documentInstance(extra = {}) {
  return Object.assign({id:'environment-1', assetId:'cabinet', name:'Cabinet', position:[0,0,0], rotation:[0,0,0], linkOffsets:{}, jointPositions:{}}, extra);
}

function worldPosition(object) {
  object.updateWorldMatrix(true, true);
  return object.getWorldPosition(new THREE.Vector3()).toArray().map(value => Math.round(value * 1e9) / 1e9);
}

test('moving an instance relocates the complete robot or environment', () => {
  const root = model();
  const instance = new SceneBuilderInstance(THREE, documentInstance(), root);
  instance.apply(documentInstance({position:[1,2,3], rotation:[0,0,Math.PI / 2]}));
  assert.deepEqual(worldPosition(root.links.drawer), [1,4,4]);
});

test('furniture offsets preserve children and reset to authored transforms', () => {
  const root = model();
  const instance = new SceneBuilderInstance(THREE, documentInstance(), root);
  instance.apply(documentInstance({linkOffsets:{cabinet:{position:[.5,1,0],rotation:[0,0,Math.PI / 2]}}}));
  assert.deepEqual(worldPosition(root.links.drawer), [.5,3,1]);
  instance.apply(documentInstance());
  assert.deepEqual(worldPosition(root.links.drawer), [2,0,1]);
});

test('gizmo exports offsets relative to the original URDF parent frame', () => {
  const root = model();
  const instance = new SceneBuilderInstance(THREE, documentInstance(), root);
  root.links.cabinet.position.set(3, .4, 0);
  const offset = instance.transform('cabinet');
  assert.deepEqual(offset.position, [1,.4,0]);
  assert.deepEqual(offset.rotation, [0,0,0]);
});

test('distinct copies never share editable scene graph state', () => {
  const first = new SceneBuilderInstance(THREE, documentInstance(), model());
  const second = new SceneBuilderInstance(THREE, documentInstance({id:'second'}), model());
  first.apply(documentInstance({position:[4,0,0]}));
  assert.deepEqual(worldPosition(second.root.links.cabinet), [2,0,0]);
});

test('mesh unit scaling applies around the instance origin', () => {
  const root = model();
  const instance = new SceneBuilderInstance(THREE, documentInstance({position:[1,0,0],scale:[.001,.001,.001]}), root);
  assert.deepEqual(worldPosition(root.links.drawer), [1.002,0,.001]);
  assert.deepEqual(instance.transform().scale, [.001,.001,.001]);
});

test('roll pitch yaw follows URDF fixed-axis rotation order', () => {
  const root = model();
  new SceneBuilderInstance(THREE, documentInstance({rotation:[Math.PI / 2, Math.PI / 2, 0]}), root);
  assert.deepEqual(worldPosition(root.links.drawer), [0,-1,-2]);
});

test('joint edits preserve articulated descendants and restore authored state', () => {
  const root = model();
  const slide = new THREE.Group();
  slide.name = 'drawer_slide'; slide.jointValue = [0]; slide.jointType = 'prismatic'; slide.limit = {lower:0,upper:.4};
  slide.setJointValue = value => { slide.jointValue[0] = value; slide.position.y = value; };
  root.links.cabinet.add(slide); slide.add(root.links.drawer); root.joints = {drawer_slide:slide};
  const instance = new SceneBuilderInstance(THREE, documentInstance(), root);
  instance.apply(documentInstance({jointPositions:{drawer_slide:.3},linkOffsets:{cabinet:{position:[1,0,0],rotation:[0,0,0]}}}));
  assert.deepEqual(worldPosition(root.links.drawer), [3,.3,1]);
  instance.apply(documentInstance());
  assert.deepEqual(worldPosition(root.links.drawer), [2,0,1]);
});

test('nonfinite placement is rejected before touching the scene graph', () => {
  const root = model();
  const instance = new SceneBuilderInstance(THREE, documentInstance(), root);
  assert.throws(() => instance.apply(documentInstance({position:[NaN,0,0]})), TypeError);
  assert.deepEqual(worldPosition(root.links.cabinet), [2,0,0]);
});

test('asset requests reject off-origin URLs before loading', () => {
  const loader = new SceneBuilderAssetLoader({THREE, baseURL:'https://example.test/workshop/scene-builder.html'});
  assert.throws(() => loader.resolveURL('https://other.test/model.glb'), TypeError);
  assert.equal(loader.resolveURL('api/scene-builder/assets/chair/model.glb'), 'https://example.test/workshop/api/scene-builder/assets/chair/model.glb');
});

test('WebGL failure leaves document editing available without starting downloads', async () => {
  const errors = [];
  const view = new SceneBuilderView({container:{}, THREE, rendererFactory:() => { throw new Error('No graphics context'); }, onError:error => errors.push(error)});
  const document = {instances:[documentInstance()]};
  await view.setDocument(document, new Map());
  assert.equal(view.available, false);
  assert.equal(errors.length, 1);
  assert.deepEqual(view.document, document);
  view.dispose();
});

// %% asynchronous model ownership
function pendingView() {
  const loads = [];
  const view = Object.create(SceneBuilderView.prototype);
  Object.assign(view, {THREE, available:true, disposed:false, document:{instances:[]}, instances:new Map(), pending:new Map(), selection:null, content:new THREE.Group(), options:{}, onError:error => { throw error; }});
  view.assetLoader = {load:asset => new Promise(resolve => loads.push({asset,resolve}))};
  return {view,loads,assets:new Map([['cabinet',{url:'cabinet.urdf'}],['new-cabinet',{url:'other.urdf'}]])};
}

test('a late model download receives the newest placement without duplicate loads', async () => {
  const {view,loads,assets} = pendingView();
  const first = view.setDocument({instances:[documentInstance()]}, assets);
  const second = view.setDocument({instances:[documentInstance({position:[5,0,0]})]}, assets);
  assert.equal(loads.length, 1);
  loads[0].resolve(model());
  await Promise.all([first,second]);
  assert.deepEqual(worldPosition(view.instances.get('environment-1').root.links.cabinet), [7,0,0]);
});

test('a deleted pending model cannot reappear after its download completes', async () => {
  const {view,loads,assets} = pendingView();
  const loading = view.setDocument({instances:[documentInstance()]}, assets);
  await view.setDocument({instances:[]}, assets);
  loads[0].resolve(model()); await loading;
  assert.equal(view.instances.size, 0);
  assert.equal(view.content.children.length, 0);
});

test('replacing an asset prevents an older request from overwriting the new model', async () => {
  const {view,loads,assets} = pendingView();
  const first = view.setDocument({instances:[documentInstance()]}, assets);
  const second = view.setDocument({instances:[documentInstance({assetId:'new-cabinet',position:[1,0,0]})]}, assets);
  loads[1].resolve(model()); await second;
  loads[0].resolve(model()); await first;
  assert.equal(view.instances.get('environment-1').assetId, 'new-cabinet');
  assert.equal(view.content.children.length, 1);
});

test('refreshing the selected inspector does not detach an active transform drag', () => {
  const {view} = pendingView();
  const instance = new SceneBuilderInstance(THREE, documentInstance(), model());
  view.instances.set(instance.id, instance);
  let detachments = 0;
  view.gizmo = {object:instance.group, detach:() => { detachments += 1; }, attach:object => { view.gizmo.object = object; }, setMode() {}};
  view.selectionBox = {visible:true,setFromObject() {}};
  view.setSelection(instance.id);
  assert.equal(detachments, 0);
});

test('one gizmo drag commits one undoable placement after release', () => {
  const {view} = pendingView();
  const instance = new SceneBuilderInstance(THREE, documentInstance(), model());
  view.instances.set(instance.id, instance); view.selection = {instanceId:instance.id,linkName:null};
  const changes = []; view.options.onTransform = change => changes.push(change);
  view.beginTransform();
  instance.group.position.x = 1; view.changeTransform();
  instance.group.position.x = 2; view.changeTransform();
  assert.equal(changes.length, 0);
  view.finishTransform();
  assert.equal(changes.length, 1);
  assert.deepEqual(changes[0].position, [2,0,0]);
  view.finishTransform();
  assert.equal(changes.length, 1);
});

test('URDF readiness waits for every asynchronous mesh callback', async () => {
  class DelayedAsset {
    constructor(manager) { this.manager = manager; }
    load(url, finish) {
      const root = model();
      this.manager.itemStart(url);
      this.loadMeshCb('late.obj', this.manager, object => root.add(object));
      finish(root); this.manager.itemEnd(url);
    }
  }
  const loader = new SceneBuilderAssetLoader({THREE,URDFLoader:DelayedAsset,baseURL:'https://example.test/scene-builder.html'});
  let finishMesh;
  loader.mesh = () => new Promise(resolve => { finishMesh = resolve; });
  let ready = false;
  const loading = loader.load({url:'robot.urdf',format:'urdf'}).then(root => { ready = true; return root; });
  await Promise.resolve();
  assert.equal(ready, false);
  const mesh = new THREE.Group(); mesh.name = 'delayedMesh';
  finishMesh(mesh);
  const root = await loading;
  assert.equal(root.getObjectByName('delayedMesh'), mesh);
});
