'use strict';

const assert = require('node:assert/strict');
const filesystem = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const WEB = path.join(__dirname, '../../../cramera/src/cramera/web');
const THREE = require(path.join(WEB, 'vendor/three.min.js'));

// %% loaded robot camera sources
function loadRobotCamera() {
  const scope = {};
  new Function('window', filesystem.readFileSync(path.join(WEB, 'core/robot-camera.js'), 'utf8'))(scope);
  return scope.RobotCamera;
}

function makeModel(identifier, linkNames, robot = true) {
  const object = new THREE.Group();
  object.links = {};
  for (const name of linkNames) {
    const link = new THREE.Group();
    link.name = name;
    link.isURDFLink = true;
    object.links[name] = link;
    object.add(link);
  }
  return {identifier, name: identifier, prefix: identifier, robot, obj: object};
}

test('only loaded robot models contribute camera sources', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('robot', ['camera_optical_frame']);
  const environment = makeModel('room', ['camera_optical_frame'], false);
  const pending = {identifier: 'pending', robot: true, obj: null};

  const sources = RobotCamera.sources([environment, pending, robot]);

  assert.deepEqual(sources.map(source => source.link), [robot.obj.links.camera_optical_frame]);
  assert.equal(sources[0].optical, true);
});

test('optical camera frames suppress body frame fallbacks for the same robot', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('pr2', [
    'head_pan_link', 'head_tilt_link', 'camera_link',
    'r_forearm_cam_optical_frame', 'wide_stereo_optical_frame',
  ]);

  const sources = RobotCamera.sources([robot]);

  assert.deepEqual(sources.map(source => source.link), [
    robot.obj.links.wide_stereo_optical_frame,
    robot.obj.links.r_forearm_cam_optical_frame,
  ]);
  assert.deepEqual(sources.map(source => source.optical), [true, true]);
});

test('a robot without optical frames can use its ordinary camera link', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('camera_robot', ['base_link', 'camera_link']);

  const sources = RobotCamera.sources([robot]);

  assert.deepEqual(sources.map(source => source.link), [robot.obj.links.camera_link]);
  assert.equal(sources[0].optical, false);
});

test('a robot without camera frames can use its head link', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('head_robot', ['base_link', 'head_link']);

  const sources = RobotCamera.sources([robot]);

  assert.deepEqual(sources.map(source => source.link), [robot.obj.links.head_link]);
  assert.equal(sources[0].optical, false);
});

test('HSRB camera frames use optical axes even without optical in their names', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('hsrb', ['hand_camera_frame', 'head_center_camera_frame']);

  const sources = RobotCamera.sources([robot]);

  assert.deepEqual(sources.map(source => source.link), [
    robot.obj.links.head_center_camera_frame,
    robot.obj.links.hand_camera_frame,
  ]);
  assert.deepEqual(sources.map(source => source.optical), [true, true]);
});

test('simulator camera frames do not duplicate the physical camera choices', () => {
  const RobotCamera = loadRobotCamera();
  const robot = makeModel('hsrb', ['head_center_camera_gazebo_frame', 'head_center_camera_frame']);

  const sources = RobotCamera.sources([robot]);

  assert.deepEqual(sources.map(source => source.link), [robot.obj.links.head_center_camera_frame]);
});

test('repeated camera names have distinct stable identities across robot instances', () => {
  const RobotCamera = loadRobotCamera();
  const first = makeModel('first', ['camera_optical_frame']);
  const second = makeModel('second', ['camera_optical_frame']);

  const sources = RobotCamera.sources([second, first]);
  const reordered = RobotCamera.sources([first, second]);

  assert.equal(new Set(sources.map(source => source.id)).size, 2);
  assert.equal(new Set(sources.map(source => source.label)).size, 2);
  assert.deepEqual(sources.map(source => source.id), reordered.map(source => source.id));
  assert.deepEqual(sources.map(source => source.link), reordered.map(source => source.link));
});

test('a scene without robot camera or head frames has no source', () => {
  const RobotCamera = loadRobotCamera();
  assert.deepEqual(RobotCamera.sources([makeModel('arm', ['base_link', 'gripper_link'])]), []);
});

// %% articulated viewpoints in the rotated viewer world
class ArticulatedViewpoint {
  constructor() {
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    this.robot = new THREE.Group();
    this.robot.position.set(3, -2, 0.5);
    this.head = new THREE.Group();
    this.head.position.set(0.2, 0, 1.4);
    this.world.add(this.robot);
    this.robot.add(this.head);
  }
}

function assertVector(actual, expected) {
  const tolerance = 1e-10;
  assert.ok(actual.distanceTo(new THREE.Vector3(...expected)) < tolerance,
    'expected ' + expected + ', received ' + actual.toArray());
}

function cameraUp(camera) {
  return new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
}

test('ordinary frames look along local positive X with positive Z up', () => {
  const RobotCamera = loadRobotCamera();
  const scene = new ArticulatedViewpoint();
  const pose = new RobotCamera.Pose(THREE);

  assert.equal(pose.update({link: scene.head, optical: false}, 640, 360), true);

  assert.equal(pose.camera.isPerspectiveCamera, true);
  assertVector(pose.camera.position, [3.2, 1.9, 2]);
  assertVector(pose.camera.getWorldDirection(new THREE.Vector3()), [1, 0, 0]);
  assertVector(cameraUp(pose.camera), [0, 1, 0]);
});

test('optical frames look along local positive Z with negative Y up', () => {
  const RobotCamera = loadRobotCamera();
  const scene = new ArticulatedViewpoint();
  const pose = new RobotCamera.Pose(THREE);

  assert.equal(pose.update({link: scene.head, optical: true}, 640, 360), true);

  assertVector(pose.camera.position, [3.2, 1.9, 2]);
  assertVector(pose.camera.getWorldDirection(new THREE.Vector3()), [0, 1, 0]);
  assertVector(cameraUp(pose.camera), [0, 0, 1]);
});

test('camera follows changed robot placement and articulated head without a prior scene render', () => {
  const RobotCamera = loadRobotCamera();
  const scene = new ArticulatedViewpoint();
  const pose = new RobotCamera.Pose(THREE);
  const source = {link: scene.head, optical: false};
  pose.update(source, 640, 360);
  scene.robot.position.set(5, 1, 0.5);
  scene.head.position.z = 2;
  scene.head.rotation.z = Math.PI / 2;

  assert.equal(pose.update(source, 640, 360), true);

  assertVector(pose.camera.position, [5.2, 2.5, -1]);
  assertVector(pose.camera.getWorldDirection(new THREE.Vector3()), [0, 0, -1]);
  assertVector(cameraUp(pose.camera), [0, 1, 0]);
});

test('resizing updates camera projection to the viewport aspect ratio', () => {
  const RobotCamera = loadRobotCamera();
  const scene = new ArticulatedViewpoint();
  const pose = new RobotCamera.Pose(THREE);
  const source = {link: scene.head, optical: true};
  pose.update(source, 320, 180);
  const previousProjection = pose.camera.projectionMatrix.clone();

  pose.update(source, 500, 500);

  assert.equal(pose.camera.aspect, 1);
  assert.equal(pose.camera.projectionMatrix.equals(previousProjection), false);
  assert.equal(pose.camera.projectionMatrix.elements[0], pose.camera.projectionMatrix.elements[5]);
});

test('an absent source leaves the camera pose unchanged and cannot render a robot view', () => {
  const RobotCamera = loadRobotCamera();
  const scene = new ArticulatedViewpoint();
  const pose = new RobotCamera.Pose(THREE);
  pose.update({link: scene.head, optical: false}, 320, 180);
  const position = pose.camera.position.clone();
  const quaternion = pose.camera.quaternion.clone();

  assert.equal(pose.update(null, 320, 180), false);
  assert.equal(pose.camera.position.equals(position), true);
  assert.equal(pose.camera.quaternion.equals(quaternion), true);
});

test('a viewport without positive finite dimensions cannot render a robot view', () => {
  const RobotCamera = loadRobotCamera();
  const scene = new ArticulatedViewpoint();
  const pose = new RobotCamera.Pose(THREE);
  const source = {link: scene.head, optical: false};
  pose.update(source, 320, 180);
  const projection = pose.camera.projectionMatrix.clone();

  for (const size of [[0, 180], [320, 0], [-10, 180], [320, -10], [NaN, 180], [320, Infinity]]) {
    assert.equal(pose.update(source, ...size), false);
    assert.equal(pose.camera.projectionMatrix.equals(projection), true);
  }
});
