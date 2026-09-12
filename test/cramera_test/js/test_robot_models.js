// Unit tests for web/core/robot-models.js (node:test): the robots among the loaded
// models, and what describes each of them.
//
// A scene used to have one robot, so the viewer kept one robot slot and drove it from
// the single `base` pose the bridge and the recording carry. A Siemens demo runs three
// (a continuum arm and two humanoids), and the bundle then lists one model per robot,
// each under its own world prefix, with one base pose per prefix beside it. What has to
// hold: the robots keep the bundle's order however their URDFs finish loading, the first
// of them stays the one a lone `base` describes, each one finds its own part table, and
// a pose re-sent unchanged does not count as a move -- that is what decides which robot
// the camera follows.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/robot-models.js'), 'utf8'))(scope);
  return scope.RobotModels;
}

// a loaded model as the shell holds it: what the bundle said about it, plus where in the
// bundle it was listed (`order`) -- the models load asynchronously, so that is not the
// order they arrive in
function model(name, prefix, robot, order) {
  return { name: name, prefix: prefix, robot: robot, order: order };
}

// the Siemens demo's bundle: the environment first, then one entry per robot
const ENVIRONMENT = model('apartment', '', false, 0);
const CONTINUUM = model('continuum_robot', 'continuum_robot', true, 1);
const WALKER = model('walker_s2', 'walker_s2_description', true, 2);
const UME = model('ume', 'uMe', true, 3);

function pose(x, y) {
  return [x, y, 0, 0, 0, 0, 1];
}

function standing(x, y) {
  return { position: { x: x, y: y, z: 0 }, quaternion: { x: 0, y: 0, z: 0, w: 1 } };
}

// %% the robots of a scene
test('the robot models are the ones the bundle flagged as robots', function () {
  const RobotModels = load();
  assert.deepStrictEqual(RobotModels.robots([ENVIRONMENT, CONTINUUM, WALKER, UME]),
    [CONTINUUM, WALKER, UME]);
});

test('the robots keep the bundle order, not the order their URDFs loaded in', function () {
  const RobotModels = load();
  // the humanoids' meshes are heavy, so the arm's little URDF lands first
  const loaded = [CONTINUUM, UME, ENVIRONMENT, WALKER];

  assert.deepStrictEqual(RobotModels.robots(loaded), [CONTINUUM, WALKER, UME]);
  assert.strictEqual(RobotModels.primary(loaded), CONTINUUM);
});

test('the first robot of the bundle is the primary one', function () {
  const RobotModels = load();
  assert.strictEqual(RobotModels.primary([ENVIRONMENT, WALKER, UME]), WALKER);
});

test('a scene whose models have not loaded yet has no robot', function () {
  const RobotModels = load();
  assert.strictEqual(RobotModels.primary([]), null);
  assert.strictEqual(RobotModels.primary([ENVIRONMENT]), null);
  assert.deepStrictEqual(RobotModels.robots(undefined), []);
});

// %% one payload per robot
test('every robot of the bundle has its own payload', function () {
  const RobotModels = load();
  const scene = {
    robot: { name: 'walker_s2', prefix: 'walker_s2_description' },
    robots: [
      { name: 'walker_s2', prefix: 'walker_s2_description' },
      { name: 'ume', prefix: 'uMe' },
    ],
  };

  assert.deepStrictEqual(RobotModels.payloads(scene), scene.robots);
});

test('a bundle from before robots were a list carries its one robot alone', function () {
  const RobotModels = load();
  const scene = { robot: { name: 'pr2', prefix: 'pr2' } };

  assert.deepStrictEqual(RobotModels.payloads(scene), [scene.robot]);
  assert.deepStrictEqual(RobotModels.payloads({}), []);
});

test('a payload belongs to the model whose world prefix it names', function () {
  const RobotModels = load();
  const payloads = [
    { name: 'walker_s2', prefix: 'walker_s2_description', parts: {} },
    { name: 'ume', prefix: 'uMe', parts: {} },
  ];

  assert.strictEqual(RobotModels.payloadFor(payloads, WALKER, 3), payloads[0]);
  assert.strictEqual(RobotModels.payloadFor(payloads, UME, 3), payloads[1]);
});

test('a robot no payload names keeps no parts rather than another robot\'s', function () {
  const RobotModels = load();
  const payloads = [{ name: 'ume', prefix: 'uMe', parts: { left_arm: ['l_upper_arm'] } }];

  assert.strictEqual(RobotModels.payloadFor(payloads, WALKER, 3), null);
  assert.deepStrictEqual(RobotModels.linkToPart(null), {});
});

test('one robot and one payload belong together whatever they were named', function () {
  /*
   * A bundle recorded before the world prefixes were written out names its model and its
   * robot differently; with a single robot there is nothing to confuse it with.
   */
  const RobotModels = load();
  const payloads = [{ name: 'pr2', prefix: 'pr2', parts: {} }];
  const onlyRobot = model('robot', '', true, 1);

  assert.strictEqual(RobotModels.payloadFor(payloads, onlyRobot, 1), payloads[0]);
  assert.strictEqual(RobotModels.payloadFor(payloads, onlyRobot, 3), null);
});

test('a payload turns into the link -> part table the viewer looks parts up in', function () {
  const RobotModels = load();
  const payload = {
    name: 'pr2',
    parts: { left_arm: ['l_shoulder_link', 'l_wrist_link'], left_gripper: ['l_gripper_link'] },
  };

  assert.deepStrictEqual(RobotModels.linkToPart(payload), {
    l_shoulder_link: 'left_arm',
    l_wrist_link: 'left_arm',
    l_gripper_link: 'left_gripper',
  });
});

// %% where each robot stands
test('a streamed snapshot poses every robot by its own prefix', function () {
  const RobotModels = load();
  const bases = {
    continuum_robot: pose(0, 0),
    walker_s2_description: pose(1, 2),
    uMe: pose(3, 4),
  };

  assert.deepStrictEqual(RobotModels.baseOf(bases, WALKER), pose(1, 2));
  assert.deepStrictEqual(RobotModels.baseOf(bases, UME), pose(3, 4));
  assert.deepStrictEqual(RobotModels.baseOf(bases, ENVIRONMENT), null);
});

test('a robot the snapshot says nothing about is left where it stands', function () {
  const RobotModels = load();

  assert.strictEqual(RobotModels.baseOf({ uMe: pose(3, 4) }, WALKER), null);
  assert.strictEqual(RobotModels.baseOf({}, WALKER), null);
  assert.strictEqual(RobotModels.baseOf(null, WALKER), null);
  // a prefix that happens to name something on Object.prototype is not a pose
  assert.strictEqual(RobotModels.baseOf({}, model('odd', 'constructor', true, 1)), null);
});

test('a recorded frame carries the pose of every robot that was moving', function () {
  const RobotModels = load();
  const trajectory = {
    base: [pose(0, 0), pose(0, 1)],
    modelBases: [
      { walker_s2_description: pose(0, 0), uMe: pose(5, 5) },
      { walker_s2_description: pose(0, 1), uMe: pose(5, 6) },
    ],
  };

  assert.deepStrictEqual(RobotModels.baseOf(RobotModels.basesAt(trajectory, 1), UME), pose(5, 6));
  assert.deepStrictEqual(RobotModels.baseOf(RobotModels.basesAt(trajectory, 0), WALKER), pose(0, 0));
});

test('a recording made before robots were a list has no per-robot poses', function () {
  const RobotModels = load();
  const trajectory = { base: [pose(0, 0), pose(0, 1)] };

  assert.strictEqual(RobotModels.basesAt(trajectory, 0), null);
  assert.strictEqual(RobotModels.basesAt({ modelBases: [] }, 0), null);
  assert.strictEqual(RobotModels.basesAt(null, 0), null);
});

// %% which robot is acting
test('a pose that puts a robot somewhere else is a move', function () {
  const RobotModels = load();

  assert.strictEqual(RobotModels.moved(standing(0, 0), pose(0, 1)), true);
  assert.strictEqual(
    RobotModels.moved(standing(0, 0), [0, 0, 0, 0, 0, 0.7071, 0.7071]), true);
});

test('a pose re-sent unchanged does not make a standing robot the acting one', function () {
  const RobotModels = load();

  assert.strictEqual(RobotModels.moved(standing(1, 2), pose(1, 2)), false);
  // the bridge rounds its poses, so "unchanged" has to survive the last decimals
  assert.strictEqual(
    RobotModels.moved(standing(1, 2), [1, 2, 1e-9, 0, 0, 0, 1]), false);
});

test('nothing to pose is no move', function () {
  const RobotModels = load();

  assert.strictEqual(RobotModels.moved(null, pose(0, 0)), false);
  assert.strictEqual(RobotModels.moved(standing(0, 0), null), false);
});
