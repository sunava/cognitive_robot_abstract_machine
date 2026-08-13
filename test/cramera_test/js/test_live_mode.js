// Unit tests for web/core/live-mode.js (node:test): where the live pose stream may attach.
//
// A recorded scene and a running demo are separate worlds, and an object drag means a
// different thing in each -- a client-side offset on a playback pose versus a real move of
// the simulated world posted to the bridge. So the stream attaches only on the reserved
// live scene, and the control offers a navigation rather than a toggle anywhere else.
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');

function load() {
  const scope = {};
  new Function('window', fs.readFileSync(path.join(WEB, 'core/live-mode.js'), 'utf8'))(scope);
  return scope.LiveMode;
}

// %% where the stream may attach
test('the pose stream attaches only on the live scene', function () {
  const live = load();
  assert.strictEqual(live.attachable(live.SCENE_NAME), true);
  assert.strictEqual(live.attachable('PR2_Apartment'), false);
});

test('a landing page that names no scene is not attachable either', function () {
  /*
   * With no ?scene= the viewer is showing the scene index's default, which is a recording
   * like any other.
   */
  const live = load();
  assert.strictEqual(live.attachable(''), false);
  assert.strictEqual(live.attachable(null), false);
  assert.strictEqual(live.attachable(undefined), false);
});

// %% what the control does
test('pressing live on a recorded scene navigates rather than attaching', function () {
  const live = load();
  assert.strictEqual(live.actionFor('PR2_Apartment'), live.NAVIGATE);
  assert.strictEqual(live.actionFor(''), live.NAVIGATE);
});

test('pressing live on the live scene toggles the stream in place', function () {
  const live = load();
  assert.strictEqual(live.actionFor(live.SCENE_NAME), live.TOGGLE);
});

test('navigating and toggling are distinguishable', function () {
  const live = load();
  assert.notStrictEqual(live.NAVIGATE, live.TOGGLE);
});

// %% what the control says
test('a recorded scene is offered a view switch, not an attach', function () {
  const live = load();
  assert.strictEqual(live.labelFor('PR2_Apartment', false), '◉ Live view');
  assert.match(live.titleFor('PR2_Apartment'), /leaves the recorded scene/);
});

test('the live scene reports whether it is attached', function () {
  const live = load();
  assert.notStrictEqual(
    live.labelFor(live.SCENE_NAME, true), live.labelFor(live.SCENE_NAME, false));
  assert.match(live.labelFor(live.SCENE_NAME, true), /attached/);
});

// %% when the live scene must be rebuilt and reloaded
test('a bundle whose signature no longer matches the bridge reloads', function () {
  /*
   * The signature digests everything the bundle is built from (sources, prefixes,
   * robot, world binding), so one comparison covers every way the loaded bundle can
   * stop describing the running demo: a world attach after an early bundle, a model
   * loaded mid-run, a switch to a different world.
   */
  const live = load();
  const scene = { bundleSignature: 'world-a' };
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, true, scene, { bundleSignature: 'world-b' }),
    true);
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, true, scene, { bundleSignature: 'world-a' }),
    false);
});

test('a page whose bundle carries no signature is stale once the bridge reports one', function () {
  /*
   * Only a bundle written before signatures existed lacks one; the first reload
   * rebuilds it with a signature and converges.
   */
  const live = load();
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, true, { models: [] }, { bundleSignature: 'world-a' }),
    true);
});

test('a recorded scene is never reloaded away by a live event', function () {
  const live = load();
  const scene = { bundleSignature: 'world-a' };
  assert.strictEqual(
    live.needsLiveSceneReload('PR2_Apartment', true, scene, { bundleSignature: 'world-b' }),
    false);
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, false, scene, { bundleSignature: 'world-b' }),
    false);
});

test('a bridge that reports no signature never triggers a reload', function () {
  /*
   * Without the bridge's answer there is no evidence of staleness, and reloading on
   * none would loop.
   */
  const live = load();
  assert.strictEqual(
    live.needsLiveSceneReload(live.SCENE_NAME, true, { bundleSignature: 'world-a' }, { running: true }),
    false);
  assert.strictEqual(live.needsLiveSceneReload(live.SCENE_NAME, true, null, { bundleSignature: 'a' }), false);
  assert.strictEqual(live.needsLiveSceneReload(live.SCENE_NAME, true, { bundleSignature: 'a' }, null), false);
});

// %% when the stream attaches by itself
test('auto-attach fires once per page by default', function () {
  const live = load();
  assert.strictEqual(live.shouldAutoAttach(live.SCENE_NAME, true, false, false, false), true);
  assert.strictEqual(live.shouldAutoAttach(live.SCENE_NAME, true, false, true, false), false);
});

test('always-live keeps re-attaching whenever the stream is down', function () {
  /*
   * A demo ending detaches the stream after enough failed polls; with always-live on,
   * the next demo run must attach again without a click -- the "attached once
   * already" guard only applies to the one-shot default.
   */
  const live = load();
  assert.strictEqual(live.shouldAutoAttach(live.SCENE_NAME, true, false, true, true), true);
  assert.strictEqual(live.shouldAutoAttach(live.SCENE_NAME, true, true, true, true), false);
});

test('a deliberately chosen recorded scene is never auto-attached away from', function () {
  /*
   * A URL naming a recorded scene is a deliberate choice; even always-live must not
   * navigate away from it. The landing page (no explicit scene) is fair game.
   */
  const live = load();
  assert.strictEqual(live.shouldAutoAttach('PR2_Apartment', true, false, false, true), false);
  assert.strictEqual(live.shouldAutoAttach('', false, false, false, true), true);
});
