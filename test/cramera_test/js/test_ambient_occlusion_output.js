// Unit tests for the composer of web/core/ambient-occlusion.js (node:test).
// With ambient occlusion on, a frame is drawn into render targets first and copied to
// the screen last. three r128 writes into a render target in that target's encoding
// (linear here) and only encodes for the screen when a shader asks for it, so the
// pass that copies to the screen has to do what a direct render does: encode the
// linear colour as sRGB. Without it every frame with ambient occlusion shows its
// midtones too dark.
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const WEB = path.join(__dirname, '..', '..', '..', 'cramera', 'src', 'cramera', 'web');
//: the scripts index.html loads for the composer, in its order
const SCRIPTS = [
  'vendor/CopyShader.js', 'vendor/SSAOShader.js', 'vendor/SimplexNoise.js', 'vendor/EffectComposer.js',
  'vendor/RenderPass.js', 'vendor/ShaderPass.js', 'vendor/SSAOPass.js', 'core/ambient-occlusion.js',
];

function load() {
  global.THREE = require(path.join(WEB, 'vendor/three.min.js'));
  global.window = {};
  SCRIPTS.forEach(function (file) { new Function(fs.readFileSync(path.join(WEB, file), 'utf8'))(); });
  return window.AmbientOcclusion;
}

const AmbientOcclusion = load();

// %% the composer with a renderer that only reports its size; nothing here draws
function composed() {
  const renderer = {
    outputEncoding: THREE.sRGBEncoding,
    getPixelRatio: function () { return 1; },
    getSize: function (target) { return target.set(320, 200); },
  };
  return AmbientOcclusion.composer(renderer, new THREE.Scene(), new THREE.PerspectiveCamera(), 320, 200);
}

// the fragment shader as three compiles it, with every chunk it includes filled in
function resolved(shader) {
  return shader.replace(/#include <(\w+)>/g, function (include, chunk) { return THREE.ShaderChunk[chunk]; });
}

function screenPass(composer) {
  return composer.passes.filter(function (pass) { return pass.renderToScreen; }).pop();
}

// %% where colour is linear and where it is encoded
test('the composer keeps every image before the screen in linear colour', function () {
  const { composer, pass } = composed();
  assert.deepEqual(
    [composer.renderTarget1.texture.encoding, composer.renderTarget2.texture.encoding, pass.beautyRenderTarget.texture.encoding],
    [THREE.LinearEncoding, THREE.LinearEncoding, THREE.LinearEncoding]);
});

test('the pass that draws on the screen encodes linear colour for the screen, as a direct render does', function () {
  const output = screenPass(composed().composer);
  assert.ok(resolved(output.material.fragmentShader).includes('linearToOutputTexel('),
    'the last pass copies linear colour to the screen unencoded');
});

test('the pass that draws on the screen does not tone-map a second time', function () {
  const output = screenPass(composed().composer);
  assert.ok(!resolved(output.material.fragmentShader).includes('toneMapping('));
});

test('the screen pass is the composer\'s last', function () {
  const { composer } = composed();
  assert.strictEqual(screenPass(composer), composer.passes[composer.passes.length - 1]);
});

// %% the occlusion pass itself
test('the occlusion is computed by the pass that hides the scene background', function () {
  const { composer, pass } = composed();
  assert.ok(pass instanceof window.BackgroundIgnoringSSAOPass && composer.passes.includes(pass));
});
