/* ============================================================================
 * core/ambient-occlusion.js — the SSAO pass the 3D scene panel renders through.
 *
 * three's SSAOPass fills its depth and normal buffers by rendering the scene with an
 * override material. A textured scene background is a 2x2 plane at the origin in the
 * render list, so under that override it is drawn as real geometry, and its occlusion
 * edges appear as a rectangle standing on the floor of every scene. This pass hides
 * the background for exactly those renders; the beauty render keeps it.
 *
 * The composed frame lives in render targets, which three r128 writes in their own
 * (linear) encoding; only a shader that includes `encodings_fragment` encodes for the
 * screen. The last pass therefore encodes for the screen itself, so a frame with
 * ambient occlusion has the colours of a direct render.
 *
 * Defined only when three's SSAOPass is on the page, like the pass it extends.
 * ==========================================================================*/
(function () {
  'use strict';
  if (typeof THREE === 'undefined' || !THREE.SSAOPass) return;

  class BackgroundIgnoringSSAOPass extends THREE.SSAOPass {
    renderOverride(renderer, overrideMaterial, renderTarget, clearColor, clearAlpha) {
      const background = this.scene.background;
      this.scene.background = null;
      super.renderOverride(renderer, overrideMaterial, renderTarget, clearColor, clearAlpha);
      this.scene.background = background;
    }
  }

  // %% output to the screen
  /** Copies the composed linear frame to the screen in the renderer's output encoding. */
  const ScreenOutputShader = {
    uniforms: { tDiffuse: { value: null } },
    vertexShader: [
      'varying vec2 vUv;',
      'void main() {',
      '  vUv = uv;',
      '  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );',
      '}',
    ].join('\n'),
    fragmentShader: [
      'uniform sampler2D tDiffuse;',
      'varying vec2 vUv;',
      'void main() {',
      '  gl_FragColor = texture2D( tDiffuse, vUv );',
      '  #include <encodings_fragment>',
      '}',
    ].join('\n'),
  };

  // %% the composer
  //: how far the occlusion samples reach around a point, in metres
  const KERNEL_RADIUS = 0.12;
  //: depth difference below which a sample counts as the same surface, as a share of the camera's depth range
  const MIN_DISTANCE = 0.001;
  //: depth difference above which a sample no longer occludes, as a share of the camera's depth range
  const MAX_DISTANCE = 0.04;

  /**
   * The composer a frame with ambient occlusion is drawn through, and its occlusion pass.
   *
   * @returns {{composer: THREE.EffectComposer, pass: BackgroundIgnoringSSAOPass}}
   */
  function composer(renderer, scene, camera, width, height) {
    const frames = new THREE.EffectComposer(renderer);
    frames.addPass(new THREE.RenderPass(scene, camera));
    const pass = new BackgroundIgnoringSSAOPass(scene, camera, width, height);
    pass.kernelRadius = KERNEL_RADIUS;
    pass.minDistance = MIN_DISTANCE;
    pass.maxDistance = MAX_DISTANCE;
    frames.addPass(pass);
    const output = new THREE.ShaderPass(ScreenOutputShader);
    output.renderToScreen = true;
    frames.addPass(output);
    return { composer: frames, pass: pass };
  }

  window.BackgroundIgnoringSSAOPass = BackgroundIgnoringSSAOPass;
  window.AmbientOcclusion = { composer: composer, ScreenOutputShader: ScreenOutputShader };
})();
