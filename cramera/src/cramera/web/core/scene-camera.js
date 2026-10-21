/* ============================================================================
 * core/scene-camera.js — the lens and reach of the scene camera.
 *
 * scene.json's `camera` block may set `fov` (vertical field of view in degrees) and
 * `far` (far clipping plane in metres) next to its position and target. A large
 * imported hall needs a far plane beyond the viewer's and an orbit that can zoom out
 * far enough to see it; a field the bundle does not set leaves the viewer's camera as
 * it is.
 *
 * No DOM, and three.js only through the camera it is handed, so it is testable under node.
 * ==========================================================================*/
(function (global) {
  'use strict';

  //: widest field of view a perspective lens can have, in degrees (exclusive)
  const STRAIGHT_ANGLE = 180;

  class SceneCamera {
    /** @param {object|undefined} camera the scene's `camera` block */
    constructor(camera) {
      this.camera = camera || {};
    }

    /** The camera settings of a scene.json payload, which may lack the block. */
    static of(scene) {
      return new SceneCamera(scene ? scene.camera : undefined);
    }

    /** Vertical field of view in degrees, or null when the scene sets no usable one. */
    get fov() {
      const fov = this.camera.fov;
      return Number.isFinite(fov) && fov > 0 && fov < STRAIGHT_ANGLE ? fov : null;
    }

    /** Far clipping plane in metres, or null when the scene sets no usable one. */
    get far() {
      const far = this.camera.far;
      return Number.isFinite(far) && far > 0 ? far : null;
    }

    /**
     * Set the lens and far plane the scene asks for on a perspective camera, and let its
     * orbit controls zoom out to at least half that far plane.
     *
     * @param {THREE.PerspectiveCamera} camera
     * @param {THREE.OrbitControls} controls
     */
    applyTo(camera, controls) {
      const fov = this.fov, far = this.far;
      if (fov === null && far === null) return;
      if (fov !== null) camera.fov = fov;
      if (far !== null) {
        camera.far = far;
        controls.maxDistance = Math.max(controls.maxDistance, far / 2);
      }
      camera.updateProjectionMatrix();
    }
  }

  global.SceneCamera = SceneCamera;
})(window);
