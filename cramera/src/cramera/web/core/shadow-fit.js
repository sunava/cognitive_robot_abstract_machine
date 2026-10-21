/* ============================================================================
 * core/shadow-fit.js — fit the key light's shadow camera to the scene's shadow bounds.
 *
 * The key light renders its shadow map through an orthographic camera of a fixed
 * texel count. Fitted to `rendering.shadowBounds` instead of a fixed region around
 * the origin, every part of an imported hall receives shadows and no texel is spent
 * outside it. The light is aimed at the centre of the bounds from the given
 * direction, and the camera is sized to the bounds' corners as the light sees them.
 *
 * Bounds and directions are in the bundle frame (Z up, metres); the viewer draws that
 * frame under a root turned to three's Y-up frame, and so does this module.
 * three.js is handed in, so it is testable under node.
 * ==========================================================================*/
(function (global) {
  'use strict';

  //: |vertical component| of a unit direction from which a light counts as straight above
  const VERTICAL = 0.999;

  class DirectionError extends Error {
    /** @param {number[]} direction the direction that has no length */
    constructor(direction) {
      super('A light direction needs a length, got [' + direction + ']; give the way the light travels.');
      this.name = 'DirectionError';
    }
  }

  class ShadowFit {
    /**
     * @param {SceneBounds} bounds the region that must receive shadows, bundle frame
     * @param {number[]} direction the way the light travels, bundle frame
     */
    constructor(bounds, direction) {
      if (!direction.some(function (value) { return value !== 0; })) throw new DirectionError(direction);
      this.bounds = bounds;
      this.direction = direction;
    }

    /** Three's Y-up vector of a point or direction given in the bundle's Z-up frame. */
    static toThree(THREE, vector) {
      return new THREE.Vector3(vector[0], vector[2], -vector[1]);
    }

    /** The bundle-frame coordinates of a three vector. */
    static fromThree(vector) {
      return [vector.x, -vector.z, vector.y];
    }

    /**
     * The light placement and the shadow camera that just encloses the bounds, in
     * three's frame: {position, target, up, left, right, top, bottom, near, far}.
     */
    frustum(THREE) {
      const margin = ShadowFit.MARGIN;
      const target = ShadowFit.toThree(THREE, this.bounds.center());
      const travel = ShadowFit.toThree(THREE, this.direction).normalize();
      const size = this.bounds.size();
      const reach = Math.hypot(size[0], size[1], size[2]) / 2 + margin;
      const position = target.clone().addScaledVector(travel, -reach);
      // a camera looking straight down cannot keep "up" as its up; north serves instead
      const up = Math.abs(travel.y) > VERTICAL ? ShadowFit.toThree(THREE, [0, 1, 0]) : new THREE.Vector3(0, 1, 0);
      const camera = new THREE.OrthographicCamera();
      camera.up.copy(up);
      camera.position.copy(position);
      camera.lookAt(target);
      camera.updateMatrixWorld();
      const seen = new THREE.Box3().setFromPoints(this.bounds.corners().map(function (corner) {
        return ShadowFit.toThree(THREE, corner).applyMatrix4(camera.matrixWorldInverse);
      }));
      return {
        position: position, target: target, up: up,
        left: seen.min.x - margin, right: seen.max.x + margin,
        bottom: seen.min.y - margin, top: seen.max.y + margin,
        // the camera looks along its -z axis, so depth is the negated z
        near: Math.max(0, -seen.max.z - margin), far: -seen.min.z + margin,
      };
    }

    /** Aim a directional light at the bounds and fit its shadow camera to them. */
    applyTo(THREE, light) {
      const frustum = this.frustum(THREE);
      light.position.copy(frustum.position);
      light.target.position.copy(frustum.target);
      light.target.updateMatrixWorld();
      const camera = light.shadow.camera;
      camera.up.copy(frustum.up);
      camera.left = frustum.left;
      camera.right = frustum.right;
      camera.top = frustum.top;
      camera.bottom = frustum.bottom;
      camera.near = frustum.near;
      camera.far = frustum.far;
      camera.updateProjectionMatrix();
    }
  }

  //: metres the shadow camera reaches past the bounds on every side, so geometry on their faces still casts
  ShadowFit.MARGIN = 0.05;
  ShadowFit.DirectionError = DirectionError;

  global.ShadowFit = ShadowFit;
})(window);
