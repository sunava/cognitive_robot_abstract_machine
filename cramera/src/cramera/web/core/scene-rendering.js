/* ============================================================================
 * core/scene-rendering.js — the renderer settings a scene bundle asks for.
 *
 * scene.json may carry a `rendering` block; this module reads its fields
 *   {groundPlane,
 *    shadowBounds: [[minx,miny,minz],[maxx,maxy,maxz]],   bundle frame, Z up, metres
 *    environment: "lighting.json" | {sun, dome, panels}}  see core/lab-environment.js
 * Every field a bundle leaves out keeps the viewer's own default, so a bundle written
 * before a field existed renders exactly as it did.
 *
 * Pure data, no three.js and no DOM, so it is testable under node.
 * ==========================================================================*/
(function (global) {
  'use strict';

  //: number of coordinates of a point in the bundle frame
  const DIMENSIONS = 3;

  // %% axis-aligned box in the bundle frame
  class SceneBounds {
    /**
     * @param {number[]} min corner with the smallest coordinates, Z up, metres
     * @param {number[]} max corner with the largest coordinates
     */
    constructor(min, max) {
      this.min = min;
      this.max = max;
    }

    /** Bounds from a `[[minx,miny,minz],[maxx,maxy,maxz]]` pair, or null when it is malformed. */
    static of(pair) {
      if (!Array.isArray(pair) || pair.length !== 2 || !pair.every(SceneBounds.isPoint)) return null;
      const [min, max] = pair;
      if (min.some(function (value, axis) { return value > max[axis]; })) return null;
      return new SceneBounds(min.slice(), max.slice());
    }

    /** Whether a value is a point of finite coordinates. */
    static isPoint(value) {
      return Array.isArray(value) && value.length === DIMENSIONS && value.every(Number.isFinite);
    }

    /** The point halfway between the two corners. */
    center() {
      const max = this.max;
      return this.min.map(function (value, axis) { return (value + max[axis]) / 2; });
    }

    /** The extent along every axis. */
    size() {
      const max = this.max;
      return this.min.map(function (value, axis) { return max[axis] - value; });
    }

    /** The eight corners of the box. */
    corners() {
      const min = this.min, max = this.max;
      const corners = [];
      [min[0], max[0]].forEach(function (x) {
        [min[1], max[1]].forEach(function (y) {
          [min[2], max[2]].forEach(function (z) { corners.push([x, y, z]); });
        });
      });
      return corners;
    }
  }

  // %% the rendering block
  class SceneRendering {
    /** @param {object|undefined} rendering the scene's `rendering` block */
    constructor(rendering) {
      this.rendering = rendering || {};
    }

    /** The rendering settings of a scene.json payload, which may lack the block. */
    static of(scene) {
      return new SceneRendering(scene ? scene.rendering : undefined);
    }

    /** Whether the procedural plank floor, which also catches the shadows, is shown. */
    get groundPlane() {
      return this.rendering.groundPlane !== false;
    }

    /** The region the key light's shadows must cover, or null to keep the fixed default region. */
    get shadowBounds() {
      return SceneBounds.of(this.rendering.shadowBounds);
    }

    /** The lighting description as written: a bundle-relative path, an inline object, or null. */
    get environment() {
      return this.rendering.environment || null;
    }

    /**
     * The lighting description: fetched next to scene.json when the bundle names a
     * file, as given when inline, else null. A file that cannot be fetched also gives
     * null, so the viewer keeps its default studio lighting.
     *
     * @param {function(string): Promise<object|null>} fetchJson reads one JSON document
     * @param {string} sceneBase URL of the bundle directory, ending in a slash
     * @returns {Promise<object|null>}
     */
    lighting(fetchJson, sceneBase) {
      const environment = this.environment;
      if (!environment) return Promise.resolve(null);
      if (typeof environment === 'object') return Promise.resolve(environment);
      return Promise.resolve()
        .then(function () { return fetchJson(sceneBase + environment); })
        .then(function (payload) { return payload || null; }, function () { return null; });
    }
  }

  global.SceneRendering = SceneRendering;
  global.SceneBounds = SceneBounds;
})(window);
