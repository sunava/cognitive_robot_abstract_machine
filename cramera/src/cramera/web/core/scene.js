/* ============================================================================
 * core/scene.js — the active scene, as read from the page URL.
 *
 * robot_scene/panel.js switches scenes by reloading the page with a new
 * ?scene= query param; every panel that talks to the /api/* routes must read
 * the same param so its requests target that same scene instead of whatever
 * the server falls back to.
 * ==========================================================================*/
(function () {
  'use strict';

  function name() {
    // a scene name is now a robot/environment/task/version path (see
    // cramera.onboard.scene_index.compose_scene_name) — URLSearchParams decodes its
    // percent-encoded '/'s, where a hand-rolled [\w-]+ regex would truncate at the
    // first one
    return new URLSearchParams(window.location.search).get('scene');
  }

  function withScene(url) {
    const active = name();
    if (!active) return url;
    return url + (url.indexOf('?') >= 0 ? '&' : '?') + 'scene=' + encodeURIComponent(active);
  }

  window.SceneContext = { name: name, withScene: withScene };
})();
