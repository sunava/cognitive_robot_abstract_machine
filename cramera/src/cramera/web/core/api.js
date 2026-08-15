/* ============================================================================
 * core/api.js — where the JSON API lives, as seen from the page.
 *
 * The viewer is served both at a host's root (`cramera` on localhost) and under
 * a path prefix, where it is one route among many — Binder mounts it at
 * <base>/cramera/. Every request therefore resolves relative to the page; a
 * root-absolute one would drop the prefix and hit the host's own routes.
 * ==========================================================================*/
(function () {
  'use strict';

  const API_ROOT = 'api/';

  function urlFor(route) {
    return API_ROOT + route;
  }

  window.ServerApi = { urlFor: urlFor };
})();
