/* ============================================================================
 * config.js — *which* panels are shown *where*. This is the file you edit to swap
 * a visualization: remove an id, add your own (define it via Panels.define in
 * a new panels/<name>/panel.js and include that script in index.html).
 *
 * Slots are the data-slot elements in index.html ('left', 'right'); a slot
 * with several panel ids stacks them vertically.
 * ==========================================================================*/
// ?replay= (recorded clip popup) and ?scene (Plan Builder's embedded 3D view) both want
// just the 3D scene; ?scene also drops the topbar chrome (see app.css .scene-only).
var _sceneOnly = /[?&](replay=|scene(\b|=))/.test(window.location.search);
if (/[?&]scene(\b|=)/.test(window.location.search)) {
  document.documentElement.classList.add('scene-only');
}
window.CRAMERA_CONFIG = {
  layout: _sceneOnly
    ? { left: ['robot-scene'] }
    : {
        left: ['robot-scene'],
        right: ['eql', 'graph'],
      },
};
