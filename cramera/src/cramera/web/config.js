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
// Safety net: only strip the chrome / reduce the layout when we're actually embedded
// in an iframe (Plan Builder / replay popup). If ?scene ends up in a *top-level* tab,
// keep the full page (topbar + EQL + graph) so the user isn't stranded on a bare scene
// with no way to navigate back.
var _framed = window.self !== window.top;
var _sceneOnly = _framed && /[?&](replay=|scene(\b|=))/.test(window.location.search);
if (_sceneOnly) {
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
