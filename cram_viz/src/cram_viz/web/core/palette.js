/* ============================================================================
 * core/palette.js — shared entity-group → colour/label palette.
 *
 * Both the EQL answer panel (tag colours) and the graph panel (node fill,
 * legend, ring size) render the same taxonomy groups and must agree on how
 * each group looks. This is the single source of truth both consume, so
 * they cannot drift the way two independent copies once did.
 * ==========================================================================*/
(function () {
  'use strict';

  window.EntityPalette = {
    // ---- TBox ----
    root:    { color: '#e8eefb', ring: '#ffffff', size: 24, label: 'Root concept' },
    klass:   { color: '#5b8cff', ring: '#a9c2ff', size: 15, label: 'Subpackage' },
    pyclass: { color: '#ffb648', ring: '#ffd89a', size: 13, label: 'Python class' },
    upper:   { color: '#8c9bbd', ring: '#c3ccdf', size: 14, label: 'Upper ontology (DUL)' },
    // ---- ABox individuals, bucketed by their asserted type ----
    robot:   { color: '#ff7a9c', ring: '#ffb3c6', size: 20, label: 'Robot / body' },
    object:  { color: '#39d5c8', ring: '#8ff0e7', size: 15, label: 'Object / substance' },
    event:   { color: '#b98cff', ring: '#d9c2ff', size: 16, label: 'Event / episode' },
    goal:    { color: '#ffb648', ring: '#ffd89a', size: 15, label: 'Goal' },
    concept: { color: '#4bd38a', ring: '#a6ecc6', size: 14, label: 'Problem / phase / fluent' },
    ind:     { color: '#7f8db0', ring: '#b6c0d8', size: 13, label: 'Individual' },
  };
})();
