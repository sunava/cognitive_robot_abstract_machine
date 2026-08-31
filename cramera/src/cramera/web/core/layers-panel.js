/* ============================================================================
 * core/layers-panel.js — whether the layers overlay is folded away.
 *
 * The overlay sits on top of the 3D view, so it has to be foldable down to its
 * title; and once folded it stays folded, on this page and the next. The state
 * and what the fold button says about it live here, without touching the DOM.
 * ==========================================================================*/
(function () {
  'use strict';

  const STORAGE_KEY = 'layersCollapsed';

  // reading a store that refuses to answer (a private window, storage turned off)
  // is not worth a broken viewer: the overlay then simply opens unfolded
  function collapsed(store) {
    try {
      return store.getItem(STORAGE_KEY) === 'yes';
    } catch (error) {
      return false;
    }
  }

  function remember(store, isCollapsed) {
    try {
      store.setItem(STORAGE_KEY, isCollapsed ? 'yes' : 'no');
    } catch (error) {
      /* a viewer that cannot remember the choice still honours it while it is open */
    }
  }

  // the fold button, as it reads in each state
  function button(isCollapsed) {
    return isCollapsed
      ? { glyph: '▸', title: 'Show the layers' }
      : { glyph: '▾', title: 'Fold the layers away' };
  }

  window.LayersPanel = {
    STORAGE_KEY: STORAGE_KEY,
    collapsed: collapsed,
    remember: remember,
    button: button,
  };
})();
