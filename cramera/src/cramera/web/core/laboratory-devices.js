/* Identity and selection for independently described laboratory instruments. */
(function (global, factory) {
  'use strict';
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.LaboratoryDevices = api;
})(typeof window === 'undefined' ? globalThis : window, function () {
  'use strict';

  // %% authored instrument identities
  const Definitions = Object.freeze([
    {id:'centrifuge', title:'Centrifuge', factory:'centrifuge', field:'centrifuge', pattern:/^(centrifuge\b|laboratory centrifuge\b)/},
    {id:'dispenser', title:'Dual dispenser', factory:'dispenser', field:'pipetting', pattern:/^(pipetting\b|dispenser aicor logo\b|laboratory pipetting\b)/},
    {id:'scale', title:'Scale', factory:'scale', field:'scale', pattern:/^(balance\b|instrument base\b|laboratory scale\b)/},
    {id:'rack', title:'Tube rack', factory:'rack', field:null, pattern:/^(rack\b|socket identifier\b|laboratory rack\b)/},
  ]);
  function normalized(value) { return String(value || '').replace(/_/g,' ').toLowerCase(); }

  /** Attach each device's own authored metadata to its manual definition. */
  function catalog(config) {
    return Definitions.map(device => Object.assign({},device,{metadata:device.field ? config[device.field] || {} : {slots:config.rackSlots}}));
  }

  /** Recognize the nearest named component without treating the whole room as one object. */
  function identify(node) {
    for (let current=node; current; current=current.parent) {
      const names=[current.name, current.userData && current.userData.name].map(normalized);
      const device=Definitions.find(candidate => names.some(name=>candidate.pattern.test(name)));
      if (device) return device.id;
    }
    return null;
  }

  function visible(node) {
    for (let current=node; current; current=current.parent) if (current.visible===false) return false;
    return true;
  }

  /** Respect the nearest visible surface so selection cannot pass through a robot or glass. */
  function pick(hits) {
    const front=hits.find(hit=>visible(hit.object));
    return front ? identify(front.object) : null;
  }

  return {catalog, identify, pick};
});
