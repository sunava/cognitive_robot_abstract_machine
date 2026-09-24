'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const Devices = require('../../../cramera/src/cramera/web/core/laboratory-devices.js');

// %% instrument identity follows authored and reusable model names
for (const [id, names] of Object.entries({
  centrifuge:['Centrifuge lid grip','Centrifuge AICOR logo','laboratory_centrifuge_rotor'],
  dispenser:['Pipetting rear housing','Pipetting_amber_bottle','Dispenser AICOR logo','laboratory_pipetting_station'],
  scale:['Instrument base','Balance weighing pan','Balance AICOR logo','laboratory_scale_pan'],
  rack:['Rack perforated plate','Socket identifier.002','laboratory_rack'],
})) {
  test('recognizes only the selected '+id+' through its actual mesh or link',()=>{
    for (const name of names) assert.equal(Devices.identify({name}),id);
    assert.equal(Devices.identify({name:'',parent:{name:names[0]}}),id);
  });
}
test('unrelated room, glass and robot meshes do not become instruments',()=>{
  for (const name of ['Porcelain worktop','Hollow borosilicate tube','r_gripper_palm_link','Wall AICOR logo']) {
    assert.equal(Devices.identify({name}),null);
  }
});
test('a robot or glass in front prevents selecting an occluded instrument',()=>{
  const instrument={name:'Centrifuge enamel chassis',visible:true};
  assert.equal(Devices.pick([{object:{name:'r_gripper_palm_link'}},{object:instrument}]),null);
  assert.equal(Devices.pick([{object:instrument}]),'centrifuge');
  assert.equal(Devices.pick([{object:{name:'Rack base',visible:false}},{object:instrument}]),'centrifuge');
  assert.equal(Devices.pick([{object:{name:'Rack base',parent:{visible:false}}},{object:instrument}]),'centrifuge');
});
test('device catalog keeps each manual bound to its own supplied metadata',()=>{
  const config={centrifuge:{origin:[0,0,0]},pipetting:{origin:[1,0,0]},scale:{panPosition:[2,0,0]},rackSlots:[{id:'A1'}]};
  const devices=Devices.catalog(config);
  assert.deepEqual(devices.map(d=>d.id),['centrifuge','dispenser','scale','rack']);
  assert.equal(devices[0].metadata,config.centrifuge);
  assert.equal(devices[1].metadata,config.pipetting);
  assert.equal(devices[2].metadata,config.scale);
  assert.equal(devices[3].metadata.slots,config.rackSlots);
});
