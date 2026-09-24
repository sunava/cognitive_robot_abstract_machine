'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const THREE = require('../../../cramera/src/cramera/web/vendor/three.min.js');
const Refrigerator = require('../../../cramera/src/cramera/web/core/laboratory-refrigerator.js');
const Physics = require('../../../cramera/src/cramera/web/core/laboratory-physics.js');
const Devices = require('../../../cramera/src/cramera/web/core/laboratory-devices.js');
const Parts = require('../../../cramera/src/cramera/web/core/laboratory-manual-parts.js');

// %% measured cold storage fixture
class ColdStorage {
  constructor() {
    this.config = {objects:[{key:'tube'}], bounds:{min:[-1.55,-.85,.15],max:[.8,.45,1.6]},rackSlots:[{id:'A1',position:[.16,-.065,.906]}],
      refrigerator:{origin:[-1.23,0,0],body:'laboratory_refrigerator_base',doorJoint:'laboratory_refrigerator_door_joint',
        slots:[{id:'R1',position:[-1.37,-.1,.24]},{id:'R2',position:[-1.23,-.1,.24]}]}};
    this.snapshot = {ok:true,state:'running',time:0,objects:{tube:[.16,-.065,.906,0,0,0,1]},contacts:[],
      liquid:{tubes:{tube:{volumeMl:1,capacityMl:10}}},
      refrigerator:{available:true,doorOpen:true,doorClosed:false,doorAngle:1.75,temperatureC:6.2,targetTemperatureC:4,
        slots:this.config.refrigerator.slots.map(slot=>({...slot,key:null})),joints:{laboratory_refrigerator_door_joint:1.75}}};
    this.commands=[];this.observations=[];
    this.service={state:async()=>this.snapshot,
      target:async target=>{this.commands.push({type:'target',...target});return {ok:true};},
      release:async()=>{this.commands.push({type:'release'});return {ok:true};},
      refrigerator:async payload=>{this.commands.push({type:'refrigerator',...payload});return {ok:true,refrigerator:this.snapshot.refrigerator};}};
    this.controller=new Physics.Controller(this.config,this.service,{setObjectPose(){},setJointFrames:frames=>this.observations.push(frames),setRefrigeratorState:state=>this.observations.push(state)});
  }
  async refresh() {this.snapshot.time+=.1;await this.controller.refresh();}
  async reach(position) {this.snapshot.objects.tube=position.concat([0,0,0,1]);await this.refresh();await this.refresh();}
}

test('refrigerator service keeps robot and standalone commands isolated',async()=>{
  for(const robot of [false,true]) {
    const calls=[];
    const client=new Physics.Client(async(url,options)=>{calls.push({url,body:JSON.parse(options.body)});return {ok:true,json:async()=>({ok:true})};},{robot});
    await client.refrigerator({action:'open'});
    assert.deepEqual(calls,[{url:(robot?Physics.RobotRoutes:Physics.Routes).REFRIGERATOR,body:{action:'open'}}]);
  }
});

test('observed door joints and state reach the scene adapter',async()=>{
  const scene=new ColdStorage();await scene.refresh();
  assert.equal(scene.controller.refrigerator,scene.snapshot.refrigerator);
  assert.ok(scene.observations.includes(scene.snapshot.refrigerator));
  assert.ok(scene.observations.includes(scene.snapshot.refrigerator.joints));
});

test('storage approaches through the open front and never lowers through the cabinet roof',async()=>{
  const scene=new ColdStorage();await scene.refresh();
  const original=scene.snapshot.objects.tube.slice();
  assert.equal(await scene.controller.storeRefrigerator(),true);
  const path=scene.controller.refrigeratorTransfer.waypoints;
  const slot=scene.snapshot.refrigerator.slots[0].position;
  assert.deepEqual(path.at(-1),slot);
  const lower=path.findIndex((point,index)=>index>0 && point[2]<path[index-1][2]);
  assert.ok(path[lower][1] <= Refrigerator.Settings.FRONT);
  assert.ok(path[lower][0]<-.9);
  assert.deepEqual(scene.snapshot.objects.tube,original);
  assert.equal(scene.commands[0].type,'target');
});

test('storage waits for physical support and settling before release and completion',async()=>{
  const scene=new ColdStorage();await scene.refresh();await scene.controller.storeRefrigerator();
  const transfer=scene.controller.refrigeratorTransfer;
  for(const waypoint of transfer.waypoints) await scene.reach(waypoint);
  assert.equal(scene.commands.some(command=>command.type==='release'),false);
  scene.snapshot.contacts=[{bodyA:'tube',bodyB:scene.config.refrigerator.body,force:.2}];
  scene.snapshot.refrigerator.slots[0].key='tube';
  await scene.refresh();
  assert.equal(scene.commands.at(-1).type,'release');
  await scene.refresh();
  assert.equal(scene.controller.refrigeratorTransfer,null);
});

test('closed door, occupied holders, secured samples and concurrent dispensing reject storage',async()=>{
  const scene=new ColdStorage();await scene.refresh();
  scene.snapshot.refrigerator.doorOpen=false;
  assert.equal(await scene.controller.storeRefrigerator(),false);
  scene.snapshot.refrigerator.doorOpen=true;scene.snapshot.refrigerator.slots[0].key='other';
  assert.equal(await scene.controller.storeRefrigerator(),false);
  scene.snapshot.refrigerator.slots[0].key=null;scene.controller.centrifuge={slots:[{key:'tube',secured:true}]};
  assert.equal(await scene.controller.storeRefrigerator(),false);
  scene.controller.centrifuge=null;scene.controller.pipetting={state:'dispensing'};
  assert.equal(await scene.controller.storeRefrigerator(),false);
  assert.deepEqual(scene.commands,[]);
});

test('retrieval clears its holder and exits the front before lifting to the bench',async()=>{
  const scene=new ColdStorage();scene.snapshot.refrigerator.slots[0].key='tube';
  scene.snapshot.objects.tube=scene.snapshot.refrigerator.slots[0].position.concat([0,0,0,1]);
  await scene.refresh();assert.equal(await scene.controller.retrieveRefrigerator(),true);
  const path=scene.controller.refrigeratorTransfer.waypoints;
  const firstHigh=path.findIndex(point=>point[2]>.7);
  assert.ok(path[firstHigh][1]<=Refrigerator.Settings.FRONT);
  assert.ok(path.slice(0,firstHigh).some(point=>point[1]<=Refrigerator.Settings.FRONT));
  assert.ok(path.at(-1)[0]>-.9);
});

test('door closing and other transport cannot interrupt active storage',async()=>{
  const scene=new ColdStorage();await scene.refresh();await scene.controller.storeRefrigerator();
  assert.equal(await scene.controller.commandRefrigerator('close'),false);
  assert.equal(await scene.controller.moveToScale(),false);
  assert.equal(await scene.controller.moveToPipette(),false);
});

test('retrieval requires an authored free rack support',async()=>{
  const scene=new ColdStorage();scene.snapshot.refrigerator.slots[0].key='tube';
  scene.snapshot.objects.tube=scene.snapshot.refrigerator.slots[0].position.concat([0,0,0,1]);
  scene.snapshot.objects.other=scene.config.rackSlots[0].position.concat([0,0,0,1]);await scene.refresh();
  assert.equal(await scene.controller.retrieveRefrigerator(),false);
  assert.equal(scene.commands.length,0);
  assert.ok(scene.controller.refrigeratorError.includes('Free a rack position'));
});

test('measured temperature replaces authored digits at the actual screen bounds',()=>{
  const root=new THREE.Group(),screen=new THREE.Mesh(new THREE.BoxGeometry(.145,.002,.036),new THREE.MeshBasicMaterial());
  screen.name='Refrigerator_display';screen.position.set(-1.14,-.293,.755);root.add(screen);
  const words=[];const document={createElement:()=>({getContext:()=>({fillRect(){},fillText:text=>words.push(text)})})};
  const view=Refrigerator.create(THREE,root,document);
  view.update({available:true,temperatureC:6.2,targetTemperatureC:4,doorOpen:true,doorClosed:false});
  assert.ok(words.includes('6.2 °C'));assert.ok(words.includes('DOOR OPEN'));
  assert.equal(view.mesh.visible,true);assert.ok(Math.abs(view.mesh.position.x-screen.position.x)<1e-6);
  const count=words.length;view.update({available:true,temperatureC:6.2,targetTemperatureC:4,doorOpen:true,doorClosed:false});assert.equal(words.length,count);
  view.update(null);assert.ok(words.includes('-- °C'));
  view.destroy();assert.deepEqual(root.children,[screen]);
});

test('refrigerator has an independent manual and reports only observed occupancy',()=>{
  const scene=new ColdStorage();
  assert.equal(Devices.identify({name:'Refrigerator door glass'}),'refrigerator');
  assert.equal(Devices.catalog(scene.config).find(device=>device.id==='refrigerator').metadata,scene.config.refrigerator);
  const parts=Parts.refrigerator(scene.config.refrigerator);
  assert.ok(parts.find(part=>part.id==='door').describe(scene.snapshot.refrigerator).includes('Open'));
  assert.ok(parts.find(part=>part.id==='holders').describe(scene.snapshot.refrigerator).includes('0 / 2'));
  assert.ok(parts.find(part=>part.id==='display').describe(scene.snapshot.refrigerator).includes('6.2 °C'));
});

// %% independently mounted instrument controls
class InstrumentControl {
  constructor(tag,ownerDocument) {this.tagName=tag;this.ownerDocument=ownerDocument;this.children=[];this.listeners=new Map();this.dataset={};this.attributes=new Map();}
  appendChild(child) {child.parent=this;this.children.push(child);}
  remove() {this.parent.children=this.parent.children.filter(child=>child!==this);}
  addEventListener(event,handler) {this.listeners.set(event,handler);}
  removeEventListener(event) {this.listeners.delete(event);}
  setAttribute(name,value) {this.attributes.set(name,value);}
  flatten() {return [this,...this.children.flatMap(child=>child.flatten())];}
}
class InstrumentControls {
  constructor() {this.host=new InstrumentControl('div',this);}
  createElement(tag) {return new InstrumentControl(tag,this);}
  find(className) {return this.host.flatten().find(item=>item.className==='laboratory-physics-refrigerator-'+className);}
  setInterval() {return 1;}
  clearInterval() {}
}
test('mounted controls show simulated temperature and observed holder occupancy',async()=>{
  const scene=new ColdStorage(),surface=new InstrumentControls(),focus=[];
  const mounted=Physics.mount(surface.host,scene.config,scene.service,{setObjectPose(){},focus:(...args)=>focus.push(args)},surface);
  await mounted.initialized;
  assert.equal(surface.find('temperature').textContent,scene.snapshot.refrigerator.temperatureC.toFixed(1)+' °C · simulated');
  assert.equal(surface.find('store').disabled,false);
  assert.equal(surface.find('retrieve').disabled,true);
  surface.find('view').listeners.get('click')();assert.equal(focus[0][1][0],scene.config.refrigerator.origin[0]);
  scene.snapshot.refrigerator.doorOpen=false;scene.snapshot.refrigerator.doorClosed=true;await mounted.refresh();
  assert.equal(surface.find('store').disabled,true);
  scene.snapshot.refrigerator.doorOpen=true;scene.snapshot.refrigerator.slots[0].key='tube';
  scene.snapshot.objects.other=scene.config.rackSlots[0].position.concat([0,0,0,1]);await mounted.refresh();
  assert.equal(surface.find('retrieve').disabled,true);
  assert.ok(surface.find('status').textContent.includes('Free a rack position'));
  assert.ok(surface.find('slot').children[0].textContent.includes('tube'));
  mounted.destroy();assert.deepEqual(surface.host.children,[]);
});

test('an in-flight old door observation cannot overwrite an acknowledged command',async()=>{
  const scene=new ColdStorage();await scene.refresh();let finish;
  scene.service.state=()=>new Promise(resolve=>{finish=resolve;});
  const stale=structuredClone(scene.snapshot),pending=scene.controller.refresh();
  const newer={...scene.snapshot.refrigerator,doorOpen:false,doorAngle:0,doorClosed:true};
  scene.service.refrigerator=async()=>({ok:true,refrigerator:newer});
  await scene.controller.commandRefrigerator('close');finish(stale);await pending;
  assert.equal(scene.controller.refrigerator,newer);
});

test('cancelling retrieval holds the last measured position even before leaving its occupied collar',async()=>{
  const scene=new ColdStorage();scene.snapshot.refrigerator.slots[0].key='tube';
  scene.snapshot.objects.tube=scene.snapshot.refrigerator.slots[0].position.concat([0,0,0,1]);
  await scene.refresh();await scene.controller.retrieveRefrigerator();await scene.controller.stopRefrigeratorTransfer();
  assert.equal(scene.controller.refrigeratorTransfer,null);
  assert.deepEqual(scene.commands.at(-1).position,scene.snapshot.objects.tube.slice(0,3));
});

test('storing a dosed glass withdraws from beneath the nozzle before lifting',()=>{
  const pose=[-.1,-.11,.916,0,0,0,1],destination=[-1.37,-.1,.24];
  const path=Refrigerator.storagePath(pose,destination,{targetPosition:pose.slice(0,3),nozzlePosition:[-.1,-.11,1.08]});
  const lift=path.findIndex(point=>point[2]>pose[2]);
  assert.ok(lift>0);
  assert.ok(path[lift][1]<=Refrigerator.Settings.FRONT);
});

test('stored samples and active storage prevent PR2 resets and reject non-glass storage',async()=>{
  const scene=new ColdStorage();scene.config.robot=true;await scene.refresh();
  scene.snapshot.refrigerator.slots[0].key='tube';
  assert.equal(await scene.controller.run(),false);
  scene.snapshot.refrigerator.slots[0].key=null;scene.controller.liquid={tubes:{}};
  assert.equal(await scene.controller.storeRefrigerator(),false);
  assert.equal(scene.commands.length,0);
});

test('failed observations clear refrigerator graphics rather than retaining a live temperature',async()=>{
  const scene=new ColdStorage();await scene.refresh();
  scene.service.state=async()=>{throw new Error('Connection lost');};await scene.refresh();
  assert.equal(scene.controller.refrigerator,null);
  assert.equal(scene.observations.at(-1),null);
});

test('a blocked waypoint stops at the observed position rather than pushing into the obstruction',async()=>{
  const scene=new ColdStorage();await scene.refresh();await scene.controller.storeRefrigerator();
  const observed=scene.snapshot.objects.tube.slice(0,3);
  scene.controller.refrigeratorTransfer.startedAt=Date.now()-Refrigerator.Settings.TIMEOUT-1;
  await scene.refresh();
  assert.equal(scene.controller.refrigeratorTransfer,null);
  assert.deepEqual(scene.commands.at(-1).position,observed);
  assert.ok(scene.controller.refrigeratorError.includes('Motion stopped'));
});

test('retrieval requires support from the destination rack before release',async()=>{
  const scene=new ColdStorage();scene.snapshot.refrigerator.slots[0].key='tube';
  scene.snapshot.objects.tube=scene.snapshot.refrigerator.slots[0].position.concat([0,0,0,1]);
  await scene.refresh();await scene.controller.retrieveRefrigerator();
  const transfer=scene.controller.refrigeratorTransfer;
  for(const waypoint of transfer.waypoints)await scene.reach(waypoint);
  scene.snapshot.contacts=[{bodyA:'tube',bodyB:'other_glass',force:.2}];await scene.refresh();
  assert.equal(scene.commands.some(command=>command.type==='release'),false);
  scene.snapshot.contacts=[{bodyA:'tube',bodyB:Refrigerator.Settings.RACK_BODY,force:.2}];await scene.refresh();
  assert.equal(scene.commands.at(-1).type,'release');
});

test('the final force target gently seats the sample below the nominal support surface',async()=>{
  const scene=new ColdStorage();await scene.refresh();await scene.controller.storeRefrigerator();
  const transfer=scene.controller.refrigeratorTransfer;
  for(const waypoint of transfer.waypoints.slice(0,-1))await scene.reach(waypoint);
  const nominal=scene.snapshot.refrigerator.slots[0].position;
  assert.deepEqual(scene.commands.at(-1).position,[nominal[0],nominal[1],nominal[2]-Refrigerator.Settings.SEATING_DEPTH]);
});
