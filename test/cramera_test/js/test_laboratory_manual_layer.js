'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const THREE = require('../../../cramera/src/cramera/web/vendor/three.min.js');
const Devices = require('../../../cramera/src/cramera/web/core/laboratory-devices.js');
const source = fs.readFileSync(path.join(__dirname, '../../../cramera/src/cramera/web/panels/robot_scene/panel.js'), 'utf8');

// %% actual panel bindings without a graphics context
class InstrumentControl {
  constructor() { this.hidden=false; this.style={}; this.attributes={}; this.children=[]; this.listeners=new Map(); }
  appendChild(element) { this.children.push(element); return element; }
  setAttribute(name,value) { this.attributes[name]=String(value); }
  addEventListener(name,callback) { this.listeners.set(name,callback); }
  removeEventListener(name) { this.listeners.delete(name); }
  remove() { this.removed=true; }
  getBoundingClientRect() { return {bottom:80}; }
}
class ManualLayerScene {
  constructor(centrifuge = {origin: [-.53,-.07,.9]}) {
    this.listeners = new Map();
    this.focusListeners = new Map();
    this.checkbox = {checked:false, addEventListener:(event, callback)=>this.listeners.set(event,callback), removeEventListener:event=>this.listeners.delete(event)};
    this.button = {disabled:true, addEventListener:(event, callback)=>this.focusListeners.set(event,callback), removeEventListener:event=>this.focusListeners.delete(event)};
    this.row = {style:{display:'none'}};
    this.controlListeners = new Map();
    this.controlsButton = {hidden:false, attributes:{}, setAttribute(name,value){this.attributes[name]=value;},
      addEventListener:(event,callback)=>this.controlListeners.set(event,callback), removeEventListener:event=>this.controlListeners.delete(event),
      remove(){this.removed=true;}};
    this.stageClasses = new Set();
    this.elements=[];
    this.focuses = [];
    this.overlay = {visible:false, observations:[], renders:0, destroyed:false,
      setParts(parts) {this.parts=parts;},
      setVisible(visible) {this.visible=visible;}, update(value) {this.observations.push(value);},
      render() {this.renders+=1;}, destroy() {this.destroyed=true;}};
    const stage = {clientWidth:1200,clientHeight:800,getBoundingClientRect:()=>({top:20,left:0,right:1200,bottom:820}),
      appendChild:element=>{this.appended=element;},
      classList:{toggle:(name,enabled)=>enabled?this.stageClasses.add(name):this.stageClasses.delete(name),remove:name=>this.stageClasses.delete(name)}};
    const panel = {getBoundingClientRect:()=>({left:900,right:1184})};
    const layers = {getBoundingClientRect:()=>({top:34,bottom:330})};
    const worldRoot=new THREE.Group();
    for (const [index,name] of ['Centrifuge enamel chassis','Pipetting rear housing','Instrument base','Rack base'].entries()) {
      const mesh=new THREE.Mesh(new THREE.BoxGeometry(.3,.2,.2)); mesh.name=name; mesh.position.set(index,0,.9); worldRoot.add(mesh);
    }
    this.context = vm.createContext({SCENE:{physics:{centrifuge}}, THREE, worldRoot, camera:new THREE.PerspectiveCamera(45,1.5,.01,200), needsRender:false, layersFolded:false, showFold(){},
      root:{ownerDocument:{createElement:()=>{const element=this.elements.length?new InstrumentControl():this.controlsButton;this.elements.push(element);return element;}},querySelector:selector=>selector==='.stage'?stage:selector==='.laboratory-physics'?panel:null},
      $:id=>({'lyr-manual':this.checkbox,'lyr-manual-row':this.row,'manual-focus':this.button,'layers-panel':layers})[id],
      LaboratoryManual:{create:config=>{this.config=config; return this.overlay;}},
      LaboratoryDevices:Devices,
      LaboratoryManualParts:{centrifuge:()=>[{id:'centrifuge'}],dispenser:()=>[{id:'dispenser'}],scale:()=>[{id:'scale'}],rack:()=>[{id:'rack'}]},
      focusSceneCamera:(position,target)=>this.focuses.push({position,target}),
    });
    const first = source.indexOf('  function bindLaboratoryManualMode()');
    assert.notEqual(first, -1);
    vm.runInContext(source.slice(first, source.indexOf('  // %% physical laboratory feedback',first)),this.context);
    this.binding = vm.runInContext('bindLaboratoryManualMode()',this.context);
  }
}

test('Manual mode begins off, focuses the centrifuge when enabled and leaves observations unchanged',()=>{
  const scene = new ManualLayerScene();
  assert.equal(scene.overlay.visible,false);
  assert.equal(scene.row.style.display,'');
  assert.equal(scene.focuses.length,0);
  scene.checkbox.checked=true;
  scene.listeners.get('change')();
  assert.equal(scene.overlay.visible,true);
  assert.equal(scene.focuses.length,1);
  assert.equal(scene.context.needsRender,true);
  const snapshot=Object.freeze({available:true,lidOpen:true,rpm:0});
  scene.binding.update(snapshot);
  scene.binding.render();
  assert.equal(scene.overlay.observations.at(-1),snapshot);
  assert.equal(scene.overlay.renders,1);
  scene.checkbox.checked=false;
  scene.listeners.get('change')();
  assert.equal(scene.overlay.visible,false);
  assert.equal(scene.focuses.length,1);
  scene.binding.destroy();
});

test('clicking an instrument offers its own manual and switching never leaks another device observation',()=>{
  const scene=new ManualLayerScene();
  const centrifuge=Object.freeze({available:true,rpm:120});
  const scale=Object.freeze({available:true,grams:2.5});
  const dispenser=Object.freeze({available:true,deliveredMl:1});
  scene.binding.update(centrifuge,'centrifuge');
  scene.binding.update(scale,'scale');
  scene.binding.update(dispenser,'dispenser');
  assert.equal(scene.binding.selectDevice('scale'),true);
  assert.equal(scene.overlay.visible,false);
  assert.equal(scene.overlay.parts[0].id,'scale');
  assert.equal(scene.overlay.observations.at(-1),scale);
  scene.elements.find(element=>element.className==='laboratory-device-open').listeners.get('click')({preventDefault(){},stopPropagation(){}});
  assert.equal(scene.overlay.visible,true);
  assert.equal(scene.focuses.length,1);
  scene.binding.selectDevice('dispenser');
  assert.equal(scene.overlay.parts[0].id,'dispenser');
  assert.equal(scene.overlay.observations.at(-1),dispenser);
  assert.equal(scene.focuses.length,2);
  scene.binding.selectDevice('rack');
  assert.equal(scene.overlay.parts[0].id,'rack');
  assert.equal(scene.overlay.observations.at(-1),null);
  assert.equal(scene.binding.selectDevice('robot'),false);
  scene.binding.destroy();
});

test('annotation rails reserve the actual control panels and refocusing does not toggle simulation state',()=>{
  const scene=new ManualLayerScene();
  assert.ok(scene.config.reservedRight()>=300);
  assert.ok(scene.config.reservedTopLeft()>=310);
  scene.focusListeners.get('click')({preventDefault(){},stopPropagation(){}});
  assert.equal(scene.checkbox.checked,true);
  assert.equal(scene.overlay.visible,true);
  assert.equal(scene.focuses.length,1);
  scene.binding.destroy();
  assert.equal(scene.listeners.size,0);
  assert.equal(scene.focusListeners.size,0);
  assert.equal(scene.overlay.destroyed,true);
  assert.equal(scene.row.style.display,'none');
});

test('unrelated scenes do not expose a centrifuge manual control',()=>{
  const scene=new ManualLayerScene(null);
  assert.equal(scene.binding,null);
  assert.equal(scene.row.style.display,'none');
  assert.equal(scene.listeners.size,0);
});

test('manual view folds physical controls reversibly and reserves space only while they are shown',()=>{
  const scene=new ManualLayerScene();
  assert.equal(scene.controlsButton.hidden,true);
  scene.checkbox.checked=true;
  scene.listeners.get('change')();
  assert.equal(scene.controlsButton.hidden,false);
  assert.equal(scene.controlsButton.textContent,'Show controls');
  assert.equal(scene.controlsButton.attributes['aria-expanded'],'false');
  assert.equal(scene.stageClasses.has('laboratory-manual-hide-controls'),true);
  assert.equal(scene.config.reservedRight(),0);
  scene.controlListeners.get('click')({stopPropagation(){}});
  assert.equal(scene.controlsButton.textContent,'Hide controls');
  assert.equal(scene.controlsButton.attributes['aria-expanded'],'true');
  assert.equal(scene.stageClasses.has('laboratory-manual-hide-controls'),false);
  assert.ok(scene.config.reservedRight()>=300);
  scene.checkbox.checked=false;
  scene.listeners.get('change')();
  assert.equal(scene.controlsButton.hidden,true);
  assert.equal(scene.stageClasses.has('laboratory-manual-hide-controls'),false);
  scene.binding.destroy();
  assert.equal(scene.controlsButton.removed,true);
  assert.equal(scene.controlListeners.size,0);
});

test('folding Layers schedules a redraw even without physics polling',()=>{
  const context=vm.createContext({needsRender:false,layersFolded:true,Folding:{button:()=>({glyph:'+',title:'Expand'})},
    layersFoldEl:{},layersPanelEl:{classList:{toggle(){}}}});
  const start=source.indexOf('  function showFold()');
  const end=source.indexOf('  showFold();',start);
  vm.runInContext(source.slice(start,end),context);
  vm.runInContext('showFold()',context);
  assert.equal(context.needsRender,true);
});

test('manual mode makes room for the diagram and restores the previous Layers layout on exit',()=>{
  const scene=new ManualLayerScene();
  scene.checkbox.checked=true;
  scene.listeners.get('change')();
  assert.equal(scene.context.layersFolded,true);
  scene.focusListeners.get('click')({preventDefault(){},stopPropagation(){}});
  scene.checkbox.checked=false;
  scene.listeners.get('change')();
  assert.equal(scene.context.layersFolded,false);
  scene.context.layersFolded=true;
  scene.checkbox.checked=true;
  scene.listeners.get('change')();
  scene.checkbox.checked=false;
  scene.listeners.get('change')();
  assert.equal(scene.context.layersFolded,true);
});
