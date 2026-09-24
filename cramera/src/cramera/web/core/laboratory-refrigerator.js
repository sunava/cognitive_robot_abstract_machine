/* Contact-guided sample storage and observed cold-cabinet display. */
(function (global, factory) {
  'use strict';
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.LaboratoryRefrigerator = api;
})(typeof window === 'undefined' ? globalThis : window, function () {
  'use strict';

  // %% cabinet access paths
  const Settings = Object.freeze({FRONT: -.55, CLEARANCE: .045, BENCH_CLEARANCE: 1.20,
    STEP: .025, TOLERANCE: .006, SPEED: .003, ANGULAR_SPEED: .03, ANGLE: .05, TIMEOUT: 20000,
    FREE_DISTANCE: .036, RACK_BODY: 'laboratory_rack', SEATING_DEPTH: .0002});
  const Actions = Object.freeze({OPEN:'open',CLOSE:'close',STORE:'store',RETRIEVE:'retrieve'});
  const Upright = Object.freeze([0,0,0,1]);

  /** Subdivide long force targets so horizontal acceleration leaves vertical support. */
  function subdivide(points) {
    const result=[points[0].slice()];
    for(let index=1;index<points.length;index+=1) {
      const from=points[index-1],to=points[index];
      const steps=Math.max(1,Math.ceil(Math.hypot(...to.map((value,axis)=>value-from[axis]))/Settings.STEP));
      for(let step=1;step<=steps;step+=1) result.push(step===steps ? to.slice() : to.map((value,axis)=>from[axis]+(value-from[axis])*step/steps));
    }
    return result;
  }

  /** Enter below each shelf's ceiling, after lowering outside the cabinet front. */
  function storagePath(pose, destination, station) {
    const elevation=Math.max(Settings.BENCH_CLEARANCE,pose[2]+.22);
    const front=Math.min(Settings.FRONT,pose[1]-.20);
    const points=[pose.slice(0,3),[pose[0],pose[1],elevation],[pose[0],front,elevation],
      [destination[0],front,elevation],[destination[0],front,destination[2]+Settings.CLEARANCE],
      [destination[0],destination[1],destination[2]+Settings.CLEARANCE],destination.slice()];
    if(station && Math.hypot(pose[0]-station.targetPosition[0],pose[1]-station.targetPosition[1])<.04 && pose[2]<station.nozzlePosition[2]) {
      points.splice(1,1,[pose[0],front,pose[2]]);
    }
    return subdivide(points);
  }

  /** Exit the open cabinet before raising the sample toward a free bench support. */
  function retrievalPath(pose,destination) {
    const elevation=Math.max(Settings.BENCH_CLEARANCE,destination[2]+.22);
    const raised=pose[2]+Settings.CLEARANCE;
    return subdivide([pose.slice(0,3),[pose[0],pose[1],raised],[pose[0],Settings.FRONT,raised],
      [pose[0],Settings.FRONT,elevation],[destination[0],Settings.FRONT,elevation],
      [destination[0],destination[1],elevation],destination.slice()]);
  }

  /** Choose an unoccupied return support using the latest measured glass positions. */
  function returnPosition(config,objects,key) {
    const candidates=config.refrigerator.returnPositions || (config.rackSlots || []).map(slot=>(slot.position || slot.pose).slice(0,3));
    return candidates.find(position=>Object.entries(objects).every(([other,pose])=>other===key || Math.hypot(...position.map((value,axis)=>value-pose[axis]))>Settings.FREE_DISTANCE));
  }

  /** Compare successive observed poses without estimating motion from desired targets. */
  function settled(transfer,pose,time) {
    const elapsed=time-transfer.previousTime;
    const travel=Math.hypot(...pose.slice(0,3).map((value,axis)=>value-transfer.previousPose[axis]));
    const previous=transfer.previousPose.slice(3),rotation=pose.slice(3);
    const agreement=Math.min(1,Math.abs(previous.reduce((sum,value,axis)=>sum+value*rotation[axis],0))/(Math.hypot(...previous)*Math.hypot(...rotation)));
    const angularTravel=2*Math.acos(agreement);
    transfer.previousPose=pose.slice();transfer.previousTime=time;
    return elapsed>0 && travel/elapsed<=Settings.SPEED && angularTravel/elapsed<=Settings.ANGULAR_SPEED;
  }

  // %% observed temperature display
  const DISPLAY_NAME='laboratory-refrigerator-reading';
  const CANVAS_SIZE=Object.freeze([768,192]);
  const SCREEN_NAMES=new Set(['Refrigerator display','Refrigerator_display']);
  const LETTER_NAMES=new Set(['Refrigerator temperature lettering','Refrigerator_temperature_lettering']);

  /** Format the server's thermal model and observed door state without a local clock. */
  function reading(snapshot) {
    if(!snapshot || !snapshot.available || !Number.isFinite(snapshot.temperatureC)) return {temperature:'-- °C',status:'NO DATA'};
    return {temperature:snapshot.temperatureC.toFixed(1)+' °C',status:snapshot.doorOpen?'DOOR OPEN':snapshot.doorClosed?'COOLING':'DOOR MOVING'};
  }

  /** Fit a live readout to the authored screen after its mesh has loaded. */
  class Display {
    constructor(THREE,root,document) {
      this.THREE=THREE;this.root=root;this.hidden=new Map();this.screen=null;this.signature=null;
      this.canvas=document.createElement('canvas');[this.canvas.width,this.canvas.height]=CANVAS_SIZE;
      this.context=this.canvas.getContext('2d');this.texture=new THREE.CanvasTexture(this.canvas);this.texture.encoding=THREE.sRGBEncoding;
      this.mesh=new THREE.Mesh(new THREE.PlaneGeometry(1,1),new THREE.MeshBasicMaterial({map:this.texture,toneMapped:false}));
      this.mesh.name=DISPLAY_NAME;this.mesh.rotation.x=Math.PI/2;this.mesh.visible=false;root.add(this.mesh);
    }
    attach() {
      this.root.traverse(node=>{
        if(LETTER_NAMES.has(node.name)||LETTER_NAMES.has(node.userData.name)) {if(!this.hidden.has(node))this.hidden.set(node,node.visible);node.visible=false;}
        if(!this.screen&&node.geometry&&(SCREEN_NAMES.has(node.name)||SCREEN_NAMES.has(node.userData.name)))this.screen=node;
      });
      if(!this.screen||this.mesh.visible)return;
      const THREE=this.THREE;this.root.updateMatrixWorld(true);
      const matrix=new THREE.Matrix4().copy(this.root.matrixWorld).invert().multiply(this.screen.matrixWorld);
      this.screen.geometry.computeBoundingBox();const bounds=this.screen.geometry.boundingBox.clone().applyMatrix4(matrix);
      const size=bounds.getSize(new THREE.Vector3()),center=bounds.getCenter(new THREE.Vector3());
      this.mesh.geometry.dispose();this.mesh.geometry=new THREE.PlaneGeometry(size.x*.94,size.z*.94);
      this.mesh.position.set(center.x,bounds.min.y-.0001,center.z);this.mesh.visible=true;
    }
    update(snapshot) {
      this.attach();const value=reading(snapshot),signature=JSON.stringify(value);if(signature===this.signature)return;this.signature=signature;
      const context=this.context;context.fillStyle='#102d30';context.fillRect(0,0,...CANVAS_SIZE);
      context.textAlign='left';context.textBaseline='middle';context.fillStyle='#95d8c9';context.font='600 36px monospace';context.fillText(value.status,24,47,720);
      context.fillStyle='#edfff8';context.font='600 82px monospace';context.fillText(value.temperature,24,130,720);this.texture.needsUpdate=true;
    }
    destroy() {
      for(const [node,visible] of this.hidden)node.visible=visible;
      this.hidden.clear();this.root.remove(this.mesh);this.mesh.geometry.dispose();this.mesh.material.dispose();this.texture.dispose();
    }
  }
  return {Settings,Actions,Upright,storagePath,retrievalPath,returnPosition,settled,DISPLAY_NAME,reading,create:(THREE,root,document)=>new Display(THREE,root,document)};
});
