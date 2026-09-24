/* Force-driven laboratory manipulation with authoritative simulation feedback. */
(function (root, factory) {
  'use strict';
  const refrigerator = typeof module !== 'undefined' && module.exports ? require('./laboratory-refrigerator.js') : root.LaboratoryRefrigerator;
  const workflow = typeof module !== 'undefined' && module.exports ? require('./laboratory-workflow.js') : root.LaboratoryWorkflow;
  const laboratory = factory(refrigerator, workflow);
  if (typeof module !== 'undefined' && module.exports) module.exports = laboratory;
  if (root) root.LaboratoryPhysics = laboratory;
})(typeof window !== 'undefined' ? window : null, function (Refrigerator, Workflow) {
  'use strict';

  // %% simulation service
  const Routes = Object.freeze({
    START: '/api/laboratory/physics/start', STATE: '/api/laboratory/physics/state',
    TARGET: '/api/laboratory/physics/target', RELEASE: '/api/laboratory/physics/release',
    RESET: '/api/laboratory/physics/reset', STOP: '/api/laboratory/physics/stop',
    LIQUID: '/api/laboratory/physics/liquid',
    TARE: '/api/laboratory/physics/scale/tare',
    CENTRIFUGE: '/api/laboratory/physics/centrifuge',
    MANUAL: '/api/laboratory/physics/manual',
    PIPETTING: '/api/laboratory/physics/pipetting',
    REFRIGERATOR: '/api/laboratory/physics/refrigerator',
  });
  const RobotRoutes = Object.freeze({
    START: '/api/laboratory/physics/robot/start', STATE: '/api/laboratory/physics/robot/state',
    TARGET: '/api/laboratory/physics/robot/target', RELEASE: '/api/laboratory/physics/robot/release',
    RESET: '/api/laboratory/physics/robot/reset', STOP: '/api/laboratory/physics/robot/stop',
    RUN: '/api/laboratory/physics/robot/run', PAUSE: '/api/laboratory/physics/robot/pause',
    MIX: '/api/laboratory/physics/robot/mix',
    LIQUID: '/api/laboratory/physics/robot/liquid',
    TARE: '/api/laboratory/physics/robot/scale/tare',
    CENTRIFUGE: '/api/laboratory/physics/robot/centrifuge',
    MANUAL: '/api/laboratory/physics/robot/manual',
    PIPETTING: '/api/laboratory/physics/robot/pipetting',
    PIPETTE: '/api/laboratory/physics/robot/pipette',
    WORKFLOW: '/api/laboratory/physics/robot/workflow',
    REFRIGERATOR: '/api/laboratory/physics/robot/refrigerator',
  });
  const RobotStates = Object.freeze({IDLE: 'idle', RUNNING: 'running', PAUSED: 'paused', SUCCEEDED: 'succeeded', FAILED: 'failed'});
  const RobotSources = Object.freeze({TRANSFER: 'recorded_cram_joint_targets', MIXING: 'laboratory_liquid_mixing', PIPETTING: 'laboratory_pipetting', WORKFLOW: Workflow.Source});
  const RobotLabels = Object.freeze({idle: 'Ready', running: 'PR2 is moving the glass', paused: 'PR2 paused', succeeded: 'Glass transferred', failed: 'Movement stopped'});
  const MixingLabels = Object.freeze({...RobotLabels, running: 'PR2 is mixing the liquids', succeeded: 'Mixture placed in A2'});
  const PipettingLabels = Object.freeze({...RobotLabels, running: 'PR2 is dosing the sample', succeeded: 'Dosed sample returned to A1'});
  const RobotPhases = Object.freeze({approach: 'Approaching the glass', grasp: 'Checking the contact grasp', lift: 'Lifting the glass', transport: 'Transporting the glass', pour: 'Pouring the liquid', return: 'Returning the source glass', swirl: 'Gently swirling the mixture', place: 'Placing the glass in the rack', release: 'Setting down the glass', settle: 'Checking stable placement', complete: 'Complete'});
  const POLL_INTERVAL = 33;
  const RETRY_INTERVAL = 1500;
  const MANUAL_HISTORY_ENTRIES = 6;
  const MANUAL_DIGEST_LENGTH = 12;
  const MOVE_STEP = .01;
  const TILT_STEP = 15;
  const LIFT_CLEARANCE = .22;
  const VOLUME_STEP = .5;
  const GlassCamera = Object.freeze({POSITION: [.32, -.42, .28], TARGET: [0, 0, .075]});
  const ProgramCommands = Object.freeze(['run', 'runMixing', 'runPipetting', 'runWorkflow']);
  const LiquidCommands = Object.freeze(['fillLiquid', 'reset', ...ProgramCommands]);
  const ScaleCommands = Object.freeze(['tareScale', 'reset', ...ProgramCommands]);
  const SCALE_MOVE_TIMEOUT = 20000;
  const SCALE_MOVE_TOLERANCE = .006;
  const SCALE_SEATING_DEPTH = .0002;
  const SCALE_ROTATION_TOLERANCE = .05;
  const SCALE_SETTLING_SPEED = .003;
  const SCALE_SETTLING_ANGULAR_SPEED = .03;
  const SCALE_SEATING_ANGLE = Math.PI / 45;
  const STOPPER_CAP_RADIUS = .0105;
  const CentrifugeActions = Object.freeze({OPEN: 'open', CLOSE: 'close', SECURE: 'secure', RELEASE: 'release', START: 'start', STOP: 'stop'});
  const CentrifugeTransfer = Object.freeze({LOAD: 'load', UNLOAD: 'unload'});
  const CentrifugeSettings = Object.freeze({RPM: 120, DURATION: 20, MAXIMUM_DURATION: 300, STOPPED_RPM: .5, DISTANCE: .0025, ANGLE: .035, ROTOR_BODY: 'laboratory_centrifuge_rotor', SAMPLE_VOLUME: 5, TRANSPORT_STEP: .025});

  /** Exchange desired targets and observed simulation states with the local service. */
  class Client {
    constructor(request, config) {
      this.request = request;
      this.robot = Boolean(config && config.robot);
      this.routes = this.robot ? RobotRoutes : Routes;
    }
    async read(url, options) {
      const response = await this.request(url, options);
      const answer = await response.json();
      if (!response.ok || answer.ok === false) throw new Error(answer.error || 'The simulation is unreachable.');
      return answer;
    }
    post(url, body) {
      return this.read(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
    }
    start() { return this.post(this.routes.START); }
    state() { return this.read(this.routes.STATE, {method: 'GET', cache: 'no-store'}); }
    target(target) { return this.post(this.routes.TARGET, target); }
    release() { return this.post(this.routes.RELEASE); }
    reset() { return this.post(this.routes.RESET); }
    stop() { return this.post(this.routes.STOP); }
    run() { return this.post(this.routes.RUN); }
    runMixing() { return this.post(this.routes.MIX); }
    runPipetting() { return this.post(this.routes.PIPETTE); }
    runWorkflow(prepare = true) { return this.post(this.routes.WORKFLOW, {prepare}); }
    pipetting(command) { return this.post(this.routes.PIPETTING, command); }
    refrigerator(command) { return this.post(this.routes.REFRIGERATOR, command); }
    pause() { return this.post(this.routes.PAUSE); }
    fillLiquid(key, volumeMl) { return this.post(this.routes.LIQUID, {key, volumeMl}); }
    tareScale() { return this.post(this.routes.TARE); }
    centrifuge(command) { return this.post(this.routes.CENTRIFUGE, command); }
    importManual(document) { return this.post(this.routes.MANUAL, document); }
  }

  // %% target commands and observed contacts
  /** Serialize targets without letting local interaction replace physical object poses. */
  class Controller {
    constructor(config, service, adapter, changed) {
      this.config = config;
      this.service = service;
      this.adapter = adapter;
      this.changed = changed || (() => {});
      this.objects = {};
      this.contacts = [];
      this.liquid = null;
      this.scale = null;
      this.scaleRevision = 0;
      this.centrifuge = null;
      this.pipetting = null;
      this.pipettingRevision = 0;
      this.refrigerator = null;
      this.refrigeratorSlot = config.refrigerator && config.refrigerator.slots.length ? config.refrigerator.slots[0].id : null;
      this.refrigeratorTransfer = null;
      this.refrigeratorError = null;
      this.refrigeratorRevision = 0;
      this.pipettingTransfer = null;
      this.pipettingError = null;
      this.centrifugeSlot = 'C1';
      this.centrifugeTransfer = null;
      this.centrifugePair = null;
      this.centrifugeTransferError = null;
      this.centrifugeRevision = 0;
      this.manualError = null;
      this.scaleTransfer = null;
      this.scaleTransferError = null;
      this.simulationTime = null;
      this.target = null;
      this.robot = {state: RobotStates.IDLE, phase: '', progress: 0};
      this.robotRequest = null;
      this.workflowUncertain = false;
      this.workflowError = null;
      this.selected = config.objects.length ? config.objects[0].key : null;
      this.dragged = null;
      this.state = 'idle';
      this.starting = false;
      this.runRevision = 0;
      this.error = null;
      this.revision = 0;
      this.liquidRevision = 0;
      this.queue = [];
      this.activeCommand = null;
      this.draining = null;
      this.polling = false;
      this.disposed = false;
      this.nextPoll = 0;
    }
    ready() { return !this.disposed && !this.starting && this.state === 'running'; }
    robotBusy() { return this.config.robot && (this.robotRequest || this.robot.state === RobotStates.RUNNING); }
    manualReady() { return this.ready() && !this.robotBusy() && !this.workflowUncertain && !Workflow.active(this.robot) && !this.manualBusy(); }
    manualBusy() { return this.activeCommand === 'manual' || this.queue.some(command => command.type === 'manual'); }
    pipettingBusy() { return this.activeCommand === 'pipetting' || this.queue.some(command => command.type === 'pipetting'); }
    dispensing() { return Boolean(this.pipetting && this.pipetting.state === 'dispensing'); }
    refrigeratorBusy() { return this.activeCommand === 'refrigerator' || this.queue.some(command => command.type === 'refrigerator'); }
    stored(key) { return Boolean(this.refrigerator && this.refrigerator.slots.some(slot => slot.key === key)); }
    refrigeratorReady() {
      return this.manualReady() && this.refrigerator && this.refrigerator.available && !this.refrigeratorBusy()
        && !this.refrigeratorTransfer && !this.scaleTransfer && !this.centrifugeTransfer && !this.centrifugePair
        && !this.pipettingTransfer && !this.dispensing() && !this.pipettingBusy() && !this.liquidBusy() && !this.draining;
    }
    selectedRefrigeratorSlot() { return this.refrigerator && this.refrigerator.slots.find(slot => slot.id === this.refrigeratorSlot); }
    commandRefrigerator(action) {
      if (!this.refrigeratorReady() || ![Refrigerator.Actions.OPEN, Refrigerator.Actions.CLOSE].includes(action)) return Promise.resolve(false);
      this.refrigeratorRevision += 1;
      this.refrigeratorError = null;
      return this.enqueue({type:'refrigerator',payload:{action}});
    }
    /** Move an existing unowned glass through the open cabinet front into a free holder. */
    storeRefrigerator() {
      const slot=this.selectedRefrigeratorSlot(),pose=this.objects[this.selected];
      if(!this.refrigeratorReady() || !this.refrigerator.doorOpen || !slot || slot.key || !pose || !this.selectedLiquid() || this.secured(this.selected) || this.stored(this.selected)) return Promise.resolve(false);
      return this.beginRefrigeratorTransfer(Refrigerator.Actions.STORE,this.selected,slot,Refrigerator.storagePath(pose,slot.position,this.config.pipetting));
    }
    /** Return a stored sample to an unoccupied rack support after exiting the cabinet. */
    retrieveRefrigerator() {
      const slot=this.selectedRefrigeratorSlot();
      if(!this.refrigeratorReady() || !this.refrigerator.doorOpen || !slot || !slot.key || !this.objects[slot.key]) return Promise.resolve(false);
      const destination=Refrigerator.returnPosition(this.config,this.objects,slot.key);
      if(!destination) {this.refrigeratorError='Free a rack position before retrieving a sample.';this.changed();return Promise.resolve(false);}
      return this.beginRefrigeratorTransfer(Refrigerator.Actions.RETRIEVE,slot.key,slot,Refrigerator.retrievalPath(this.objects[slot.key],destination));
    }
    beginRefrigeratorTransfer(type,key,slot,waypoints) {
      if(waypoints.some(position=>position.some((value,axis)=>value<this.config.bounds.min[axis] || value>this.config.bounds.max[axis]))) return Promise.resolve(false);
      this.refrigeratorError=null;this.error=null;
      this.refrigeratorTransfer={type,key,slot:slot.id,index:0,startedAt:Date.now(),waypoints,previousPose:this.objects[key].slice(),previousTime:this.simulationTime};
      return this.targetCommand(key,waypoints[0],Refrigerator.Upright);
    }
    stopRefrigeratorTransfer() {
      const transfer=this.refrigeratorTransfer;
      if(!transfer || !this.objects[transfer.key])return Promise.resolve(false);
      const pose=this.objects[transfer.key];
      const command=this.targetCommand(transfer.key,pose.slice(0,3),pose.slice(3));
      this.refrigeratorTransfer=null;
      this.refrigeratorError=null;
      this.changed();
      return command;
    }
    /** Advance only from measured poses and release after stable physical support. */
    async advanceRefrigeratorTransfer() {
      const transfer=this.refrigeratorTransfer;
      if(!transfer || this.refrigeratorBusy() || this.draining)return;
      if(!this.manualReady() || this.error || !this.refrigerator || !this.refrigerator.doorOpen) {this.refrigeratorTransfer=null;return;}
      const pose=this.objects[transfer.key];
      if(!pose) {this.refrigeratorTransfer=null;return;}
      if(Date.now()-transfer.startedAt>Refrigerator.Settings.TIMEOUT) {
        if(transfer.released) {
          this.refrigeratorTransfer=null;
          this.refrigeratorError='The released sample has not settled securely. Check its placement before continuing.';
        } else {
          await this.stopRefrigeratorTransfer();
          this.refrigeratorError='The sample path is blocked. Motion stopped at the observed position; adjust the held glass or release it.';
        }
        return;
      }
      const stable=Refrigerator.settled(transfer,pose,this.simulationTime);
      const target=transfer.waypoints[transfer.index];
      const distance=Math.hypot(...target.map((value,axis)=>value-pose[axis]));
      const upright=Math.abs(pose[6])/Math.hypot(...pose.slice(3))>=Math.cos(Refrigerator.Settings.ANGLE/2);
      if(distance>Refrigerator.Settings.TOLERANCE || !upright)return;
      if(transfer.index===transfer.waypoints.length-1) {
        if(!stable)return;
        const support=transfer.type===Refrigerator.Actions.RETRIEVE ? Refrigerator.Settings.RACK_BODY : this.config.refrigerator.body;
        const supported=this.contacts.some(contact=>contact.force>0 &&
          ((contact.bodyA===transfer.key && contact.bodyB===support) || (contact.bodyB===transfer.key && contact.bodyA===support)));
        if(!supported)return;
        if(!transfer.released) {transfer.released=true;await this.enqueue({type:'release'});this.target=null;return;}
        if(transfer.type===Refrigerator.Actions.STORE && !this.refrigerator.slots.some(slot=>slot.id===transfer.slot && slot.key===transfer.key))return;
        this.refrigeratorTransfer=null;this.changed();return;
      }
      transfer.index+=1;transfer.startedAt=Date.now();
      const next=transfer.waypoints[transfer.index].slice();
      if(transfer.index===transfer.waypoints.length-1)next[2]-=Refrigerator.Settings.SEATING_DEPTH;
      await this.targetCommand(transfer.key,next,Refrigerator.Upright);
    }
    pipettingReady() {
      return this.manualReady() && this.pipetting && this.pipetting.available && !this.dispensing() && !this.pipettingBusy()
        && !this.pipettingTransfer && !this.scaleTransfer && !this.centrifugeTransfer && !this.centrifugePair && !this.refrigeratorTransfer && !this.refrigeratorBusy() && !this.draining;
    }
    canDispense(source, volumeMl) {
      if (!this.pipettingReady() || this.secured(this.selected) || !Number.isFinite(volumeMl)) return false;
      const station = this.pipetting;
      const reservoir = station.reservoirs.find(item => item.id === source);
      const tube = this.selectedLiquid();
      const pose = this.objects[this.selected];
      return Boolean(reservoir && tube && pose && volumeMl >= station.minimumMl && volumeMl <= station.maximumMl
        && volumeMl <= reservoir.volumeMl && volumeMl <= tube.capacityMl - tube.volumeMl
        && Math.hypot(...station.targetPosition.map((value, axis) => value - pose[axis])) <= .008
        && this.observedTilt() <= 3);
    }
    dispense(source, volumeMl) {
      if (!this.canDispense(source, volumeMl)) return Promise.resolve(false);
      return this.commandPipetting({action: 'dispense', key: this.selected, source, volumeMl});
    }
    commandPipetting(payload) {
      if (!this.ready() || this.pipettingBusy()) return Promise.resolve(false);
      if (payload.action !== 'stop' && !this.pipettingReady()) return Promise.resolve(false);
      this.pipettingRevision += 1;
      this.pipettingError = null;
      return this.enqueue({type: 'pipetting', payload});
    }
    /** Approach the nozzle horizontally after clearing the rack in front. */
    moveToPipette() {
      const pose = this.objects[this.selected];
      if (!this.pipettingReady() || !pose || !this.selectedLiquid() || this.secured(this.selected) || this.stored(this.selected)) return Promise.resolve(false);
      const destination = this.pipetting.targetPosition;
      if (Math.hypot(...destination.map((value, axis) => value - pose[axis])) <= SCALE_MOVE_TOLERANCE && this.observedTilt() <= 3) return Promise.resolve(true);
      const elevation = Math.max(pose[2], destination[2]) + LIFT_CLEARANCE;
      const front = Math.min(pose[1], destination[1]) - .20;
      this.pipettingTransfer = {key: this.selected, index: 0, startedAt: Date.now(), waypoints: [
        [pose[0], pose[1], elevation], [pose[0], front, elevation],
        [destination[0], front, elevation], [destination[0], front, destination[2]], destination.slice(),
      ]};
      this.pipettingError = null;
      return this.targetCommand(this.selected, this.pipettingTransfer.waypoints[0], [0, 0, 0, 1]);
    }
    async advancePipettingTransfer() {
      const transfer = this.pipettingTransfer;
      if (!transfer) return;
      if (!this.manualReady() || this.error) { this.pipettingTransfer = null; return; }
      const observed = this.objects[transfer.key];
      if (!observed) { this.pipettingTransfer = null; return; }
      if (Date.now() - transfer.startedAt > SCALE_MOVE_TIMEOUT) {
        this.pipettingTransfer = null;
        this.pipettingError = 'The path to the pipetting station is blocked. Move the held glass manually or release it.';
        return;
      }
      const target = transfer.waypoints[transfer.index];
      if (Math.hypot(...target.map((value, axis) => value - observed[axis])) > SCALE_MOVE_TOLERANCE || this.observedTilt() > 3) return;
      transfer.index += 1;
      transfer.startedAt = Date.now();
      if (transfer.index === transfer.waypoints.length) { this.pipettingTransfer = null; return; }
      await this.targetCommand(transfer.key, transfer.waypoints[transfer.index], [0, 0, 0, 1]);
    }
    liquidBusy() { return LiquidCommands.includes(this.activeCommand) || this.queue.some(command => LiquidCommands.includes(command.type)); }
    scaleBusy() { return ScaleCommands.includes(this.activeCommand) || this.queue.some(command => ScaleCommands.includes(command.type)); }
    centrifugeBusy() { return this.manualBusy() || this.activeCommand === 'centrifuge' || this.queue.some(command => command.type === 'centrifuge'); }
    canImportManual() {
      const instrument = this.centrifuge;
      return this.manualReady() && instrument && instrument.available && !instrument.locked && !instrument.running
        && Math.abs(instrument.rpm) <= CentrifugeSettings.STOPPED_RPM && !this.centrifugeBusy()
        && !this.centrifugeTransfer && !this.centrifugePair && !this.scaleTransfer && !this.draining;
    }
    importManual(payload) {
      if (!this.canImportManual()) return Promise.resolve(false);
      this.manualError = null;
      this.centrifugeRevision += 1;
      const command = {type: 'manual', payload, accepted: false};
      const importing = this.enqueue(command);
      this.changed();
      return importing.then(() => command.accepted);
    }
    manualAllowsStart() {
      const knowledge = this.centrifuge && this.centrifuge.knowledge;
      return !knowledge || Boolean(knowledge.evaluation && knowledge.evaluation.allowed);
    }
    centrifugeOwnsObjects() { return Boolean(this.centrifuge && (this.centrifuge.locked || this.centrifuge.slots.some(slot => slot.secured))); }
    secured(key) { return Boolean(this.centrifuge && this.centrifuge.slots.some(slot => slot.secured && slot.key === key)); }
    selectedCentrifugeSlot() { return this.centrifuge && this.centrifuge.slots.find(slot => slot.id === this.centrifugeSlot); }
    selectCentrifugeSlot(slot) {
      if (this.centrifugeTransfer || this.centrifugePair || !this.centrifuge || !this.centrifuge.slots.some(item => item.id === slot)) return false;
      this.centrifugeSlot = slot;
      this.changed();
      return true;
    }
    centrifugeAccessible() {
      return this.manualReady() && this.centrifuge && this.centrifuge.available && this.centrifuge.lidOpen
        && !this.centrifuge.locked && Math.abs(this.centrifuge.rpm) <= CentrifugeSettings.STOPPED_RPM && !this.centrifugeBusy() && !this.refrigeratorTransfer && !this.refrigeratorBusy();
    }
    commandCentrifuge(action, options) {
      const instrument = this.centrifuge;
      if (!this.ready() || !instrument || !instrument.available || this.centrifugeBusy()) return Promise.resolve(false);
      if (action !== CentrifugeActions.STOP && (!this.manualReady() || this.centrifugeTransfer || this.centrifugePair)) return Promise.resolve(false);
      if ([CentrifugeActions.OPEN, CentrifugeActions.CLOSE].includes(action) && (instrument.locked || Math.abs(instrument.rpm) > CentrifugeSettings.STOPPED_RPM)) return Promise.resolve(false);
      if (action === CentrifugeActions.START && (!instrument.lidClosed || !instrument.balanced || instrument.running || instrument.locked || !this.manualAllowsStart())) return Promise.resolve(false);
      if (![CentrifugeActions.OPEN, CentrifugeActions.CLOSE, CentrifugeActions.START, CentrifugeActions.STOP].includes(action)) return Promise.resolve(false);
      const payload = {action, ...(options || {})};
      if (action === CentrifugeActions.START && (
        (payload.rpm !== undefined && (!Number.isFinite(payload.rpm) || payload.rpm <= 0 || payload.rpm > instrument.maximumRpm)) ||
        (payload.duration !== undefined && (!Number.isFinite(payload.duration) || payload.duration < 1 || payload.duration > CentrifugeSettings.MAXIMUM_DURATION)))) return Promise.resolve(false);
      return this.sendCentrifuge(payload);
    }
    sendCentrifuge(payload) {
      this.centrifugeRevision += 1;
      return this.enqueue({type: 'centrifuge', payload});
    }
    /** Find unoccupied glass tubes in selection order, excluding secured samples. */
    freeCentrifugeTubes() {
      const keys = this.config.objects.map(object => object.key).filter(key => this.objects[key]
        && this.liquid && this.liquid.tubes && this.liquid.tubes[key] && !this.secured(key) && !this.stored(key));
      return keys.includes(this.selected) ? [this.selected, ...keys.filter(key => key !== this.selected)] : keys;
    }
    /** Fill free opposing holders or complete one unmatched secured sample. */
    centrifugePairSamples() {
      if (!this.centrifuge) return [];
      const slots = this.centrifuge.slots;
      const occupied = slots.filter(slot => slot.key);
      if (occupied.some(slot => !slot.secured)) return [];
      const unmatched = occupied.filter(slot => !slots.some(item => item.id === slot.opposite && item.key));
      const keys = this.freeCentrifugeTubes();
      if (unmatched.length === 1) {
        const paired = occupied.filter(slot => slot !== unmatched[0]);
        if (paired.some(slot => !Number.isFinite(slot.grams)
          || slot.grams !== slots.find(item => item.id === slot.opposite).grams)) return [];
        const source = this.liquid && this.liquid.tubes[unmatched[0].key];
        const opposite = slots.find(slot => slot.id === unmatched[0].opposite && !slot.key);
        if (!source || !opposite || !Number.isFinite(source.volumeMl)) return [];
        const key = keys.find(key => this.liquid.tubes[key].capacityMl >= source.volumeMl);
        return key ? [{key, slot: opposite.id, volumeMl: source.volumeMl}] : [];
      }
      if (unmatched.length || (occupied.length && !this.centrifuge.balanced)) return [];
      const free = slots.filter(slot => !slot.key && slots.some(item => item.id === slot.opposite && !item.key));
      const slot = free.find(item => item.id === this.centrifugeSlot) || free[0];
      const opposite = slot && free.find(item => item.id === slot.opposite);
      const eligible = keys.filter(key => this.liquid.tubes[key].capacityMl >= CentrifugeSettings.SAMPLE_VOLUME);
      if (!slot || !opposite || eligible.length < 2) return [];
      return [slot, opposite].map((item, index) => ({key: eligible[index], slot: item.id, volumeMl: CentrifugeSettings.SAMPLE_VOLUME}));
    }
    /** Explain the current loading state and the next usable instrument action. */
    centrifugePairGuidance() {
      const instrument = this.centrifuge;
      const occupied = instrument ? instrument.slots.filter(slot => slot.key) : [];
      const samples = this.centrifugePairSamples();
      const unmatched = occupied.filter(slot => !instrument.slots.some(item => item.id === slot.opposite && item.key));
      const counterpart = unmatched.length === 1 && unmatched[0].secured;
      const label = counterpart ? 'Load counterbalance sample'
        : (occupied.length === 2 && samples.length === 2 ? 'Load second test pair (2 × 5 ml)' : 'Load test pair (2 × 5 ml)');
      if (this.centrifugePair) return {label: 'Loading test pair …', note: 'Sample ' + (this.centrifugePair.index + 1) + ' of ' + this.centrifugePair.samples.length + ' is being prepared and loaded. Please wait for the transfer.'};
      if (!this.ready()) return {label, note: 'First select “Start / resume laboratory”.'};
      if (this.robotBusy()) return {label, note: 'First pause PR2 or wait for its task to finish.'};
      if (!instrument || !instrument.available) return {label, note: 'The centrifuge is currently unavailable.'};
      if (this.centrifugeTransfer) return {label, note: 'A sample is being moved. Please wait for the transfer.'};
      if (occupied.some(slot => !slot.secured)) return {label: 'Check loading', note: 'At least one sample is not secured. Check its placement and complete loading.'};
      if (occupied.length >= 2 && !samples.length) {
        if (!instrument.balanced) return {label: 'Check loading', note: 'The loaded samples are not balanced. Load opposite slots with equal masses.'};
        const full = occupied.length === instrument.slots.length;
        const prepared = {label: full ? 'Centrifuge fully loaded' : 'Test pair ready', note: occupied.length + ' samples are already loaded and balanced. '};
        if (!full) prepared.note += 'A second pair requires two available glasses with sufficient capacity. ';
        if (instrument.locked || instrument.running || Math.abs(instrument.rpm) > CentrifugeSettings.STOPPED_RPM) prepared.note += 'Stop the rotor or wait for the current cycle to finish.';
        else if (instrument.lidOpen) prepared.note += 'Now select “Close lid”, then “Start centrifuge”.';
        else if (instrument.lidClosed) prepared.note += 'Now select “Start centrifuge”.';
        else prepared.note += 'Wait for the lid to reach its final position.';
        return prepared;
      }
      if (instrument.locked || instrument.running || Math.abs(instrument.rpm) > CentrifugeSettings.STOPPED_RPM) return {label, note: 'Before loading, stop the rotor and wait for the lid to unlock.'};
      if (!instrument.lidOpen) return {label, note: 'To load samples, first select “Open lid” and wait until it is fully open.'};
      if (this.scaleTransfer) return {label, note: 'First wait for the scale transfer to finish or stop it.'};
      if (this.centrifugeBusy() || this.liquidBusy()) return {label, note: 'The previous command is still running. Please wait.'};
      if (!samples.length) return {label, note: 'Loading requires available glasses with sufficient capacity and free opposite rotor slots.'};
      return {label, note: counterpart
        ? 'The counterbalance sample receives the same volume as the loaded sample (' + samples[0].volumeMl.toFixed(1) + ' ml). The existing sample stays unchanged.'
        : 'Loads two glasses containing 5 ml each into opposite slots in sequence. Then close the lid and start the centrifuge; no PR2 demo is needed.'};
    }
    /** Fill and seat balanced samples sequentially using measured contact feedback. */
    loadCentrifugePair() {
      if (!this.centrifugeAccessible() || this.centrifugeTransfer || this.centrifugePair || this.scaleTransfer || this.liquidBusy()) return Promise.resolve(false);
      const samples = this.centrifugePairSamples();
      if (!samples.length) return Promise.resolve(false);
      const pair = {samples, index: 0};
      this.centrifugePair = pair;
      this.centrifugeTransferError = null;
      this.error = null;
      return this.prepareCentrifugeSample(pair);
    }
    /** Continue only the uncancelled sample preparation after each server acknowledgement. */
    async prepareCentrifugeSample(pair) {
      if (this.centrifugePair !== pair) return false;
      if (pair.index === pair.samples.length) {
        this.centrifugePair = null;
        this.changed();
        return true;
      }
      const sample = pair.samples[pair.index];
      this.selected = sample.key;
      this.centrifugeSlot = sample.slot;
      this.changed();
      if (!await this.fillLiquidCommand(sample.key, sample.volumeMl)) {
        this.failCentrifugePair(pair);
        return false;
      }
      if (this.centrifugePair !== pair) return false;
      const loaded = await this.beginCentrifugeLoad(sample.key, sample.slot);
      if (!loaded) this.failCentrifugePair(pair);
      this.changed();
      return loaded;
    }
    /** Retain preparation failures until the user resumes manual interaction. */
    failCentrifugePair(pair) {
      if (this.centrifugePair !== pair && (this.centrifugePair || !this.error)) return;
      this.centrifugePair = null;
      this.centrifugeTransferError = this.error || 'Unable to load the sample. Check that the rotor slot is free and the lid is open.';
      this.error = this.centrifugeTransferError;
      this.changed();
    }
    /** Offer the opposite empty holder and another glass after an individual load. */
    selectCentrifugeCounterpart(slotId) {
      const completed = this.centrifuge && this.centrifuge.slots.find(slot => slot.id === slotId && slot.secured);
      const opposite = completed && this.centrifuge.slots.find(slot => slot.id === completed.opposite && !slot.key);
      const keys = this.freeCentrifugeTubes();
      if (!opposite || !keys.length) return;
      this.centrifugeSlot = opposite.id;
      this.selected = keys[0];
      this.changed();
    }
    loadCentrifuge() {
      if (this.centrifugePair) return Promise.resolve(false);
      return this.beginCentrifugeLoad(this.selected, this.centrifugeSlot);
    }
    beginCentrifugeLoad(key, slotId) {
      const slot = this.centrifuge && this.centrifuge.slots.find(item => item.id === slotId), pose = this.objects[key];
      if (!this.centrifugeAccessible() || !slot || slot.key || !pose || !this.liquid || !this.liquid.tubes[key] || this.secured(key) || this.stored(key) || this.centrifugeTransfer) return Promise.resolve(false);
      const elevation = Math.max(pose[2], slot.position[2]) + LIFT_CLEARANCE;
      if (elevation > this.config.bounds.max[2]) return Promise.resolve(false);
      this.scaleTransfer = null;
      this.centrifugeTransferError = null;
      this.error = null;
      const upright = [0, 0, 0, 1];
      // Short horizontal targets leave force available to support the sample's weight.
      const horizontalDistance = Math.hypot(slot.position[0] - pose[0], slot.position[1] - pose[1]);
      const stepCount = Math.max(1, Math.ceil(horizontalDistance / CentrifugeSettings.TRANSPORT_STEP));
      const traverse = Array.from({length: stepCount}, (_, index) => ({
        position: [0, 1].map(axis => pose[axis] + (slot.position[axis] - pose[axis]) * (index + 1) / stepCount).concat(elevation),
        orientation: upright,
      }));
      this.centrifugeTransfer = {type: CentrifugeTransfer.LOAD, key, slot: slot.id, index: 0,
        startedAt: Date.now(), previousPose: pose.slice(), previousTime: this.simulationTime, waypoints: [
          {position: [pose[0], pose[1], elevation], orientation: pose.slice(3)},
          {position: [pose[0], pose[1], elevation], orientation: upright},
          ...traverse,
          {position: slot.position.slice(), orientation: upright},
        ]};
      const first = this.centrifugeTransfer.waypoints[0];
      return this.targetCommand(key, first.position, first.orientation);
    }
    async unloadCentrifuge() {
      const slot = this.selectedCentrifugeSlot();
      if (!this.centrifugeAccessible() || !slot || !slot.secured || !slot.key || this.centrifugeTransfer || this.centrifugePair) return false;
      const key = slot.key, pose = this.objects[key];
      if (!pose || pose[2] + LIFT_CLEARANCE > this.config.bounds.max[2]) return false;
      const transfer = {type: CentrifugeTransfer.UNLOAD, key, slot: slot.id, index: 0, startedAt: Date.now(),
        previousPose: pose.slice(), previousTime: this.simulationTime, waypoints: [
          {position: pose.slice(0, 3), orientation: pose.slice(3)},
          {position: [pose[0], pose[1], pose[2] + LIFT_CLEARANCE], orientation: [0, 0, 0, 1]},
        ]};
      this.scaleTransfer = null;
      this.centrifugeTransferError = null;
      this.centrifugeTransfer = transfer;
      if (!await this.sendCentrifuge({action: CentrifugeActions.RELEASE, slot: slot.id}) || this.centrifugeTransfer !== transfer || !this.manualReady()) return false;
      // The accepted fixture release authorizes a force target at the last measured pose.
      this.centrifuge = {...this.centrifuge, slots: this.centrifuge.slots.map(item => item.id === slot.id ? {...item, key: null, secured: false} : item)};
      return this.targetCommand(key, pose.slice(0, 3), pose.slice(3));
    }
    async advanceCentrifugeTransfer() {
      const transfer = this.centrifugeTransfer;
      if (!transfer || this.centrifugeBusy() || this.draining) return;
      if (!this.manualReady() || this.error || !this.centrifuge || !this.centrifuge.lidOpen) { this.centrifugeTransfer = null; this.centrifugePair = null; return; }
      const pose = this.objects[transfer.key];
      if (!pose) { this.centrifugeTransfer = null; this.centrifugePair = null; return; }
      if (Date.now() - transfer.startedAt > SCALE_MOVE_TIMEOUT) {
        this.centrifugeTransfer = null;
        this.centrifugePair = null;
        this.centrifugeTransferError = transfer.released ? 'The sample is not yet seated securely in the rotor. Check its placement before securing it.' : 'The path to the centrifuge is blocked. The glass remains held; move it manually or release it.';
        this.error = this.centrifugeTransferError;
        return;
      }
      const target = transfer.waypoints[transfer.index];
      const elapsed = this.simulationTime === null || transfer.previousTime === null ? null : this.simulationTime - transfer.previousTime;
      const distance = Math.hypot(...target.position.map((value, axis) => value - pose[axis]));
      const travel = Math.hypot(...pose.slice(0, 3).map((value, axis) => value - transfer.previousPose[axis]));
      const alignment = (first, second) => Math.min(1, Math.abs(first.reduce((sum, value, axis) => sum + value * second[axis], 0)) / (Math.hypot(...first) * Math.hypot(...second)));
      const angularTravel = 2 * Math.acos(alignment(pose.slice(3), transfer.previousPose.slice(3)));
      transfer.previousPose = pose.slice();
      transfer.previousTime = this.simulationTime;
      if (elapsed === null || !(elapsed > 0) || travel / elapsed > SCALE_SETTLING_SPEED || angularTravel / elapsed > SCALE_SETTLING_ANGULAR_SPEED) return;
      if (distance > CentrifugeSettings.DISTANCE || alignment(pose.slice(3), target.orientation) < Math.cos(CentrifugeSettings.ANGLE / 2)) return;
      const last = transfer.index === transfer.waypoints.length - 1;
      if (last && transfer.type === CentrifugeTransfer.LOAD) {
        if (!transfer.released) {
          transfer.released = true;
          await this.enqueue({type: 'release'});
          return;
        }
        const supported = this.contacts.some(contact => contact.force > 0 &&
          ((contact.bodyA === transfer.key && contact.bodyB === CentrifugeSettings.ROTOR_BODY) || (contact.bodyB === transfer.key && contact.bodyA === CentrifugeSettings.ROTOR_BODY)));
        if (!supported) return;
        const pair = this.centrifugePair;
        this.centrifugeTransfer = null;
        if (await this.sendCentrifuge({action: CentrifugeActions.SECURE, slot: transfer.slot, key: transfer.key})) {
          this.target = null;
          if (pair && this.centrifugePair === pair) {
            pair.index += 1;
            await this.prepareCentrifugeSample(pair);
          } else if (!pair) this.selectCentrifugeCounterpart(transfer.slot);
        }
        return;
      }
      if (last) { this.centrifugeTransfer = null; return; }
      transfer.index += 1;
      transfer.startedAt = Date.now();
      const next = transfer.waypoints[transfer.index];
      await this.targetCommand(transfer.key, next.position, next.orientation);
    }
    tareScale() {
      if (!this.manualReady() || this.centrifugePair || !this.scale || !this.scale.available || !this.scale.stable || this.scaleBusy()) return Promise.resolve(false);
      this.scaleRevision += 1;
      return this.enqueue({type: 'tareScale'});
    }
    async start() {
      if (this.disposed || this.starting || this.ready()) return false;
      this.starting = true;
      this.runRevision += 1;
      this.liquidRevision += 1;
      this.scaleRevision += 1;
      this.target = null;
      this.dragged = null;
      this.liquid = null;
      this.scale = null;
      this.centrifuge = null;
      this.centrifugeTransfer = null;
      this.centrifugePair = null;
      if (this.adapter.setCentrifugeState) this.adapter.setCentrifugeState(null);
      this.centrifugeTransferError = null;
      this.centrifugeRevision += 1;
      this.scaleTransfer = null;
      this.scaleTransferError = null;
      this.queue = [];
      this.error = null;
      this.changed();
      try {
        if (this.draining) await this.draining;
        const answer = await this.service.start();
        if (this.disposed) return false;
        this.state = answer.state;
        this.starting = false;
        this.nextPoll = 0;
        await this.refresh();
        return this.ready();
      } catch (error) {
        this.state = 'failed';
        this.error = error.message;
        this.nextPoll = Date.now() + RETRY_INTERVAL;
        return false;
      } finally {
        this.starting = false;
        if (!this.disposed) this.changed();
      }
    }
    known(key) { return this.config.objects.some(object => object.key === key); }
    bounded(position) {
      if (!Array.isArray(position) || position.length !== 3 || !position.every(Number.isFinite)) return null;
      return position.map((value, axis) => Math.min(this.config.bounds.max[axis], Math.max(this.config.bounds.min[axis], value)));
    }
    setTarget(key, position, orientation) {
      this.refrigeratorTransfer = null;
      this.refrigeratorError = null;
      this.pipettingTransfer = null;
      this.centrifugeTransfer = null;
      this.centrifugePair = null;
      this.centrifugeTransferError = null;
      this.scaleTransfer = null;
      this.scaleTransferError = null;
      return this.targetCommand(key, position, orientation);
    }
    targetCommand(key, position, orientation) {
      const bounded = this.bounded(position);
      if (!this.manualReady() || !this.known(key) || this.secured(key) || (this.stored(key) && !(this.refrigeratorTransfer && this.refrigeratorTransfer.key === key)) || !bounded) return Promise.resolve(false);
      const rotation = orientation || (this.target && this.target.key === key && this.target.orientation);
      if (rotation && (!Array.isArray(rotation) || rotation.length !== 4 || !rotation.every(Number.isFinite) || Math.hypot(...rotation) === 0)) return Promise.resolve(false);
      this.selected = key;
      this.target = {key, position: bounded};
      if (rotation) this.target.orientation = rotation.slice();
      this.revision += 1;
      this.changed();
      return this.enqueue({type: 'target', target: {...this.target, position: bounded.slice()}});
    }
    moveToScale() {
      if (this.pipettingTransfer || this.dispensing() || this.pipettingBusy() || this.refrigeratorTransfer || this.refrigeratorBusy() || this.stored(this.selected)) return Promise.resolve(false);
      const pose = this.objects[this.selected];
      const scale = this.config.scale;
      if (!this.manualReady() || !pose || !scale || !scale.panPosition || this.scaleTransfer || this.centrifugeTransfer || this.centrifugePair || this.secured(this.selected)) return Promise.resolve(false);
      const pan = scale.panPosition.slice();
      if (scale.holder && !this.selectedLiquid() && scale.holder.innerRadius < STOPPER_CAP_RADIUS) {
        pan[0] += (scale.panRadius + scale.holder.outerRadius) / 2;
      }
      const elevation = Math.max(pose[2] + LIFT_CLEARANCE, pan[2] + LIFT_CLEARANCE);
      if (elevation > this.config.bounds.max[2]) return Promise.resolve(false);
      const upright = [0, 0, 0, 1];
      const seated = scale.holder && this.selectedLiquid()
        ? [0, Math.sin(SCALE_SEATING_ANGLE / 2), 0, Math.cos(SCALE_SEATING_ANGLE / 2)] : upright;
      this.scaleTransferError = null;
      this.scaleTransfer = {key: this.selected, index: 0, startedAt: Date.now(), previousPose: pose.slice(), previousTime: this.simulationTime, waypoints: [
        {position: [pose[0], pose[1], elevation], orientation: pose.slice(3)},
        {position: [pose[0], pose[1], elevation], orientation: upright},
        {position: [pan[0], pan[1], elevation], orientation: upright},
        {position: [pan[0], pan[1], pan[2] - SCALE_SEATING_DEPTH], orientation: seated},
      ]};
      const station = this.config.pipetting;
      if (station && Math.hypot(pose[0] - station.targetPosition[0], pose[1] - station.targetPosition[1]) < .04
        && pose[2] < station.nozzlePosition[2]) {
        const front = station.targetPosition[1] - .20;
        this.scaleTransfer.waypoints[0].position[1] = front;
        this.scaleTransfer.waypoints[1].position[1] = front;
        this.scaleTransfer.waypoints.unshift({position: [pose[0], front, pose[2]], orientation: upright});
      }
      const target = this.scaleTransfer.waypoints[0];
      return this.targetCommand(this.selected, target.position, target.orientation);
    }
    async advanceScaleTransfer() {
      const transfer = this.scaleTransfer;
      if (!transfer) return;
      if (!this.manualReady() || this.error) { this.scaleTransfer = null; return; }
      const observed = this.objects[transfer.key];
      if (!observed) { this.scaleTransfer = null; return; }
      if (Date.now() - transfer.startedAt > SCALE_MOVE_TIMEOUT) {
        this.scaleTransfer = null;
        this.scaleTransferError = 'The path to the scale is blocked. The glass remains held; move it manually or release it.';
        this.error = this.scaleTransferError;
        return;
      }
      const target = transfer.waypoints[transfer.index];
      const elapsed = this.simulationTime === null || transfer.previousTime === null ? null : this.simulationTime - transfer.previousTime;
      const travel = Math.hypot(...observed.slice(0, 3).map((value, axis) => value - transfer.previousPose[axis]));
      const previousRotation = transfer.previousPose.slice(3);
      const currentRotation = observed.slice(3);
      const rotationAgreement = Math.abs(previousRotation.reduce((sum, value, axis) => sum + value * currentRotation[axis], 0))
        / (Math.hypot(...previousRotation) * Math.hypot(...currentRotation));
      const rotationTravel = 2 * Math.acos(Math.min(1, rotationAgreement));
      transfer.previousPose = observed.slice();
      transfer.previousTime = this.simulationTime;
      if (elapsed !== null && (!(elapsed > 0) || travel / elapsed > SCALE_SETTLING_SPEED || rotationTravel / elapsed > SCALE_SETTLING_ANGULAR_SPEED)) return;
      const distance = Math.hypot(...target.position.map((value, axis) => value - observed[axis]));
      const observedRotation = observed.slice(3);
      const alignment = Math.abs(target.orientation.reduce((sum, value, axis) => sum + value * observedRotation[axis], 0))
        / (Math.hypot(...target.orientation) * Math.hypot(...observedRotation));
      if (distance > SCALE_MOVE_TOLERANCE || alignment < Math.cos(SCALE_ROTATION_TOLERANCE / 2)) return;
      if (transfer.index === transfer.waypoints.length - 1 && this.config.scale.holder
        && !(this.scale && this.scale.available && this.scale.objects.includes(transfer.key) && this.scale.forceNewtons > 0)) return;
      transfer.index += 1;
      transfer.startedAt = Date.now();
      if (transfer.index === transfer.waypoints.length) {
        this.scaleTransfer = null;
        await this.release();
        return;
      }
      const next = transfer.waypoints[transfer.index];
      await this.targetCommand(transfer.key, next.position, next.orientation);
    }
    hold(key) {
      if (this.refrigeratorTransfer && this.refrigeratorTransfer.key === key) return this.stopRefrigeratorTransfer();
      const pose = this.objects[key];
      return pose ? this.setTarget(key, pose.slice(0, 3)) : Promise.resolve(false);
    }
    beginDrag(key) {
      if (!this.manualReady() || this.centrifugePair || !this.known(key) || this.secured(key) || !this.objects[key]) return false;
      this.dragged = key;
      this.hold(key);
      return true;
    }
    dragTo(position) {
      if (!this.dragged || !this.target || this.target.key !== this.dragged) return Promise.resolve(false);
      return this.setTarget(this.dragged, [position[0], position[1], this.target.position[2]]);
    }
    nudge(delta) {
      const pose = this.target && this.target.key === this.selected ? this.target.position : this.objects[this.selected];
      if (!pose) return Promise.resolve(false);
      return this.setTarget(this.selected, delta.map((value, axis) => pose[axis] + value));
    }
    selectedLiquid() { return this.liquid && this.liquid.tubes && this.liquid.tubes[this.selected]; }
    tilt(degrees) {
      if (!this.selectedLiquid() || !Number.isFinite(degrees)) return Promise.resolve(false);
      const pose = this.objects[this.selected];
      if (!pose) return Promise.resolve(false);
      const target = this.target && this.target.key === this.selected ? this.target : null;
      const [x, y, z, w] = target && target.orientation ? target.orientation : pose.slice(3);
      const angle = degrees * Math.PI / 360;
      const sine = Math.sin(angle), cosine = Math.cos(angle);
      const orientation = [cosine * x + sine * z, cosine * y + sine * w, cosine * z - sine * x, cosine * w - sine * y];
      return this.setTarget(this.selected, target ? target.position : pose.slice(0, 3), orientation);
    }
    upright() {
      if (!this.selectedLiquid()) return Promise.resolve(false);
      const pose = this.target && this.target.key === this.selected ? this.target.position : this.objects[this.selected];
      return pose ? this.setTarget(this.selected, pose.slice(0, 3), [0, 0, 0, 1]) : Promise.resolve(false);
    }
    fillLiquid(volumeMl) {
      if (this.centrifugePair) return Promise.resolve(false);
      return this.fillLiquidCommand(this.selected, volumeMl);
    }
    fillLiquidCommand(key, volumeMl) {
      if (this.dispensing() || this.pipettingBusy() || this.pipettingTransfer || this.refrigeratorTransfer || this.refrigeratorBusy() || this.stored(this.selected)) return Promise.resolve(false);
      const tube = this.liquid && this.liquid.tubes && this.liquid.tubes[key];
      if (!this.manualReady() || this.secured(key) || !tube || !Number.isFinite(volumeMl) || volumeMl < 0 || volumeMl > tube.capacityMl) return Promise.resolve(false);
      this.revision += 1;
      this.liquidRevision += 1;
      return this.enqueue({type: 'fillLiquid', key, volumeMl, revision: this.revision});
    }
    observedTilt() {
      const pose = this.objects[this.selected];
      if (!pose) return 0;
      const [x, y, z, w] = pose.slice(3);
      const length = x * x + y * y + z * z + w * w;
      return length ? Math.acos(Math.max(-1, Math.min(1, 1 - 2 * (x * x + y * y) / length))) * 180 / Math.PI : 0;
    }
    viewGlass() {
      const pose = this.objects[this.selected];
      if (!pose || !this.adapter.focus) return false;
      this.adapter.focus(
        GlassCamera.POSITION.map((offset, axis) => pose[axis] + offset),
        GlassCamera.TARGET.map((offset, axis) => pose[axis] + offset),
      );
      return true;
    }
    release() { return this.clear('release'); }
    reset() { return this.clear('reset'); }
    run() { return this.startProgram('run'); }
    runMixing() { return this.startProgram('runMixing'); }
    runPipetting() { return this.startProgram('runPipetting'); }
    workflowReady() {
      const resuming = this.robot.source === RobotSources.WORKFLOW && this.robot.state === RobotStates.PAUSED;
      return Boolean(this.config.robot && this.ready() && !this.robotBusy() && !this.workflowUncertain && !this.draining
        && !this.manualBusy() && !this.scaleTransfer && !this.centrifugeTransfer && !this.centrifugePair
        && !this.pipettingTransfer && !this.refrigeratorTransfer && (resuming || !this.dispensing())
        && this.pipetting && this.pipetting.available && this.scale && this.scale.available
        && this.centrifuge && this.centrifuge.available && this.refrigerator && this.refrigerator.available);
    }
    runWorkflow() {
      if (!this.workflowReady()) return Promise.resolve(false);
      this.workflowError = null;
      this.robotRequest = 'runWorkflow';
      this.workflowUncertain = true;
      return this.clear('runWorkflow');
    }
    startProgram(type) {
      if (this.workflowUncertain || Workflow.active(this.robot)) return Promise.resolve(false);
      if (this.dispensing() || this.pipettingBusy() || this.pipettingTransfer || this.refrigeratorTransfer || this.refrigeratorBusy() || this.stored(this.selected)) return Promise.resolve(false);
      if ((this.refrigerator && this.refrigerator.slots.some(slot => slot.key)) || !this.config.robot || !this.ready() || this.robotBusy() || this.centrifugeTransfer || this.centrifugePair || this.centrifugeOwnsObjects()) return Promise.resolve(false);
      this.robotRequest = type;
      return this.clear(type);
    }
    pause() {
      if (!this.config.robot || !this.ready() || !this.robotBusy() || this.robotRequest === 'pause') return Promise.resolve(false);
      this.robotRequest = 'pause';
      return this.clear('pause');
    }
    clear(type) {
      this.refrigeratorTransfer = null;
      this.refrigeratorError = null;
      this.refrigeratorRevision += 1;
      if (type === 'reset' && this.adapter.setRefrigeratorState) this.adapter.setRefrigeratorState(null);
      this.pipettingTransfer = null;
      this.pipettingError = null;
      this.pipettingRevision += 1;
      if (type === 'reset' && this.adapter.setPipettingState) this.adapter.setPipettingState(null);
      this.centrifugeTransfer = null;
      this.centrifugePair = null;
      this.centrifugeTransferError = null;
      if (type === 'reset' || ProgramCommands.includes(type)) this.centrifugeRevision += 1;
      if ((type === 'reset' || ProgramCommands.includes(type)) && this.adapter.setCentrifugeState) this.adapter.setCentrifugeState(null);
      this.scaleTransfer = null;
      this.scaleTransferError = null;
      this.dragged = null;
      this.target = null;
      this.revision += 1;
      if (LiquidCommands.includes(type)) this.liquidRevision += 1;
      if (ScaleCommands.includes(type)) this.scaleRevision += 1;
      this.queue = this.queue.filter(command => command.type !== 'target');
      this.changed();
      return this.enqueue({type});
    }
    enqueue(command) {
      if (this.disposed) return Promise.resolve(false);
      if (command.type === 'target' && this.queue.length && this.queue[this.queue.length - 1].type === 'target') {
        this.queue[this.queue.length - 1] = command;
      } else this.queue.push(command);
      if (!this.draining) this.draining = this.drain();
      return this.draining;
    }
    async drain() {
      while (this.queue.length && !this.disposed) {
        const command = this.queue.shift();
        this.activeCommand = command.type;
        try {
          let answer;
          if (command.type === 'runWorkflow') answer = await this.service.runWorkflow(!(this.robot.source === RobotSources.WORKFLOW && this.robot.state === RobotStates.PAUSED));
          else if (command.type === 'target') answer = await this.service.target(command.target);
          else if (command.type === 'refrigerator') answer = await this.service.refrigerator(command.payload);
          else if (command.type === 'pipetting') answer = await this.service.pipetting(command.payload);
          else if (command.type === 'centrifuge') answer = await this.service.centrifuge(command.payload);
          else if (command.type === 'manual') answer = await this.service.importManual(command.payload);
          else if (command.type === 'fillLiquid') answer = await this.service.fillLiquid(command.key, command.volumeMl);
          else answer = await this.service[command.type]();
          if (answer.refrigerator && !this.disposed) {
            this.refrigerator = answer.refrigerator;
            if (this.adapter.setRefrigeratorState) this.adapter.setRefrigeratorState(this.refrigerator);
          }
          if (answer.pipetting && !this.disposed) {
            this.pipetting = answer.pipetting;
            if (this.adapter.setPipettingState) this.adapter.setPipettingState(this.pipetting);
          }
          if (['centrifuge', 'manual', 'reset', ...ProgramCommands].includes(command.type) && !this.disposed && answer.centrifuge) {
            this.centrifuge = answer.centrifuge;
            if (this.adapter.setCentrifugeState) this.adapter.setCentrifugeState(this.centrifuge);
          }
          if (command.type === 'manual') {
            this.manualError = null;
            command.accepted = true;
          }
          if (command.type === 'tareScale' && !this.disposed && answer.scale) {
            this.scale = answer.scale;
            if (this.adapter.setScaleState) this.adapter.setScaleState(this.scale);
          }
          if (command.type === 'fillLiquid' && !this.disposed && command.revision === this.revision && answer.liquid) {
            if (answer.objects) {
              this.objects = answer.objects;
              for (const [key, pose] of Object.entries(this.objects)) this.adapter.setObjectPose(key, pose);
            }
            this.liquid = answer.liquid;
            if (this.adapter.setLiquidState) this.adapter.setLiquidState(this.liquid);
          }
          if (this.config.robot && [...ProgramCommands, 'pause', 'reset'].includes(command.type)) {
            const startingProgram = ProgramCommands.includes(command.type);
            const state = startingProgram ? RobotStates.RUNNING : (command.type === 'pause' ? RobotStates.PAUSED : RobotStates.IDLE);
            const source = startingProgram ? (command.type === 'runWorkflow' ? RobotSources.WORKFLOW : command.type === 'runPipetting' ? RobotSources.PIPETTING : command.type === 'runMixing' ? RobotSources.MIXING : RobotSources.TRANSFER) : this.robot.source;
            if (answer.robot) {
              this.robot = answer.robot;
              this.workflowUncertain = false;
              if (command.type === 'runWorkflow' || command.type === 'reset') this.workflowError = null;
            } else if (command.type !== 'runWorkflow') {
              this.robot = {...this.robot, state, source};
              if (command.type === 'reset') {this.workflowUncertain = false;this.workflowError = null;}
            } else throw new Error('Workflow status was not returned. Refresh before starting again.');
          }
          this.error = null;
        } catch (error) {
          this.error = error.message;
          if (command.type === 'runWorkflow') this.workflowError = error.message;
          if (command.type === 'refrigerator') this.refrigeratorError = error.message;
          if (command.type === 'pipetting') this.pipettingError = error.message;
          if (command.type === 'manual') this.manualError = error.message;
          else {
            if (this.centrifugePair || this.centrifugeTransfer) this.centrifugeTransferError = this.error;
            this.target = null;
            this.dragged = null;
            this.scaleTransfer = null;
            this.pipettingTransfer = null;
            this.refrigeratorTransfer = null;
            this.centrifugeTransfer = null;
            this.centrifugePair = null;
            this.queue = this.queue.filter(pending => pending.type !== 'target');
          }
        }
        if (['refrigerator', 'reset', ...ProgramCommands].includes(command.type)) this.refrigeratorRevision += 1;
        if (LiquidCommands.includes(command.type)) this.liquidRevision += 1;
        if (['pipetting', 'reset', ...ProgramCommands].includes(command.type)) this.pipettingRevision += 1;
        if (ScaleCommands.includes(command.type)) this.scaleRevision += 1;
        if (['centrifuge', 'manual', 'reset'].includes(command.type) || ProgramCommands.includes(command.type)) this.centrifugeRevision += 1;
        this.activeCommand = null;
        if (this.robotRequest === command.type) this.robotRequest = null;
        this.changed();
      }
      this.draining = null;
      return !this.error;
    }
    async refresh() {
      if (this.polling || this.disposed || this.starting || Date.now() < this.nextPoll) return;
      const revision = this.revision;
      const runRevision = this.runRevision;
      const commandPending = Boolean(this.draining);
      const liquidRevision = this.liquidRevision;
      const liquidPending = this.liquidBusy();
      const scaleRevision = this.scaleRevision;
      const scalePending = this.scaleBusy();
      const centrifugeRevision = this.centrifugeRevision;
      const centrifugePending = this.centrifugeBusy();
      const pipettingRevision = this.pipettingRevision;
      const pipettingPending = this.pipettingBusy();
      const refrigeratorRevision = this.refrigeratorRevision;
      const refrigeratorPending = this.refrigeratorBusy();
      this.polling = true;
      try {
        const answer = await this.service.state();
        if (this.disposed || runRevision !== this.runRevision) return;
        this.state = answer.state;
        this.simulationTime = Number.isFinite(answer.time) ? answer.time : null;
        this.objects = answer.objects || {};
        this.contacts = answer.contacts || [];
        if (!commandPending && !this.draining && revision === this.revision) {
          if (!this.centrifugeTransfer) this.target = answer.target || null;
          this.robot = answer.robot || this.robot;
          if (answer.robot) {
            this.workflowUncertain = false;
            if (answer.robot.source === RobotSources.WORKFLOW && answer.robot.state === RobotStates.RUNNING) this.workflowError = null;
          }
        }
        if (answer.frames && this.adapter.setJointFrames) this.adapter.setJointFrames(answer.frames);
        for (const [key, pose] of Object.entries(this.objects)) this.adapter.setObjectPose(key, pose);
        if (!liquidPending && !this.liquidBusy() && liquidRevision === this.liquidRevision) {
          this.liquid = answer.liquid || null;
          if (this.liquid && this.adapter.setLiquidState) this.adapter.setLiquidState(this.liquid);
        }
        if (this.adapter.showContacts) this.adapter.showContacts(this.contacts, this.target);
        if (!pipettingPending && !this.pipettingBusy() && pipettingRevision === this.pipettingRevision) {
          this.pipetting = answer.pipetting || null;
          if (this.adapter.setPipettingState) this.adapter.setPipettingState(this.ready() ? this.pipetting : null, this.liquid, this.objects);
        }
        if (!scalePending && !this.scaleBusy() && scaleRevision === this.scaleRevision) {
          this.scale = answer.scale || null;
          if (this.adapter.setScaleState) this.adapter.setScaleState(this.scale);
        }
        if (!refrigeratorPending && !this.refrigeratorBusy() && refrigeratorRevision === this.refrigeratorRevision) {
          this.refrigerator = answer.refrigerator || null;
          if (this.refrigerator && this.refrigerator.joints && this.adapter.setJointFrames) this.adapter.setJointFrames(this.refrigerator.joints);
          if (this.adapter.setRefrigeratorState) this.adapter.setRefrigeratorState(this.ready() ? this.refrigerator : null);
        }
        if (!centrifugePending && !this.centrifugeBusy() && centrifugeRevision === this.centrifugeRevision) {
          this.centrifuge = answer.centrifuge || null;
          if (this.centrifuge && this.centrifuge.joints && this.adapter.setJointFrames) this.adapter.setJointFrames(this.centrifuge.joints);
          if (this.adapter.setCentrifugeState) this.adapter.setCentrifugeState(this.ready() ? this.centrifuge : null);
        }
        this.error = answer.error || this.scaleTransferError || this.centrifugeTransferError || this.pipettingError || this.refrigeratorError || null;
        await this.advanceRefrigeratorTransfer();
        await this.advancePipettingTransfer();
        await this.advanceScaleTransfer();
        await this.advanceCentrifugeTransfer();
        this.nextPoll = 0;
      } catch (error) {
        if (runRevision !== this.runRevision) return;
        this.state = 'failed';
        this.refrigerator = null;
        this.refrigeratorTransfer = null;
        if (this.adapter.setRefrigeratorState) this.adapter.setRefrigeratorState(null);
        this.pipetting = null;
        this.pipettingTransfer = null;
        if (this.adapter.setPipettingState) this.adapter.setPipettingState(null);
        this.scale = null;
        this.centrifuge = null;
        this.centrifugeTransfer = null;
        this.centrifugePair = null;
        if (this.adapter.setScaleState) this.adapter.setScaleState(null);
        if (this.adapter.setCentrifugeState) this.adapter.setCentrifugeState(null);
        this.error = error.message;
        this.nextPoll = Date.now() + RETRY_INTERVAL;
      } finally {
        this.polling = false;
        if (!this.disposed) this.changed();
      }
    }
    feedback() {
      const relevant = this.contacts.filter(contact => contact.bodyA === this.selected || contact.bodyB === this.selected);
      const force = relevant.reduce((sum, contact) => sum + Math.max(0, Number(contact.force) || 0), 0);
      const pose = this.target && this.objects[this.target.key];
      const distance = pose ? Math.hypot(...this.target.position.map((value, axis) => value - pose[axis])) : 0;
      return {contactCount: relevant.length, force, distance};
    }
    destroy() { this.disposed = true; this.queue = []; this.scaleTransfer = null; this.centrifugeTransfer = null; this.centrifugePair = null; this.pipettingTransfer = null; this.refrigeratorTransfer = null; }
  }

  // %% compact manipulation controls
  /** Mount object, movement and measured contact controls beside the live scene. */
  function mount(host, config, service, adapter, clock) {
    const document = host.ownerDocument;
    const scheduler = clock || document.defaultView;
    const listeners = [];
    let disposed = false;
    function element(tag, parent, className, text) {
      const item = document.createElement(tag);
      item.className = className || '';
      if (text) item.textContent = text;
      parent.appendChild(item);
      return item;
    }
    function listen(item, event, callback) {
      item.addEventListener(event, callback);
      listeners.push(() => item.removeEventListener(event, callback));
    }
    function button(parent, text, callback, className) {
      const item = element('button', parent, className, text);
      item.type = 'button';
      listen(item, 'click', callback);
      return item;
    }
    const panel = element('section', host, 'laboratory-physics');
    panel.setAttribute('aria-label', 'Laboratory with contact physics');
    element('h3', panel, '', config.robot ? 'PR2 & contact' : 'Glass & contact');
    element('p', panel, 'laboratory-physics-intro', config.robot ? 'PR2 grasps and moves the glass using contact physics.' : 'Drag, lift and release glasses. Obstacles resist their movement.');
    const label = element('label', panel, 'laboratory-physics-field', 'Object');
    const select = element('select', label, 'laboratory-physics-object');
    for (const object of config.objects) {
      const option = element('option', select, '', object.label || object.key);
      option.value = object.key;
    }
    const controller = new Controller(config, service, adapter, render);
    const workflow = config.robot ? new Workflow.View(panel, {start: () => controller.runWorkflow(), pause: () => controller.pause(), reset: () => controller.reset()}) : null;
    const refrigerator = element('section', panel, 'laboratory-physics-refrigerator');
    element('h4', refrigerator, '', 'Sample refrigerator');
    const refrigeratorTemperature = element('p', refrigerator, 'laboratory-physics-refrigerator-temperature');
    const refrigeratorStatus = element('p', refrigerator, 'laboratory-physics-refrigerator-status');
    refrigeratorStatus.setAttribute('role', 'status');
    const refrigeratorSlotLabel = element('label', refrigerator, 'laboratory-physics-field', 'Storage position');
    const refrigeratorSlot = element('select', refrigeratorSlotLabel, 'laboratory-physics-refrigerator-slot');
    for (const slot of config.refrigerator && config.refrigerator.slots || []) {
      const option = element('option', refrigeratorSlot, '', slot.id);
      option.value = slot.id;
    }
    listen(refrigeratorSlot, 'change', () => { if (!controller.refrigeratorTransfer) controller.refrigeratorSlot = refrigeratorSlot.value; render(); });
    const refrigeratorOpen = button(refrigerator, 'Open door', () => controller.commandRefrigerator('open'), 'laboratory-physics-refrigerator-open');
    const refrigeratorClose = button(refrigerator, 'Close door', () => controller.commandRefrigerator('close'), 'laboratory-physics-refrigerator-close');
    const refrigeratorStore = button(refrigerator, 'Store selected glass', () => controller.storeRefrigerator(), 'laboratory-physics-refrigerator-store');
    const refrigeratorRetrieve = button(refrigerator, 'Retrieve sample to rack', () => controller.retrieveRefrigerator(), 'laboratory-physics-refrigerator-retrieve');
    const refrigeratorCancel = button(refrigerator, 'Stop transfer', () => controller.stopRefrigeratorTransfer(), 'laboratory-physics-refrigerator-cancel');
    if (adapter.focus && config.refrigerator) button(refrigerator, 'View refrigerator', () => {
      const origin = config.refrigerator.origin;
      adapter.focus([origin[0]+.65,origin[1]-1.1,origin[2]+.90],[origin[0],origin[1]-.05,origin[2]+.42]);
    }, 'laboratory-physics-refrigerator-view');
    element('p', refrigerator, 'laboratory-physics-note', 'Open the door, select a free position and store an existing glass. Manual transfers use contact physics. Cabinet temperature is simulated; close the door to cool toward 4 °C. Retrieve stored samples before starting a PR2 task.');
    const pipetting = element('section', panel, 'laboratory-physics-pipetting');
    element('h4', pipetting, '', 'Pipetting station');
    const pipettingStatus = element('p', pipetting, 'laboratory-physics-pipetting-status');
    pipettingStatus.setAttribute('role', 'status');
    const reservoirStatus = element('p', pipetting, 'laboratory-physics-pipetting-reservoirs');
    const reagentLabel = element('label', pipetting, 'laboratory-physics-field', 'Reagent');
    const reagent = element('select', reagentLabel, 'laboratory-physics-pipetting-source');
    for (const [id, title] of [['amber', 'Amber reagent'], ['teal', 'Teal reagent']]) {
      const option = element('option', reagent, '', title); option.value = id;
    }
    reagent.value = 'amber';
    const doseLabel = element('label', pipetting, 'laboratory-physics-field', 'Dose (ml)');
    const dose = element('input', doseLabel, 'laboratory-physics-pipetting-volume');
    dose.type = 'number'; dose.min = '.1'; dose.max = '5'; dose.step = '.1'; dose.value = '1';
    listen(dose, 'input', render); listen(reagent, 'change', render);
    const positionPipette = button(pipetting, 'Position selected glass', () => controller.moveToPipette(), 'laboratory-physics-pipetting-position');
    const dispense = button(pipetting, 'Dispense dose', () => controller.dispense(reagent.value, Number(dose.value)), 'laboratory-physics-pipetting-dispense');
    const stopPipetting = button(pipetting, 'Stop dosing', () => controller.commandPipetting({action: 'stop'}), 'laboratory-physics-pipetting-stop');
    const cancelPipetting = button(pipetting, 'Stop transfer', () => controller.hold(controller.selected), 'laboratory-physics-pipetting-cancel');
    const refillPipetting = button(pipetting, 'Refill selected reagent', () => controller.commandPipetting({action: 'refill', source: reagent.value}), 'laboratory-physics-pipetting-refill');
    const viewPipetting = adapter.focus ? button(pipetting, 'View pipetting station', () => {
      const position = controller.pipetting.targetPosition;
      adapter.focus([position[0] + .30, position[1] - .48, position[2] + .34], [position[0], position[1] + .04, position[2] + .15]);
    }, 'laboratory-physics-pipetting-view') : null;
    element('p', pipetting, 'laboratory-physics-note', 'Position an open, upright glass under the nozzle, then dispense. Two separate reagent lines supply measured doses from finite reservoirs.');
    const scale = element('section', panel, 'laboratory-physics-scale');
    element('h4', scale, '', 'Scale');
    const scaleReading = element('p', scale, 'laboratory-physics-scale-reading');
    const scaleStatus = element('p', scale, 'laboratory-physics-scale-status');
    element('p', scale, 'laboratory-physics-note', 'Liquids are weighed using water density: 1 ml = 1 g.');
    const tare = button(scale, 'Tare / zero', () => controller.tareScale(), 'laboratory-physics-scale-tare');
    const moveToScale = button(scale, 'Move object to scale', () => controller.moveToScale(), 'laboratory-physics-scale-move');
    const cancelScale = button(scale, 'Stop transfer', () => controller.hold(controller.selected), 'laboratory-physics-scale-cancel');
    const viewScale = adapter.focus && config.scale ? button(scale, 'View scale', () => {
      const position = config.scale.panPosition;
      adapter.focus([position[0] + .28, position[1] - .45, position[2] + .3], position);
    }, 'laboratory-physics-scale-view') : null;
    const centrifuge = element('section', panel, 'laboratory-physics-centrifuge');
    element('h4', centrifuge, '', 'Centrifuge');
    const centrifugeStatus = element('p', centrifuge, 'laboratory-physics-centrifuge-status');
    centrifugeStatus.setAttribute('role', 'status');
    const manual = element('section', centrifuge, 'laboratory-physics-manual');
    element('h5', manual, '', 'Manual & robot knowledge');
    const manualVersion = element('p', manual, 'laboratory-physics-manual-version');
    const manualButtons = element('div', manual, 'laboratory-physics-manual-buttons');
    const manualV1 = button(manualButtons, 'Read manual V1', () => controller.importManual({edition: 'v1'}), 'laboratory-physics-manual-v1');
    const manualV2 = button(manualButtons, 'Read manual V2', () => controller.importManual({edition: 'v2'}), 'laboratory-physics-manual-v2');
    const manualChanges = element('p', manual, 'laboratory-physics-manual-changes');
    const manualSource = element('p', manual, 'laboratory-physics-manual-source');
    const manualObserved = element('p', manual, 'laboratory-physics-manual-observed');
    const manualReady = element('p', manual, 'laboratory-physics-manual-ready');
    const manualPlan = element('p', manual, 'laboratory-physics-manual-plan');
    const manualError = element('p', manual, 'laboratory-physics-manual-error');
    manualError.setAttribute('role', 'alert');
    const requirementsDetails = element('details', manual, 'laboratory-physics-manual-requirements-details');
    element('summary', requirementsDetails, '', 'Requirements and evidence');
    const manualRequirements = element('p', requirementsDetails, 'laboratory-physics-manual-requirements');
    const historyDetails = element('details', manual, 'laboratory-physics-manual-history-details');
    element('summary', historyDetails, '', 'Recently read manuals');
    const manualHistory = element('p', historyDetails, 'laboratory-physics-manual-history');
    const documentDetails = element('details', manual, 'laboratory-physics-manual-document-details');
    element('summary', documentDetails, '', 'Read / edit manual');
    const manualDigest = element('p', documentDetails, 'laboratory-physics-manual-digest');
    const manualDocument = element('textarea', documentDetails, 'laboratory-physics-manual-document');
    manualDocument.rows = 13;
    manualDocument.setAttribute('aria-label', 'Markdown manual with Operating rules table');
    manualDocument.spellcheck = false;
    let manualDraftDirty = false;
    let manualDocumentDigest = null;
    listen(manualDocument, 'input', () => { manualDraftDirty = true; render(); });
    const manualDraftStatus = element('p', documentDetails, 'laboratory-physics-manual-draft-status');
    const manualApply = button(documentDetails, 'Apply edited manual', async () => {
      const submitted = manualDocument.value;
      if (await controller.importManual({document: submitted}) && manualDocument.value === submitted) {
        manualDraftDirty = false;
        manualDocumentDigest = null;
        render();
      }
    }, 'laboratory-physics-manual-apply');
    const manualRestore = button(documentDetails, 'Show active document', () => {
      manualDraftDirty = false;
      manualDocumentDigest = null;
      render();
    }, 'laboratory-physics-manual-restore');
    element('p', manual, 'laboratory-physics-note', 'This demo reads the Operating rules table.');
    function sourceLabel(source) {
      if (!source) return 'Unknown source';
      if (source.section) return source.section + ' · line ' + source.line + (source.quote ? ': ' + source.quote : '');
      return source.label || source.kind || 'Unknown source';
    }
    function renderManual(instrument) {
      const knowledge = instrument && instrument.knowledge;
      manual.hidden = !knowledge;
      if (!knowledge) return;
      const active = knowledge.manual, evaluation = knowledge.evaluation;
      const importing = controller.manualBusy();
      const available = controller.canImportManual();
      manualVersion.textContent = active.title + ' · ' + active.version + ' · Knowledge revision ' + knowledge.revision;
      manualV1.disabled = manualV2.disabled = !available;
      manualRestore.disabled = importing;
      const changes = knowledge.changes || [];
      manualChanges.textContent = changes.map(change => change.label + ': ' + String(change.before) + ' → ' + String(change.after)).join('\n');
      manualChanges.hidden = !changes.length;
      manualSource.textContent = 'Manual source: ' + sourceLabel(active.source);
      const measured = instrument.slots.filter(item => item.secured && item.key).length;
      const observation = evaluation.observationSource;
      manualObserved.textContent = 'Observed: ' + measured + ' secured samples. Rule: at least ' + active.minimumSamples + '.\n'
        + (observation ? sourceLabel(observation) : 'Direct simulation state (no visual recognition).');
      manualReady.textContent = importing ? 'Reading manual …' : evaluation.allowed ? 'Derived: start requirements met.' : 'Derived: start is not yet allowed.';
      manualReady.dataset.allowed = String(Boolean(evaluation.allowed));
      manualPlan.textContent = 'Recommended steps:\n' + (evaluation.plan || []).join('\n');
      manualPlan.hidden = !(evaluation.plan || []).length;
      manualRequirements.textContent = (evaluation.requirements || []).map(requirement =>
        (requirement.satisfied === true ? '✓ ' : requirement.satisfied === false ? '✗ ' : '? ') + requirement.label
        + ' · observed: ' + String(requirement.actual) + ' · required: ' + String(requirement.required)
        + '\n' + sourceLabel(requirement.source)).join('\n\n');
      manualHistory.textContent = (knowledge.history || []).slice(-MANUAL_HISTORY_ENTRIES).map(entry =>
        'Knowledge revision ' + entry.revision + ' · ' + entry.version + ' · SHA-256: ' + entry.digest.slice(0, MANUAL_DIGEST_LENGTH)
        + (entry.changes.length ? '\n' + entry.changes.map(change => change.label + ': ' + String(change.before) + ' → ' + String(change.after)
          + '\n' + sourceLabel(change.source)).join('\n') : '\nNo operating rules changed.')).join('\n\n') || 'No previous documents yet.';
      manualError.textContent = controller.manualError || '';
      manualError.hidden = !controller.manualError;
      manualDigest.textContent = 'Active document · SHA-256: ' + active.digest.slice(0, MANUAL_DIGEST_LENGTH);
      manualDigest.title = active.digest;
      if (!manualDraftDirty && manualDocumentDigest !== active.digest) {
        manualDocument.value = active.document;
        manualDocumentDigest = active.digest;
      }
      manualApply.disabled = !available || !String(manualDocument.value || '').trim();
      manualDraftStatus.textContent = manualDraftDirty ? 'Unsaved draft – active version remains ' + active.version + '.' : 'Showing the active manual.';
    }
    const slotLabel = element('label', centrifuge, 'laboratory-physics-field', 'Rotor slot');
    const slotSelect = element('select', slotLabel, 'laboratory-physics-centrifuge-slot');
    const slotOptions = new Map();
    for (const id of ['C1', 'C2', 'C3', 'C4']) {
      const option = element('option', slotSelect, '', id);
      option.value = id;
      slotOptions.set(id, option);
    }
    listen(slotSelect, 'change', () => controller.selectCentrifugeSlot(slotSelect.value));
    const lidOpen = button(centrifuge, 'Open lid', () => controller.commandCentrifuge(CentrifugeActions.OPEN), 'laboratory-physics-centrifuge-open');
    const lidClose = button(centrifuge, 'Close lid', () => controller.commandCentrifuge(CentrifugeActions.CLOSE), 'laboratory-physics-centrifuge-close');
    const loadPair = button(centrifuge, 'Load test pair (2 × 5 ml)', () => controller.loadCentrifugePair(), 'laboratory-physics-centrifuge-pair');
    const pairNote = element('p', centrifuge, 'laboratory-physics-centrifuge-pair-note');
    const loadSample = button(centrifuge, 'Load sample', () => controller.loadCentrifuge(), 'laboratory-physics-centrifuge-load');
    const unloadSample = button(centrifuge, 'Lift sample out', () => controller.unloadCentrifuge(), 'laboratory-physics-centrifuge-unload');
    const centrifugeCancel = button(centrifuge, 'Stop transfer', () => controller.hold(controller.selected), 'laboratory-physics-centrifuge-cancel');
    element('p', centrifuge, 'laboratory-physics-note', 'Load opposite slots with equal masses (C1/C3, C2/C4). Samples are secured after bottom contact is measured in the rotor. Remove secured samples before starting PR2 tasks.');
    const rpmLabel = element('label', centrifuge, 'laboratory-physics-field', 'Speed (RPM)');
    const rpm = element('input', rpmLabel, 'laboratory-physics-centrifuge-rpm');
    rpm.type = 'number'; rpm.min = 1; rpm.step = 1; rpm.value = String(CentrifugeSettings.RPM);
    const durationLabel = element('label', centrifuge, 'laboratory-physics-field', 'Duration (simulation seconds)');
    const duration = element('input', durationLabel, 'laboratory-physics-centrifuge-duration');
    duration.type = 'number'; duration.min = 1; duration.max = CentrifugeSettings.MAXIMUM_DURATION; duration.step = 1; duration.value = String(CentrifugeSettings.DURATION);
    listen(rpm, 'input', render); listen(duration, 'input', render);
    const centrifugeStart = button(centrifuge, 'Start centrifuge', () => controller.commandCentrifuge(CentrifugeActions.START, {rpm: Number(rpm.value), duration: Number(duration.value)}), 'laboratory-physics-centrifuge-start');
    const centrifugeStop = button(centrifuge, 'Stop rotor', () => controller.commandCentrifuge(CentrifugeActions.STOP), 'laboratory-physics-centrifuge-stop');
    const viewCentrifuge = adapter.focus ? button(centrifuge, 'View centrifuge', () => {
      const slots = controller.centrifuge && controller.centrifuge.slots;
      if (!slots || !slots.length) return;
      const center = [0, 1, 2].map(axis => slots.reduce((sum, slot) => sum + slot.position[axis], 0) / slots.length);
      adapter.focus([center[0] + .32, center[1] - .52, center[2] + .4], center);
    }, 'laboratory-physics-centrifuge-view') : null;
    const liquid = element('section', panel, 'laboratory-physics-liquid');
    element('h4', liquid, '', 'Liquid');
    const liquidStatus = element('p', liquid, 'laboratory-physics-liquid-status');
    const volumeLabel = element('label', liquid, 'laboratory-physics-volume-label', 'Volume (ml)');
    const volume = element('input', volumeLabel, 'laboratory-physics-volume');
    volume.type = 'number';
    volume.min = 0;
    volume.step = VOLUME_STEP;
    let volumeKey = null;
    listen(volume, 'input', render);
    const fill = button(liquid, 'Set volume', () => controller.fillLiquid(Number(volume.value)), 'laboratory-physics-fill');
    const liftClear = button(liquid, 'Lift out of rack', () => controller.nudge([0, 0, LIFT_CLEARANCE]), 'laboratory-physics-lift-clear');
    liftClear.title = 'Raise target by ' + Math.round(LIFT_CLEARANCE * 100) + ' cm';
    const tilts = element('div', liquid, 'laboratory-physics-tilts');
    const tiltLeft = button(tilts, '−' + TILT_STEP + '°', () => controller.tilt(-TILT_STEP), 'laboratory-physics-tilt-left');
    tiltLeft.setAttribute('aria-label', 'Tilt left · ' + TILT_STEP + ' degrees');
    const upright = button(tilts, 'Upright', () => controller.upright(), 'laboratory-physics-upright');
    const tiltRight = button(tilts, '+' + TILT_STEP + '°', () => controller.tilt(TILT_STEP), 'laboratory-physics-tilt-right');
    tiltRight.setAttribute('aria-label', 'Tilt right · ' + TILT_STEP + ' degrees');
    const tiltStatus = element('p', liquid, 'laboratory-physics-tilt-status');
    const viewGlass = adapter.focus ? button(liquid, 'View glass', () => controller.viewGlass(), 'laboratory-physics-view-glass') : null;
    element('p', liquid, 'laboratory-physics-note', 'Lift before tilting. Simplified liquid model.');
    const start = button(panel, 'Start / resume laboratory', () => controller.start(), 'laboratory-physics-start');
    let robotRun, robotMix, robotPipette, robotPause, robotStatus, robotProgress, robotMeasurements, robotMotionNote;
    if (config.robot) {
      robotPipette = button(pipetting, 'Start pipetting demo', () => controller.runPipetting(), 'laboratory-physics-robot-pipette');
      element('p', pipetting, 'laboratory-physics-note', 'PR2 holds the clear glass under the nozzle, adds 1 ml of each reagent and returns the sample to A1.');
      robotMix = button(panel, 'Start mixing demo', () => controller.runMixing(), 'laboratory-physics-robot-mix');
      element('p', panel, 'laboratory-physics-robot-recipe', 'Pour 5 ml each of amber and teal liquid into the clear glass, gently swirl, then place it in A2. Starting a new run restores these volumes and the initial positions.');
      robotRun = button(panel, 'Start PR2: A1 → A3', () => controller.run(), 'laboratory-physics-robot-run');
      element('p', panel, 'laboratory-physics-robot-reset-note', 'A new run resets PR2 and the glasses; liquid volumes are preserved. Paused runs can be resumed.');
      robotPause = button(panel, 'Pause PR2', () => controller.pause(), 'laboratory-physics-robot-pause');
      robotStatus = element('p', panel, 'laboratory-physics-robot-status');
      robotStatus.setAttribute('role', 'status');
      robotProgress = element('progress', panel, 'laboratory-physics-robot-progress');
      robotProgress.max = 1;
      robotProgress.setAttribute('aria-label', 'PR2 movement progress');
      robotMeasurements = element('p', panel, 'laboratory-physics-robot-measurements');
      robotMotionNote = element('p', panel, 'laboratory-physics-robot-motion-note');
    }
    select.value = controller.selected;
    listen(select, 'change', () => { controller.refrigeratorTransfer = null; controller.pipettingTransfer = null; controller.scaleTransfer = null; controller.centrifugeTransfer = null; controller.centrifugePair = null; controller.selected = select.value; render(); });
    const hold = button(panel, 'Hold glass', () => controller.hold(select.value), 'laboratory-physics-hold');
    const moves = element('div', panel, 'laboratory-physics-moves');
    const motion = [
      ['←', [-MOVE_STEP, 0, 0], 'Move left'], ['↑', [0, MOVE_STEP, 0], 'Move back'],
      ['↓', [0, -MOVE_STEP, 0], 'Move forward'], ['→', [MOVE_STEP, 0, 0], 'Move right'],
      ['Lift', [0, 0, MOVE_STEP], 'Lift'], ['Lower', [0, 0, -MOVE_STEP], 'Lower'],
    ].map(([text, delta, title]) => {
      const control = button(moves, text, () => controller.nudge(delta));
      control.setAttribute('aria-label', title + ' · 1 cm');
      control.title = title + ' · 1 cm';
      return control;
    });
    const release = button(panel, 'Release glass', () => controller.release(), 'laboratory-physics-release');
    const status = element('p', panel, 'laboratory-physics-status');
    status.setAttribute('role', 'status');
    const force = element('p', panel, 'laboratory-physics-force');
    const distance = element('p', panel, 'laboratory-physics-distance');
    element('p', panel, 'laboratory-physics-note', 'Orange: desired position · Red: contact points');
    const camera = element('div', panel, 'laboratory-physics-cameras');
    if (config.cameras && adapter.focus) {
      for (const [name, text] of [['closeup', 'Close-up'], ['overview', 'Overview']]) {
        if (config.cameras[name]) button(camera, text, () => adapter.focus(config.cameras[name].position, config.cameras[name].target));
      }
    }
    const reset = button(panel, 'Reset laboratory', () => controller.reset(), 'laboratory-physics-reset');
    function render() {
      if (disposed) return;
      const ready = controller.ready();
      const manualReady = controller.manualReady();
      const reading = controller.scale;
      const instrument = controller.centrifuge;
      const station = controller.pipetting;
      const cabinet = controller.refrigerator;
      const storage = controller.selectedRefrigeratorSlot();
      const cabinetReady = controller.refrigeratorReady();
      const returnSupport = storage && storage.key && config.refrigerator ? Refrigerator.returnPosition(config, controller.objects, storage.key) : null;
      refrigerator.hidden = !config.refrigerator;
      refrigeratorTemperature.textContent = cabinet && cabinet.available ? cabinet.temperatureC.toFixed(1) + ' °C · simulated' : '-- °C';
      refrigeratorStatus.textContent = controller.refrigeratorError || (!cabinet || !cabinet.available ? 'Refrigerator unavailable'
        : (cabinet.doorOpen ? 'Door open' : cabinet.doorClosed ? 'Door closed' : 'Door moving')
          + ' · ' + cabinet.slots.filter(slot => slot.key).length + ' / ' + cabinet.slots.length + ' samples stored'
          + (controller.refrigeratorTransfer ? ' · ' + (controller.refrigeratorTransfer.type === 'store' ? 'Storing sample …' : 'Returning sample to rack …') : ''));
      if (storage && storage.key && !returnSupport && !controller.refrigeratorTransfer) refrigeratorStatus.textContent += ' · Free a rack position before retrieving a sample.';
      refrigeratorSlot.disabled = !cabinetReady;
      refrigeratorSlot.value = controller.refrigeratorSlot;
      for (const option of refrigeratorSlot.children) {
        const item = cabinet && cabinet.slots.find(slot => slot.id === option.value);
        option.textContent = option.value + (item ? ' · ' + (item.key || 'Empty') : ' · Unknown');
      }
      refrigeratorOpen.disabled = !cabinetReady || cabinet.doorOpen;
      refrigeratorClose.disabled = !cabinetReady || cabinet.doorClosed;
      refrigeratorStore.disabled = !cabinetReady || !cabinet.doorOpen || !storage || Boolean(storage.key) || !controller.selectedLiquid() || controller.secured(controller.selected) || controller.stored(controller.selected);
      refrigeratorRetrieve.disabled = !cabinetReady || !cabinet.doorOpen || !storage || !storage.key || !returnSupport;
      refrigeratorCancel.hidden = !controller.refrigeratorTransfer;
      pipetting.hidden = !config.pipetting && !station;
      pipettingStatus.textContent = controller.pipettingError || (controller.pipettingTransfer ? 'Positioning the glass …'
        : !station || !station.available ? 'Pipetting station unavailable'
          : station.error || (({idle: 'Ready', dispensing: 'Dispensing', complete: 'Dose complete', blocked: 'Stopped'}[station.state] || station.state)
            + ' · ' + Number(station.deliveredMl || 0).toFixed(2) + ' / ' + Number(station.requestedMl || 0).toFixed(2) + ' ml'));
      reservoirStatus.textContent = station ? station.reservoirs.map(item => item.label + ': ' + item.volumeMl.toFixed(1) + ' ml').join(' · ') : '';
      dose.disabled = reagent.disabled = !controller.pipettingReady();
      dispense.disabled = !String(dose.value).trim() || !controller.canDispense(reagent.value, Number(dose.value));
      positionPipette.disabled = !controller.pipettingReady() || !controller.selectedLiquid() || controller.secured(controller.selected);
      stopPipetting.disabled = !ready || !controller.dispensing() || controller.pipettingBusy();
      cancelPipetting.hidden = !controller.pipettingTransfer;
      refillPipetting.disabled = !controller.pipettingReady();
      if (viewPipetting) viewPipetting.disabled = !station || !station.available;
      renderManual(instrument);
      const slot = controller.selectedCentrifugeSlot();
      const secured = controller.secured(controller.selected) || controller.stored(controller.selected);
      const pair = controller.centrifugePair;
      const instrumentBusy = controller.centrifugeBusy() || Boolean(controller.centrifugeTransfer) || Boolean(pair) || Boolean(controller.refrigeratorTransfer) || controller.refrigeratorBusy();
      const accessible = controller.centrifugeAccessible() && !instrumentBusy;
      centrifuge.hidden = !config.centrifuge && !instrument;
      if (viewCentrifuge) viewCentrifuge.disabled = !instrument || !instrument.available;
      centrifugeStatus.textContent = !instrument || !instrument.available ? 'Centrifuge unavailable'
        : Math.round(instrument.rpm) + ' RPM · ' + Number(instrument.remainingSeconds || 0).toFixed(1) + ' s remaining · '
          + Number(instrument.rcf || 0).toFixed(1) + ' × g · ' + (instrument.balanced ? 'Balanced' : 'Balance the load')
          + ' · ' + (instrument.locked ? 'Locked' : (instrument.lidOpen ? 'Lid open' : (instrument.lidClosed ? 'Lid closed' : 'Lid moving')));
      if (pair) centrifugeStatus.textContent += ' · Sample ' + (pair.index + 1) + ' of ' + pair.samples.length + ' is being prepared and loaded …';
      else if (controller.centrifugeTransfer) centrifugeStatus.textContent += ' · Sample is being ' + (controller.centrifugeTransfer.type === CentrifugeTransfer.LOAD ? 'loaded …' : 'lifted out …');
      const pairSamples = controller.centrifugePairSamples();
      const pairGuidance = controller.centrifugePairGuidance();
      loadPair.textContent = pairGuidance.label;
      loadPair.disabled = !accessible || !pairSamples.length || Boolean(controller.scaleTransfer) || controller.liquidBusy();
      pairNote.textContent = pairGuidance.note;
      slotSelect.value = controller.centrifugeSlot;
      slotSelect.disabled = !manualReady || instrumentBusy;
      if (instrument) for (const item of instrument.slots) {
        const option = slotOptions.get(item.id);
        if (option) option.textContent = item.id + ' ↔ ' + item.opposite + (item.key ? ' · ' + item.key + ' · ' + item.grams.toFixed(1) + ' g' : ' · empty');
      }
      lidOpen.disabled = !manualReady || !instrument || instrumentBusy || instrument.locked || instrument.lidOpen || Math.abs(instrument.rpm) > CentrifugeSettings.STOPPED_RPM;
      lidClose.disabled = !manualReady || !instrument || instrumentBusy || instrument.locked || instrument.lidClosed || Math.abs(instrument.rpm) > CentrifugeSettings.STOPPED_RPM;
      loadSample.disabled = !accessible || !slot || Boolean(slot.key) || secured || !controller.selectedLiquid();
      unloadSample.disabled = !accessible || !slot || !slot.secured;
      centrifugeCancel.hidden = !controller.centrifugeTransfer && !pair;
      rpm.max = instrument ? instrument.maximumRpm : CentrifugeSettings.RPM;
      rpm.disabled = duration.disabled = !manualReady || !instrument || instrument.locked || instrumentBusy;
      const validCycle = String(rpm.value).trim() && String(duration.value).trim() && Number.isFinite(Number(rpm.value)) && Number(rpm.value) > 0
        && instrument && Number(rpm.value) <= instrument.maximumRpm && Number.isFinite(Number(duration.value)) && Number(duration.value) >= 1 && Number(duration.value) <= CentrifugeSettings.MAXIMUM_DURATION;
      centrifugeStart.disabled = !manualReady || !instrument || instrumentBusy || !instrument.lidClosed || !instrument.balanced || instrument.locked || instrument.running || !validCycle || !controller.manualAllowsStart();
      centrifugeStop.disabled = !ready || !instrument || controller.centrifugeBusy() || (!instrument.running && Math.abs(instrument.rpm) <= CentrifugeSettings.STOPPED_RPM);
      scale.hidden = !config.scale && !reading;
      scaleReading.textContent = reading && reading.available && Number.isFinite(reading.grams)
        ? (Math.abs(reading.grams) < .0005 ? 0 : reading.grams).toFixed(3) + ' g' : '— g';
      scaleStatus.textContent = !reading || !reading.available ? 'Scale unavailable'
        : (reading.stable ? 'Stable' : 'Settling …') + ' · Tare: ' + reading.tareGrams.toFixed(3) + ' g';
      tare.disabled = !manualReady || Boolean(pair) || !reading || !reading.available || !reading.stable || controller.scaleBusy();
      moveToScale.disabled = !manualReady || Boolean(pair) || !config.scale || !controller.objects[controller.selected] || Boolean(controller.scaleTransfer) || Boolean(controller.centrifugeTransfer) || Boolean(controller.refrigeratorTransfer) || controller.refrigeratorBusy() || secured;
      cancelScale.hidden = !controller.scaleTransfer;
      if (controller.scaleTransfer) scaleStatus.textContent = 'Lifting the object, moving it to the scale and setting it down …';
      start.hidden = ready;
      start.disabled = controller.starting;
      select.value = controller.selected;
      select.disabled = !manualReady || Boolean(pair) || Boolean(controller.refrigeratorTransfer);
      for (const control of [hold, ...motion]) control.disabled = !manualReady || Boolean(pair) || secured;
      reset.disabled = !ready;
      release.disabled = !manualReady || !controller.target;
      const tube = controller.selectedLiquid();
      liquid.hidden = !tube;
      if (viewGlass) viewGlass.disabled = !controller.objects[controller.selected];
      for (const control of [volume, fill, liftClear, tiltLeft, upright, tiltRight]) control.disabled = !manualReady || Boolean(pair) || !tube || secured;
      if (tube) {
        volume.max = tube.capacityMl;
        if (volumeKey !== controller.selected) {
          volume.value = String(Math.round(tube.volumeMl / VOLUME_STEP) * VOLUME_STEP);
          volumeKey = controller.selected;
        }
        const requestedVolume = Number(volume.value);
        fill.disabled = !manualReady || Boolean(pair) || secured || !String(volume.value).trim() || !Number.isFinite(requestedVolume) || requestedVolume < 0 || requestedVolume > tube.capacityMl;
        liquidStatus.textContent = tube.volumeMl.toFixed(1) + ' / ' + tube.capacityMl.toFixed(1) + ' ml · Spilled: ' + (Number(controller.liquid.spilledMl) || 0).toFixed(1) + ' ml';
        tiltStatus.textContent = 'Tilt: ' + Math.round(controller.observedTilt()) + '°';
      }
      status.textContent = controller.error || (controller.starting ? 'Starting laboratory …' : (ready ? (controller.robotBusy() ? 'Manual movement is paused while PR2 is running a task.' : Workflow.active(controller.robot) ? Workflow.PausedNote : (controller.target ? 'Glass held · Use the arrows to move the target.' : 'Select a glass or drag it directly in the scene.')) : 'The simulation is inactive. Start the laboratory to continue.'));
      status.dataset.error = String(Boolean(controller.error));
      if (config.robot) {
        const mixing = controller.robot.source === RobotSources.MIXING;
        const dosing = controller.robot.source === RobotSources.PIPETTING;
        const fullDemo = controller.robot.source === RobotSources.WORKFLOW;
        const workflowOwned = Workflow.active(controller.robot);
        workflow.render(controller.robot, {ready, canStart: controller.workflowReady(), canPause: fullDemo && controller.robot.state === RobotStates.RUNNING, pending: Boolean(controller.draining || controller.robotRequest), error: controller.workflowError || controller.error});
        const paused = controller.robot.state === RobotStates.PAUSED;
        robotMotionNote.textContent = mixing || dosing || fullDemo ? 'Motion: local contact controller. Contacts: MuJoCo.' : 'Joint targets: recorded CRAM plan. Motion and contacts: MuJoCo.';
        robotRun.disabled = workflowOwned || !ready || Boolean(controller.robotBusy()) || Boolean(controller.centrifugeTransfer) || Boolean(pair) || controller.centrifugeOwnsObjects() || controller.dispensing() || controller.pipettingBusy() || Boolean(controller.pipettingTransfer) || Boolean(controller.refrigeratorTransfer) || controller.refrigeratorBusy() || Boolean(cabinet && cabinet.slots.some(slot => slot.key));
        robotRun.textContent = paused && !mixing && !dosing && !fullDemo ? 'Resume PR2' : 'Start PR2: A1 → A3';
        robotPipette.disabled = robotRun.disabled || !station || !station.available;
        robotPipette.textContent = paused && dosing ? 'Resume pipetting demo' : 'Start pipetting demo';
        robotMix.disabled = robotRun.disabled;
        robotMix.textContent = paused && mixing ? 'Resume mixing demo' : 'Start mixing demo';
        robotPause.disabled = !ready || !controller.robotBusy() || controller.robotRequest === 'pause';
        const phase = controller.robot.message || RobotPhases[controller.robot.phase];
        const label = (fullDemo ? Workflow.Labels : dosing ? PipettingLabels : mixing ? MixingLabels : RobotLabels)[controller.robot.state] || controller.robot.state;
        robotStatus.textContent = controller.robot.error || (label + (phase && controller.robot.state === RobotStates.RUNNING ? ' · ' + phase : ''));
        robotStatus.dataset.error = String(Boolean(controller.robot.error));
        robotProgress.value = Math.max(0, Math.min(1, Number(controller.robot.progress) || 0));
        const measurements = [];
        if (Number.isFinite(controller.robot.gripperForce)) measurements.push('Grasp contact: ' + controller.robot.gripperForce.toFixed(2) + ' N');
        if (Number.isFinite(controller.robot.maximumLift)) measurements.push('Max. lift: ' + (controller.robot.maximumLift * 100).toFixed(1) + ' cm');
        robotMeasurements.textContent = measurements.join(' · ');
        robotMeasurements.hidden = !measurements.length;
      }
      const feedback = controller.feedback();
      force.textContent = 'Contacts: ' + feedback.contactCount + ' · ' + feedback.force.toFixed(2) + ' N';
      if (config.robot && controller.robot.holding) {
        distance.textContent = 'Held by PR2 · Contact physics active';
      } else if (config.robot && controller.robot.state === RobotStates.RUNNING) {
        distance.textContent = 'PR2 task running · Contact physics active';
      } else {
        distance.textContent = controller.target ? 'Distance to target: ' + (feedback.distance * 1000).toFixed(1) + ' mm' : 'Released · Gravity active';
      }
    }
    render();
    const initialized = controller.refresh();
    const timer = scheduler.setInterval(() => controller.refresh(), POLL_INTERVAL);
    return {
      controller, initialized, refresh: () => controller.refresh(),
      destroy() {
        disposed = true;
        controller.destroy();
        scheduler.clearInterval(timer);
        listeners.forEach(remove => remove());
        if (workflow) workflow.destroy();
        panel.remove();
      },
    };
  }

  /** Start or resume the laboratory physics scene from a non-physics page. */
  function mountStart(host, service, navigate) {
    const document = host.ownerDocument;
    const panel = document.createElement('section');
    panel.className = 'laboratory-physics-entry';
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = service.robot ? 'PR2 with contact physics' : 'Open laboratory with contact physics';
    const status = document.createElement('p');
    status.setAttribute('role', 'status');
    panel.appendChild(button);
    panel.appendChild(status);
    host.appendChild(panel);
    let pending = false;
    const start = async () => {
      if (pending) return;
      pending = true;
      button.disabled = true;
      status.textContent = 'Preparing laboratory …';
      try {
        const answer = await service.start();
        if (!answer.viewerUrl) throw new Error('The laboratory view is not ready yet.');
        navigate(answer.viewerUrl);
      } catch (error) {
        status.textContent = error.message;
        pending = false;
        button.disabled = false;
      }
    };
    button.addEventListener('click', start);
    return {destroy() { button.removeEventListener('click', start); panel.remove(); }};
  }

  return {Client, Controller, Routes, RobotRoutes, RobotStates, RobotSources, POLL_INTERVAL, MOVE_STEP, TILT_STEP, LIFT_CLEARANCE, SCALE_MOVE_TIMEOUT, CentrifugeActions, CentrifugeSettings, mount, mountStart};
});
