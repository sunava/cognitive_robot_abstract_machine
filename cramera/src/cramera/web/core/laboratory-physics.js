/* Force-driven laboratory manipulation with authoritative simulation feedback. */
(function (root, factory) {
  'use strict';
  const laboratory = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = laboratory;
  if (root) root.LaboratoryPhysics = laboratory;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  // %% simulation service
  const Routes = Object.freeze({
    START: '/api/laboratory/physics/start', STATE: '/api/laboratory/physics/state',
    TARGET: '/api/laboratory/physics/target', RELEASE: '/api/laboratory/physics/release',
    RESET: '/api/laboratory/physics/reset', STOP: '/api/laboratory/physics/stop',
    LIQUID: '/api/laboratory/physics/liquid',
    TARE: '/api/laboratory/physics/scale/tare',
    CENTRIFUGE: '/api/laboratory/physics/centrifuge',
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
  });
  const RobotStates = Object.freeze({IDLE: 'idle', RUNNING: 'running', PAUSED: 'paused', SUCCEEDED: 'succeeded', FAILED: 'failed'});
  const RobotSources = Object.freeze({TRANSFER: 'recorded_cram_joint_targets', MIXING: 'laboratory_liquid_mixing'});
  const RobotLabels = Object.freeze({idle: 'Bereit', running: 'PR2 bewegt das Glas', paused: 'PR2 pausiert', succeeded: 'Glas umgesetzt', failed: 'Bewegung angehalten'});
  const MixingLabels = Object.freeze({...RobotLabels, running: 'PR2 mischt die Flüssigkeiten', succeeded: 'Mischung in A2 abgelegt'});
  const RobotPhases = Object.freeze({approach: 'Glas anfahren', grasp: 'Kontaktgriff prüfen', lift: 'Glas anheben', transport: 'Glas transportieren', pour: 'Flüssigkeit eingießen', return: 'Spenderglas zurückstellen', swirl: 'Mischung sanft schwenken', place: 'Glas im Ständer abstellen', release: 'Glas absetzen', settle: 'Stabilen Stand prüfen', complete: 'Abgeschlossen'});
  const POLL_INTERVAL = 33;
  const RETRY_INTERVAL = 1500;
  const MOVE_STEP = .01;
  const TILT_STEP = 15;
  const LIFT_CLEARANCE = .22;
  const VOLUME_STEP = .5;
  const GlassCamera = Object.freeze({POSITION: [.32, -.42, .28], TARGET: [0, 0, .075]});
  const ProgramCommands = Object.freeze(['run', 'runMixing']);
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
      if (!response.ok || answer.ok === false) throw new Error(answer.error || 'Die Simulation ist nicht erreichbar.');
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
    pause() { return this.post(this.routes.PAUSE); }
    fillLiquid(key, volumeMl) { return this.post(this.routes.LIQUID, {key, volumeMl}); }
    tareScale() { return this.post(this.routes.TARE); }
    centrifuge(command) { return this.post(this.routes.CENTRIFUGE, command); }
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
      this.centrifugeSlot = 'C1';
      this.centrifugeTransfer = null;
      this.centrifugePair = null;
      this.centrifugeTransferError = null;
      this.centrifugeRevision = 0;
      this.scaleTransfer = null;
      this.scaleTransferError = null;
      this.simulationTime = null;
      this.target = null;
      this.robot = {state: RobotStates.IDLE, phase: '', progress: 0};
      this.robotRequest = null;
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
    manualReady() { return this.ready() && !this.robotBusy(); }
    liquidBusy() { return LiquidCommands.includes(this.activeCommand) || this.queue.some(command => LiquidCommands.includes(command.type)); }
    scaleBusy() { return ScaleCommands.includes(this.activeCommand) || this.queue.some(command => ScaleCommands.includes(command.type)); }
    centrifugeBusy() { return this.activeCommand === 'centrifuge' || this.queue.some(command => command.type === 'centrifuge'); }
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
        && !this.centrifuge.locked && Math.abs(this.centrifuge.rpm) <= CentrifugeSettings.STOPPED_RPM && !this.centrifugeBusy();
    }
    commandCentrifuge(action, options) {
      const instrument = this.centrifuge;
      if (!this.ready() || !instrument || !instrument.available || this.centrifugeBusy()) return Promise.resolve(false);
      if (action !== CentrifugeActions.STOP && (!this.manualReady() || this.centrifugeTransfer || this.centrifugePair)) return Promise.resolve(false);
      if ([CentrifugeActions.OPEN, CentrifugeActions.CLOSE].includes(action) && (instrument.locked || Math.abs(instrument.rpm) > CentrifugeSettings.STOPPED_RPM)) return Promise.resolve(false);
      if (action === CentrifugeActions.START && (!instrument.lidClosed || !instrument.balanced || instrument.running || instrument.locked)) return Promise.resolve(false);
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
        && this.liquid && this.liquid.tubes && this.liquid.tubes[key] && !this.secured(key));
      return keys.includes(this.selected) ? [this.selected, ...keys.filter(key => key !== this.selected)] : keys;
    }
    /** Plan a fresh test pair or the matching counterweight for one secured sample. */
    centrifugePairSamples() {
      if (!this.centrifuge) return [];
      const occupied = this.centrifuge.slots.filter(slot => slot.key);
      const keys = this.freeCentrifugeTubes();
      if (occupied.length === 1 && occupied[0].secured) {
        const source = this.liquid && this.liquid.tubes[occupied[0].key];
        const opposite = this.centrifuge.slots.find(slot => slot.id === occupied[0].opposite && !slot.key);
        if (!source || !opposite || !Number.isFinite(source.volumeMl)) return [];
        const key = keys.find(key => this.liquid.tubes[key].capacityMl >= source.volumeMl);
        return key ? [{key, slot: opposite.id, volumeMl: source.volumeMl}] : [];
      }
      if (occupied.length) return [];
      const slot = this.selectedCentrifugeSlot();
      const opposite = slot && this.centrifuge.slots.find(item => item.id === slot.opposite && !item.key);
      const eligible = keys.filter(key => this.liquid.tubes[key].capacityMl >= CentrifugeSettings.SAMPLE_VOLUME);
      if (!slot || !opposite || eligible.length < 2) return [];
      return [slot, opposite].map((item, index) => ({key: eligible[index], slot: item.id, volumeMl: CentrifugeSettings.SAMPLE_VOLUME}));
    }
    /** Explain the current loading state and the next usable instrument action. */
    centrifugePairGuidance() {
      const instrument = this.centrifuge;
      const occupied = instrument ? instrument.slots.filter(slot => slot.key) : [];
      const samples = this.centrifugePairSamples();
      const counterpart = occupied.length === 1 && occupied[0].secured;
      const label = counterpart ? 'Gegenprobe einsetzen' : 'Testpaar laden (2 × 5 ml)';
      if (this.centrifugePair) return {label: 'Testpaar wird geladen …', note: 'Probe ' + (this.centrifugePair.index + 1) + ' von ' + this.centrifugePair.samples.length + ' wird vorbereitet und eingesetzt. Bitte den Transport abwarten.'};
      if (!this.ready()) return {label, note: 'Zuerst „Labor starten / fortsetzen“ wählen.'};
      if (this.robotBusy()) return {label, note: 'Zuerst den PR2 pausieren oder seinen Auftrag abwarten.'};
      if (!instrument || !instrument.available) return {label, note: 'Die Zentrifuge ist momentan nicht verfügbar.'};
      if (this.centrifugeTransfer) return {label, note: 'Eine Probe wird gerade bewegt. Bitte den Transport abwarten.'};
      if (occupied.some(slot => !slot.secured)) return {label: 'Beladung prüfen', note: 'Mindestens eine Probe ist nicht gesichert. Sitz der Probe prüfen und den Einsetzvorgang abschließen.'};
      if (occupied.length >= 2) {
        if (!instrument.balanced) return {label: 'Beladung prüfen', note: 'Die eingesetzten Proben sind nicht ausgewogen. Gegenüberliegende Plätze gleich schwer beladen.'};
        const prepared = {label: 'Testpaar bereit', note: occupied.length + ' Proben sind bereits eingesetzt und ausgewogen. '};
        if (instrument.locked || instrument.running || Math.abs(instrument.rpm) > CentrifugeSettings.STOPPED_RPM) prepared.note += 'Rotor stoppen oder den laufenden Durchgang abwarten.';
        else if (instrument.lidOpen) prepared.note += 'Jetzt „Deckel schließen“ und anschließend „Zentrifugieren“ wählen.';
        else if (instrument.lidClosed) prepared.note += 'Jetzt „Zentrifugieren“ wählen.';
        else prepared.note += 'Warten, bis der Deckel seine Endposition erreicht.';
        return prepared;
      }
      if (instrument.locked || instrument.running || Math.abs(instrument.rpm) > CentrifugeSettings.STOPPED_RPM) return {label, note: 'Vor dem Beladen den Rotor stoppen und warten, bis der Deckel entriegelt ist.'};
      if (!instrument.lidOpen) return {label, note: 'Zum Beladen zuerst „Deckel öffnen“ wählen und warten, bis er vollständig offen ist.'};
      if (this.scaleTransfer) return {label, note: 'Zuerst den Transport zur Waage abwarten oder anhalten.'};
      if (this.centrifugeBusy() || this.liquidBusy()) return {label, note: 'Der vorherige Auftrag wird noch ausgeführt. Bitte kurz warten.'};
      if (!samples.length) return {label, note: 'Zum Beladen werden freie Gläser mit passender Kapazität und freie gegenüberliegende Rotorplätze benötigt.'};
      return {label, note: counterpart
        ? 'Die Gegenprobe erhält dieselbe Füllmenge wie die eingesetzte Probe (' + samples[0].volumeMl.toFixed(1) + ' ml). Die vorhandene Probe bleibt unverändert.'
        : 'Setzt zwei Gläser mit je 5 ml nacheinander in gegenüberliegende Plätze. Danach Deckel schließen und zentrifugieren; keine PR2-Demo nötig.'};
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
      this.centrifugeTransferError = this.error || 'Probe kann nicht eingesetzt werden. Freien Rotorplatz und offenen Deckel prüfen.';
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
      if (!this.centrifugeAccessible() || !slot || slot.key || !pose || !this.liquid || !this.liquid.tubes[key] || this.secured(key) || this.centrifugeTransfer) return Promise.resolve(false);
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
        this.centrifugeTransferError = transfer.released ? 'Probe im Rotor noch nicht sicher abgesetzt. Sitz prüfen, bevor sie gesichert wird.' : 'Weg zur Zentrifuge blockiert. Das Glas bleibt gehalten; manuell weiterbewegen oder loslassen.';
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
      this.centrifugeTransfer = null;
      this.centrifugePair = null;
      this.centrifugeTransferError = null;
      this.scaleTransfer = null;
      this.scaleTransferError = null;
      return this.targetCommand(key, position, orientation);
    }
    targetCommand(key, position, orientation) {
      const bounded = this.bounded(position);
      if (!this.manualReady() || !this.known(key) || this.secured(key) || !bounded) return Promise.resolve(false);
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
        this.scaleTransferError = 'Weg zur Waage blockiert. Das Glas bleibt gehalten; manuell weiterbewegen oder loslassen.';
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
    startProgram(type) {
      if (!this.config.robot || !this.ready() || this.robotBusy() || this.centrifugeTransfer || this.centrifugePair || this.centrifugeOwnsObjects()) return Promise.resolve(false);
      this.robotRequest = type;
      return this.clear(type);
    }
    pause() {
      if (!this.config.robot || !this.ready() || !this.robotBusy() || this.robotRequest === 'pause') return Promise.resolve(false);
      this.robotRequest = 'pause';
      return this.clear('pause');
    }
    clear(type) {
      this.centrifugeTransfer = null;
      this.centrifugePair = null;
      this.centrifugeTransferError = null;
      if (type === 'reset' || ProgramCommands.includes(type)) this.centrifugeRevision += 1;
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
          if (command.type === 'target') answer = await this.service.target(command.target);
          else if (command.type === 'centrifuge') answer = await this.service.centrifuge(command.payload);
          else if (command.type === 'fillLiquid') answer = await this.service.fillLiquid(command.key, command.volumeMl);
          else answer = await this.service[command.type]();
          if (command.type === 'centrifuge' && !this.disposed && answer.centrifuge) this.centrifuge = answer.centrifuge;
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
            const source = startingProgram ? (command.type === 'runMixing' ? RobotSources.MIXING : RobotSources.TRANSFER) : this.robot.source;
            this.robot = answer.robot || {...this.robot, state, source};
          }
          this.error = null;
        } catch (error) {
          this.error = error.message;
          if (this.centrifugePair || this.centrifugeTransfer) this.centrifugeTransferError = this.error;
          this.target = null;
          this.dragged = null;
          this.scaleTransfer = null;
          this.centrifugeTransfer = null;
          this.centrifugePair = null;
          this.queue = this.queue.filter(pending => pending.type !== 'target');
        }
        if (LiquidCommands.includes(command.type)) this.liquidRevision += 1;
        if (ScaleCommands.includes(command.type)) this.scaleRevision += 1;
        if (command.type === 'centrifuge' || command.type === 'reset' || ProgramCommands.includes(command.type)) this.centrifugeRevision += 1;
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
        }
        if (answer.frames && this.adapter.setJointFrames) this.adapter.setJointFrames(answer.frames);
        for (const [key, pose] of Object.entries(this.objects)) this.adapter.setObjectPose(key, pose);
        if (!liquidPending && !this.liquidBusy() && liquidRevision === this.liquidRevision) {
          this.liquid = answer.liquid || null;
          if (this.liquid && this.adapter.setLiquidState) this.adapter.setLiquidState(this.liquid);
        }
        if (this.adapter.showContacts) this.adapter.showContacts(this.contacts, this.target);
        if (!scalePending && !this.scaleBusy() && scaleRevision === this.scaleRevision) {
          this.scale = answer.scale || null;
          if (this.adapter.setScaleState) this.adapter.setScaleState(this.scale);
        }
        if (!centrifugePending && !this.centrifugeBusy() && centrifugeRevision === this.centrifugeRevision) {
          this.centrifuge = answer.centrifuge || null;
          if (this.centrifuge && this.centrifuge.joints && this.adapter.setJointFrames) this.adapter.setJointFrames(this.centrifuge.joints);
        }
        this.error = answer.error || this.scaleTransferError || this.centrifugeTransferError || null;
        await this.advanceScaleTransfer();
        await this.advanceCentrifugeTransfer();
        this.nextPoll = 0;
      } catch (error) {
        if (runRevision !== this.runRevision) return;
        this.state = 'failed';
        this.scale = null;
        this.centrifuge = null;
        this.centrifugeTransfer = null;
        this.centrifugePair = null;
        if (this.adapter.setScaleState) this.adapter.setScaleState(null);
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
    destroy() { this.disposed = true; this.queue = []; this.scaleTransfer = null; this.centrifugeTransfer = null; this.centrifugePair = null; }
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
    panel.setAttribute('aria-label', 'Labor mit Kontaktphysik');
    element('h3', panel, '', config.robot ? 'PR2 & Kontakt' : 'Glas & Kontakt');
    element('p', panel, 'laboratory-physics-intro', config.robot ? 'Der PR2 greift und bewegt das Glas mit Kontaktphysik.' : 'Gläser ziehen, anheben und loslassen. Hindernisse halten das Glas zurück.');
    const label = element('label', panel, 'laboratory-physics-field', 'Objekt');
    const select = element('select', label, 'laboratory-physics-object');
    for (const object of config.objects) {
      const option = element('option', select, '', object.label || object.key);
      option.value = object.key;
    }
    const controller = new Controller(config, service, adapter, render);
    const scale = element('section', panel, 'laboratory-physics-scale');
    element('h4', scale, '', 'Waage');
    const scaleReading = element('p', scale, 'laboratory-physics-scale-reading');
    const scaleStatus = element('p', scale, 'laboratory-physics-scale-status');
    element('p', scale, 'laboratory-physics-note', 'Flüssigkeiten werden mit Wasserdichte gewogen: 1 ml = 1 g.');
    const tare = button(scale, 'Tara / Nullstellen', () => controller.tareScale(), 'laboratory-physics-scale-tare');
    const moveToScale = button(scale, 'Objekt auf die Waage', () => controller.moveToScale(), 'laboratory-physics-scale-move');
    const cancelScale = button(scale, 'Transport anhalten', () => controller.hold(controller.selected), 'laboratory-physics-scale-cancel');
    const viewScale = adapter.focus && config.scale ? button(scale, 'Waage ansehen', () => {
      const position = config.scale.panPosition;
      adapter.focus([position[0] + .28, position[1] - .45, position[2] + .3], position);
    }, 'laboratory-physics-scale-view') : null;
    const centrifuge = element('section', panel, 'laboratory-physics-centrifuge');
    element('h4', centrifuge, '', 'Zentrifuge');
    const centrifugeStatus = element('p', centrifuge, 'laboratory-physics-centrifuge-status');
    centrifugeStatus.setAttribute('role', 'status');
    const slotLabel = element('label', centrifuge, 'laboratory-physics-field', 'Rotorplatz');
    const slotSelect = element('select', slotLabel, 'laboratory-physics-centrifuge-slot');
    const slotOptions = new Map();
    for (const id of ['C1', 'C2', 'C3', 'C4']) {
      const option = element('option', slotSelect, '', id);
      option.value = id;
      slotOptions.set(id, option);
    }
    listen(slotSelect, 'change', () => controller.selectCentrifugeSlot(slotSelect.value));
    const lidOpen = button(centrifuge, 'Deckel öffnen', () => controller.commandCentrifuge(CentrifugeActions.OPEN), 'laboratory-physics-centrifuge-open');
    const lidClose = button(centrifuge, 'Deckel schließen', () => controller.commandCentrifuge(CentrifugeActions.CLOSE), 'laboratory-physics-centrifuge-close');
    const loadPair = button(centrifuge, 'Testpaar laden (2 × 5 ml)', () => controller.loadCentrifugePair(), 'laboratory-physics-centrifuge-pair');
    const pairNote = element('p', centrifuge, 'laboratory-physics-centrifuge-pair-note');
    const loadSample = button(centrifuge, 'Probe einsetzen', () => controller.loadCentrifuge(), 'laboratory-physics-centrifuge-load');
    const unloadSample = button(centrifuge, 'Probe herausheben', () => controller.unloadCentrifuge(), 'laboratory-physics-centrifuge-unload');
    const centrifugeCancel = button(centrifuge, 'Transport anhalten', () => controller.hold(controller.selected), 'laboratory-physics-centrifuge-cancel');
    element('p', centrifuge, 'laboratory-physics-note', 'Gegenüberliegende Plätze gleich schwer beladen (C1/C3, C2/C4). Die Probe wird nach gemessenem Bodenkontakt im Rotor gesichert. Vor PR2-Aufträgen gesicherte Proben herausheben.');
    const rpmLabel = element('label', centrifuge, 'laboratory-physics-field', 'Drehzahl (U/min)');
    const rpm = element('input', rpmLabel, 'laboratory-physics-centrifuge-rpm');
    rpm.type = 'number'; rpm.min = 1; rpm.step = 1; rpm.value = String(CentrifugeSettings.RPM);
    const durationLabel = element('label', centrifuge, 'laboratory-physics-field', 'Dauer (Simulationssekunden)');
    const duration = element('input', durationLabel, 'laboratory-physics-centrifuge-duration');
    duration.type = 'number'; duration.min = 1; duration.max = CentrifugeSettings.MAXIMUM_DURATION; duration.step = 1; duration.value = String(CentrifugeSettings.DURATION);
    listen(rpm, 'input', render); listen(duration, 'input', render);
    const centrifugeStart = button(centrifuge, 'Zentrifugieren', () => controller.commandCentrifuge(CentrifugeActions.START, {rpm: Number(rpm.value), duration: Number(duration.value)}), 'laboratory-physics-centrifuge-start');
    const centrifugeStop = button(centrifuge, 'Rotor stoppen', () => controller.commandCentrifuge(CentrifugeActions.STOP), 'laboratory-physics-centrifuge-stop');
    const viewCentrifuge = adapter.focus ? button(centrifuge, 'Zentrifuge ansehen', () => {
      const slots = controller.centrifuge && controller.centrifuge.slots;
      if (!slots || !slots.length) return;
      const center = [0, 1, 2].map(axis => slots.reduce((sum, slot) => sum + slot.position[axis], 0) / slots.length);
      adapter.focus([center[0] + .32, center[1] - .52, center[2] + .4], center);
    }, 'laboratory-physics-centrifuge-view') : null;
    const liquid = element('section', panel, 'laboratory-physics-liquid');
    element('h4', liquid, '', 'Flüssigkeit');
    const liquidStatus = element('p', liquid, 'laboratory-physics-liquid-status');
    const volumeLabel = element('label', liquid, 'laboratory-physics-volume-label', 'Füllmenge (ml)');
    const volume = element('input', volumeLabel, 'laboratory-physics-volume');
    volume.type = 'number';
    volume.min = 0;
    volume.step = VOLUME_STEP;
    let volumeKey = null;
    listen(volume, 'input', render);
    const fill = button(liquid, 'Füllung setzen', () => controller.fillLiquid(Number(volume.value)), 'laboratory-physics-fill');
    const liftClear = button(liquid, 'Aus dem Ständer heben', () => controller.nudge([0, 0, LIFT_CLEARANCE]), 'laboratory-physics-lift-clear');
    liftClear.title = 'Ziel um ' + Math.round(LIFT_CLEARANCE * 100) + ' cm anheben';
    const tilts = element('div', liquid, 'laboratory-physics-tilts');
    const tiltLeft = button(tilts, '−' + TILT_STEP + '°', () => controller.tilt(-TILT_STEP), 'laboratory-physics-tilt-left');
    tiltLeft.setAttribute('aria-label', 'Nach links kippen · ' + TILT_STEP + ' Grad');
    const upright = button(tilts, 'Aufrecht', () => controller.upright(), 'laboratory-physics-upright');
    const tiltRight = button(tilts, '+' + TILT_STEP + '°', () => controller.tilt(TILT_STEP), 'laboratory-physics-tilt-right');
    tiltRight.setAttribute('aria-label', 'Nach rechts kippen · ' + TILT_STEP + ' Grad');
    const tiltStatus = element('p', liquid, 'laboratory-physics-tilt-status');
    const viewGlass = adapter.focus ? button(liquid, 'Glas ansehen', () => controller.viewGlass(), 'laboratory-physics-view-glass') : null;
    element('p', liquid, 'laboratory-physics-note', 'Erst anheben, dann kippen. Vereinfachtes Flüssigkeitsmodell.');
    const start = button(panel, 'Labor starten / fortsetzen', () => controller.start(), 'laboratory-physics-start');
    let robotRun, robotMix, robotPause, robotStatus, robotProgress, robotMeasurements, robotMotionNote;
    if (config.robot) {
      robotMix = button(panel, 'Misch-Demo starten', () => controller.runMixing(), 'laboratory-physics-robot-mix');
      element('p', panel, 'laboratory-physics-robot-recipe', 'Je 5 ml Bernstein und Türkis ins klare Glas gießen, sanft schwenken und in A2 ablegen. Ein neuer Start setzt diese Füllmengen und die Ausgangspositionen.');
      robotRun = button(panel, 'PR2: A1 → A3 starten', () => controller.run(), 'laboratory-physics-robot-run');
      element('p', panel, 'laboratory-physics-robot-reset-note', 'Neuer Durchlauf stellt PR2 und Gläser zurück; die Füllmengen bleiben erhalten. Pausierte Abläufe lassen sich fortsetzen.');
      robotPause = button(panel, 'PR2 pausieren', () => controller.pause(), 'laboratory-physics-robot-pause');
      robotStatus = element('p', panel, 'laboratory-physics-robot-status');
      robotStatus.setAttribute('role', 'status');
      robotProgress = element('progress', panel, 'laboratory-physics-robot-progress');
      robotProgress.max = 1;
      robotProgress.setAttribute('aria-label', 'Fortschritt der PR2-Bewegung');
      robotMeasurements = element('p', panel, 'laboratory-physics-robot-measurements');
      robotMotionNote = element('p', panel, 'laboratory-physics-robot-motion-note');
    }
    select.value = controller.selected;
    listen(select, 'change', () => { controller.scaleTransfer = null; controller.centrifugeTransfer = null; controller.centrifugePair = null; controller.selected = select.value; render(); });
    const hold = button(panel, 'Glas halten', () => controller.hold(select.value), 'laboratory-physics-hold');
    const moves = element('div', panel, 'laboratory-physics-moves');
    const motion = [
      ['←', [-MOVE_STEP, 0, 0], 'Nach links'], ['↑', [0, MOVE_STEP, 0], 'Nach hinten'],
      ['↓', [0, -MOVE_STEP, 0], 'Nach vorne'], ['→', [MOVE_STEP, 0, 0], 'Nach rechts'],
      ['Anheben', [0, 0, MOVE_STEP], 'Anheben'], ['Absenken', [0, 0, -MOVE_STEP], 'Absenken'],
    ].map(([text, delta, title]) => {
      const control = button(moves, text, () => controller.nudge(delta));
      control.setAttribute('aria-label', title + ' · 1 cm');
      control.title = title + ' · 1 cm';
      return control;
    });
    const release = button(panel, 'Glas loslassen', () => controller.release(), 'laboratory-physics-release');
    const status = element('p', panel, 'laboratory-physics-status');
    status.setAttribute('role', 'status');
    const force = element('p', panel, 'laboratory-physics-force');
    const distance = element('p', panel, 'laboratory-physics-distance');
    element('p', panel, 'laboratory-physics-note', 'Orange: gewünschte Position · Rot: Kontaktpunkte');
    const camera = element('div', panel, 'laboratory-physics-cameras');
    if (config.cameras && adapter.focus) {
      for (const [name, text] of [['closeup', 'Nahansicht'], ['overview', 'Übersicht']]) {
        if (config.cameras[name]) button(camera, text, () => adapter.focus(config.cameras[name].position, config.cameras[name].target));
      }
    }
    const reset = button(panel, 'Labor zurücksetzen', () => controller.reset(), 'laboratory-physics-reset');
    function render() {
      if (disposed) return;
      const ready = controller.ready();
      const manualReady = controller.manualReady();
      const reading = controller.scale;
      const instrument = controller.centrifuge;
      const slot = controller.selectedCentrifugeSlot();
      const secured = controller.secured(controller.selected);
      const pair = controller.centrifugePair;
      const instrumentBusy = controller.centrifugeBusy() || Boolean(controller.centrifugeTransfer) || Boolean(pair);
      const accessible = controller.centrifugeAccessible() && !instrumentBusy;
      centrifuge.hidden = !config.centrifuge && !instrument;
      if (viewCentrifuge) viewCentrifuge.disabled = !instrument || !instrument.available;
      centrifugeStatus.textContent = !instrument || !instrument.available ? 'Zentrifuge nicht verfügbar'
        : Math.round(instrument.rpm) + ' U/min · ' + Number(instrument.remainingSeconds || 0).toFixed(1) + ' s verbleibend · '
          + Number(instrument.rcf || 0).toFixed(1) + ' × g · ' + (instrument.balanced ? 'Ausgewogen' : 'Beladung ausgleichen')
          + ' · ' + (instrument.locked ? 'Verriegelt' : (instrument.lidOpen ? 'Deckel offen' : (instrument.lidClosed ? 'Deckel geschlossen' : 'Deckel bewegt sich')));
      if (pair) centrifugeStatus.textContent += ' · Probe ' + (pair.index + 1) + ' von ' + pair.samples.length + ' wird vorbereitet und eingesetzt …';
      else if (controller.centrifugeTransfer) centrifugeStatus.textContent += ' · Probe wird ' + (controller.centrifugeTransfer.type === CentrifugeTransfer.LOAD ? 'eingesetzt …' : 'herausgehoben …');
      const pairSamples = controller.centrifugePairSamples();
      const pairGuidance = controller.centrifugePairGuidance();
      loadPair.textContent = pairGuidance.label;
      loadPair.disabled = !accessible || !pairSamples.length || Boolean(controller.scaleTransfer) || controller.liquidBusy();
      pairNote.textContent = pairGuidance.note;
      slotSelect.value = controller.centrifugeSlot;
      slotSelect.disabled = !manualReady || instrumentBusy;
      if (instrument) for (const item of instrument.slots) {
        const option = slotOptions.get(item.id);
        if (option) option.textContent = item.id + ' ↔ ' + item.opposite + (item.key ? ' · ' + item.key + ' · ' + item.grams.toFixed(1) + ' g' : ' · leer');
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
      centrifugeStart.disabled = !manualReady || !instrument || instrumentBusy || !instrument.lidClosed || !instrument.balanced || instrument.locked || instrument.running || !validCycle;
      centrifugeStop.disabled = !ready || !instrument || controller.centrifugeBusy() || (!instrument.running && Math.abs(instrument.rpm) <= CentrifugeSettings.STOPPED_RPM);
      scale.hidden = !config.scale && !reading;
      scaleReading.textContent = reading && reading.available && Number.isFinite(reading.grams)
        ? (Math.abs(reading.grams) < .0005 ? 0 : reading.grams).toFixed(3) + ' g' : '— g';
      scaleStatus.textContent = !reading || !reading.available ? 'Waage nicht verfügbar'
        : (reading.stable ? 'Stabil' : 'Einpendeln …') + ' · Tara: ' + reading.tareGrams.toFixed(3) + ' g';
      tare.disabled = !manualReady || Boolean(pair) || !reading || !reading.available || !reading.stable || controller.scaleBusy();
      moveToScale.disabled = !manualReady || Boolean(pair) || !config.scale || !controller.objects[controller.selected] || Boolean(controller.scaleTransfer) || Boolean(controller.centrifugeTransfer) || secured;
      cancelScale.hidden = !controller.scaleTransfer;
      if (controller.scaleTransfer) scaleStatus.textContent = 'Objekt wird angehoben, zur Waage bewegt und abgesetzt …';
      start.hidden = ready;
      start.disabled = controller.starting;
      select.value = controller.selected;
      select.disabled = !manualReady || Boolean(pair);
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
        liquidStatus.textContent = tube.volumeMl.toFixed(1) + ' / ' + tube.capacityMl.toFixed(1) + ' ml · Ausgelaufen: ' + (Number(controller.liquid.spilledMl) || 0).toFixed(1) + ' ml';
        tiltStatus.textContent = 'Neigung: ' + Math.round(controller.observedTilt()) + '°';
      }
      status.textContent = controller.error || (controller.starting ? 'Labor wird gestartet …' : (ready ? (controller.robotBusy() ? 'Manuelles Bewegen ist während des PR2-Auftrags pausiert.' : (controller.target ? 'Glas gehalten · Pfeile bewegen das Ziel.' : 'Glas wählen oder direkt in der Szene ziehen.')) : 'Die Simulation ist nicht aktiv. Labor starten, um fortzufahren.'));
      status.dataset.error = String(Boolean(controller.error));
      if (config.robot) {
        const mixing = controller.robot.source === RobotSources.MIXING;
        const paused = controller.robot.state === RobotStates.PAUSED;
        robotMotionNote.textContent = mixing ? 'Bewegung: lokaler Kontaktregler. Kontakte: MuJoCo.' : 'Gelenkziele: gespeicherter CRAM-Plan. Bewegung und Kontakte: MuJoCo.';
        robotRun.disabled = !ready || Boolean(controller.robotBusy()) || Boolean(controller.centrifugeTransfer) || Boolean(pair) || controller.centrifugeOwnsObjects();
        robotRun.textContent = paused && !mixing ? 'PR2 fortsetzen' : 'PR2: A1 → A3 starten';
        robotMix.disabled = robotRun.disabled;
        robotMix.textContent = paused && mixing ? 'Misch-Demo fortsetzen' : 'Misch-Demo starten';
        robotPause.disabled = !ready || !controller.robotBusy() || controller.robotRequest === 'pause';
        const phase = controller.robot.message || RobotPhases[controller.robot.phase];
        const label = (mixing ? MixingLabels : RobotLabels)[controller.robot.state] || controller.robot.state;
        robotStatus.textContent = controller.robot.error || (label + (phase && controller.robot.state === RobotStates.RUNNING ? ' · ' + phase : ''));
        robotStatus.dataset.error = String(Boolean(controller.robot.error));
        robotProgress.value = Math.max(0, Math.min(1, Number(controller.robot.progress) || 0));
        const measurements = [];
        if (Number.isFinite(controller.robot.gripperForce)) measurements.push('Greifkontakt: ' + controller.robot.gripperForce.toFixed(2) + ' N');
        if (Number.isFinite(controller.robot.maximumLift)) measurements.push('Max. Hub: ' + (controller.robot.maximumLift * 100).toFixed(1) + ' cm');
        robotMeasurements.textContent = measurements.join(' · ');
        robotMeasurements.hidden = !measurements.length;
      }
      const feedback = controller.feedback();
      force.textContent = 'Kontakt: ' + feedback.contactCount + ' · ' + feedback.force.toFixed(2) + ' N';
      if (config.robot && controller.robot.holding) {
        distance.textContent = 'Vom PR2 gehalten · Kontaktphysik aktiv';
      } else if (config.robot && controller.robot.state === RobotStates.RUNNING) {
        distance.textContent = 'PR2-Auftrag aktiv · Kontaktphysik aktiv';
      } else {
        distance.textContent = controller.target ? 'Abstand zum Ziel: ' + (feedback.distance * 1000).toFixed(1) + ' mm' : 'Losgelassen · Schwerkraft aktiv';
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
    button.textContent = service.robot ? 'PR2 mit Kontaktphysik' : 'Labor mit Kontaktphysik öffnen';
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
      status.textContent = 'Labor wird vorbereitet …';
      try {
        const answer = await service.start();
        if (!answer.viewerUrl) throw new Error('Die Laboransicht ist noch nicht bereit.');
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
