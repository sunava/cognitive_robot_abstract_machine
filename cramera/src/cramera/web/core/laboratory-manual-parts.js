/* Authored instrument parts and explanations grounded in current observations. */
(function (global, factory) {
  'use strict';
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.LaboratoryManualParts = api;
})(typeof window === 'undefined' ? globalThis : window, function () {
  'use strict';

  // %% measured operating explanations
  const SECONDS_PER_MINUTE = 60;

  /** Read one centrifuge observation without inferring missing measurements. */
  class CentrifugeObservation {
    constructor(snapshot, metadata) {
      this.snapshot = snapshot || {};
      this.metadata = metadata || {};
      this.available = this.snapshot.available === true;
      this.knowledge = this.snapshot.knowledge || {};
    }

    /** Describe the independently observed lid position and lock. */
    lid() {
      const state = this.snapshot;
      const position = !this.available ? 'Lid position unknown'
        : state.lidOpen === true ? 'Open'
        : state.lidClosed === true ? 'Closed'
        : state.lidOpen === false && state.lidClosed === false ? 'Partly open'
        : 'Lid position unknown';
      const lock = !this.available ? 'lock state unknown'
        : state.locked === true ? 'locked while the rotor is active'
        : state.locked === false ? 'unlocked' : 'lock state unknown';
      return position + ' · ' + lock + '.';
    }

    /** Report whether the simulated drive is running or braking. */
    rotor() {
      const state = this.snapshot;
      if (!this.available || typeof state.running !== 'boolean') return 'Rotor motion unknown.';
      if (!state.running) return 'Rotor stopped.';
      if (!Number.isFinite(state.targetRpm)) return 'Rotor active; drive target unknown.';
      return state.targetRpm > 0 ? 'Rotor running.' : 'Braking; wait for the rotor to stop.';
    }

    /** Combine observed secured holders with the actual document loading rule. */
    holders() {
      const slots = this.snapshot.slots;
      const expected = this.metadata.slots;
      const complete = this.available && Array.isArray(slots) && slots.length > 0
        && (!Array.isArray(expected) || slots.length === expected.length)
        && slots.every(slot => slot && typeof slot.secured === 'boolean');
      const count = complete ? slots.filter(slot => slot.secured).length + ' secured' : 'Secured count unknown';
      const balance = !this.available ? 'balance unknown'
        : this.snapshot.balanced === true ? 'balanced'
        : this.snapshot.balanced === false ? 'not balanced' : 'balance unknown';
      const manual = this.knowledge.manual || {};
      const rule = Number.isInteger(manual.minimumSamples) && manual.minimumSamples > 0
        ? 'Manual' + (typeof manual.version === 'string' && manual.version ? ' ' + manual.version : '')
          + ': at least ' + manual.minimumSamples + ' secured samples.'
        : 'Manual loading rule unknown.';
      return count + ' · ' + balance + '. ' + rule;
    }

    /** Show simulation time and measured speed, never a browser countdown. */
    display() {
      const state = this.snapshot;
      if (!this.available || !Number.isFinite(state.rpm) || !Number.isFinite(state.remainingSeconds)) {
        return 'Speed and remaining time unknown.';
      }
      const seconds = Math.max(0, Math.ceil(state.remainingSeconds));
      const clock = String(Math.floor(seconds / SECONDS_PER_MINUTE)).padStart(2, '0')
        + ':' + String(seconds % SECONDS_PER_MINUTE).padStart(2, '0');
      return Math.round(Math.abs(state.rpm)) + ' RPM · ' + clock + ' remaining';
    }

    /** Explain the backend's start assessment without bypassing operator checks. */
    start() {
      const evaluation = this.knowledge.evaluation;
      if (!this.available || !evaluation || typeof evaluation.allowed !== 'boolean') return 'Start readiness unknown.';
      if (evaluation.allowed) return 'Start checks passed.';
      const requirements = Array.isArray(evaluation.requirements) ? evaluation.requirements : [];
      const unmet = requirements.filter(requirement => requirement && requirement.satisfied !== true && typeof requirement.label === 'string');
      return unmet.length ? 'Start checks pending: ' + unmet.map(requirement => requirement.label).join('; ') + '.' : 'Start checks not met.';
    }

    /** Identify the configured simulation speed limit without implying a real instrument specification. */
    housing() {
      const maximum = this.available && Number.isFinite(this.snapshot.maximumRpm)
        ? this.snapshot.maximumRpm : this.metadata.maximumRpm;
      return Number.isFinite(maximum) ? 'Simulation speed limit: ' + maximum + ' RPM.' : 'Simulation speed limit unknown.';
    }
  }

  // %% authored part anchors
  /** Return numbered callouts tied to actual Blender geometry and instrument state. */
  function centrifuge(metadata) {
    const describe = method => snapshot => new CentrifugeObservation(snapshot, metadata)[method]();
    const interiorVisible = snapshot => Boolean(snapshot && snapshot.available === true && snapshot.lidOpen === true);
    return [
      {
        id: 'lid', number: 1, title: 'Protective lid', side: 'left',
        description: 'Closes the chamber. Opening is locked while the rotor is active.',
        meshNames: ['Centrifuge lid grip'], describe: describe('lid'),
      },
      {
        id: 'hinge', number: 2, title: 'Lid hinge', side: 'left',
        description: 'Rear pivot of the articulated cover. Use Open lid or Close lid in the Centrifuge panel.',
        meshNames: ['Centrifuge lid hinge'],
      },
      {
        id: 'rotor', number: 3, title: 'Rotor', side: 'left',
        description: 'Rotates the sample holders around the central vertical drive axis.',
        meshNames: ['Centrifuge central hub'], describe: describe('rotor'), visible: interiorVisible,
      },
      {
        id: 'holders', number: 4, title: 'Sample holders', side: 'left',
        description: 'Load and secure samples in opposite holders. Their masses must be balanced.',
        meshNames: ['Centrifuge C1 hollow sleeve'], describe: describe('holders'), visible: interiorVisible,
      },
      {
        id: 'display', number: 5, title: 'Cycle display', side: 'right',
        description: 'Shows measured rotor speed and the remaining simulated cycle time.',
        meshNames: ['Centrifuge display'], describe: describe('display'),
      },
      {
        id: 'start', number: 6, title: 'Start control', side: 'right',
        description: 'Use Start in the Centrifuge panel. The active manual and physical interlocks govern starting.',
        meshNames: ['Centrifuge run key'], describe: describe('start'),
      },
      {
        id: 'housing', number: 7, title: 'Instrument housing', side: 'right',
        description: 'Stationary chamber and bench support. Open the lid to inspect the rotor and holders.',
        meshNames: ['Centrifuge enamel chassis'], describe: describe('housing'),
      },
    ];
  }

  // %% dispensing observations and parts
  const DOSE_STATE = Object.freeze({idle: 'Ready', dispensing: 'Dispensing', complete: 'Dose complete', blocked: 'Stopped'});

  class DispenserObservation {
    constructor(snapshot) {
      this.snapshot = snapshot || {};
      this.available = this.snapshot.available === true;
    }

    /** Preserve the measured inventory of each independent reagent channel. */
    reservoir(channel) {
      const reservoirs = this.snapshot.reservoirs;
      const reservoir = this.available && Array.isArray(reservoirs) && reservoirs.find(item => item.id === channel);
      if (!reservoir || !Number.isFinite(reservoir.volumeMl) || !Number.isFinite(reservoir.capacityMl)) return 'Reservoir volume unknown.';
      return reservoir.volumeMl.toFixed(2) + ' / ' + reservoir.capacityMl.toFixed(2) + ' ml remaining.';
    }

    /** Report server-confirmed dose progress rather than extrapolating flow. */
    display() {
      const state = this.snapshot;
      if (!this.available || !Number.isFinite(state.deliveredMl) || !Number.isFinite(state.requestedMl)) return 'Dose progress unknown.';
      const status = DOSE_STATE[state.state] || 'State unknown';
      return status + ' · ' + state.deliveredMl.toFixed(2) + ' / ' + state.requestedMl.toFixed(2) + ' ml';
    }
  }

  /** Describe separate reagent feeds and the front-open vessel receiving area. */
  function dispenser(metadata) {
    return [
      {id: 'amber', number: 1, title: 'Amber reservoir', side: 'left', meshNames: ['Pipetting amber bottle'],
        description: 'Supplies the amber reagent through its own feed line. Refill selected reagent in the Pipetting station panel.',
        describe: snapshot => new DispenserObservation(snapshot).reservoir('amber')},
      {id: 'teal', number: 2, title: 'Teal reservoir', side: 'right', meshNames: ['Pipetting teal bottle'],
        description: 'Supplies the teal reagent through a separate feed line. The two reservoirs retain independent inventories.',
        describe: snapshot => new DispenserObservation(snapshot).reservoir('teal')},
      {id: 'nozzles', number: 3, title: 'Dosing outlets', side: 'right', meshNames: ['Pipetting amber nozzle'],
        description: 'Two adjacent outlets dispense into an open, upright glass. Select the reagent and dose in the Pipetting station panel.'},
      {id: 'receiving', number: 4, title: 'Sample position', side: 'right', meshNames: ['Pipetting receiving inset'],
        description: 'The front-open cradle supports the receiving glass. Use Position selected glass before dispensing.'},
      {id: 'display', number: 5, title: 'Dose display', side: 'left', meshNames: ['Pipetting display'],
        description: 'Shows the confirmed amount delivered and the requested dose in millilitres.',
        describe: snapshot => new DispenserObservation(snapshot).display()},
      {id: 'head', number: 6, title: 'Dispensing head', side: 'left', meshNames: ['Pipetting head cover'],
        description: 'Supports the reagent tubing and both dosing outlets above the receiving glass.'},
    ];
  }

  // %% balance observations and parts
  class ScaleObservation {
    constructor(snapshot) {
      this.snapshot = snapshot || {};
      this.available = this.snapshot.available === true;
    }

    /** Retain the distinction between measured net mass and settling state. */
    display() {
      const state = this.snapshot;
      if (!this.available || !Number.isFinite(state.grams)) return 'Mass reading unknown.';
      const stability = state.stable === true ? 'stable' : state.stable === false ? 'settling' : 'stability unknown';
      return state.grams.toFixed(3) + ' g · ' + stability;
    }

    /** Expose only the offset explicitly confirmed by the scale. */
    tare() {
      return this.available && Number.isFinite(this.snapshot.tareGrams)
        ? 'Stored tare: ' + this.snapshot.tareGrams.toFixed(3) + ' g.' : 'Stored tare unknown.';
    }
  }

  /** Describe the contact-measuring pan, live display and tare operation. */
  function scale(metadata) {
    return [
      {id: 'pan', number: 1, title: 'Weighing pan', side: 'left', meshNames: ['Balance weighing pan'],
        description: 'Measures the load transmitted through physical contacts. Release the glass onto the pan and wait for the reading to settle.'},
      {id: 'display', number: 2, title: 'Mass display', side: 'right', meshNames: ['Balance display'],
        description: 'Shows the measured net load in grams after subtracting the stored tare.',
        describe: snapshot => new ScaleObservation(snapshot).display()},
      {id: 'housing', number: 3, title: 'Balance housing', side: 'left', meshNames: ['Instrument base'],
        description: 'Supports the weighing assembly. Use Tare / zero in the Scale panel after the measurement has settled.',
        describe: snapshot => new ScaleObservation(snapshot).tare()},
    ];
  }

  // %% reusable rack parts
  /** Describe socket geometry without inferring occupation from an authored location. */
  function rack(metadata) {
    const slots = metadata && metadata.slots;
    const identifiers = Array.isArray(slots) && slots.length ? ' Positions: ' + slots.map(slot => slot.id).join(', ') + '.' : ' Identify each position using the labels on the upper plate.';
    return [
      {id: 'sockets', number: 1, title: 'Tube sockets', side: 'right', meshNames: ['Rack perforated plate'],
        description: 'Through-holes guide upright sample glasses.' + identifiers},
      {id: 'base', number: 2, title: 'Support plate', side: 'left', meshNames: ['Rack base'],
        description: 'Supports the bottoms of inserted glasses. Lift a glass above the upper plate before moving it sideways.'},
      {id: 'standoffs', number: 3, title: 'Rack columns', side: 'left', meshNames: ['Rack standoff'],
        description: 'Connect the lower support and perforated upper plate, keeping the socket height fixed.'},
    ];
  }

  // %% simulated cold storage
  /** Explain articulated access, physical support and the reduced cabinet thermal model. */
  function refrigerator(metadata) {
    return [
      {id:'door',number:1,title:'Glass door',side:'left',meshNames:['Refrigerator door glass'],
        description:'Open the door in the Sample refrigerator panel before moving a glass. The hinged door and cabinet resist contact.',
        describe:snapshot=>!snapshot || !snapshot.available ? 'Door state unknown.' : snapshot.doorOpen ? 'Open · sample access available.' : snapshot.doorClosed ? 'Closed.' : 'Door moving.'},
      {id:'handle',number:2,title:'Door handle',side:'right',meshNames:['Refrigerator handle'],
        description:'Use Open door and Close door in the controls. Sample transfers require the door to remain fully open.'},
      {id:'holders',number:3,title:'Sample holders',side:'left',meshNames:['Refrigerator R1 sample collar'],
        description:'Six open collars support existing glasses on two shelves. Select R1–R6 and Store selected glass. Retrieve sample returns it to a free rack position.',
        visible:snapshot=>Boolean(snapshot && snapshot.doorOpen),
        describe:snapshot=>!snapshot || !snapshot.available ? 'Occupancy unknown.' : snapshot.slots.filter(slot=>slot.key).length+' / '+snapshot.slots.length+' positions occupied.'},
      {id:'shelf',number:4,title:'Storage shelf',side:'right',meshNames:['Refrigerator shelf 1'],
        description:'The shelf supports the glass through measured contacts. Manual transfers enter through the front and release only after the sample settles.',
        visible:snapshot=>Boolean(snapshot && snapshot.doorOpen)},
      {id:'display',number:5,title:'Cabinet temperature',side:'right',meshNames:['Refrigerator display'],
        description:'Shows the simulated cabinet temperature. Opening the door warms the cabinet; closing it cools toward the set point. Sample temperatures are not individually simulated.',
        describe:snapshot=>!snapshot || !snapshot.available || !Number.isFinite(snapshot.temperatureC) ? 'Temperature unknown.' : snapshot.temperatureC.toFixed(1)+' °C · simulated · target '+snapshot.targetTemperatureC.toFixed(1)+' °C'},
    ];
  }

  return {centrifuge, dispenser, scale, rack, refrigerator};
});
