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

  return {centrifuge};
});
