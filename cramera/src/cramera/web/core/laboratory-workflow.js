/* Authoritative sample observations for the complete laboratory demonstration. */
(function (root, factory) {
  'use strict';
  const workflow = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = workflow;
  if (root) root.LaboratoryWorkflow = workflow;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  // %% observed workflow vocabulary
  const Source = 'laboratory_workflow';
  const States = Object.freeze({IDLE: 'idle', RUNNING: 'running', PAUSED: 'paused', SUCCEEDED: 'succeeded', FAILED: 'failed'});
  const Labels = Object.freeze({idle: 'Ready for the full laboratory demo', running: 'Laboratory workflow running',
    paused: 'Laboratory workflow paused', succeeded: 'Laboratory workflow complete', failed: 'Laboratory workflow stopped'});
  const Stages = Object.freeze({prepare: 'Bringing the sample rack from the drawer to the bench', fill: 'Filling at the dual dispenser', weigh: 'Measuring sample mass',
    load_centrifuge: 'Loading the centrifuge', spin: 'Running the balanced centrifuge', store: 'Storing samples in the cooler', complete: 'Complete'});
  const Locations = Object.freeze({rack: 'In the sample rack', robot: 'PR2 transfer in progress', dispenser: 'At the dual dispenser',
    scale: 'On the scale', regrasp: 'At the grasp-change position', centrifuge: 'In the centrifuge', refrigerator: 'In the sample cooler'});
  const Buttons = Object.freeze({START: 'Prepare & run full demo', RESUME: 'Resume full demo', PAUSE: 'Pause full demo', RESET: 'Reset laboratory'});
  const PausedNote = 'Samples are reserved for the paused workflow. Resume the demo or reset the laboratory.';
  const Recipe = 'Resets the demo with four empty samples in the drawer. PR2 opens the drawer, carries the loaded rack onto the bench, then closes the drawer. For each sample: dispense 1 ml of reagent A and 1 ml of reagent B, weigh it, then load the centrifuge. The robot uses A1 to change between front and top grasps. Once all four are loaded, run the balanced cycle and store them in the cooler.';
  const ObservationNote = 'Measured in simulation. Mass is recorded after the scale settles. PR2 opens and closes the drawer, lid and cooler door through physical gripper contacts with their handles.';

  /** Whether the server reports that this workflow still owns its samples. */
  function active(robot) {
    return Boolean(robot && robot.source === Source && [States.RUNNING, States.PAUSED].includes(robot.state));
  }

  /** Return a standalone handle request, excluding ordinary hinge observations. */
  function instrument(robot) {
    const command = robot && robot.instrument;
    return command && typeof command === 'object' && typeof command.instrument === 'string' && typeof command.action === 'string' ? command : null;
  }

  /** Format a measurement without treating absent values as measured zeros. */
  function measurement(value, digits) {
    return Number.isFinite(value) ? value.toFixed(digits) : '—';
  }

  // %% measured sample table and execution controls
  /** Render only reported stages and measurements, with explicit execution controls. */
  class View {
    constructor(host, commands) {
      this.document = host.ownerDocument;
      this.listeners = [];
      this.rows = new Map();
      this.panel = this.element('section', host, 'laboratory-workflow');
      this.element('h4', this.panel, '', 'Full laboratory demo');
      this.element('p', this.panel, 'laboratory-workflow-recipe', Recipe);
      this.start = this.button(Buttons.START, commands.start, 'laboratory-workflow-start');
      this.pause = this.button(Buttons.PAUSE, commands.pause, 'laboratory-workflow-pause');
      this.reset = this.button(Buttons.RESET, commands.reset, 'laboratory-workflow-reset');
      this.status = this.element('p', this.panel, 'laboratory-workflow-status');
      this.status.setAttribute('role', 'status');
      this.stage = this.element('p', this.panel, 'laboratory-workflow-stage');
      this.progress = this.element('progress', this.panel, 'laboratory-workflow-progress');
      this.progress.max = 1;
      this.progress.setAttribute('aria-label', 'Laboratory workflow progress');
      this.samples = this.element('ul', this.panel, 'laboratory-workflow-samples');
      this.samples.setAttribute('aria-label', 'Observed sample measurements');
      this.element('p', this.panel, 'laboratory-workflow-note', ObservationNote);
    }
    element(tag, parent, className, text) {
      const item = this.document.createElement(tag);
      item.className = className;
      if (text) item.textContent = text;
      parent.appendChild(item);
      return item;
    }
    button(text, command, className) {
      const control = this.element('button', this.panel, className, text);
      control.type = 'button';
      control.addEventListener('click', command);
      this.listeners.push(() => control.removeEventListener('click', command));
      return control;
    }
    render(robot, controls) {
      const matching = robot && robot.source === Source && !instrument(robot);
      const observed = matching && robot.workflow;
      const state = matching ? robot.state : States.IDLE;
      const error = matching && (robot.error || observed && observed.failure);
      this.panel.dataset.state = state;
      this.start.textContent = state === States.PAUSED ? Buttons.RESUME : Buttons.START;
      this.start.disabled = !controls.ready || !controls.canStart || controls.pending;
      this.pause.disabled = !controls.ready || !controls.canPause || controls.pending;
      this.reset.disabled = !controls.ready || controls.pending;
      this.status.textContent = error || controls.error || Labels[state] || state;
      this.status.dataset.error = String(Boolean(error || controls.error));
      this.stage.textContent = observed ? observed.stageLabel || Stages[observed.stage] || observed.stage || '' : '';
      this.progress.value = matching && Number.isFinite(robot.progress) ? Math.max(0, Math.min(1, robot.progress)) : 0;
      const samples = observed && Array.isArray(observed.samples) ? observed.samples : [];
      const keys = new Set(samples.map(sample => sample.key));
      for (const [key, row] of this.rows) {
        if (!keys.has(key)) {row.item.remove();this.rows.delete(key);}
      }
      for (const sample of samples) this.renderSample(sample);
    }
    renderSample(sample) {
      let row = this.rows.get(sample.key);
      if (!row) {
        const item = this.element('li', this.samples, 'laboratory-workflow-sample');
        row = {item, label: this.element('strong', item, ''), dose: this.element('span', item, ''),
          mass: this.element('span', item, ''), location: this.element('span', item, '')};
        this.rows.set(sample.key, row);
      }
      row.label.textContent = sample.label || sample.key;
      row.dose.textContent = 'A: ' + measurement(sample.doseAMl, 2) + ' ml · B: ' + measurement(sample.doseBMl, 2) + ' ml';
      row.mass.textContent = 'Mass: ' + measurement(sample.weightGrams, 3) + ' g';
      const slot = sample.location === 'refrigerator' ? sample.refrigeratorSlot : sample.location === 'centrifuge' ? sample.centrifugeSlot : sample.location === 'regrasp' ? sample.rackSlot : null;
      row.location.textContent = (Locations[sample.location] || sample.location || 'Location not reported') + (slot ? ' · ' + slot : '');
    }
    destroy() {
      this.listeners.forEach(remove => remove());
      this.rows.clear();
      this.panel.remove();
    }
  }

  return {Source, States, Labels, Stages, Locations, Buttons, Recipe, ObservationNote, PausedNote, active, instrument, View};
});
