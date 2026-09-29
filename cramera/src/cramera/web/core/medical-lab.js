/* Scene-scoped station navigation and observed instrument mechanism controls. */
(function (root, factory) {
  'use strict';
  const medicalLab = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = medicalLab;
  if (root) root.MedicalLab = medicalLab;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  // %% commands and measured-state presentation
  const Actions = Object.freeze({RUN: 'run', OPEN: 'open', CLOSE: 'close', PAUSE: 'pause', RESET: 'reset'});
  const AllDevices = 'all';
  const States = Object.freeze({IDLE: 'idle', RUNNING: 'running', PAUSED: 'paused', COMPLETE: 'complete', FAILED: 'failed'});
  const Labels = Object.freeze({idle: 'Instruments ready', running: 'Instrument demo running', paused: 'Instrument demo paused',
    complete: 'Instrument demo complete', failed: 'Instrument demo stopped'});
  const RobotDemos = Object.freeze([
    {id: 'workflow', label: 'Full sample workflow', description: 'Retrieve the rack, dispense, weigh, centrifuge and store four samples.'},
    {id: 'pipetting', label: 'Dose one sample', description: 'PR2 holds a sample under the dual dispenser and adds both reagents.'},
    {id: 'mixing', label: 'Pour and mix', description: 'Combine two liquids, swirl the mixture and return it to the rack.'},
    {id: 'transfer', label: 'Transfer A1 → A3', description: 'Move a glass between rack positions using contact grasping.'},
    {id: 'centrifuge', label: 'Open and close the centrifuge', description: 'PR2 grasps the lid handle and operates the hinge.'},
    {id: 'refrigerator', label: 'Open and close the cooler', description: 'PR2 grasps the door handle and operates the hinge.'},
  ]);

  /** Format only finite observations in the unit of the authored joint. */
  function jointReading(joint, position) {
    if (!Number.isFinite(position)) return 'Not reported';
    return joint.type === 'prismatic' ? (position * 1000).toFixed(1) + ' mm'
      : (position * 180 / Math.PI).toFixed(1) + '°';
  }

  // %% English controls for the larger room
  class View {
    constructor(host, config, commands) {
      this.document = host.ownerDocument;
      this.config = config;
      this.commands = commands;
      this.listeners = [];
      this.observation = null;
      this.controls = null;
      this.rows = [];
      this.panel = this.element('section', host, 'medical-lab');
      const navigation = this.element('section', this.panel, 'medical-lab-navigation');
      this.element('h4', navigation, '', 'Explore the lab');
      const stationLabel = this.element('label', navigation, 'medical-lab-field', 'Workstation');
      this.station = this.element('select', stationLabel, 'medical-lab-station');
      this.station.setAttribute('aria-label', 'Laboratory workstation');
      for (const station of config.stations || []) {
        this.element('option', this.station, '', station.label || station.id).value = station.id;
      }
      this.station.value = config.stations && config.stations[0] ? config.stations[0].id : '';
      this.focus = this.button(navigation, 'View station', () => this.focusStation(), 'medical-lab-focus');

      const robot = this.element('section', this.panel, 'medical-lab-robot');
      this.element('h4', robot, '', 'PR2 sample demos');
      const robotLabel = this.element('label', robot, 'medical-lab-field', 'Robot task');
      this.robot = this.element('select', robotLabel, 'medical-lab-robot-demo');
      this.robot.setAttribute('aria-label', 'PR2 sample demonstration');
      for (const demo of RobotDemos) this.element('option', this.robot, '', demo.label).value = demo.id;
      this.robot.value = RobotDemos[0].id;
      this.robotDescription = this.element('p', robot, 'medical-lab-note');
      this.listen(this.robot, 'change', () => this.render(this.observation, this.controls));
      const robotButtons = this.element('div', robot, 'medical-lab-buttons');
      this.runRobot = this.button(robotButtons, 'Run robot demo', () => commands.robot(this.robot.value), 'medical-lab-robot-run');
      this.pauseRobot = this.button(robotButtons, 'Pause PR2', () => commands.pauseRobot(), 'medical-lab-robot-pause');
      this.robotStatus = this.element('p', robot, 'medical-lab-robot-status');
      this.robotStatus.setAttribute('role', 'status');

      const mechanisms = this.element('section', this.panel, 'medical-lab-mechanisms');
      this.element('h4', mechanisms, '', 'Instrument mechanisms');
      const deviceLabel = this.element('label', mechanisms, 'medical-lab-field', 'Instrument');
      this.device = this.element('select', deviceLabel, 'medical-lab-device');
      this.device.setAttribute('aria-label', 'Instrument mechanism demonstration');
      this.element('option', this.device, '', 'All motorized instruments').value = AllDevices;
      for (const instrument of config.instruments || []) {
        this.element('option', this.device, '', instrument.label || instrument.id).value = instrument.id;
      }
      this.device.value = AllDevices;
      this.listen(this.device, 'change', () => {
        const selected = this.selectedDevice();
        if (selected && selected.station) this.station.value = selected.station;
        this.render(this.observation, this.controls);
      });
      this.description = this.element('p', mechanisms, 'medical-lab-description');
      this.run = this.button(mechanisms, 'Run mechanism demo', () => this.command(Actions.RUN), 'medical-lab-run');
      const positions = this.element('div', mechanisms, 'medical-lab-buttons');
      this.inspect = this.button(positions, 'Inspect position', () => this.command(Actions.OPEN), 'medical-lab-inspect');
      this.rest = this.button(positions, 'Rest position', () => this.command(Actions.CLOSE), 'medical-lab-rest');
      const playback = this.element('div', mechanisms, 'medical-lab-buttons');
      this.pause = this.button(playback, 'Pause', () => this.command(Actions.PAUSE), 'medical-lab-pause');
      this.reset = this.button(playback, 'Reset instruments', () => this.command(Actions.RESET), 'medical-lab-reset');
      this.status = this.element('p', mechanisms, 'medical-lab-status');
      this.status.setAttribute('role', 'status');
      this.progress = this.element('progress', mechanisms, 'medical-lab-progress');
      this.progress.max = 1;
      this.progress.setAttribute('aria-label', 'Observed instrument demonstration progress');
      this.step = this.element('p', mechanisms, 'medical-lab-step');
      const observed = this.element('details', mechanisms, 'medical-lab-observations');
      this.element('summary', observed, '', 'Observed joint positions');
      const positionsList = this.element('dl', observed, 'medical-lab-joints');
      for (const instrument of config.instruments || []) {
        for (const joint of instrument.joints || []) {
          const item = this.element('div', positionsList, 'medical-lab-joint');
          this.element('dt', item, '', (instrument.label || instrument.id) + ' · ' + joint.name);
          const value = this.element('dd', item, 'medical-lab-joint-value');
          this.rows.push({instrument: instrument.id, joint, item, value});
        }
      }
      this.element('p', mechanisms, 'medical-lab-note', 'These demos move instrument joints through their inspection and rest positions. They do not perform sample processing or robot manipulation.');
      this.render(null, null);
    }
    element(tag, parent, className, text) {
      const element = this.document.createElement(tag);
      element.className = className || '';
      if (text) element.textContent = text;
      parent.appendChild(element);
      return element;
    }
    listen(element, event, callback) {
      element.addEventListener(event, callback);
      this.listeners.push(() => element.removeEventListener(event, callback));
    }
    button(parent, label, callback, className) {
      const button = this.element('button', parent, className, label);
      button.type = 'button';
      this.listen(button, 'click', callback);
      return button;
    }
    selectedDevice() {
      return (this.config.instruments || []).find(device => device.id === this.device.value);
    }
    command(action) {
      const command = [Actions.PAUSE, Actions.RESET].includes(action) ? {action} : {action, device: this.device.value};
      return this.commands.devices(command);
    }
    focusStation() {
      const station = (this.config.stations || []).find(item => item.id === this.station.value);
      if (station && station.camera && this.commands.focus) this.commands.focus(station.camera.position, station.camera.target);
    }
    render(observation, controls) {
      this.observation = observation;
      this.controls = controls;
      const selected = this.selectedDevice();
      const joints = selected ? selected.joints || [] : (this.config.instruments || []).flatMap(device => device.joints || []);
      const all = this.device.value === AllDevices;
      const demo = observation && observation.demo;
      const busy = Boolean(controls && controls.pending);
      const allowed = action => Boolean(joints.length && controls && !busy && controls.canCommand(action, this.device.value));
      this.run.disabled = !allowed(Actions.RUN);
      this.inspect.disabled = all || !allowed(Actions.OPEN);
      this.rest.disabled = all || !allowed(Actions.CLOSE);
      this.pause.disabled = !controls || busy || !controls.canCommand(Actions.PAUSE);
      this.reset.disabled = !controls || busy || !controls.canCommand(Actions.RESET);
      this.run.textContent = demo && demo.state === States.PAUSED && demo.device === this.device.value ? 'Resume mechanism demo' : 'Run mechanism demo';
      this.description.textContent = !joints.length ? 'This accessory has no driven joints. Place it independently in the scene.'
        : all ? 'Cycle the motorized instruments one after another.' : (selected.label || selected.id) + ' · ' + joints.length + ' driven joints';
      const error = controls && controls.error || demo && demo.error;
      this.status.textContent = error || (busy ? 'Sending instrument command…'
        : !controls || !controls.ready ? 'Start the laboratory to operate the instruments.'
        : !demo ? 'Awaiting instrument state…' : Labels[demo.state] || 'Instrument state unavailable');
      this.status.dataset.error = String(Boolean(error));
      this.progress.value = demo && Number.isFinite(demo.completed) && Number.isFinite(demo.total) && demo.total > 0
        ? Math.max(0, Math.min(1, demo.completed / demo.total)) : 0;
      this.step.textContent = demo ? [demo.action, Number.isFinite(demo.completed) && Number.isFinite(demo.total) && demo.total > 0
        ? demo.completed + ' / ' + demo.total + ' steps' : ''].filter(Boolean).join(' · ') : '';
      for (const row of this.rows) {
        row.item.hidden = !all && row.instrument !== this.device.value;
        row.value.textContent = jointReading(row.joint, observation && observation.joints && observation.joints[row.joint.name]);
      }
      this.focus.disabled = !(this.config.stations || []).length || !this.commands.focus;
      const robotDemo = RobotDemos.find(item => item.id === this.robot.value);
      this.robotDescription.textContent = robotDemo ? robotDemo.description : '';
      this.runRobot.disabled = !controls || !controls.robotReady || !controls.robotReady[this.robot.value];
      this.pauseRobot.disabled = !controls || !controls.robotCanPause;
      const robot = controls && controls.robot;
      this.runRobot.textContent = robot && robot.state === 'paused' && !this.runRobot.disabled ? 'Resume robot demo' : 'Run robot demo';
      this.robotStatus.textContent = controls && controls.robotError || robot && robot.error
        || (robot && robot.state === 'running' ? robot.message || 'PR2 demo running'
          : robot && robot.state === 'paused' ? 'PR2 paused · resume its current task below.'
          : robot && robot.state === 'succeeded' ? 'PR2 task complete' : 'Select a PR2 sample demo.');
    }
    destroy() {
      this.listeners.forEach(remove => remove());
      this.rows.length = 0;
      this.panel.remove();
    }
  }

  return {Actions, AllDevices, States, Labels, RobotDemos, jointReading, View};
});
