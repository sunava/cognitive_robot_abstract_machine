/* Independent articulated drawers and loose parts in the AICOR wood workshop. */
(function (root, factory) {
  'use strict';
  const workshop = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = workshop;
  if (root) root.Workshop = workshop;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  // %% manual manipulation vocabulary
  const Motion = Object.freeze({DRAWER_DURATION: 450, PART_DURATION: 420, STEP: .05});
  const Codes = Object.freeze({
    DONE: 'done', NOT_READY: 'not_ready', INVALID: 'invalid', BUSY: 'busy',
    HANDS_FULL: 'hands_full', NOT_HELD: 'not_held', OCCUPIED: 'occupied',
    WRONG_PART: 'wrong_part', OUTSIDE: 'outside', CANCELLED: 'cancelled',
  });
  const Messages = Object.freeze({
    [Codes.DONE]: 'Ready.',
    [Codes.NOT_READY]: 'Manual controls become available when the workshop finishes loading and live control is off.',
    [Codes.INVALID]: 'This part or drawer is unavailable.',
    [Codes.BUSY]: 'Wait for the current movement to finish.',
    [Codes.HANDS_FULL]: 'Return or insert the lifted part first.',
    [Codes.NOT_HELD]: 'Lift a part first.',
    [Codes.OCCUPIED]: 'The pilot hole already contains a screw. Lift it out first.',
    [Codes.WRONG_PART]: 'Choose a screw for the pilot hole.',
    [Codes.OUTSIDE]: 'That move would leave the workbench area.',
    [Codes.CANCELLED]: 'Movement stopped.',
  });

  /** Describe a completed command without coupling its state to the controls. */
  class Outcome {
    constructor(code, message) {
      this.code = code;
      this.ok = code === Codes.DONE;
      this.message = message || Messages[code];
    }
  }

  /** Supply a browser animation clock which can be replaced for deterministic checks. */
  class BrowserClock {
    now() { return performance.now(); }
    request(callback) { return requestAnimationFrame(callback); }
    cancel(identifier) { cancelAnimationFrame(identifier); }
  }

  /** Advance one bounded motion and resolve cancellation without another scene write. */
  class Animation {
    constructor(clock) { this.clock = clock; this.pending = null; this.finish = null; }

    run(duration, update) {
      const started = this.clock.now();
      return new Promise(resolve => {
        this.finish = resolve;
        const tick = time => {
          const progress = Math.min(1, Math.max(0, (time - started) / duration));
          const eased = progress * progress * (3 - 2 * progress);
          const accepted = update(eased);
          if (accepted === false || progress === 1) {
            this.pending = null;
            this.finish = null;
            resolve(accepted === false ? Codes.NOT_READY : Codes.DONE);
            return;
          }
          this.pending = this.clock.request(tick);
        };
        this.pending = this.clock.request(tick);
      });
    }

    cancel() {
      if (this.pending !== null) this.clock.cancel(this.pending);
      this.pending = null;
      if (this.finish) this.finish(Codes.CANCELLED);
      this.finish = null;
    }
  }

  /** Blend pose positions and unit quaternions while retaining exact segment endpoints. */
  function interpolatePose(start, end, amount) {
    if (amount === 1) return end.slice();
    const pose = start.map((value, index) => value + (end[index] - value) * amount);
    const dot = start.slice(3).reduce((sum, value, index) => sum + value * end[index + 3], 0);
    for (let index = 3; index < 7; index += 1) {
      pose[index] = start[index] * (1 - amount) + end[index] * (dot < 0 ? -amount : amount);
    }
    const length = Math.hypot(...pose.slice(3));
    for (let index = 3; index < 7; index += 1) pose[index] /= length;
    return pose;
  }

  // %% independent parts, reserved socket and bounded drawer joints
  /** Coordinate explicit manual moves above fixtures and into declared destinations. */
  class Controller {
    constructor(config, adapter, clock = new BrowserClock()) {
      this.config = config;
      this.adapter = adapter;
      this.parts = new Map(config.parts.map(part => [part.key, part]));
      this.drawers = new Map(config.drawers.map(drawer => [drawer.id, drawer]));
      this.drawerPositions = new Map(config.drawers.map(drawer => [drawer.id, 0]));
      this.animation = new Animation(clock);
      this.heldPart = null;
      this.socketOccupant = null;
      this.busy = false;
      this.destroyed = false;
    }

    availability() {
      if (this.destroyed) return new Outcome(Codes.CANCELLED);
      if (!this.adapter.isReady()) return new Outcome(Codes.NOT_READY);
      return this.busy ? new Outcome(Codes.BUSY) : null;
    }

    label(key) { return this.parts.has(key) ? this.parts.get(key).label : key; }

    /** Read the actual drawer position, including changes made through joint controls. */
    drawerPosition(id) {
      const drawer = this.drawers.get(id);
      return this.adapter.getJoint ? this.adapter.getJoint(drawer.joint) : this.drawerPositions.get(id);
    }

    /** Complete a serialized animation without exposing an intermediate command state. */
    async operate(operation, complete) {
      this.busy = true;
      this.adapter.pause();
      const result = await operation();
      this.busy = false;
      if (result !== Codes.DONE) return new Outcome(result);
      return new Outcome(Codes.DONE, complete());
    }

    /** Follow pose waypoints, restoring the original pose if the viewer rejects a write. */
    async movePart(key, destinations) {
      const original = this.adapter.getPose(key);
      if (!original) return Codes.NOT_READY;
      let previous = original;
      for (const destination of destinations) {
        const result = await this.animation.run(Motion.PART_DURATION, amount => {
          if (!this.adapter.isReady()) return false;
          return this.adapter.setPose(key, interpolatePose(previous, destination, amount));
        });
        if (result !== Codes.DONE) {
          if (result !== Codes.CANCELLED) this.adapter.setPose(key, original);
          return result;
        }
        previous = destination;
      }
      return Codes.DONE;
    }

    /** Raise a single loose part vertically before permitting horizontal movement. */
    async lift(key) {
      const unavailable = this.availability();
      if (unavailable) return unavailable;
      if (!this.parts.has(key)) return new Outcome(Codes.INVALID);
      if (this.heldPart) return new Outcome(Codes.HANDS_FULL);
      const pose = this.adapter.getPose(key);
      if (!pose) return new Outcome(Codes.NOT_READY);
      pose[2] = this.config.liftHeight;
      return this.operate(() => this.movePart(key, [pose]), () => {
        this.heldPart = key;
        if (this.socketOccupant === key) this.socketOccupant = null;
        return this.label(key) + ' lifted. Move it above the bench, insert it, or return it to its box.';
      });
    }

    /** Shift the lifted part by a bounded horizontal increment at the transit height. */
    async move(x, y) {
      const unavailable = this.availability();
      if (unavailable) return unavailable;
      if (!this.heldPart) return new Outcome(Codes.NOT_HELD);
      const pose = this.adapter.getPose(this.heldPart);
      if (!pose) return new Outcome(Codes.NOT_READY);
      pose[0] += x;
      pose[1] += y;
      const bounds = this.config.bounds;
      if (!Number.isFinite(x) || !Number.isFinite(y) || pose[0] < bounds.min[0] || pose[0] > bounds.max[0]
        || pose[1] < bounds.min[1] || pose[1] > bounds.max[1]) return new Outcome(Codes.OUTSIDE);
      pose[2] = this.config.liftHeight;
      return this.operate(() => this.movePart(this.heldPart, [pose]), () => this.label(this.heldPart) + ' moved above the workbench.');
    }

    /** Approach a configured placement vertically after travelling at the clear height. */
    placementRoute(destination) {
      const above = destination.slice();
      above[2] = this.config.liftHeight;
      return [above, destination.slice()];
    }

    /** Seat one screw in the empty pilot hole; this command does not simulate threading. */
    async insert() {
      const unavailable = this.availability();
      if (unavailable) return unavailable;
      if (!this.heldPart) return new Outcome(Codes.NOT_HELD);
      if (this.parts.get(this.heldPart).kind !== 'screw') return new Outcome(Codes.WRONG_PART);
      if (this.socketOccupant) return new Outcome(Codes.OCCUPIED);
      const key = this.heldPart;
      return this.operate(() => this.movePart(key, this.placementRoute(this.config.assembly.socketPose)), () => {
        this.socketOccupant = key;
        this.heldPart = null;
        return this.label(key) + ' seated in the pilot hole.';
      });
    }

    /** Restore a lifted part to its own non-overlapping location inside its storage box. */
    async returnPart() {
      const unavailable = this.availability();
      if (unavailable) return unavailable;
      if (!this.heldPart) return new Outcome(Codes.NOT_HELD);
      const key = this.heldPart;
      return this.operate(() => this.movePart(key, this.placementRoute(this.parts.get(key).home)), () => {
        this.heldPart = null;
        return this.label(key) + ' returned to its box.';
      });
    }

    /** Animate one drawer joint between its authored closed and open positions. */
    async setDrawer(id, open) {
      const unavailable = this.availability();
      if (unavailable) return unavailable;
      const drawer = this.drawers.get(id);
      if (!drawer) return new Outcome(Codes.INVALID);
      const start = this.drawerPosition(id);
      if (!Number.isFinite(start)) return new Outcome(Codes.NOT_READY);
      const end = open ? drawer.open : 0;
      return this.operate(async () => {
        const result = await this.animation.run(Motion.DRAWER_DURATION, amount => {
          return this.adapter.isReady() && this.adapter.setJoint(drawer.joint, start + (end - start) * amount);
        });
        if (result !== Codes.DONE && result !== Codes.CANCELLED) this.adapter.setJoint(drawer.joint, start);
        return result;
      }, () => {
        this.drawerPositions.set(id, end);
        return drawer.label + (open ? ' opened.' : ' closed.');
      });
    }

    /** Use a saved workshop camera without modifying parts or joint positions. */
    focus(name) {
      if (!this.adapter.isReady()) return new Outcome(Codes.NOT_READY);
      const camera = this.config.cameras[name];
      if (!camera) return new Outcome(Codes.INVALID);
      this.adapter.focus(camera.position.slice(), camera.target.slice());
      return new Outcome(Codes.DONE, 'Workshop view updated.');
    }

    /** Restore the initial inventory and drawers as one reversible scene reset. */
    async reset() {
      const unavailable = this.availability();
      if (unavailable) return unavailable;
      const previousParts = this.config.parts.map(part => [part.key, this.adapter.getPose(part.key)]);
      const previousDrawers = this.config.drawers.map(drawer => [drawer.joint, this.drawerPosition(drawer.id)]);
      if (previousParts.some(([, pose]) => !pose)) return new Outcome(Codes.NOT_READY);
      this.adapter.pause();
      const accepted = this.config.parts.every(part => this.adapter.setPose(part.key, part.home.slice()) !== false)
        && this.config.drawers.every(drawer => this.adapter.setJoint(drawer.joint, 0) !== false);
      if (!accepted) {
        previousParts.forEach(([key, pose]) => this.adapter.setPose(key, pose));
        previousDrawers.forEach(([joint, value]) => this.adapter.setJoint(joint, value));
        return new Outcome(Codes.NOT_READY);
      }
      this.heldPart = null;
      this.socketOccupant = null;
      this.config.drawers.forEach(drawer => this.drawerPositions.set(drawer.id, 0));
      return new Outcome(Codes.DONE, 'All parts returned to their boxes. Drawers closed.');
    }

    destroy() { this.destroyed = true; this.animation.cancel(); }
  }

  // %% compact, keyboard-accessible workshop controls
  /** Mount explicit manual controls alongside the viewer's native recording playback. */
  function mount(host, config, adapter, clock) {
    if (!config) return null;
    const controller = new Controller(config, adapter, clock);
    const document = host.ownerDocument;
    const listeners = [];
    const commands = [];
    const panel = document.createElement('details');
    panel.className = 'workshop-controls';
    panel.open = true;
    panel.setAttribute('aria-label', 'AICOR workshop manual controls');

    function element(tag, parent, className, text) {
      const item = document.createElement(tag);
      if (className) item.className = className;
      if (text) item.textContent = text;
      parent.appendChild(item);
      return item;
    }
    function listen(item, event, handler) {
      item.addEventListener(event, handler);
      listeners.push(() => item.removeEventListener(event, handler));
    }
    function button(parent, label, command, className) {
      const item = element('button', parent, className, label);
      item.type = 'button';
      commands.push(item);
      listen(item, 'click', async () => {
        const pending = command();
        refresh();
        const result = await pending;
        if (controller.destroyed) return;
        status.textContent = result.message;
        status.dataset.error = String(!result.ok);
        refresh();
      });
      return item;
    }

    const heading = element('summary', panel, 'workshop-heading');
    element('span', heading, 'workshop-title', 'Wood workshop');
    element('span', heading, 'workshop-badge', 'AICOR');
    const body = element('div', panel, 'workshop-body');
    element('p', body, 'workshop-intro', 'Explore the drawers and handle each screw and nail independently.');
    if (typeof config.demoUrl === 'string' && /^index\.html\?[^\s#]*$/.test(config.demoUrl)) {
      const demo = element('a', body, 'workshop-demo', 'Watch PR2 assembly demo');
      demo.href = config.demoUrl;
    }
    const cameras = element('div', body, 'workshop-camera-row');
    for (const [name, label] of [['overview', 'Overview'], ['bench', 'Workbench'], ['drawers', 'Drawers']]) {
      if (config.cameras[name]) button(cameras, label, () => controller.focus(name));
    }
    const drawers = element('fieldset', body, 'workshop-section');
    element('legend', drawers, null, 'Drawer cabinet');
    const drawerButtons = new Map();
    for (const drawer of config.drawers) {
      const row = element('div', drawers, 'workshop-drawer-row');
      element('span', row, null, drawer.label);
      const control = button(row, 'Open', () => controller.setDrawer(drawer.id, controller.drawerPosition(drawer.id) < drawer.open / 2));
      control.setAttribute('aria-label', 'Open ' + drawer.label.toLowerCase());
      drawerButtons.set(drawer.id, control);
    }
    const inventory = element('fieldset', body, 'workshop-section');
    element('legend', inventory, null, 'Loose parts');
    const label = element('label', inventory, 'workshop-part-label');
    element('span', label, null, 'Selected part');
    const select = element('select', label, 'workshop-part-select');
    for (const part of config.parts) {
      const option = element('option', select, null, part.label);
      option.value = part.key;
    }
    const lift = button(inventory, 'Lift selected part', () => controller.lift(select.value), 'workshop-primary');
    const movement = element('div', inventory, 'workshop-movement');
    movement.setAttribute('aria-label', 'Move lifted part in five centimetre steps');
    const movementButtons = [];
    for (const [text, x, y, description] of [
      ['←', -Motion.STEP, 0, 'Move left'], ['↑', 0, Motion.STEP, 'Move towards the back'],
      ['↓', 0, -Motion.STEP, 'Move towards the front'], ['→', Motion.STEP, 0, 'Move right'],
    ]) {
      const item = button(movement, text, () => controller.move(x, y));
      item.setAttribute('aria-label', description + ' by 5 cm');
      item.title = description + ' by 5 cm';
      movementButtons.push(item);
    }
    const placement = element('div', inventory, 'workshop-placement');
    const insert = button(placement, 'Insert in pilot hole', () => controller.insert(), 'workshop-primary');
    const returnPart = button(placement, 'Return to box', () => controller.returnPart());
    const held = element('p', inventory, 'workshop-held');
    const status = element('p', body, 'workshop-status', 'Choose a drawer or lift a part to explore the workbench.');
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    button(body, 'Reset workshop', () => controller.reset(), 'workshop-reset');
    element('p', body, 'workshop-note', 'Manual object positioning · no force or drilling simulation.');
    if (config.assetRights) {
      const rights = element('details', body, 'workshop-rights');
      element('summary', rights, null, 'Asset rights & credits');
      for (const field of ['principalAuthor', 'institution', 'copyright', 'permissionNotice', 'priorLicenseNotice']) {
        if (config.assetRights[field]) element('p', rights, null, config.assetRights[field]);
      }
    }
    let wasReady = adapter.isReady();

    function refresh() {
      const ready = adapter.isReady();
      const disabled = !ready || controller.busy;
      commands.forEach(command => { command.disabled = disabled; });
      select.disabled = disabled || Boolean(controller.heldPart);
      lift.disabled = disabled || Boolean(controller.heldPart);
      returnPart.disabled = disabled || !controller.heldPart;
      const heldMetadata = controller.parts.get(controller.heldPart);
      insert.disabled = disabled || !heldMetadata || heldMetadata.kind !== 'screw' || Boolean(controller.socketOccupant);
      movementButtons.forEach(command => { command.disabled = disabled || !controller.heldPart; });
      for (const drawer of config.drawers) {
        const control = drawerButtons.get(drawer.id);
        const opened = controller.drawerPosition(drawer.id) >= drawer.open / 2;
        control.textContent = opened ? 'Close' : 'Open';
        control.setAttribute('aria-label', (opened ? 'Close ' : 'Open ') + drawer.label.toLowerCase());
      }
      held.textContent = controller.heldPart ? 'Lifted: ' + controller.label(controller.heldPart)
        : controller.socketOccupant ? 'In pilot hole: ' + controller.label(controller.socketOccupant) : 'No part lifted · Pilot hole empty';
      panel.dataset.busy = String(controller.busy);
      if (!ready) status.textContent = Messages[Codes.NOT_READY];
      else if (!wasReady) status.textContent = 'Workshop ready. Choose a drawer or lift a part.';
      wasReady = ready;
    }

    listen(select, 'change', refresh);
    host.appendChild(panel);
    refresh();
    return {
      controller,
      refresh,
      select(key) {
        if (!controller.parts.has(key) || controller.heldPart || controller.busy) return false;
        select.value = key;
        refresh();
        return true;
      },
      destroy() {
        controller.destroy();
        listeners.forEach(remove => remove());
        panel.remove();
      },
    };
  }

  return {Controller, Codes, Motion, Outcome, mount};
});
