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

  // %% authenticated, authoritative physics
  const Waypoint = Object.freeze({LIFT: 'lift', TRANSIT: 'transit', FINAL: 'final'});
  const Physics = Object.freeze({POLL_MS: 100, MOVE_STEP: .025, ANGLE_STEP: Math.PI / 12, TARGET_DECIMALS: 6, CLEARANCE: 1.3, APPROACH_CLEARANCE: .15, AXIS_TOLERANCE: 1e-6, POSITION_TOLERANCE: .006, ANGLE_TOLERANCE: .08, WAYPOINT_TIMEOUT_MS: 8000, FINAL_POSITION_TOLERANCE: .0015, FINAL_ANGLE_TOLERANCE: .04, CONTACT_TIMEOUT_MS: 2000});

  /** Keep all workshop requests inside the current authenticated viewer mount. */
  class PhysicsClient {
    constructor(fetch, viewerUrl) {
      this.fetch = fetch;
      this.stateUrl = new URL('api/workshop/state', viewerUrl).href;
      this.commandUrl = new URL('api/workshop/command', viewerUrl).href;
    }

    async request(url, options) {
      const response = await this.fetch(url, Object.assign({credentials: 'same-origin', cache: 'no-store'}, options));
      const state = await response.json();
      if (!response.ok) throw new Error(state.error || 'Workshop physics request failed (' + response.status + ').');
      return state;
    }

    state() { return this.request(this.stateUrl, {method: 'GET'}); }
    command(body) {
      return this.request(this.commandUrl, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    }
  }

  /** Send spring targets while rendering only measured bodies and joints. */
  class PhysicsController {
    constructor(config, adapter, client) {
      this.config = config;
      this.adapter = adapter;
      this.client = client;
      const descriptions = [...(config.parts || []), ...(config.bins || []), config.assembly, config.driver].filter(Boolean);
      this.objects = new Map((config.objects || descriptions).map(object => [object.key, Object.assign({}, object, descriptions.find(item => item.key === object.key))]));
      this.drawers = new Map((config.drawers || []).map(drawer => [drawer.id, drawer]));
      this.snapshot = null;
      this.selected = this.objects.keys().next().value;
      this.target = null;
      this.queue = Promise.resolve();
      this.revision = 0;
      this.pending = 0;
      this.polling = false;
      this.destroyed = false;
      this.error = null;
      this.message = 'Connecting to MuJoCo…';
      this.onChange = () => {};
      this.drag = null;
      this.aligning = false;
      this.ownsHold = false;
      this.alignmentRevision = 0;
      this.clock = {now: () => Date.now(), wait: milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds))};
    }

    isReady() { return !this.destroyed && !this.error && !!this.snapshot && this.adapter.isReady(); }
    isRunning() { return this.isReady() && this.snapshot.running; }
    label(key) { const object = this.objects.get(key); return object && object.label || (key || '').replace(/_/g, ' '); }
    select(key) { if (!this.objects.has(key)) return false; this.selected = key; this.onChange(); return true; }

    accept(snapshot) {
      if (this.destroyed) return;
      this.snapshot = snapshot;
      this.error = snapshot.error || (snapshot.available === false ? 'Physics unavailable. Reset the workshop to recover.' : null);
      this.adapter.applySnapshot(snapshot);
      this.onChange();
    }

    fail(error) {
      if (this.destroyed) return {ok: false, message: 'Controls closed.'};
      this.error = error.message || String(error);
      this.message = this.error;
      this.onChange();
      return {ok: false, message: this.error};
    }

    /** Serialize commands and invalidate any older in-flight observation. */
    command(body, message) {
      this.revision += 1;
      this.pending += 1;
      const pending = this.queue.then(async () => {
        if (this.destroyed) return {ok: false, message: 'Controls closed.'};
        try {
          const snapshot = await this.client.command(body);
          this.message = message || 'Physics updated.';
          this.accept(snapshot);
          return {ok: !this.error, message: this.error || this.message};
        } catch (error) { return this.fail(error); }
      }).finally(() => { this.pending -= 1; if (!this.destroyed) this.onChange(); });
      this.queue = pending;
      return pending;
    }

    async poll() {
      if (this.destroyed || this.polling || this.pending) return;
      this.polling = true;
      const revision = this.revision;
      try {
        const state = await this.client.state();
        if (revision === this.revision) this.accept(state);
      } catch (error) { if (revision === this.revision) this.fail(error); }
      finally { this.polling = false; }
    }

    start() { this.adapter.pause(); return this.command({action: 'start'}, 'MuJoCo running. Drag an object or grab it to lift and tilt.'); }
    pause() { this.alignmentRevision += 1; return this.command({action: 'pause'}, 'Physics paused. Resume to continue.'); }
    reset() { this.alignmentRevision += 1; this.target = null; return this.command({action: 'reset'}, 'Workshop reset.'); }
    release() {
      this.alignmentRevision += 1;
      this.target = null;
      return this.command({action: 'release'}, 'Released. Gravity and contacts control the object.').then(result => {
        if (result.ok) this.ownsHold = false;
        return result;
      });
    }

    async grab(key = this.selected) {
      if (!this.isRunning() || !this.objects.has(key)) return {ok: false, message: 'Start physics and choose a movable object first.'};
      this.select(key);
      this.ownsHold = true;
      const result = await this.command({action: 'grab', object: key}, this.label(key) + ' held by a spring. Lift or tilt it with the controls.');
      if (result.ok && this.snapshot.objects[key]) this.target = this.snapshot.objects[key].slice();
      return result;
    }

    moveTo(pose) {
      if (!this.isRunning() || !this.snapshot.held || !Array.isArray(pose) || pose.length !== 7 || !pose.every(Number.isFinite)) {
        return Promise.resolve({ok: false, message: 'Grab an object first.'});
      }
      this.target = pose.slice();
      return this.command({action: 'move', object: this.snapshot.held, pose: this.target.slice()}, 'Spring target moved. Contacts may limit the actual movement.');
    }

    moveBy(offset) {
      const pose = this.target || (this.snapshot && this.snapshot.objects[this.snapshot.held]);
      if (!pose) return Promise.resolve({ok: false, message: 'Grab an object first.'});
      const desired = pose.slice();
      offset.forEach((value, axis) => { desired[axis] = Number((desired[axis] + value).toFixed(Physics.TARGET_DECIMALS)); });
      return this.moveTo(desired);
    }

    tilt(axis, angle) {
      const pose = this.target || (this.snapshot && this.snapshot.objects[this.snapshot.held]);
      if (!pose || ![0, 1, 2].includes(axis) || !Number.isFinite(angle)) return Promise.resolve({ok: false, message: 'Grab an object first.'});
      const desired = pose.slice();
      const rotation = [0, 0, 0, Math.cos(angle / 2)];
      rotation[axis] = Math.sin(angle / 2);
      const [ax, ay, az, aw] = rotation, [bx, by, bz, bw] = pose.slice(3);
      desired.splice(3, 4, aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz);
      return this.moveTo(desired);
    }

    setDrawer(id, opened) {
      const drawer = this.drawers.get(id);
      if (!drawer) return Promise.resolve({ok: false, message: 'Unknown drawer.'});
      return this.command({action: 'drawer', joint: drawer.joint, position: opened ? drawer.open : 0}, 'Drawer motor target changed. Contacts may block it.');
    }

    drive(direction) { return this.command({action: 'drive', direction}, direction ? 'Driver torque active. Alignment and thread contact are checked by physics.' : 'Driver stopped.'); }

    /** Stop at blocked contacts instead of continuing from an unachieved target. */
    async waitForPose(key, desired, revision, phase = Waypoint.TRANSIT) {
      const final = phase === Waypoint.FINAL;
      const positionTolerance = final ? Physics.FINAL_POSITION_TOLERANCE : Physics.POSITION_TOLERANCE;
      const angleTolerance = final ? Physics.FINAL_ANGLE_TOLERANCE : Physics.ANGLE_TOLERANCE;
      const deadline = this.clock.now() + Physics.WAYPOINT_TIMEOUT_MS;
      while (this.clock.now() < deadline) {
        if (!this.isRunning() || revision !== this.alignmentRevision || this.snapshot.held !== key) return {ok: false, message: 'Alignment stopped.'};
        const actual = this.snapshot.objects[key];
        const distance = Math.hypot(...actual.slice(0, 3).map((value, axis) => value - desired[axis]));
        const dot = Math.abs(actual.slice(3).reduce((sum, value, axis) => sum + value * desired[axis + 3], 0));
        const angle = 2 * Math.acos(Math.min(1, dot));
        if (distance <= positionTolerance && (phase === Waypoint.LIFT || angle <= angleTolerance)) return {ok: true};
        await this.clock.wait(Physics.POLL_MS);
        await this.poll();
      }
      return {ok: false, message: 'Alignment blocked: the object did not reach its target. Move the obstruction or lift it manually.'};
    }

    /** Confirm physical pilot capture or tool contact before reporting success. */
    async waitForThread(kind, key, revision) {
      const deadline = this.clock.now() + Physics.CONTACT_TIMEOUT_MS;
      while (this.clock.now() < deadline) {
        if (!this.isRunning() || revision !== this.alignmentRevision) return {ok: false, message: 'Alignment stopped.'};
        const thread = this.snapshot.thread || {};
        if (kind === 'driver' && thread.driverAligned) return {ok: true, message: 'Driver contact confirmed. Drive in or reverse.'};
        if (kind !== 'driver' && thread.screw === key && ['seated', 'driving', 'stopped', 'limit'].includes(thread.state)) {
          return {ok: true, message: 'Screw seated in the pilot hole. Align the driver next.'};
        }
        await this.clock.wait(Physics.POLL_MS);
        await this.poll();
      }
      return {ok: false, message: kind === 'driver' ? 'Driver contact was not confirmed. Adjust its alignment before driving.' : 'Pilot capture was not confirmed. Reposition the screw before continuing.'};
    }

    /** Lift clear of the fixtures before approaching measured screw and driver goals. */
    async align(kind) {
      if (!this.isRunning() || this.aligning || this.drag) return {ok: false, message: 'Finish the current manipulation first.'};
      const key = kind === 'driver' ? this.config.driver && this.config.driver.key : this.selected;
      const selected = this.objects.get(key);
      if (!selected || (kind !== 'driver' && selected.kind !== 'screw')) return {ok: false, message: 'Select a screw before positioning it.'};
      this.aligning = true;
      const revision = ++this.alignmentRevision;
      this.onChange();
      try {
        let result = await this.grab(key);
        if (!result.ok) return result;
        const thread = this.snapshot.thread || {};
        const goal = kind === 'driver' ? thread.driverTargetPose : thread.screwTargetPose;
        if (!goal) return {ok: false, message: 'The physics alignment target is unavailable.'};
        const actual = this.snapshot.objects[key];
        const height = Math.max(this.config.liftHeight || Physics.CLEARANCE, actual[2], goal[2] + .1);
        const raised = actual.slice(); raised[2] = height;
        const [x, y, z, w] = goal.slice(3);
        const axis = [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)];
        const clearance = axis[2] > 1 - Physics.AXIS_TOLERANCE
          ? Math.max(Physics.APPROACH_CLEARANCE, height - goal[2]) : Physics.APPROACH_CLEARANCE;
        const approach = goal.slice();
        for (let index = 0; index < 3; index += 1) approach[index] += axis[index] * clearance;
        const above = approach.slice(); above[2] = Math.max(height, approach[2]);
        const waypoints = [[raised, Waypoint.LIFT], [above, Waypoint.TRANSIT]];
        if (Math.abs(above[2] - approach[2]) > Physics.AXIS_TOLERANCE) waypoints.push([approach, Waypoint.TRANSIT]);
        waypoints.push([goal, Waypoint.FINAL]);
        for (const [waypoint, phase] of waypoints) {
          if (revision !== this.alignmentRevision) return {ok: false, message: 'Alignment stopped.'};
          result = await this.moveTo(waypoint);
          if (!result.ok) return result;
          result = await this.waitForPose(key, waypoint, revision, phase);
          if (!result.ok) return result;
        }
        if (kind !== 'driver') {
          result = await this.release();
          if (!result.ok) return result;
          return await this.waitForThread(kind, key, revision + 1);
        }
        return await this.waitForThread(kind, key, revision);
      } finally {
        this.aligning = false;
        this.onChange();
      }
    }

    focus(name) {
      const camera = this.config.cameras && this.config.cameras[name];
      if (camera) this.adapter.focus(camera.position, camera.target);
      return {ok: !!camera, message: camera ? 'View changed.' : 'View unavailable.'};
    }

    /** Capture the pointer immediately while the asynchronous grab is pending. */
    beginDrag(key) {
      if (!this.isRunning() || this.drag || this.aligning || !this.objects.has(key)) return false;
      const drag = {key, latest: null, ending: false, pump: null};
      this.drag = drag;
      drag.pump = this.grab(key).then(result => this.flushDrag(drag, result));
      return true;
    }

    async flushDrag(drag, result) {
      while (!this.destroyed && result.ok && drag.latest) {
        const position = drag.latest;
        drag.latest = null;
        const pose = (this.target || this.snapshot.objects[drag.key]).slice();
        pose.splice(0, 3, ...position);
        result = await this.moveTo(pose);
      }
      drag.pump = null;
      return result;
    }

    dragTo(position) {
      const drag = this.drag;
      if (!drag || drag.ending || !Array.isArray(position) || position.length !== 3 || !position.every(Number.isFinite)) return;
      drag.latest = position.slice();
      if (!drag.pump) drag.pump = this.flushDrag(drag, {ok: true});
    }

    async endDrag() {
      const drag = this.drag;
      if (!drag) return;
      drag.ending = true;
      if (drag.pump) await drag.pump;
      if (!this.destroyed) await this.release();
      this.drag = null;
    }

    destroy() {
      this.destroyed = true;
      this.revision += 1;
      this.onChange = () => {};
      if (this.adapter.destroy) this.adapter.destroy();
      if (this.ownsHold) {
        this.queue = this.queue.then(() => this.client.command({action: 'release'})).catch(() => {});
      }
      return this.queue;
    }
  }

  // %% physical manipulation controls
  // %% native controller recording presentation
  /** Present recorded plan stages and restart from the first measured frame. */
  function mountRecording(host, config, adapter) {
    if (!config || config.kind !== 'native-recording') return null;
    const document = host.ownerDocument;
    function element(tag, parent, className, text) {
      const item = document.createElement(tag);
      if (className) item.className = className;
      if (text) item.textContent = text;
      parent.appendChild(item);
      return item;
    }
    const panel = element('details', host, 'workshop-controls workshop-recording');
    panel.open = true;
    panel.setAttribute('aria-label', 'Recorded PR2 workshop demo');
    const heading = element('summary', panel, 'workshop-heading');
    element('span', heading, 'workshop-title', 'PR2 assembly demo');
    element('span', heading, 'workshop-badge', 'Recorded');
    const body = element('div', panel, 'workshop-body');
    const play = element('button', body, 'workshop-primary workshop-recording-start', 'Play from start');
    play.type = 'button';
    const list = element('ol', body, 'workshop-demo-steps');
    const steps = (Array.isArray(config.steps) ? config.steps : []).filter(step => typeof step === 'string' && step.trim());
    const rows = steps.map(step => element('li', list, null, step));
    element('p', body, 'workshop-demo-note', 'Native robot-plan recording. The interactive MuJoCo contact world runs separately; this playback does not drive its physics.');
    const back = element('a', body, 'workshop-demo', 'Back to interactive workshop');
    back.href = 'index.html?scene=aicor_workshop&layout=scene&offline=1';
    function refresh() {
      play.disabled = !adapter.hasTrajectory();
      const frame = adapter.playhead();
      const active = (Array.isArray(config.stepRanges) ? config.stepRanges : []).find(range =>
        range && typeof range.name === 'string' && Number.isFinite(range.startFrame)
        && Number.isFinite(range.endFrame) && range.endFrame >= range.startFrame
        && frame >= range.startFrame && frame <= range.endFrame);
      rows.forEach((row, index) => row.setAttribute('aria-current', active && active.name === steps[index] ? 'step' : 'false'));
    }
    function start() {
      if (!adapter.hasTrajectory()) return;
      adapter.seek(0);
      adapter.play();
      refresh();
    }
    play.addEventListener('click', start);
    refresh();
    return {start, refresh, destroy() { play.removeEventListener('click', start); panel.remove(); }};
  }

  /** Mount compact controls which expose forces, independent contents and thread approximation. */
  function mountPhysics(host, config, adapter, client) {
    const controller = new PhysicsController(config, adapter, client);
    const document = host.ownerDocument;
    const listeners = [];
    const commands = [];
    const panel = document.createElement('details');
    panel.className = 'workshop-controls workshop-physics';
    panel.open = true;
    panel.setAttribute('aria-label', 'AICOR workshop physics controls');
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
    function button(parent, label, command, enabled = () => controller.isRunning() && !controller.aligning && !controller.drag, className) {
      const item = element('button', parent, className, label);
      item.type = 'button';
      commands.push({item, enabled});
      listen(item, 'click', async () => {
        const result = await command();
        if (controller.destroyed) return;
        if (result) controller.message = result.message;
        refresh();
      });
      return item;
    }
    const heading = element('summary', panel, 'workshop-heading');
    element('span', heading, 'workshop-title', 'Wood workshop');
    const badge = element('span', heading, 'workshop-badge', 'MuJoCo');
    const body = element('div', panel, 'workshop-body');
    element('p', body, 'workshop-intro', 'Drag a box or loose part. Grab it to lift and tilt; release to let gravity act.');
    if (typeof config.demoUrl === 'string' && /^index\.html\?[^\s#]*$/.test(config.demoUrl)) {
      const demo = element('a', body, 'workshop-demo workshop-primary', 'Start full PR2 demo');
      const destination = new URL(config.demoUrl, 'https://cramera.invalid/');
      destination.searchParams.set('autoplay', '1');
      demo.href = destination.pathname.slice(1) + destination.search;
      element('p', body, 'workshop-demo-description', 'Retrieve the screw box from the right shelf, place it on the workbench, pick a screw and assemble it with the screwdriver.');
      element('p', body, 'workshop-demo-note', 'Opens a recorded robot plan. The interactive contact simulation is separate.');
    }
    const runRow = element('div', body, 'workshop-placement');
    const run = button(runRow, 'Start physics', () => controller.isRunning() ? controller.pause() : controller.start(), () => !controller.pending);
    button(runRow, 'Reset', () => controller.reset(), () => !!controller.snapshot && adapter.isReady() && !controller.pending);
    const cameras = element('div', body, 'workshop-camera-row');
    for (const [name, label] of [['overview', 'Overview'], ['bench', 'Bench'], ['drawers', 'Drawers']]) {
      if (config.cameras && config.cameras[name]) button(cameras, label, () => controller.focus(name), () => true);
    }
    const drawers = element('fieldset', body, 'workshop-section');
    element('legend', drawers, null, 'Drawer motors');
    const drawerButtons = new Map();
    for (const drawer of config.drawers || []) {
      const row = element('div', drawers, 'workshop-drawer-row');
      element('span', row, null, drawer.label);
      const control = button(row, 'Open', () => controller.setDrawer(drawer.id, (controller.snapshot.joints[drawer.joint] || 0) < drawer.open / 2));
      drawerButtons.set(drawer.id, control);
    }
    const inventory = element('fieldset', body, 'workshop-section');
    element('legend', inventory, null, 'Objects & loose contents');
    const label = element('label', inventory, 'workshop-part-label');
    element('span', label, null, 'Selected');
    const select = element('select', label, 'workshop-part-select');
    for (const [key] of controller.objects) {
      const option = element('option', select, null, controller.label(key));
      option.value = key;
    }
    select.value = controller.selected;
    listen(select, 'change', () => controller.select(select.value));
    const grasp = element('div', inventory, 'workshop-placement');
    const held = () => controller.isRunning() && !controller.aligning && !controller.drag && !!controller.snapshot.held;
    button(grasp, 'Grab selected', () => controller.grab(), () => controller.isRunning() && !controller.pending && !controller.aligning && !controller.drag, 'workshop-primary');
    button(grasp, 'Release', () => controller.release(), held);
    const movement = element('div', inventory, 'workshop-translation');
    for (const [label, offset] of [['X −', [-Physics.MOVE_STEP, 0, 0]], ['X +', [Physics.MOVE_STEP, 0, 0]],
      ['Y −', [0, -Physics.MOVE_STEP, 0]], ['Y +', [0, Physics.MOVE_STEP, 0]],
      ['Lower', [0, 0, -Physics.MOVE_STEP]], ['Lift', [0, 0, Physics.MOVE_STEP]]]) {
      const control = button(movement, label, () => controller.moveBy(offset), held);
      control.title = 'Move spring target by 2.5 cm';
    }
    const rotations = element('details', inventory, 'workshop-rotations');
    element('summary', rotations, null, 'Tilt / rotate by 15°');
    const axes = element('div', rotations, 'workshop-translation');
    for (const [axis, name] of [[0, 'Roll'], [1, 'Pitch'], [2, 'Yaw']]) {
      for (const direction of [-1, 1]) button(axes, name + (direction < 0 ? ' −' : ' +'), () => controller.tilt(axis, direction * Physics.ANGLE_STEP), held);
    }
    const observation = element('p', inventory, 'workshop-held');
    const editor = element('a', body, 'workshop-demo', 'Arrange a scene in Scene Builder');
    editor.href = 'scene_builder.html';
    const threadSection = element('details', body, 'workshop-thread');
    element('summary', threadSection, null, 'Screw driving');
    element('p', threadSection, 'workshop-note', 'Select a screw and use Position screw, then Align driver. Each move lifts above the bench first. The tool applies torque; an ideal thread constraint couples rotation and insertion. No material removal is simulated.');
    const align = element('div', threadSection, 'workshop-placement');
    button(align, 'Position screw', () => controller.align('screw'));
    button(align, 'Align driver', () => controller.align('driver'));
    const drive = element('div', threadSection, 'workshop-placement');
    button(drive, 'Drive in', () => controller.drive(1));
    button(drive, 'Reverse', () => controller.drive(-1));
    button(drive, 'Stop', () => controller.drive(0));
    const threadState = element('p', threadSection, 'workshop-held');
    const status = element('p', body, 'workshop-status');
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');
    if (config.assetRights) {
      const rights = element('details', body, 'workshop-rights');
      element('summary', rights, null, 'Asset rights & credits');
      for (const field of ['principalAuthor', 'institution', 'copyright', 'permissionNotice', 'priorLicenseNotice']) {
        if (config.assetRights[field]) element('p', rights, null, config.assetRights[field]);
      }
    }
    function refresh() {
      if (controller.destroyed) return;
      const snapshot = controller.snapshot;
      commands.forEach(({item, enabled}) => { item.disabled = !enabled(); });
      select.value = controller.selected;
      run.textContent = controller.isRunning() ? 'Pause physics' : controller.isReady() ? 'Resume physics' : 'Retry physics';
      badge.textContent = controller.error ? 'Offline' : snapshot && snapshot.running ? 'MuJoCo · Live' : 'MuJoCo';
      status.textContent = controller.error || controller.message;
      status.dataset.error = String(!!controller.error);
      for (const drawer of config.drawers || []) {
        const opened = snapshot && (snapshot.joints[drawer.joint] || 0) > drawer.open / 2;
        const control = drawerButtons.get(drawer.id);
        control.textContent = opened ? 'Close' : 'Open';
        control.setAttribute('aria-label', (opened ? 'Close ' : 'Open ') + drawer.label.toLowerCase());
      }
      const contacts = snapshot && snapshot.contacts || [];
      observation.textContent = (snapshot && snapshot.held ? 'Holding: ' + controller.label(snapshot.held) : 'Nothing held')
        + ' · ' + contacts.length + ' contacts';
      const thread = snapshot && snapshot.thread || {};
      threadState.textContent = (thread.engaged ? 'Thread engaged' : 'Thread free') + ' · '
        + (thread.driverAligned ? 'Driver aligned' : 'Driver unaligned')
        + (Number.isFinite(thread.depth) ? ' · ' + (thread.depth * 1000).toFixed(1) + ' mm depth' : '');
    }
    controller.onChange = refresh;
    host.appendChild(panel);
    refresh();
    controller.start();
    const timer = setInterval(() => controller.poll(), Physics.POLL_MS);
    return {
      controller, refresh,
      select: key => controller.select(key),
      destroy() { clearInterval(timer); controller.destroy(); listeners.forEach(remove => remove()); panel.remove(); },
    };
  }

  return {Controller, Codes, Motion, Outcome, mount, PhysicsClient, PhysicsController, mountPhysics, mountRecording};
});
