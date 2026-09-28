(function () {
  'use strict';

  // %% simulation transport
  const Routes = Object.freeze({STATE: '/api/cable/state', START: '/api/cable/start', COMMAND: '/api/cable/command', RESET: '/api/cable/reset', STOP: '/api/cable/stop'});
  const Actions = Object.freeze({PINCH: 'pinch', RELEASE: 'release', LIFT: 'lift', LOWER: 'lower', MOVE: 'move'});

  class Client {
    /** Submit control requests to the headless simulation. */
    constructor(fetcher) { this.fetcher = fetcher; }
    async request(route, payload) {
      const options = {cache: 'no-store'};
      if (payload !== undefined) Object.assign(options, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
      const response = await this.fetcher(route, options);
      const answer = await response.json();
      if (!response.ok || (payload !== undefined && answer.ok === false)) throw new Error(answer.error || 'The simulator could not complete this request.');
      return answer;
    }
    state() { return this.request(Routes.STATE); }
    start(config) { return this.request(Routes.START, config); }
    command(action) { return this.request(Routes.COMMAND, {action}); }
    reset() { return this.request(Routes.RESET, {}); }
    stop() { return this.request(Routes.STOP, {}); }
  }

  class Configuration {
    /** Translate form values into the native cable configuration. */
    static read(values) {
      return {
        segment_count: Number(values.segment_count), segment_length: Number(values.segment_length),
        radius: Number(values.radius), mass: Number(values.mass),
        bend: Number(values.bend), twist: Number(values.twist),
        friction: [Number(values.friction), .005, .0001],
        anchor_to_parent: Boolean(values.anchor_to_parent),
        anchor_offset: ['x', 'y', 'z'].map(axis => Number(values['anchor_offset_' + axis] || 0)),
        anchor_rpy: ['x', 'y', 'z'].map(axis => Number(values['anchor_rpy_' + axis] || 0)),
      };
    }
    static fromForm(form) {
      const values = Object.fromEntries(new FormData(form));
      values.anchor_to_parent = form.elements.anchor_to_parent.checked;
      return Configuration.read(values);
    }
    static apply(form, config) {
      for (const name of ['segment_count', 'segment_length', 'radius', 'mass', 'bend', 'twist']) {
        if (Number.isFinite(config[name])) form.elements[name].value = config[name];
      }
      if (Array.isArray(config.friction)) form.elements.friction.value = config.friction[0];
      form.elements.anchor_to_parent.checked = config.anchor_to_parent === true;
      for (const name of ['anchor_offset', 'anchor_rpy']) {
        if (Array.isArray(config[name])) ['x', 'y', 'z'].forEach((axis, index) => { form.elements[name + '_' + axis].value = config[name][index]; });
      }
    }
  }

  class Feedback {
    /** Preserve the distinction between measured contact and missing physics. */
    static read(snapshot) {
      const fallback = snapshot.physical === false;
      const grasp = snapshot.grasp;
      const labels = {open: 'Gripper open', contact: 'Cable contact', holding: 'Bilateral contact · low slip', slipping: 'Cable slipping'};
      const statuses = {idle: 'Ready to build', running: 'Physics running', failed: 'Simulation failed', stopped: 'Physics stopped'};
      return {
        status: fallback ? 'Placeholders only' : (snapshot.state === 'idle' && snapshot.config ? 'Physics stopped' : statuses[snapshot.state]) || 'Waiting for simulator',
        canCommand: snapshot.state === 'running' && snapshot.physical === true,
        grasp: grasp ? labels[grasp.state] || grasp.state : 'No contact data',
        normalForce: grasp && Number.isFinite(grasp.normal_force) ? grasp.normal_force : null,
        frictionForce: grasp && Number.isFinite(grasp.friction_force) ? grasp.friction_force : null,
        contactCount: grasp && Number.isFinite(grasp.contact_count) ? grasp.contact_count : null,
        warning: (snapshot.warnings || []).join(' ') || (fallback ? 'The physical cable could not be built. Only semantic placeholder points are available.' : ''),
      };
    }
  }

  class Controller {
    /** Publish only snapshots newer than the most recent control request. */
    constructor(client, observe) {
      this.client = client;
      this.observe = observe;
      this.revision = 0;
      this.pending = false;
      this.snapshot = null;
    }
    accept(snapshot) { this.snapshot = snapshot; this.observe(snapshot); }
    async refresh() {
      if (this.pending) return;
      const revision = this.revision;
      const snapshot = await this.client.state();
      if (revision === this.revision && !this.pending) this.accept(snapshot);
    }
    async update(operation) {
      this.revision += 1;
      this.pending = true;
      try { this.accept(await operation()); }
      finally { this.pending = false; }
    }
    start(config) { return this.update(() => this.client.start(config)); }
    command(action) { return this.update(() => this.client.command(action)); }
    reset() { return this.update(() => this.client.reset()); }
    stop() { return this.update(() => this.client.stop()); }
  }

  // %% authoritative three dimensional geometry
  class Viewport {
    /** Draw only geometry and contact positions delivered by MuJoCo. */
    constructor(host, library) {
      this.host = host;
      this.library = library;
      this.scene = new library.Scene();
      this.camera = new library.PerspectiveCamera(40, 1, .001, 30);
      this.camera.up.set(0, 0, 1);
      this.renderer = new library.WebGLRenderer({antialias: true, alpha: true});
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = library.PCFSoftShadowMap;
      this.renderer.outputEncoding = library.sRGBEncoding;
      this.host.prepend(this.renderer.domElement);
      this.controls = new library.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.minDistance = .03;
      this.controls.maxDistance = 5;
      this.scene.add(new library.HemisphereLight(0xe0f4ff, 0x627381, 1.1));
      const light = new library.DirectionalLight(0xffffff, 1.5);
      light.position.set(.3, -.5, 1.6);
      light.castShadow = true;
      light.shadow.mapSize.set(1024, 1024);
      light.shadow.camera.left = -1;
      light.shadow.camera.right = 1;
      light.shadow.camera.top = 1;
      light.shadow.camera.bottom = -1;
      light.shadow.bias = -.00015;
      this.scene.add(light);
      const grid = new library.GridHelper(2, 40, 0x355064, 0x213340);
      grid.rotation.x = Math.PI / 2;
      grid.position.z = -.001;
      this.scene.add(grid);
      this.materials = {
        cable: new library.MeshStandardMaterial({color: 0xee5867, roughness: .5}),
        pad: new library.MeshStandardMaterial({color: 0x43d5c4, roughness: .7, metalness: .15}),
        point: new library.MeshBasicMaterial({color: 0xffffff, depthTest: false, transparent: true, opacity: .85}),
        contact: new library.MeshBasicMaterial({color: 0xffc16c}),
        anchor: new library.MeshStandardMaterial({color: 0xe1b472, roughness: .4, metalness: .4}),
        table: new library.MeshStandardMaterial({color: 0x344350, roughness: .9}),
      };
      this.geometry = {
        sphere: new library.SphereGeometry(1, 14, 10),
        cylinder: new library.CylinderGeometry(1, 1, 1, 14),
        box: new library.BoxGeometry(1, 1, 1),
      };
      this.cables = [];
      this.pads = [];
      this.contacts = [];
      this.points = [];
      this.table = this.mesh(this.geometry.box, this.materials.table);
      this.anchor = this.mesh(this.geometry.sphere, this.materials.anchor);
      this.table.visible = this.anchor.visible = false;
      this.snapshot = null;
      this.showPoints = false;
      this.needsFrame = true;
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(host);
      this.home();
      this.frame = () => {
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
        this.animation = window.requestAnimationFrame(this.frame);
      };
      this.frame();
    }
    mesh(geometry, material) {
      const result = new this.library.Mesh(geometry, material);
      result.castShadow = true;
      result.receiveShadow = true;
      this.scene.add(result);
      return result;
    }
    resize() {
      this.renderer.setSize(this.host.clientWidth, this.host.clientHeight);
      this.camera.aspect = this.host.clientWidth / this.host.clientHeight;
      this.camera.updateProjectionMatrix();
    }
    sync(pool, values, create, update) {
      while (pool.length < values.length) pool.push(create());
      pool.forEach((item, index) => {
        item.visible = index < values.length;
        if (item.visible) update(item, values[index]);
      });
    }
    capsule() {
      const group = new this.library.Group();
      group.add(new this.library.Mesh(this.geometry.cylinder, this.materials.cable));
      group.add(new this.library.Mesh(this.geometry.sphere, this.materials.cable));
      group.add(new this.library.Mesh(this.geometry.sphere, this.materials.cable));
      group.children.forEach(mesh => { mesh.castShadow = true; });
      this.scene.add(group);
      return group;
    }
    render(snapshot) {
      this.snapshot = snapshot;
      const library = this.library;
      const radius = snapshot.config ? snapshot.config.radius : .005;
      const color = snapshot.config && snapshot.config.color;
      if (color) this.materials.cable.color.setRGB(...(Array.isArray(color) ? color : [color.R, color.G, color.B]).slice(0, 3));
      this.sync(this.cables, snapshot.segments || [], () => this.capsule(), (group, segment) => {
        const start = new library.Vector3().fromArray(segment.start);
        const end = new library.Vector3().fromArray(segment.end);
        const direction = end.clone().sub(start);
        const cylinder = group.children[0];
        cylinder.position.copy(start).add(end).multiplyScalar(.5);
        cylinder.scale.set(radius, direction.length(), radius);
        cylinder.quaternion.setFromUnitVectors(new library.Vector3(0, 1, 0), direction.normalize());
        group.children[1].position.copy(start);
        group.children[2].position.copy(end);
        group.children[1].scale.setScalar(radius);
        group.children[2].scale.setScalar(radius);
      });
      this.sync(this.pads, snapshot.gripper ? snapshot.gripper.pads || [] : [], () => this.mesh(this.geometry.box, this.materials.pad), (mesh, pad) => {
        mesh.position.fromArray(pad.position);
        mesh.scale.fromArray(pad.size);
        mesh.quaternion.fromArray(pad.quaternion || [0, 0, 0, 1]);
      });
      this.sync(this.contacts, snapshot.contacts || [], () => this.mesh(this.geometry.sphere, this.materials.contact), (mesh, contact) => {
        mesh.position.fromArray(contact.position);
        mesh.scale.setScalar(radius * .6);
      });
      this.sync(this.points, snapshot.points || [], () => this.mesh(this.geometry.sphere, this.materials.point), (mesh, point) => {
        mesh.position.fromArray(point.position);
        mesh.scale.setScalar(radius * .4);
        mesh.visible = this.showPoints || snapshot.physical === false;
      });
      this.table.visible = Boolean(snapshot.table);
      if (snapshot.table) { this.table.position.fromArray(snapshot.table.position); this.table.scale.fromArray(snapshot.table.size); this.table.quaternion.fromArray(snapshot.table.quaternion || [0, 0, 0, 1]); }
      this.anchor.visible = Boolean(snapshot.anchor && snapshot.anchor.enabled);
      if (this.anchor.visible) { this.anchor.position.fromArray(snapshot.anchor.position); this.anchor.scale.setScalar(radius * 1.5); }
      if (this.needsFrame && (snapshot.segments || []).length) { this.home(); this.needsFrame = false; }
    }
    home() {
      const positions = this.snapshot ? (this.snapshot.segments || []).flatMap(segment => [segment.start, segment.end]) : [];
      const center = new this.library.Vector3(.15, 0, .1);
      let distance = .7;
      if (positions.length) {
        const bounds = new this.library.Box3().setFromPoints(positions.map(position => new this.library.Vector3().fromArray(position)));
        bounds.getCenter(center);
        distance = Math.max(.35, bounds.getSize(new this.library.Vector3()).length() * 2);
      }
      this.controls.target.copy(center);
      this.camera.position.copy(center).add(new this.library.Vector3(.5, -.8, .55).multiplyScalar(distance));
      this.controls.update();
    }
    destroy() {
      window.cancelAnimationFrame(this.animation);
      this.resizeObserver.disconnect();
      this.controls.dispose();
      Object.values(this.geometry).forEach(geometry => geometry.dispose());
      Object.values(this.materials).forEach(material => material.dispose());
      this.renderer.dispose();
      this.renderer.domElement.remove();
    }
  }

  // %% canvas projection when WebGL is unavailable
  class Projection {
    /** Project world coordinates onto an isometric orthographic image. */
    static point(position) {
      return [(position[0] + position[1]) / Math.sqrt(2), (position[0] - position[1] - 2 * position[2]) / Math.sqrt(6)];
    }
    static corners(box) {
      const quaternion = box.quaternion || [0, 0, 0, 1];
      const corners = [];
      for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
        const vertex = [x * box.size[0] / 2, y * box.size[1] / 2, z * box.size[2] / 2];
        const cross = [quaternion[1] * vertex[2] - quaternion[2] * vertex[1], quaternion[2] * vertex[0] - quaternion[0] * vertex[2], quaternion[0] * vertex[1] - quaternion[1] * vertex[0]];
        const rotation = [quaternion[1] * cross[2] - quaternion[2] * cross[1], quaternion[2] * cross[0] - quaternion[0] * cross[2], quaternion[0] * cross[1] - quaternion[1] * cross[0]];
        corners.push(vertex.map((value, index) => value + 2 * (quaternion[3] * cross[index] + rotation[index]) + box.position[index]));
      }
      return corners;
    }
  }

  class CanvasViewport {
    /** Display measured cable and box geometry without a graphics context. */
    constructor(host) {
      this.host = host;
      this.canvas = host.ownerDocument.createElement('canvas');
      this.context = this.canvas.getContext('2d');
      this.host.prepend(this.canvas);
      this.snapshot = {};
      this.showPoints = false;
      this.needsFrame = true;
      this.center = [0, 0];
      this.span = .6;
      this.resizeObserver = new ResizeObserver(() => this.draw());
      this.resizeObserver.observe(host);
    }
    pixel(position) {
      const projected = Projection.point(position);
      return [this.host.clientWidth / 2 + (projected[0] - this.center[0]) * this.scale, this.host.clientHeight / 2 + (projected[1] - this.center[1]) * this.scale];
    }
    line(start, end, color, width) {
      this.context.beginPath();
      this.context.moveTo(...this.pixel(start));
      this.context.lineTo(...this.pixel(end));
      this.context.strokeStyle = color;
      this.context.lineWidth = width;
      this.context.lineCap = 'round';
      this.context.stroke();
    }
    circle(position, radius, color) {
      this.context.beginPath();
      this.context.arc(...this.pixel(position), radius, 0, 2 * Math.PI);
      this.context.fillStyle = color;
      this.context.fill();
    }
    box(box, colors) {
      const corners = Projection.corners(box);
      const faces = [[0, 2, 6, 4], [0, 4, 5, 1], [4, 6, 7, 5], [1, 5, 7, 3]];
      faces.forEach((face, index) => {
        this.context.beginPath();
        face.forEach((corner, vertex) => this.context[vertex ? 'lineTo' : 'moveTo'](...this.pixel(corners[corner])));
        this.context.closePath();
        this.context.fillStyle = colors[index % colors.length];
        this.context.fill();
        this.context.strokeStyle = '#8db1c230';
        this.context.lineWidth = 1;
        this.context.stroke();
      });
    }
    home() {
      const positions = (this.snapshot.segments || []).flatMap(segment => [segment.start, segment.end]);
      for (const box of [this.snapshot.table, ...((this.snapshot.gripper || {}).pads || [])]) {
        if (box) positions.push(...Projection.corners(box));
      }
      if (!positions.length) return;
      const projected = positions.map(position => Projection.point(position));
      const minimum = [0, 1].map(axis => Math.min(...projected.map(point => point[axis])));
      const maximum = [0, 1].map(axis => Math.max(...projected.map(point => point[axis])));
      this.center = minimum.map((value, axis) => (value + maximum[axis]) / 2);
      this.span = Math.max(.25, maximum[0] - minimum[0], maximum[1] - minimum[1]);
      this.draw();
    }
    render(snapshot) {
      this.snapshot = snapshot;
      if (this.needsFrame && (snapshot.segments || []).length) { this.home(); this.needsFrame = false; }
      this.draw();
    }
    draw() {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = this.host.clientWidth * ratio;
      this.canvas.height = this.host.clientHeight * ratio;
      this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.scale = Math.min(this.host.clientWidth - 90, this.host.clientHeight - 130) / this.span;
      if (this.snapshot.table) this.box(this.snapshot.table, ['#293742', '#304452', '#405869']);
      const radius = this.snapshot.config ? this.snapshot.config.radius : .005;
      const configured = this.snapshot.config && this.snapshot.config.color;
      const components = configured ? Array.isArray(configured) ? configured : [configured.R, configured.G, configured.B] : null;
      const color = components ? 'rgb(' + components.slice(0, 3).map(value => Math.round(value * 255)).join(',') + ')' : '#ee5867';
      for (const segment of this.snapshot.segments || []) this.line(segment.start, segment.end, color, Math.max(2, 2 * radius * this.scale));
      for (const pad of (this.snapshot.gripper || {}).pads || []) this.box(pad, ['#20756d', '#32ac9e', '#61d9ca']);
      if (this.snapshot.anchor && this.snapshot.anchor.enabled) this.circle(this.snapshot.anchor.position, Math.max(3, radius * this.scale * 1.5), '#e1b472');
      if (this.showPoints || this.snapshot.physical === false) for (const point of this.snapshot.points || []) this.circle(point.position, 2.5, '#ffffff');
      for (const contact of this.snapshot.contacts || []) this.circle(contact.position, 3, '#ffc16c');
    }
    destroy() { this.resizeObserver.disconnect(); this.canvas.remove(); }
  }

  // %% controls and status display
  class Application {
    /** Keep experiment controls usable even when WebGL is unavailable. */
    constructor(document, browser) {
      this.document = document;
      this.browser = browser;
      this.form = document.getElementById('cable-config');
      this.view = null;
      this.busy = false;
      this.active = true;
      this.polling = false;
      this.snapshot = {state: 'idle'};
      this.commandError = '';
      this.viewNotice = '';
      this.configurationSeen = false;
      this.configurationEdited = false;
      this.controller = new Controller(new Client(browser.fetch.bind(browser)), snapshot => this.render(snapshot));
      try { this.view = new Viewport(this.element('viewport'), browser.THREE); }
      catch (error) {
        this.view = new CanvasViewport(this.element('viewport'));
        this.viewNotice = '2D view · WebGL unavailable. Geometry and forces come from the live simulation.';
        this.element('viewport').setAttribute('aria-label', 'Orthographic 2D view of the simulated cable');
        this.document.querySelector('.view-hint').textContent = 'Orthographic projection · actual simulated geometry';
      }
      this.bind();
      this.poll();
    }
    element(name) { return this.document.getElementById(name); }
    notice(name, message) { this.element(name).textContent = message; this.element(name).hidden = !message; }
    format(value, unit) { return Number.isFinite(value) ? value.toFixed(2) + ' ' + unit : '—'; }
    render(snapshot) {
      this.snapshot = snapshot;
      if (snapshot.config && !this.configurationSeen) {
        if (!this.configurationEdited) { Configuration.apply(this.form, snapshot.config); this.updateConfiguration(); }
        this.configurationSeen = true;
      }
      const feedback = Feedback.read(snapshot);
      this.element('simulation-status').textContent = feedback.status;
      this.element('simulation-status').dataset.state = snapshot.state;
      this.element('grasp-status').textContent = feedback.grasp;
      this.element('reading-time').textContent = this.format(snapshot.simulation_time, 's');
      this.element('reading-normal').textContent = this.format(feedback.normalForce, 'N');
      this.element('reading-friction').textContent = this.format(feedback.frictionForce, 'N');
      this.element('reading-contacts').textContent = feedback.contactCount === null ? '—' : feedback.contactCount;
      this.element('view-empty').hidden = Boolean((snapshot.segments || []).length || (snapshot.points || []).length);
      this.element('view-caption').textContent = snapshot.physical === true ? 'Live MuJoCo poses · meters' : snapshot.physical === false ? 'Semantic placeholders · physics unavailable' : 'Waiting for the simulator';
      this.notice('lab-error', snapshot.error || this.commandError);
      this.notice('lab-warning', [this.viewNotice, feedback.warning].filter(Boolean).join(' '));
      if (this.view) this.view.render(snapshot);
      this.updateButtons();
    }
    updateButtons() {
      const feedback = Feedback.read(this.snapshot);
      this.document.querySelectorAll('[data-action]').forEach(button => { button.disabled = this.busy || !feedback.canCommand; });
      this.element('start-button').disabled = this.busy;
      this.element('start-button').textContent = this.busy ? 'Working…' : 'Build & start';
      this.element('reset-button').disabled = this.busy || !this.snapshot.config;
      this.element('stop-button').disabled = this.busy || this.snapshot.state !== 'running';
    }
    async perform(operation) {
      if (this.busy) return;
      this.busy = true;
      this.commandError = '';
      this.notice('lab-error', '');
      this.updateButtons();
      try { await operation(); }
      catch (error) { this.commandError = error.message; this.notice('lab-error', this.commandError); }
      finally { this.busy = false; this.updateButtons(); }
    }
    updateConfiguration() {
      const config = Configuration.fromForm(this.form);
      this.element('friction-value').value = config.friction[0].toFixed(2);
      this.element('cable-dimension').textContent = (config.segment_count * config.segment_length).toFixed(2) + ' m cable · ' + config.segment_count + ' elements';
    }
    preset(anchored) {
      this.configurationEdited = true;
      this.form.reset();
      this.form.elements.anchor_to_parent.checked = anchored;
      this.updateConfiguration();
    }
    bind() {
      this.form.addEventListener('submit', event => {
        event.preventDefault();
        if (this.view) this.view.needsFrame = true;
        this.perform(() => this.controller.start(Configuration.fromForm(this.form)));
      });
      this.form.addEventListener('input', () => { this.configurationEdited = true; this.updateConfiguration(); });
      this.element('preset-anchored').addEventListener('click', () => this.preset(true));
      this.element('preset-loose').addEventListener('click', () => this.preset(false));
      this.document.querySelectorAll('[data-action]').forEach(button => button.addEventListener('click', () => this.perform(() => this.controller.command(button.dataset.action))));
      this.element('reset-button').addEventListener('click', () => this.perform(() => this.controller.reset()));
      this.element('stop-button').addEventListener('click', () => this.perform(() => this.controller.stop()));
      this.element('camera-home').addEventListener('click', () => { if (this.view) this.view.home(); });
      this.element('show-points').addEventListener('change', event => {
        if (!this.view) return;
        this.view.showPoints = event.target.checked;
        this.view.render(this.snapshot);
      });
      this.browser.addEventListener('pagehide', event => {
        this.active = false;
        this.browser.clearTimeout(this.timer);
        if (!event.persisted && this.view) this.view.destroy();
      });
      this.browser.addEventListener('pageshow', () => {
        if (this.active) return;
        this.active = true;
        this.poll();
      });
    }
    async poll() {
      if (!this.active || this.polling) return;
      this.polling = true;
      try { await this.controller.refresh(); }
      catch (error) {
        this.notice('lab-error', error.message);
        this.element('simulation-status').textContent = 'Connection unavailable';
        this.document.querySelectorAll('[data-action]').forEach(button => { button.disabled = true; });
      }
      finally {
        this.polling = false;
        if (this.active) this.timer = this.browser.setTimeout(() => this.poll(), 100);
      }
    }
  }

  const api = {Routes, Actions, Client, Configuration, Feedback, Controller, Viewport, Projection, CanvasViewport, Application};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') {
    window.CableLab = api;
    if (document.getElementById('cable-config')) new Application(document, window);
  }
})();
