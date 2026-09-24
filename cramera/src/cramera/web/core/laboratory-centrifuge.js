/* Measured rotor speed and remaining simulation time on the authored instrument. */
(function (global, factory) {
  'use strict';
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.LaboratoryCentrifuge = api;
})(typeof window === 'undefined' ? globalThis : window, function () {
  'use strict';

  // %% authored screen and state labels
  const DISPLAY_NAME = 'laboratory-centrifuge-reading';
  const SCREEN_NAMES = new Set(['Centrifuge display', 'Centrifuge_display']);
  const LETTER_NAMES = new Set(['Centrifuge status lettering', 'Centrifuge_status_lettering', 'Centrifuge capacity lettering', 'Centrifuge_capacity_lettering']);
  const CANVAS_SIZE = Object.freeze([768, 192]);
  const SCREEN_INSET = .94;
  const SCREEN_OFFSET = .0001;
  const Status = Object.freeze({UNAVAILABLE: 'NO DATA', OPEN: 'OPEN', LID: 'LID', CHECK: 'CHECK', READY: 'READY', RUNNING: 'RUNNING', BRAKING: 'BRAKING'});

  /** Derive display text from one authoritative observation without advancing a clock. */
  function reading(snapshot) {
    if (!snapshot || !snapshot.available || !Number.isFinite(snapshot.rpm) || !Number.isFinite(snapshot.remainingSeconds)) {
      return {status: Status.UNAVAILABLE, clock: '--:--', speed: '-- RPM', active: false};
    }
    const knowledge = snapshot.knowledge;
    const allowed = snapshot.balanced && (!knowledge || Boolean(knowledge.evaluation && knowledge.evaluation.allowed));
    const status = snapshot.running ? (snapshot.targetRpm > 0 ? Status.RUNNING : Status.BRAKING)
      : snapshot.lidOpen ? Status.OPEN : !snapshot.lidClosed ? Status.LID : allowed ? Status.READY : Status.CHECK;
    const seconds = Math.max(0, Math.ceil(snapshot.remainingSeconds));
    return {
      status,
      clock: String(Math.floor(seconds / 60)).padStart(2, '0') + ':' + String(seconds % 60).padStart(2, '0'),
      speed: Math.round(Math.abs(snapshot.rpm)) + ' RPM',
      active: snapshot.running,
    };
  }

  /** Replace only the static screen lettering, locating the panel from its mesh bounds. */
  class Display {
    constructor(THREE, root, document) {
      this.THREE = THREE;
      this.root = root;
      this.hidden = new Map();
      this.screen = null;
      this.canvas = document.createElement('canvas');
      [this.canvas.width, this.canvas.height] = CANVAS_SIZE;
      this.context = this.canvas.getContext('2d');
      this.texture = new THREE.CanvasTexture(this.canvas);
      this.texture.encoding = THREE.sRGBEncoding;
      this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({map: this.texture, toneMapped: false}));
      this.mesh.name = DISPLAY_NAME;
      this.mesh.rotation.x = Math.PI / 2;
      this.mesh.visible = false;
      root.add(this.mesh);
      this.lastReading = null;
    }

    /** Find late-loaded meshes and fit the screen in laboratory coordinates. */
    attach() {
      this.root.traverse(node => {
        if (LETTER_NAMES.has(node.name) || LETTER_NAMES.has(node.userData.name)) {
          if (!this.hidden.has(node)) this.hidden.set(node, node.visible);
          node.visible = false;
        }
        if (!this.screen && node.geometry && (SCREEN_NAMES.has(node.name) || SCREEN_NAMES.has(node.userData.name))) this.screen = node;
      });
      if (!this.screen || this.mesh.visible) return;
      const THREE = this.THREE;
      this.root.updateMatrixWorld(true);
      const transform = new THREE.Matrix4().copy(this.root.matrixWorld).invert().multiply(this.screen.matrixWorld);
      this.screen.geometry.computeBoundingBox();
      const bounds = this.screen.geometry.boundingBox.clone().applyMatrix4(transform);
      const size = bounds.getSize(new THREE.Vector3());
      const center = bounds.getCenter(new THREE.Vector3());
      this.mesh.geometry.dispose();
      this.mesh.geometry = new THREE.PlaneGeometry(size.x * SCREEN_INSET, size.z * SCREEN_INSET);
      this.mesh.position.set(center.x, bounds.min.y - SCREEN_OFFSET, center.z);
      this.mesh.visible = true;
    }

    /** Paint measured speed and remaining seconds, reusing the existing GPU texture. */
    update(snapshot) {
      this.attach();
      const value = reading(snapshot);
      const signature = JSON.stringify(value);
      if (signature === this.lastReading) return;
      this.lastReading = signature;
      const context = this.context;
      context.fillStyle = '#11292a';
      context.fillRect(0, 0, ...CANVAS_SIZE);
      context.textBaseline = 'middle';
      context.textAlign = 'left';
      context.fillStyle = value.active ? '#cef9dd' : '#bad9ce';
      context.font = '600 38px monospace';
      context.fillText(value.status, 22, 43, 305);
      context.textAlign = 'right';
      context.font = '600 83px monospace';
      context.fillText(value.clock, CANVAS_SIZE[0] - 22, 61, 410);
      context.textAlign = 'left';
      context.fillStyle = '#82c6b5';
      context.font = '600 59px monospace';
      context.fillText(value.speed, 22, 145, CANVAS_SIZE[0] - 44);
      this.texture.needsUpdate = true;
    }

    /** Restore authored lettering and release every owned renderer resource. */
    destroy() {
      for (const [node, visible] of this.hidden) node.visible = visible;
      this.hidden.clear();
      this.root.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.material.dispose();
      this.texture.dispose();
    }
  }

  return {DISPLAY_NAME, reading, create: (THREE, root, document) => new Display(THREE, root, document)};
});
