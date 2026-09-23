/* Authoritative reagent levels, dosing display and continuous nozzle stream. */
(function (global, factory) {
  'use strict';
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.LaboratoryPipetting = api;
})(typeof window === 'undefined' ? globalThis : window, function () {
  'use strict';

  // %% measured dosing display
  const CANVAS_SIZE = Object.freeze([640, 240]);
  const STATUS = Object.freeze({idle: 'READY', dispensing: 'DISPENSING', complete: 'COMPLETE', blocked: 'STOPPED'});
  const STREAM_RADIUS = .00065;
  const INITIAL_RESERVOIR_ML = 50;

  class Display {
    /** Keep all graphics in the same Z-up metric coordinates as measured bodies. */
    constructor(THREE, root, metadata, document) {
      this.THREE = THREE;
      this.root = root;
      this.metadata = metadata;
      this.reservoirs = new Map();
      this.canvas = document.createElement('canvas');
      [this.canvas.width, this.canvas.height] = CANVAS_SIZE;
      this.context = this.canvas.getContext('2d');
      this.texture = new THREE.CanvasTexture(this.canvas);
      this.texture.encoding = THREE.sRGBEncoding;
      const size = metadata.displaySize || [.078, .028];
      this.screen = new THREE.Mesh(new THREE.PlaneGeometry(...size), new THREE.MeshBasicMaterial({map: this.texture, toneMapped: false}));
      this.screen.name = 'laboratory-pipetting-reading';
      this.screen.rotation.x = Math.PI / 2;
      this.screen.position.fromArray(metadata.displayPosition || [0,0,0]);
      this.screen.position.y -= .0008;
      this.screen.visible = Boolean(metadata.displayPosition);
      this.stream = new THREE.Mesh(new THREE.CylinderGeometry(STREAM_RADIUS, STREAM_RADIUS, 1, 16), new THREE.MeshPhysicalMaterial({color: 0xd5b556, roughness: .1, transmission: .3, transparent: true, opacity: .8}));
      this.stream.name = 'laboratory-pipetting-stream';
      this.stream.visible = false;
      root.add(this.screen, this.stream);
      this.signature = null;
    }

    /** Replace authored initial fills with reusable meshes scaled about their bottoms. */
    updateReservoirs(snapshot) {
      const THREE = this.THREE;
      this.root.updateMatrixWorld(true);
      for (const channel of this.metadata.channels || []) {
        let rendered = this.reservoirs.get(channel.id);
        if (!rendered) {
          let authored = null;
          const names = [channel.liquidMesh, channel.liquidMesh.replaceAll(' ', '_')];
          this.root.traverse(node => { if (node.geometry && names.includes(node.name)) authored = node; });
          if (!authored) continue;
          const matrix = new THREE.Matrix4().copy(this.root.matrixWorld).invert().multiply(authored.matrixWorld);
          authored.geometry.computeBoundingBox();
          const bounds = authored.geometry.boundingBox.clone().applyMatrix4(matrix);
          const mesh = new THREE.Mesh(authored.geometry, authored.material);
          mesh.name = 'laboratory-pipetting-reservoir-' + channel.id;
          mesh.matrixAutoUpdate = false;
          rendered = {mesh, authored, visible: authored.visible, matrix, bottom: bounds.min.z};
          this.reservoirs.set(channel.id, rendered);
          authored.visible = false;
          this.root.add(mesh);
        }
        const reservoir = snapshot && snapshot.reservoirs.find(item => item.id === channel.id);
        const fraction = reservoir ? Math.max(0, Math.min(1, reservoir.volumeMl / (reservoir.capacityMl || INITIAL_RESERVOIR_ML))) : 0;
        rendered.mesh.visible = Boolean(reservoir && fraction > 0);
        rendered.mesh.matrix.makeTranslation(0, 0, rendered.bottom)
          .multiply(new THREE.Matrix4().makeScale(1, 1, fraction))
          .multiply(new THREE.Matrix4().makeTranslation(0, 0, -rendered.bottom))
          .multiply(rendered.matrix);
        rendered.mesh.matrixWorldNeedsUpdate = true;
      }
    }

    /** Show only server-confirmed flow down to the receiving free surface. */
    updateStream(snapshot, liquid, objects) {
      this.stream.visible = false;
      if (!snapshot || !snapshot.available || snapshot.state !== 'dispensing' || !liquid || !objects) return;
      const tube = liquid.tubes[snapshot.key];
      const pose = objects[snapshot.key];
      if (!tube || !pose) return;
      const THREE = this.THREE;
      const channel = (this.metadata.channels || []).find(item => item.id === snapshot.source);
      const nozzle = channel && channel.nozzlePosition || snapshot.nozzlePosition;
      if (!nozzle) return;
      const normal = tube.normal || [0, 0, 1];
      if (Math.abs(normal[2]) < .5) return;
      const start = new THREE.Vector3().fromArray(nozzle);
      const height = tube.offset / normal[2];
      const end = new THREE.Vector3(0, 0, height).applyQuaternion(new THREE.Quaternion().fromArray(pose.slice(3))).add(new THREE.Vector3().fromArray(pose));
      end.x = start.x; end.y = start.y;
      const direction = new THREE.Vector3().subVectors(end, start);
      const length = direction.length();
      if (!Number.isFinite(length) || length <= 0 || end.z >= start.z) return;
      this.stream.position.copy(start).add(end).multiplyScalar(.5);
      this.stream.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
      this.stream.scale.set(1, length, 1);
      const reservoir = snapshot.reservoirs.find(item => item.id === snapshot.source);
      if (reservoir && reservoir.color) this.stream.material.color.fromArray(reservoir.color);
      this.stream.visible = true;
    }

    /** Paint measured progress without estimating elapsed time in the browser. */
    update(snapshot, liquid, objects) {
      const valid = snapshot && snapshot.available;
      const status = valid ? STATUS[snapshot.state] || 'CHECK' : 'NO DATA';
      const reading = valid ? Number(snapshot.deliveredMl || 0).toFixed(2) + ' / ' + Number(snapshot.requestedMl || 0).toFixed(2) + ' ml' : '-- / -- ml';
      const signature = status + reading;
      if (signature !== this.signature) {
        this.signature = signature;
        const context = this.context;
        context.fillStyle = '#102d30'; context.fillRect(0, 0, ...CANVAS_SIZE);
        context.textAlign = 'left'; context.textBaseline = 'middle';
        context.fillStyle = '#95d8c9'; context.font = '600 42px monospace';
        context.fillText(status, 22, 62, 595);
        context.fillStyle = '#edfff8'; context.font = '600 54px monospace';
        context.fillText(reading, 22, 159, 595);
        this.texture.needsUpdate = true;
      }
      this.updateReservoirs(valid ? snapshot : null);
      this.updateStream(snapshot, liquid, objects);
    }

    /** Release owned GPU resources and restore the authored reservoir visibility. */
    destroy() {
      this.root.remove(this.screen, this.stream);
      this.screen.geometry.dispose(); this.screen.material.dispose(); this.texture.dispose();
      this.stream.geometry.dispose(); this.stream.material.dispose();
      for (const rendered of this.reservoirs.values()) {
        this.root.remove(rendered.mesh);
        rendered.authored.visible = rendered.visible;
      }
      this.reservoirs.clear();
    }
  }
  return {create: (THREE, root, metadata, document) => new Display(THREE, root, metadata, document)};
});
