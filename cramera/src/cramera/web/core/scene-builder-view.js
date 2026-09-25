/* Scene Builder viewport. Model coordinates are metres, Z-up, with URDF RPY. */
(function (global, factory) {
  'use strict';
  const exported = factory(global);
  if (typeof module !== 'undefined' && module.exports) module.exports = exported;
  else Object.assign(global, exported);
})(typeof window !== 'undefined' ? window : globalThis, function (global) {
  'use strict';

  // %% immutable authored transforms
  const ROTATION_ORDER = 'ZYX';
  const DEFAULT_POSITION = [0, 0, 0];
  const DEFAULT_SCALE = [1, 1, 1];
  const DRAG_CLICK_DISTANCE = 5;

  function vector(values, fallback) {
    const result = values === undefined ? fallback : values;
    if (!Array.isArray(result) || result.length !== 3 || !result.every(Number.isFinite)) {
      throw new TypeError('Transforms require three finite numbers.');
    }
    return result;
  }

  function matrix(THREE, transform) {
    const position = new THREE.Vector3().fromArray(vector(transform.position, DEFAULT_POSITION));
    const rotation = vector(transform.rotation, DEFAULT_POSITION);
    const quaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(...rotation, ROTATION_ORDER));
    const scale = vector(transform.scale, DEFAULT_SCALE);
    if (scale.some(value => value <= 0)) throw new TypeError('Scale must be positive.');
    return new THREE.Matrix4().compose(position, quaternion, new THREE.Vector3().fromArray(scale));
  }

  function readTransform(THREE, transform) {
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    transform.decompose(position, quaternion, scale);
    const rotation = new THREE.Euler().setFromQuaternion(quaternion, ROTATION_ORDER);
    return {position:position.toArray(), rotation:[rotation.x, rotation.y, rotation.z].map(value => value === 0 ? 0 : value), scale:scale.toArray()};
  }

  function disposeObject(object) {
    const geometries = new Set();
    const materials = new Set();
    const textures = new Set();
    object.traverse(child => {
      if (child.geometry) geometries.add(child.geometry);
      const assigned = Array.isArray(child.material) ? child.material : [child.material];
      assigned.filter(Boolean).forEach(material => {
        materials.add(material);
        Object.values(material).filter(value => value && value.isTexture).forEach(texture => textures.add(texture));
      });
    });
    textures.forEach(texture => texture.dispose());
    materials.forEach(material => material.dispose());
    geometries.forEach(geometry => geometry.dispose());
  }

  /** An independently editable asset with offsets in each link's parent frame. */
  class SceneBuilderInstance {
    constructor(THREE, specification, root) {
      this.THREE = THREE;
      this.id = specification.id;
      this.assetId = specification.assetId;
      this.root = root;
      this.group = new THREE.Group();
      this.group.name = specification.name;
      this.group.userData.sceneBuilderInstance = this.id;
      this.group.add(root);
      this.authored = new Map();
      Object.entries(root.links || {}).forEach(([name, link]) => {
        link.updateMatrix();
        this.authored.set(name, link.matrix.clone());
        link.userData.sceneBuilderLink = name;
      });
      this.defaultJoints = new Map(Object.entries(root.joints || {}).map(([name, joint]) => [name, joint.jointValue.slice()]));
      this.apply(specification);
    }

    /** Apply a placement without modifying authored transforms or joint axes. */
    apply(specification) {
      const placement = matrix(this.THREE, specification);
      const offsets = new Map();
      Object.entries(specification.linkOffsets || {}).forEach(([name, offset]) => {
        if (!this.authored.has(name)) throw new TypeError('Unknown model link: ' + name);
        offsets.set(name, matrix(this.THREE, offset));
      });
      Object.entries(specification.jointPositions || {}).forEach(([name, value]) => {
        if (!this.root.joints || !this.root.joints[name] || !Number.isFinite(value)) throw new TypeError('Unknown or invalid joint: ' + name);
      });
      placement.decompose(this.group.position, this.group.quaternion, this.group.scale);
      this.group.name = specification.name;
      this.authored.forEach((baseline, name) => {
        const parent_T_link = offsets.has(name) ? offsets.get(name).clone().multiply(baseline) : baseline;
        const link = this.root.links[name];
        parent_T_link.decompose(link.position, link.quaternion, link.scale);
      });
      this.defaultJoints.forEach((value, name) => {
        const desired = Object.prototype.hasOwnProperty.call(specification.jointPositions || {}, name) ? [specification.jointPositions[name]] : value;
        this.root.joints[name].setJointValue(...desired);
      });
      this.group.updateMatrixWorld(true);
    }

    /** Export the editable offset, not the authored mesh or joint origin. */
    transform(linkName = null) {
      const target = this.target(linkName);
      target.updateMatrix();
      const edited = target.matrix.clone();
      if (linkName) edited.multiply(this.authored.get(linkName).clone().invert());
      return readTransform(this.THREE, edited);
    }

    target(linkName = null) {
      if (!linkName) return this.group;
      if (!this.root.links || !this.root.links[linkName]) throw new TypeError('Unknown model link: ' + linkName);
      return this.root.links[linkName];
    }

    structure() {
      const links = Object.entries(this.root.links || {}).map(([name, link]) => {
        let ancestor = link.parent;
        while (ancestor && !ancestor.userData.sceneBuilderLink) ancestor = ancestor.parent;
        return {name, parent:ancestor ? ancestor.userData.sceneBuilderLink : null};
      });
      const joints = Object.entries(this.root.joints || {}).filter(([,joint]) => ['revolute','continuous','prismatic'].includes(joint.jointType)).map(([name,joint]) => ({name, type:joint.jointType, lower:joint.limit.lower, upper:joint.limit.upper, value:joint.jointValue[0]}));
      return {links, joints};
    }
  }

  // %% same-origin authored assets
  /** Load complete assets, preserving their original materials and textures. */
  class SceneBuilderAssetLoader {
    constructor(options = {}) {
      this.THREE = options.THREE || global.THREE;
      this.baseURL = options.baseURL || global.location.href;
      this.URDFLoader = options.URDFLoader || global.URDFLoader;
    }

    resolveURL(path) {
      const resolved = new URL(path, this.baseURL);
      const base = new URL(this.baseURL);
      if (resolved.origin !== base.origin || !['https:', 'http:'].includes(resolved.protocol)) {
        throw new TypeError('Assets must use the Scene Builder server.');
      }
      return resolved.href;
    }

    async load(asset) {
      const THREE = this.THREE;
      const url = this.resolveURL(asset.url || asset.modelUrl);
      const format = (asset.format || new URL(url).pathname.split('.').pop()).toLowerCase();
      const manager = new THREE.LoadingManager();
      // Embedded GLB images are local data; every network request stays on origin.
      manager.setURLModifier(path => /^(blob:|data:)/.test(path) ? path : this.resolveURL(path));
      if (format !== 'urdf') return this.mesh(url, format, manager, false);
      return new Promise((resolve, reject) => {
        let model = null;
        let failed = false;
        const failure = error => { if (!failed) { failed = true; reject(error instanceof Error ? error : new Error('Asset dependency failed: ' + error)); } };
        manager.onError = failure;
        manager.onLoad = () => { if (!failed && model) resolve(model); };
        const loader = new this.URDFLoader(manager);
        loader.packages = asset.packages || {};
        loader.parseCollision = false;
        loader.fetchOptions = {credentials:'same-origin'};
        loader.loadMeshCb = (path, activeManager, done) => {
          const meshFormat = path.split('?')[0].split('.').pop().toLowerCase();
          // URDF completion waits for optional OBJ materials and every mesh callback.
          const dependency = 'scene-builder-mesh:' + path;
          activeManager.itemStart(dependency);
          this.mesh(path, meshFormat, activeManager, true).then(object => {
            done(object); activeManager.itemEnd(dependency);
          }, error => { failure(error); done(null, error); activeManager.itemEnd(dependency); });
        };
        loader.load(url, object => { model = object; }, undefined, failure);
      });
    }

    mesh(url, format, manager, urdf) {
      const THREE = this.THREE;
      return new Promise((resolve, reject) => {
        if (format === 'glb' || format === 'gltf') {
          new THREE.GLTFLoader(manager).load(url, result => resolve(result.scene), undefined, reject);
        } else if (format === 'stl') {
          new THREE.STLLoader(manager).load(url, geometry => resolve(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({color:0xc8d2d3, roughness:.65}))), undefined, reject);
        } else if (format === 'dae') {
          new THREE.ColladaLoader(manager).load(url, result => {
            // All editing uses Z-up. Undo ColladaLoader's standalone Y-up correction.
            result.scene.quaternion.identity();
            result.scene.traverse(child => { if (child.isLight) child.visible = false; });
            resolve(result.scene);
          }, undefined, reject);
        } else if (format === 'obj') {
          const loadObject = materials => {
            const loader = new THREE.OBJLoader(manager);
            if (materials) { materials.preload(); loader.setMaterials(materials); }
            loader.load(url, resolve, undefined, reject);
          };
          // MTL is optional; its absence must not reject an otherwise valid OBJ.
          const materialManager = new THREE.LoadingManager();
          materialManager.setURLModifier(path => this.resolveURL(path));
          new THREE.MTLLoader(materialManager).load(url.replace(/\.obj(?=\?|$)/i, '.mtl'), loadObject, undefined, () => loadObject(null));
        } else reject(new TypeError('Unsupported model format: ' + format));
      });
    }
  }

  // %% editor viewport
  /** Render editing state independently of the live simulation and plan runner. */
  class SceneBuilderView {
    constructor(options) {
      this.options = options;
      this.THREE = options.THREE || global.THREE;
      this.container = options.container;
      this.instances = new Map();
      this.pending = new Map();
      this.document = {instances:[]};
      this.selection = null;
      this.available = false;
      this.disposed = false;
      this.mode = 'translate';
      this.onError = options.onError || (() => {});
      try {
        this.renderer = options.rendererFactory ? options.rendererFactory() : new this.THREE.WebGLRenderer({antialias:true, alpha:false});
      } catch (error) {
        this.onError(error);
        return;
      }
      this.initialize();
      this.available = true;
    }

    initialize() {
      const THREE = this.THREE;
      const renderer = this.renderer;
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0xe8edee);
      this.content = new THREE.Group();
      this.scene.add(this.content);
      this.camera = new THREE.PerspectiveCamera(45, 1, .01, 500);
      this.camera.up.set(0, 0, 1);
      this.camera.position.set(4, -6, 4);
      renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, 2));
      renderer.outputEncoding = THREE.sRGBEncoding;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1;
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.domElement.setAttribute('aria-label', 'Scene Builder 3D viewport');
      this.container.appendChild(renderer.domElement);
      const light = new THREE.DirectionalLight(0xfff5e5, 1.1);
      light.position.set(3, -4, 7); light.castShadow = true;
      light.shadow.mapSize.set(2048, 2048);
      Object.assign(light.shadow.camera, {left:-8, right:8, top:8, bottom:-8});
      light.shadow.normalBias = .015;
      this.scene.add(light, new THREE.HemisphereLight(0xffffff, 0x718089, .9));
      if (THREE.RoomEnvironment) {
        const environment = new THREE.PMREMGenerator(renderer);
        const room = new THREE.RoomEnvironment();
        this.environment = environment.fromScene(room, .04);
        this.scene.environment = this.environment.texture;
        disposeObject(room); environment.dispose();
      }
      const grid = new THREE.GridHelper(20, 40, 0xa2b4b9, 0xc8d2d5);
      grid.rotation.x = Math.PI / 2;
      this.scene.add(grid);
      this.orbit = new THREE.OrbitControls(this.camera, renderer.domElement);
      this.orbit.target.set(0, 0, .7);
      this.orbit.enableDamping = true;
      this.orbit.maxPolarAngle = Math.PI * .99;
      this.gizmo = new THREE.TransformControls(this.camera, renderer.domElement);
      this.scene.add(this.gizmo);
      this.gizmo.addEventListener('dragging-changed', event => { this.orbit.enabled = !event.value; });
      this.gizmo.addEventListener('mouseDown', () => this.beginTransform());
      this.gizmo.addEventListener('objectChange', () => this.changeTransform());
      this.gizmo.addEventListener('mouseUp', () => this.finishTransform());
      this.selectionBox = new THREE.BoxHelper(new THREE.Object3D(), 0x00897b);
      this.selectionBox.visible = false;
      this.selectionBox.material.depthTest = false;
      this.scene.add(this.selectionBox);
      this.assetLoader = this.options.assetLoader || new SceneBuilderAssetLoader({THREE});
      this.pointerDown = event => { this.pointerOrigin = [event.clientX, event.clientY]; this.pointerWasGizmo = Boolean(this.gizmo.axis); };
      this.pointerUp = event => this.pick(event);
      renderer.domElement.addEventListener('pointerdown', this.pointerDown);
      renderer.domElement.addEventListener('pointerup', this.pointerUp);
      this.contextLost = event => { event.preventDefault(); this.available = false; this.onError(new Error('The graphics context was lost. Your scene document is retained. Reload to restore the viewport.')); };
      renderer.domElement.addEventListener('webglcontextlost', this.contextLost);
      this.resize = () => {
        const width = Math.max(this.container.clientWidth, 1);
        const height = Math.max(this.container.clientHeight, 1);
        renderer.setSize(width, height);
        this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
      };
      this.resizeObserver = new global.ResizeObserver(this.resize);
      this.resizeObserver.observe(this.container); this.resize();
      this.animate = () => {
        if (this.disposed) return;
        if (this.available) {
          this.orbit.update();
          if (this.selectionBox.visible) this.selectionBox.update();
          renderer.render(this.scene, this.camera);
        }
        this.frame = global.requestAnimationFrame(this.animate);
      };
      this.frame = global.requestAnimationFrame(this.animate);
    }

    async setDocument(document, assetMap) {
      this.document = document;
      if (!this.available || this.disposed) return;
      const desired = new Map(document.instances.map(instance => [instance.id, instance]));
      this.instances.forEach((instance, id) => {
        if (!desired.has(id) || desired.get(id).assetId !== instance.assetId) this.removeInstance(id);
      });
      const loading = document.instances.map(async specification => {
        const existing = this.instances.get(specification.id);
        if (existing) { existing.apply(specification); return; }
        const pending = this.pending.get(specification.id);
        if (pending && pending.assetId === specification.assetId) return pending.promise;
        const asset = assetMap instanceof Map ? assetMap.get(specification.assetId) : assetMap[specification.assetId];
        if (!asset) { this.onError(new Error('Asset is unavailable: ' + specification.assetId)); return; }
        const task = {assetId:specification.assetId};
        task.promise = this.assetLoader.load(asset).then(root => {
          const current = this.document.instances.find(candidate => candidate.id === specification.id);
          if (this.disposed || this.pending.get(specification.id) !== task || !current || current.assetId !== specification.assetId) { disposeObject(root); return; }
          root.traverse(child => {
            if (!child.isMesh) return;
            if (global.AuthoredMaterials) global.AuthoredMaterials.prepareMesh(child);
            else { child.castShadow = true; child.receiveShadow = true; }
          });
          const instance = new SceneBuilderInstance(this.THREE, current, root);
          this.instances.set(current.id, instance); this.content.add(instance.group);
          if (this.options.onStructure) this.options.onStructure(current.id, instance.structure());
          if (this.selection && this.selection.instanceId === current.id) this.setSelection(current.id, this.selection.linkName);
        }).catch(error => this.onError(error)).finally(() => { if (this.pending.get(specification.id) === task) this.pending.delete(specification.id); });
        this.pending.set(specification.id, task);
        return task.promise;
      });
      await Promise.all(loading);
      if (this.selection && !this.document.instances.some(instance => instance.id === this.selection.instanceId)) this.setSelection(null);
    }

    removeInstance(id) {
      const instance = this.instances.get(id);
      if (!instance) return;
      if (this.selection && this.selection.instanceId === id) this.gizmo.detach();
      this.content.remove(instance.group); disposeObject(instance.group); this.instances.delete(id);
    }

    setSelection(instanceId, linkName = null) {
      this.selection = instanceId ? {instanceId, linkName} : null;
      if (!this.available) return;
      const instance = this.instances.get(instanceId);
      const target = instance ? instance.target(linkName) : null;
      if (target && this.gizmo.object === target) return;
      this.gizmo.detach(); this.selectionBox.visible = false;
      if (!instance) return;
      this.gizmo.attach(target); this.gizmo.setMode(this.mode);
      this.selectionBox.setFromObject(target); this.selectionBox.visible = true;
    }

    setMode(mode) {
      if (!['translate', 'rotate'].includes(mode)) throw new TypeError('Unknown transform mode.');
      this.mode = mode;
      if (this.gizmo) this.gizmo.setMode(mode);
    }

    setSnap(enabled) {
      if (!this.gizmo) return;
      this.gizmo.setTranslationSnap(enabled ? .05 : null);
      this.gizmo.setRotationSnap(enabled ? Math.PI / 12 : null);
    }

    getStructure(instanceId) {
      const instance = this.instances.get(instanceId);
      return instance ? instance.structure() : {links:[], joints:[]};
    }

    beginTransform() { this.transformChanged = false; }

    changeTransform() { this.transformChanged = true; }

    /** Commit a drag as one history step while rendering intermediate poses. */
    finishTransform() {
      if (!this.transformChanged) return;
      this.transformChanged = false;
      this.emitTransform();
    }

    emitTransform() {
      if (!this.selection || !this.options.onTransform) return;
      const instance = this.instances.get(this.selection.instanceId);
      this.options.onTransform(Object.assign({}, this.selection, instance.transform(this.selection.linkName)));
    }

    pick(event) {
      if (!this.pointerOrigin || this.pointerWasGizmo || this.gizmo.dragging || event.button !== 0) return;
      if (Math.hypot(event.clientX - this.pointerOrigin[0], event.clientY - this.pointerOrigin[1]) > DRAG_CLICK_DISTANCE) return;
      const rectangle = this.renderer.domElement.getBoundingClientRect();
      const pointer = new this.THREE.Vector2(2 * (event.clientX - rectangle.left) / rectangle.width - 1, 1 - 2 * (event.clientY - rectangle.top) / rectangle.height);
      const ray = new this.THREE.Raycaster(); ray.setFromCamera(pointer, this.camera);
      const hit = ray.intersectObjects(this.content.children, true)[0];
      if (!hit) return;
      let current = hit.object;
      let linkName = null;
      while (current && !current.userData.sceneBuilderInstance) {
        if (!linkName && current.userData.sceneBuilderLink) linkName = current.userData.sceneBuilderLink;
        current = current.parent;
      }
      if (!current) return;
      // A normal click selects the entire model; Alt-click selects an authored link.
      const selectedLink = event.altKey ? linkName : null;
      this.setSelection(current.userData.sceneBuilderInstance, selectedLink);
      if (this.options.onSelect) this.options.onSelect(current.userData.sceneBuilderInstance, selectedLink);
    }

    frameSelection() {
      const instance = this.selection && this.instances.get(this.selection.instanceId);
      if (instance) this.frameObject(instance.target(this.selection.linkName));
    }

    frameAll() { if (this.available) this.frameObject(this.content); }

    frameObject(object) {
      const bounds = new this.THREE.Box3().setFromObject(object);
      if (bounds.isEmpty()) return;
      const center = bounds.getCenter(new this.THREE.Vector3());
      const size = bounds.getSize(new this.THREE.Vector3());
      const radius = Math.max(size.length() / 2, .15);
      const direction = this.camera.position.clone().sub(this.orbit.target).normalize();
      const field = Math.min(this.camera.fov * Math.PI / 180, 2 * Math.atan(Math.tan(this.camera.fov * Math.PI / 360) * this.camera.aspect));
      this.camera.position.copy(center).addScaledVector(direction, radius / Math.sin(field / 2) * 1.15);
      this.orbit.target.copy(center); this.orbit.update();
    }

    dispose() {
      this.disposed = true;
      if (!this.renderer) return;
      global.cancelAnimationFrame(this.frame);
      this.resizeObserver.disconnect(); this.orbit.dispose(); this.gizmo.dispose();
      this.renderer.domElement.removeEventListener('pointerdown', this.pointerDown);
      this.renderer.domElement.removeEventListener('pointerup', this.pointerUp);
      this.renderer.domElement.removeEventListener('webglcontextlost', this.contextLost);
      disposeObject(this.scene);
      if (this.environment) this.environment.dispose();
      this.renderer.dispose(); this.renderer.domElement.remove();
    }
  }

  return {SceneBuilderView, SceneBuilderInstance, SceneBuilderAssetLoader};
});
