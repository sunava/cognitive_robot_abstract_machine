/* Validated, portable scene drafts and reversible authoring operations. */
(function () {
  'use strict';

  class SceneDocumentError extends Error {}

  class SceneBuilderState {
    static VERSION = 1;
    static MAX_INSTANCES = 500;
    static HISTORY_LIMIT = 100;
    static MAX_COORDINATE = 10000;
    static MAX_SCALE = 1000;
    static POINT_CLOUD_DEFAULTS = Object.freeze({visible: true, locked: true, pointSize: .006, opacity: .65});
    static MIN_POINT_SIZE = .0001;
    static MAX_POINT_SIZE = .1;
    static MIN_OPACITY = .05;
    static FORBIDDEN_KEYS = new Set(['__proto__', 'prototype', 'constructor']);

    constructor(assets = [], document = SceneBuilderState.empty()) {
      this.assets = new Map(assets.map((asset) => [asset.id, asset]));
      this.current = SceneBuilderState.validate(document);
      this.previous = [];
      this.following = [];
      this.listeners = new Set();
      this.sequence = 0;
    }

    /** Return a standalone JSON value, never a reference into the history. */
    static copy(value) { return JSON.parse(JSON.stringify(value)); }

    static empty(name = 'Untitled scene') { return {version: this.VERSION, name, instances: []}; }

    static record(value, label, allowed = null) {
      if (!value || typeof value !== 'object' || Array.isArray(value)) throw new SceneDocumentError(`${label} must be an object.`);
      for (const key of Object.keys(value)) {
        if (this.FORBIDDEN_KEYS.has(key) || (allowed && !allowed.includes(key))) throw new SceneDocumentError(`Unsupported ${label} field: ${key}.`);
      }
      return value;
    }

    static text(value, label, maximum = 160) {
      if (typeof value !== 'string' || !value.trim() || value.length > maximum || /[\u0000-\u001f]/.test(value)) throw new SceneDocumentError(`${label} must be a nonempty string up to ${maximum} characters.`);
      return value;
    }

    static vector(value, label) {
      if (!Array.isArray(value) || value.length !== 3 || value.some((number) => typeof number !== 'number' || !Number.isFinite(number) || Math.abs(number) > this.MAX_COORDINATE)) throw new SceneDocumentError(`${label} must contain three finite numbers.`);
      return [...value];
    }

    static pose(value, label) {
      this.record(value, label, ['position', 'rotation']);
      return {position: this.vector(value.position, `${label} position`), rotation: this.vector(value.rotation, `${label} rotation`)};
    }

    static pointCloud(settings) {
      this.record(settings, 'point cloud', ['visible', 'locked', 'pointSize', 'opacity']);
      const result = {...this.POINT_CLOUD_DEFAULTS, ...settings};
      if (typeof result.visible !== 'boolean' || typeof result.locked !== 'boolean') throw new SceneDocumentError('Reference visibility and lock must be booleans.');
      if (typeof result.pointSize !== 'number' || !Number.isFinite(result.pointSize) || result.pointSize < this.MIN_POINT_SIZE || result.pointSize > this.MAX_POINT_SIZE) throw new SceneDocumentError('Point size must be between 0.1 and 100 mm.');
      if (typeof result.opacity !== 'number' || !Number.isFinite(result.opacity) || result.opacity < this.MIN_OPACITY || result.opacity > 1) throw new SceneDocumentError('Reference opacity must be between 0.05 and 1.');
      return result;
    }

    static assertEditable(instance) {
      if (instance.pointCloud?.locked) throw new SceneDocumentError('Unlock this reference to change its placement or scale.');
    }

    /** Reject unknown versions and malformed values before recording any edit. */
    static validate(document) {
      this.record(document, 'document', ['version', 'name', 'instances', 'sourceScene', 'notices']);
      if (document.version !== this.VERSION) throw new SceneDocumentError(`Unsupported scene version. Expected ${this.VERSION}.`);
      this.text(document.name, 'Scene name');
      if (!Array.isArray(document.instances) || document.instances.length > this.MAX_INSTANCES) throw new SceneDocumentError(`A scene supports up to ${this.MAX_INSTANCES} instances.`);
      const identities = new Set();
      const instances = document.instances.map((instance) => {
        this.record(instance, 'instance', ['id', 'assetId', 'name', 'position', 'rotation', 'scale', 'linkOffsets', 'jointPositions', 'pointCloud']);
        this.text(instance.id, 'Instance id', 96);
        if (!/^[A-Za-z0-9_-]+$/.test(instance.id)) throw new SceneDocumentError('Invalid instance id.');
        this.text(instance.assetId, 'Asset id', 2048);
        this.text(instance.name, 'Instance name');
        if (identities.has(instance.id)) throw new SceneDocumentError(`Duplicate instance id: ${instance.id}.`);
        identities.add(instance.id);
        const linkOffsets = {};
        this.record(instance.linkOffsets ?? {}, 'link offsets');
        for (const [link, pose] of Object.entries(instance.linkOffsets ?? {})) {
          this.text(link, 'Link name', 256);
          linkOffsets[link] = this.pose(pose, `Link ${link}`);
        }
        const jointPositions = {};
        this.record(instance.jointPositions ?? {}, 'joint positions');
        for (const [joint, value] of Object.entries(instance.jointPositions ?? {})) {
          this.text(joint, 'Joint name', 256);
          if (typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > this.MAX_COORDINATE) throw new SceneDocumentError(`Joint ${joint} must be finite.`);
          jointPositions[joint] = value;
        }
        const scale = this.vector(instance.scale ?? [1, 1, 1], 'Scale');
        if (scale.some((value) => value <= 0 || value > this.MAX_SCALE)) throw new SceneDocumentError('Scale values must be positive.');
        const result = {id: instance.id, assetId: instance.assetId, name: instance.name, position: this.vector(instance.position, 'Position'), rotation: this.vector(instance.rotation, 'Rotation'), scale, linkOffsets, jointPositions};
        if (instance.pointCloud !== undefined) result.pointCloud = this.pointCloud(instance.pointCloud);
        return result;
      });
      const result = {version: this.VERSION, name: document.name, instances, notices: []};
      if (document.sourceScene !== undefined) result.sourceScene = this.text(document.sourceScene, 'Source scene', 1024);
      if (document.notices !== undefined) {
        if (!Array.isArray(document.notices) || document.notices.length > this.MAX_INSTANCES) throw new SceneDocumentError('Notices must be an array.');
        result.notices = document.notices.map((notice) => {
          this.record(notice, 'notice', ['name', 'url']);
          const url = this.text(notice.url, 'Notice URL', 2048);
          if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(url) || url.startsWith('//') || url.includes('\\')) throw new SceneDocumentError('Notices must reference local resources.');
          return {name: this.text(notice.name, 'Notice name'), url};
        });
      }
      return result;
    }

    get document() { return SceneBuilderState.copy(this.current); }
    get canUndo() { return this.previous.length > 0; }
    get canRedo() { return this.following.length > 0; }
    instance(id) { return SceneBuilderState.copy(this.current.instances.find((instance) => instance.id === id) || null); }
    subscribe(listener) { this.listeners.add(listener); return () => this.listeners.delete(listener); }
    notify() { for (const listener of this.listeners) listener(this.document); }

    replace(document) {
      const validated = SceneBuilderState.validate(document);
      if (JSON.stringify(validated) === JSON.stringify(this.current)) return;
      this.previous.push(this.current);
      if (this.previous.length > SceneBuilderState.HISTORY_LIMIT) this.previous.shift();
      this.current = validated;
      this.following = [];
      this.notify();
    }

    edit(id, mutation) {
      const document = this.document;
      const instance = document.instances.find((item) => item.id === id);
      if (!instance) throw new SceneDocumentError('Select an object first.');
      mutation(instance);
      this.replace(document);
    }

    nextId() {
      let id;
      do { id = `instance_${Date.now().toString(36)}_${++this.sequence}`; } while (this.current.instances.some((instance) => instance.id === id));
      return id;
    }

    add(assetId, options = {}) {
      const asset = this.assets.get(assetId);
      if (!asset) throw new SceneDocumentError('This asset is unavailable in the catalog.');
      const instance = {id: this.nextId(), assetId, name: options.name ?? asset.name, position: [0, 0, 0], rotation: [0, 0, 0], scale: options.scale ?? [1, 1, 1], linkOffsets: {}, jointPositions: {}};
      if (asset.kind === 'point-cloud') instance.pointCloud = {...SceneBuilderState.POINT_CLOUD_DEFAULTS};
      const notices = new Map(this.current.notices.map((notice) => [notice.url, notice]));
      for (const notice of asset.notices || []) notices.set(notice.url, notice);
      this.replace({...this.document, instances: [...this.document.instances, instance], notices: [...notices.values()]});
      return SceneBuilderState.copy(instance);
    }

    duplicate(id) {
      const instance = this.instance(id);
      if (!instance) throw new SceneDocumentError('Select an object first.');
      instance.id = this.nextId();
      instance.name = `${instance.name.slice(0, 150)} copy`;
      instance.position[0] += 0.25;
      this.replace({...this.document, instances: [...this.document.instances, instance]});
      return instance;
    }

    remove(id) { this.replace({...this.document, instances: this.document.instances.filter((instance) => instance.id !== id)}); }
    rename(name) { this.replace({...this.document, name}); }
    renameInstance(id, name) { this.edit(id, (instance) => { instance.name = name; }); }

    transform(id, pose, linkName = null) {
      SceneBuilderState.record(pose, 'transform', ['position', 'rotation', 'scale']);
      const validated = SceneBuilderState.pose({position: pose.position, rotation: pose.rotation}, 'Transform');
      if (pose.scale !== undefined && linkName) throw new SceneDocumentError('Scale applies to whole objects, not individual URDF parts.');
      if (linkName && SceneBuilderState.FORBIDDEN_KEYS.has(linkName)) throw new SceneDocumentError('Invalid link name.');
      this.edit(id, (instance) => {
        SceneBuilderState.assertEditable(instance);
        if (linkName) instance.linkOffsets[linkName] = validated;
        else Object.assign(instance, validated, pose.scale === undefined ? {} : {scale: pose.scale});
      });
    }

    setScale(id, scale) { this.edit(id, (instance) => { SceneBuilderState.assertEditable(instance); instance.scale = scale; }); }
    setPointCloud(id, changes) {
      SceneBuilderState.record(changes, 'point cloud', ['visible', 'locked', 'pointSize', 'opacity']);
      this.edit(id, (instance) => {
        if (!instance.pointCloud) throw new SceneDocumentError('Select a point cloud reference first.');
        instance.pointCloud = SceneBuilderState.pointCloud({...instance.pointCloud, ...changes});
      });
    }
    resetLink(id, linkName) { this.edit(id, (instance) => { delete instance.linkOffsets[linkName]; }); }
    setJoint(id, joint, value) {
      if (SceneBuilderState.FORBIDDEN_KEYS.has(joint)) throw new SceneDocumentError('Invalid joint name.');
      this.edit(id, (instance) => { instance.jointPositions[joint] = value; });
    }

    undo() {
      if (!this.canUndo) return false;
      this.following.push(this.current);
      this.current = this.previous.pop();
      this.notify();
      return true;
    }

    redo() {
      if (!this.canRedo) return false;
      this.previous.push(this.current);
      this.current = this.following.pop();
      this.notify();
      return true;
    }

    export() { return JSON.stringify(this.current, null, 2) + '\n'; }
  }

  window.SceneDocumentError = SceneDocumentError;
  window.SceneBuilderState = SceneBuilderState;
})();
