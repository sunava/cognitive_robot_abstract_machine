/* Standalone placement editor; authored drafts never start the simulator. */
(function () {
  'use strict';

  class SceneBuilderApi {
    constructor(location, fetcher = window.fetch.bind(window)) {
      this.base = new URL('api/scene-builder/', location);
      this.fetcher = fetcher;
    }

    async request(path, payload = null) {
      const options = {credentials: 'same-origin', cache: 'no-store', headers: {Accept: 'application/json'}};
      if (payload !== null) Object.assign(options, {method: 'POST', body: JSON.stringify(payload), headers: {...options.headers, 'Content-Type': 'application/json'}});
      const response = await this.fetcher(new URL(path, this.base), options);
      return this.readResponse(response);
    }

    async uploadPointCloud(file, format) {
      if (!['ply', 'pcd'].includes(format)) throw new Error('Choose a PLY or PCD point cloud file.');
      const response = await this.fetcher(new URL(`point-clouds/${format}`, this.base), {method: 'POST', credentials: 'same-origin', cache: 'no-store', headers: {Accept: 'application/json', 'Content-Type': 'application/octet-stream'}, body: file});
      return this.readResponse(response);
    }

    async readResponse(response) {
      if (!(response.headers.get('content-type') || '').includes('application/json')) throw new Error('The server did not return a scene document. Please sign in again or reopen Scene Builder.');
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || `The request failed (${response.status}).`);
      return body;
    }
  }

  class SceneBuilderApp {
    static DEGREES = 180 / Math.PI;
    static CATALOG_PAGE_SIZE = 120;
    static MAX_IMPORT_BYTES = 2 * 1024 * 1024;
    static MAX_SCAN_BYTES = 64 * 1024 * 1024;
    static SOURCE_UNITS = Object.freeze({m: 1, cm: .01, mm: .001});
    static TRANSFORM_MODES = Object.freeze(['translate', 'rotate', 'scale']);

    constructor() {
      this.api = new SceneBuilderApi(window.location.href);
      this.state = new window.SceneBuilderState();
      this.assets = new Map();
      this.view = null;
      this.selection = null;
      this.part = null;
      this.draftId = null;
      this.savedDocument = null;
      this.storageKey = `cramera.scene-builder.v1:${window.location.pathname}`;
      this.renderSequence = 0;
      this.sceneRevision = 0;
      this.linkedScale = true;
      this.mode = 'translate';
      this.uploadingScan = false;
      this.bind();
      this.state.subscribe(() => { this.render(); this.storeRecovery(); });
    }

    element(id) { return document.getElementById(id); }
    status(message, error = false) { this.element('status').textContent = message; this.element('status').dataset.error = String(error); }
    attempt(action) {
      try { return Promise.resolve(action()).catch((error) => this.status(error.message, true)); }
      catch (error) { this.status(error.message, true); return Promise.resolve(); }
    }

    bind() {
      const click = (id, action) => this.element(id).addEventListener('click', () => this.attempt(action));
      const change = (id, action) => this.element(id).addEventListener('change', () => this.attempt(action));
      click('new-scene', () => { this.sceneRevision += 1; this.draftId = null; this.savedDocument = null; this.state.replace(window.SceneBuilderState.empty()); this.select(null); this.status('New scene. Add assets from the library.'); });
      click('undo', () => this.state.undo());
      click('redo', () => this.state.redo());
      click('save', () => this.save(false));
      click('save-copy', () => this.save(true));
      click('load-draft', () => this.loadDraft());
      click('load-template', () => this.loadTemplate());
      click('export', () => this.export());
      click('import', () => this.element('import-file').click());
      change('import-file', () => this.import());
      click('load-scan', () => this.element('scan-file').click());
      change('scan-file', () => this.uploadScan());
      this.element('scale-linked').checked = this.linkedScale;
      change('scale-linked', () => { this.linkedScale = this.element('scale-linked').checked; this.view?.setScaleLinked(this.linkedScale); });
      click('scale-half', () => this.multiplyScale(.5));
      click('scale-double', () => this.multiplyScale(2));
      click('scale-reset', () => this.state.setScale(this.selection, [1, 1, 1]));
      change('reference-visible', () => this.state.setPointCloud(this.selection, {visible: this.element('reference-visible').checked}));
      change('reference-locked', () => this.state.setPointCloud(this.selection, {locked: this.element('reference-locked').checked}));
      change('reference-opacity', () => this.state.setPointCloud(this.selection, {opacity: Number(this.element('reference-opacity').value)}));
      change('reference-point-size', () => this.state.setPointCloud(this.selection, {pointSize: Number(this.element('reference-point-size').value) / 1000}));
      change('scene-name', () => this.state.rename(this.element('scene-name').value.trim()));
      change('instance-name', () => this.state.renameInstance(this.selection, this.element('instance-name').value.trim()));
      change('part-select', () => this.select(this.selection, this.element('part-select').value || null));
      change('scene-template', () => { this.element('load-template').disabled = !this.element('scene-template').value; });
      change('draft-list', () => { this.element('load-draft').disabled = !this.element('draft-list').value; });
      click('duplicate', () => { const instance = this.state.duplicate(this.selection); this.select(instance.id); });
      click('delete', () => { this.state.remove(this.selection); this.select(null); });
      click('reset-part', () => this.state.resetLink(this.selection, this.part));
      click('frame-selection', () => this.view?.frameSelection());
      click('frame-all', () => this.view?.frameAll());
      for (const mode of SceneBuilderApp.TRANSFORM_MODES) click(`mode-${mode}`, () => this.setMode(mode));
      for (const axis of ['x', 'y', 'z']) {
        change(`position-${axis}`, () => this.applyPose());
        change(`rotation-${axis}`, () => this.applyPose());
        change(`scale-${axis}`, () => this.applyScale(axis));
      }
      this.element('asset-search').addEventListener('input', () => this.renderLibrary());
      change('asset-category', () => this.renderLibrary());
      document.addEventListener('keydown', (event) => {
        if (event.target.closest('input,select,textarea,[contenteditable]')) return;
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
          event.preventDefault();
          event.shiftKey ? this.state.redo() : this.state.undo();
        }
      });
    }

    async start() {
      const catalog = await this.api.request('catalog');
      this.assets = new Map(catalog.assets.map((asset) => [asset.id, asset]));
      this.state.assets = this.assets;
      this.populate(this.element('scene-template'), catalog.scenes || [], 'Choose a scene…');
      this.restoreRecovery();
      this.renderLibrary();
      this.view = new window.SceneBuilderView({
        container: this.element('viewport'),
        onSelect: (instanceId, linkName) => this.select(instanceId, linkName),
        onTransform: ({instanceId, linkName, position, rotation, scale}) => this.attempt(() => this.state.transform(instanceId, {...{position, rotation}, ...(!linkName && scale !== undefined ? {scale} : {})}, linkName)),
        onStructure: (instanceId) => { if (instanceId === this.selection) this.renderInspector(); },
        onError: (error) => this.status(error.message || String(error), true),
      });
      this.view.setScaleLinked(this.linkedScale);
      if (!this.view.available) {
        this.element('view-fallback').hidden = false;
        this.element('view-error').textContent = 'This browser could not create a WebGL context.';
      }
      this.render();
      await this.refreshDrafts();
      this.status(`${this.assets.size} assets available. Open a scene copy or add assets to arrange them.`);
    }

    populate(select, entries, placeholder, selected = '') {
      select.replaceChildren(new Option(placeholder, ''));
      for (const entry of entries) select.add(new Option(entry.name, entry.id));
      select.value = selected;
    }

    async refreshDrafts() {
      const result = await this.api.request('drafts');
      this.populate(this.element('draft-list'), result.drafts || [], 'Choose a draft…', this.draftId || '');
      this.element('load-draft').disabled = !this.element('draft-list').value;
    }

    async save(copy) {
      const revision = this.sceneRevision;
      const document = this.state.document;
      const result = await this.api.request('drafts', {...(!copy && this.draftId ? {id: this.draftId} : {}), document});
      if (revision !== this.sceneRevision) {
        await this.refreshDrafts();
        this.status('The previous draft was saved. Your current scene is unchanged.');
        return;
      }
      this.draftId = result.id;
      const canonical = window.SceneBuilderState.validate(result.document || document);
      if (JSON.stringify(this.state.document) === JSON.stringify(document)) this.state.replace(canonical);
      this.savedDocument = JSON.stringify(canonical);
      this.storeRecovery();
      await this.refreshDrafts();
      this.renderSaveIndicator();
      this.status(`Saved “${document.name}”. The original source assets are unchanged.`);
    }

    async loadDraft() {
      const id = this.element('draft-list').value;
      if (!id) return;
      const revision = ++this.sceneRevision;
      const result = await this.api.request(`drafts/${encodeURIComponent(id)}`);
      if (revision !== this.sceneRevision) return;
      this.state.replace(result.document);
      this.draftId = result.id;
      this.savedDocument = JSON.stringify(this.state.document);
      this.select(null);
      this.storeRecovery();
      this.renderSaveIndicator();
      this.view?.frameAll();
      this.status(`Loaded “${this.state.document.name}”.`);
    }

    async loadTemplate() {
      const asset = this.element('scene-template').value;
      if (!asset) return;
      const revision = ++this.sceneRevision;
      const document = await this.api.request(`template?asset=${encodeURIComponent(asset)}`);
      if (revision !== this.sceneRevision) return;
      this.state.replace(document);
      this.draftId = null;
      this.savedDocument = null;
      this.select(null);
      this.storeRecovery();
      this.renderSaveIndicator();
      this.view?.frameAll();
      this.status('Opened an editable scene copy. Select a URDF part to rearrange furniture inside the environment.');
    }

    export() {
      const blob = new Blob([this.state.export()], {type: 'application/json'});
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `${this.state.document.name.replace(/[^a-zA-Z0-9_-]+/g, '_') || 'scene'}.scene.json`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
      this.status('Scene JSON exported with asset references and authored poses.');
    }

    async import() {
      const file = this.element('import-file').files[0];
      if (!file) return;
      if (file.size > SceneBuilderApp.MAX_IMPORT_BYTES) throw new Error('Scene documents must be smaller than 2 MB.');
      const revision = ++this.sceneRevision;
      const document = JSON.parse(await file.text());
      if (revision !== this.sceneRevision) return;
      this.state.replace(document);
      this.draftId = null;
      this.savedDocument = null;
      this.select(null);
      this.element('import-file').value = '';
      this.storeRecovery();
      this.renderSaveIndicator();
      const missing = this.state.document.instances.filter((instance) => !this.assets.has(instance.assetId));
      this.status(missing.length ? `Imported scene; ${missing.length} asset references are unavailable in this catalog. Their poses remain editable.` : 'Imported scene. Save a draft to keep a server copy.', missing.length > 0);
    }

    async uploadScan() {
      const file = this.element('scan-file').files[0];
      if (!file || this.uploadingScan) return;
      const format = file.name.split('.').pop().toLowerCase();
      if (!['ply', 'pcd'].includes(format)) throw new Error('Choose a PLY or PCD point cloud file.');
      if (!file.size || file.size > SceneBuilderApp.MAX_SCAN_BYTES) throw new Error('Point cloud files must be nonempty and no larger than 64 MiB.');
      const unitScale = SceneBuilderApp.SOURCE_UNITS[this.element('scan-units').value];
      if (!unitScale) throw new Error('Select the source units for this scan.');
      const revision = this.sceneRevision;
      this.uploadingScan = true;
      this.element('load-scan').disabled = true;
      try {
        this.status('Checking the point cloud…');
        const metadata = window.PointCloudSupport.inspect(await file.arrayBuffer(), format);
        this.status(`Uploading ${metadata.finitePointCount.toLocaleString()} points…`);
        const asset = await this.api.uploadPointCloud(file, format);
        this.assets.set(asset.id, asset);
        this.state.assets = this.assets;
        this.renderLibrary();
        if (revision !== this.sceneRevision) {
          this.status(`Point cloud stored in the asset library. The current scene was left unchanged; its source scale is ${unitScale}.`);
          return;
        }
        const name = file.name.replace(/\.(ply|pcd)$/i, '').replace(/[\u0000-\u001f]/g, '').trim().slice(0, 160) || asset.name;
        const instance = this.state.add(asset.id, {name, scale: [unitScale, unitScale, unitScale]});
        this.select(instance.id);
        this.renderLibrary();
        this.status(`Loaded “${name}” as a locked reference. Stored with your drafts on this server. Save the draft to keep its placement.`);
      } finally {
        this.uploadingScan = false;
        this.element('load-scan').disabled = false;
        this.element('scan-file').value = '';
      }
    }

    storeRecovery() {
      try { window.localStorage.setItem(this.storageKey, JSON.stringify({draftId: this.draftId, document: this.state.document})); }
      catch (_) { /* Server save and JSON export remain available when browser storage is full. */ }
    }

    restoreRecovery() {
      try {
        const recovery = JSON.parse(window.localStorage.getItem(this.storageKey) || 'null');
        if (!recovery) return;
        const validated = window.SceneBuilderState.validate(recovery.document);
        this.draftId = typeof recovery.draftId === 'string' ? recovery.draftId : null;
        this.state.replace(validated);
      } catch (_) { this.status('A browser recovery draft could not be read. Saved server drafts are still available.', true); }
    }

    renderLibrary() {
      const search = this.element('asset-search').value.trim().toLowerCase();
      const category = this.element('asset-category').value;
      const assets = [...this.assets.values()].filter((asset) => (!category || asset.category === category) && `${asset.name} ${asset.id}`.toLowerCase().includes(search));
      this.element('asset-count').textContent = `${assets.length} assets`;
      const library = this.element('asset-library');
      library.replaceChildren();
      for (const asset of assets.slice(0, SceneBuilderApp.CATALOG_PAGE_SIZE)) {
        const button = document.createElement('button');
        button.className = 'asset-card';
        button.title = `Add ${asset.name}\n${asset.id}`;
        const text = document.createElement('span');
        const name = document.createElement('span');
        name.className = 'asset-name';
        name.textContent = asset.name;
        const metadata = document.createElement('span');
        metadata.className = 'asset-meta';
        metadata.textContent = `${asset.category} · ${(asset.format || asset.kind).toUpperCase()}`;
        text.append(name, metadata);
        const add = document.createElement('span');
        add.className = 'asset-add';
        add.textContent = '+';
        add.setAttribute('aria-hidden', 'true');
        button.append(text, add);
        button.addEventListener('click', () => this.attempt(() => {
          const instance = this.state.add(asset.id);
          this.select(instance.id);
          this.status(`Added ${asset.name}. Drag its handles or edit its pose.`);
        }));
        library.append(button);
      }
      if (!assets.length || assets.length > SceneBuilderApp.CATALOG_PAGE_SIZE) {
        const note = document.createElement('p');
        note.className = 'library-note';
        note.textContent = assets.length ? `Showing the first ${SceneBuilderApp.CATALOG_PAGE_SIZE} results. Search to narrow the library.` : 'No matching assets.';
        library.append(note);
      }
    }

    select(instanceId, linkName = null) {
      this.selection = this.state.instance(instanceId) ? instanceId : null;
      this.part = this.selection ? linkName : null;
      if (this.part && this.mode === 'scale') this.setMode('translate');
      this.view?.setSelection(this.selection, this.part);
      this.renderInstances();
      this.renderInspector();
    }

    render() {
      const currentDocument = this.state.document;
      if (document.activeElement !== this.element('scene-name')) this.element('scene-name').value = currentDocument.name;
      this.element('undo').disabled = !this.state.canUndo;
      this.element('redo').disabled = !this.state.canRedo;
      if (!this.state.instance(this.selection)) { this.selection = null; this.part = null; }
      this.renderInstances();
      this.renderInspector();
      this.renderSaveIndicator();
      if (this.view) {
        const sequence = ++this.renderSequence;
        Promise.resolve(this.view.setDocument(currentDocument, this.assets)).then(() => {
          if (sequence !== this.renderSequence) return;
          this.view.setSelection(this.selection, this.part);
          this.renderInspector();
        }).catch((error) => this.status(error.message, true));
      }
    }

    renderSaveIndicator() {
      const saved = this.savedDocument === JSON.stringify(this.state.document);
      this.element('save-indicator').textContent = saved ? 'Saved' : 'Unsaved changes';
    }

    renderInstances() {
      const instances = this.state.document.instances;
      this.element('instance-count').textContent = String(instances.length);
      const list = this.element('instance-list');
      list.replaceChildren();
      for (const instance of instances) {
        const asset = this.assets.get(instance.assetId);
        const button = document.createElement('button');
        button.className = 'instance-item';
        button.setAttribute('aria-selected', String(instance.id === this.selection));
        const icon = document.createElement('span');
        icon.className = 'instance-icon';
        icon.textContent = asset?.category === 'robot' ? 'ROBOT' : (asset?.kind || '?').toUpperCase();
        const name = document.createElement('span');
        name.textContent = instance.name;
        button.append(icon, name);
        button.addEventListener('click', () => this.select(instance.id));
        list.append(button);
      }
    }

    renderInspector() {
      const instance = this.state.instance(this.selection);
      this.element('selection-editor').hidden = !instance;
      this.element('empty-inspector').hidden = Boolean(instance);
      this.renderTransformAvailability(instance);
      if (!instance) return;
      const asset = this.assets.get(instance.assetId);
      this.element('selection-kind').textContent = asset?.category || 'Missing asset';
      this.element('instance-name').value = instance.name;
      const structure = this.view?.getStructure(instance.id) || {links: [], joints: []};
      const partSelect = this.element('part-select');
      partSelect.replaceChildren(new Option('Whole object / robot spawn', ''));
      const linkNames = new Set([...(structure.links || []).map((link) => link.name), ...Object.keys(instance.linkOffsets)]);
      for (const name of linkNames) partSelect.add(new Option(name, name));
      if (this.part && ![...partSelect.options].some((option) => option.value === this.part)) partSelect.add(new Option(this.part, this.part));
      partSelect.value = this.part || '';
      partSelect.disabled = asset?.kind !== 'urdf';
      const pose = this.part ? (instance.linkOffsets[this.part] || {position: [0, 0, 0], rotation: [0, 0, 0]}) : instance;
      this.writeVector('position', pose.position);
      this.writeVector('rotation', pose.rotation.map((value) => value * SceneBuilderApp.DEGREES));
      this.writeVector('scale', instance.scale);
      this.element('scale-fields').hidden = Boolean(this.part);
      this.element('scale-shortcuts').hidden = Boolean(this.part);
      this.renderReference(instance);
      this.element('reset-part').hidden = !this.part;
      this.element('position-reference').textContent = this.part ? 'offset in parent frame' : 'in world';
      this.element('part-help').textContent = this.part ? 'Move this URDF subtree with its attached parts. Joint motion remains relative to the new placement. Alt-click a part in 3D to select it.' : 'The whole model moves together. For robots this sets the spawn pose. Select an environment part to move furniture individually.';
      this.renderJoints(instance, structure.joints || []);
      this.renderNotices(asset);
    }

    renderTransformAvailability(instance) {
      const locked = Boolean(instance?.pointCloud?.locked);
      for (const mode of SceneBuilderApp.TRANSFORM_MODES) this.element(`mode-${mode}`).disabled = !instance || locked || (mode === 'scale' && Boolean(this.part));
      for (const prefix of ['position', 'rotation', 'scale']) {
        for (const axis of ['x', 'y', 'z']) this.element(`${prefix}-${axis}`).disabled = locked;
      }
      for (const identifier of ['scale-half', 'scale-double', 'scale-reset', 'scale-linked']) this.element(identifier).disabled = locked || Boolean(this.part);
    }

    renderReference(instance) {
      this.element('reference-settings').hidden = !instance.pointCloud;
      if (!instance.pointCloud) return;
      const settings = instance.pointCloud;
      this.element('reference-visible').checked = settings.visible;
      this.element('reference-locked').checked = settings.locked;
      if (document.activeElement !== this.element('reference-opacity')) this.element('reference-opacity').value = String(settings.opacity);
      if (document.activeElement !== this.element('reference-point-size')) this.element('reference-point-size').value = String(settings.pointSize * 1000);
    }

    writeVector(prefix, values) {
      ['x', 'y', 'z'].forEach((axis, index) => {
        const input = this.element(`${prefix}-${axis}`);
        if (document.activeElement !== input) input.value = String(Number(values[index].toFixed(6)));
      });
    }

    readVector(prefix) {
      return ['x', 'y', 'z'].map((axis) => {
        const input = this.element(`${prefix}-${axis}`);
        if (!input.value.trim()) throw new Error('Enter a number for each transform field.');
        return Number(input.value);
      });
    }

    applyPose() {
      this.state.transform(this.selection, {position: this.readVector('position'), rotation: this.readVector('rotation').map((value) => value / SceneBuilderApp.DEGREES)}, this.part);
    }

    applyScale(axis) {
      if (this.part) throw new Error('Select the whole object to change its scale.');
      const instance = this.state.instance(this.selection);
      const index = ['x', 'y', 'z'].indexOf(axis);
      const requested = this.readVector('scale');
      const scale = this.linkedScale ? instance.scale.map((value) => value * requested[index] / instance.scale[index]) : requested;
      this.state.setScale(this.selection, scale);
    }

    multiplyScale(multiplier) {
      if (this.part) throw new Error('Select the whole object to change its scale.');
      this.state.setScale(this.selection, this.state.instance(this.selection).scale.map((value) => value * multiplier));
    }

    renderJoints(instance, joints) {
      const editable = joints.filter((joint) => !['fixed', 'floating', 'planar'].includes(joint.type));
      this.element('joints-section').hidden = !editable.length;
      const fields = this.element('joint-fields');
      fields.replaceChildren();
      for (const joint of editable) {
        const degrees = joint.type !== 'prismatic';
        const multiplier = degrees ? SceneBuilderApp.DEGREES : 1;
        const label = document.createElement('label');
        label.textContent = `${joint.name} · ${degrees ? '°' : 'm'}`;
        const input = document.createElement('input');
        input.type = 'number';
        input.step = degrees ? '5' : '0.01';
        input.value = String(Number(((instance.jointPositions[joint.name] ?? joint.value ?? 0) * multiplier).toFixed(5)));
        if (joint.type !== 'continuous' && Number.isFinite(joint.lower)) input.min = String(joint.lower * multiplier);
        if (joint.type !== 'continuous' && Number.isFinite(joint.upper)) input.max = String(joint.upper * multiplier);
        input.addEventListener('change', () => this.attempt(() => {
          if (!input.value.trim() || !input.checkValidity()) throw new Error(`Enter a valid position for ${joint.name}.`);
          this.state.setJoint(instance.id, joint.name, Number(input.value) / multiplier);
        }));
        label.append(input);
        fields.append(label);
      }
    }

    renderNotices(asset) {
      const list = this.element('notice-list');
      list.replaceChildren();
      const path = document.createElement('p');
      path.textContent = asset?.id || this.state.instance(this.selection).assetId;
      list.append(path);
      for (const notice of asset?.notices || []) {
        const url = new URL(notice.url, window.location.href);
        if (url.origin !== window.location.origin || !['https:', 'http:'].includes(url.protocol)) continue;
        const link = document.createElement('a');
        link.href = url.href;
        link.target = '_blank';
        link.rel = 'noopener';
        link.textContent = notice.name;
        list.append(link);
      }
    }

    setMode(mode) {
      if (!SceneBuilderApp.TRANSFORM_MODES.includes(mode)) throw new Error('Unknown transform mode.');
      if (mode === 'scale' && this.part) throw new Error('Scale applies to whole objects, not individual URDF parts.');
      this.mode = mode;
      this.view?.setMode(mode);
      for (const value of SceneBuilderApp.TRANSFORM_MODES) this.element(`mode-${value}`).setAttribute('aria-pressed', String(value === mode));
    }
  }

  window.SceneBuilderApi = SceneBuilderApi;
  window.SceneBuilderApp = SceneBuilderApp;
  if (document.getElementById('scene-builder')) {
    const app = new SceneBuilderApp();
    app.start().catch((error) => app.status(error.message, true));
  }
})();
