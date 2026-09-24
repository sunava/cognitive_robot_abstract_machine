/* Read-only instrument annotations anchored to the authored articulated meshes. */
(function (global, factory) {
  'use strict';
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else global.LaboratoryManual = api;
})(typeof window === 'undefined' ? globalThis : window, function () {
  'use strict';

  // %% projected labels and local leaders
  const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';
  const Layout = Object.freeze({RAILS: 'rails', COMPACT: 'compact'});
  const DIMENSIONS = Object.freeze({cardWidth: 208, estimatedHeight: 98, margin: 18, gap: 12, leaderGap: 36, maximumLeader: 180, minimumRailWidth: 700, minimumCompactWidth: 220, maximumCompactWidth: 360, leftTop: 300, rightTop: 70});

  function normalizedName(value) { return String(value || '').replace(/[\s_]+/g, ' ').trim().toLowerCase(); }

  /** Project a world point, rejecting the entire camera clipping volume. */
  function projectAnchor(THREE, point, camera, width, height) {
    const clip = new THREE.Vector4(point.x, point.y, point.z, 1).applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix);
    if (!(clip.w > 0)) return null;
    const horizontal = clip.x / clip.w;
    const vertical = clip.y / clip.w;
    const depth = clip.z / clip.w;
    if (![horizontal, vertical, depth].every(Number.isFinite) || Math.abs(horizontal) > 1 || Math.abs(vertical) > 1 || Math.abs(depth) > 1) return null;
    return {x: (horizontal + 1) * width / 2, y: (1 - vertical) * height / 2};
  }

  /** Place labels close to their anchors without overlapping or leaving their rail. */
  function layoutRail(items, top, bottom, gap) {
    const ordered = items.slice().sort((left, right) => left.point.y - right.point.y);
    const required = ordered.reduce((sum, item) => sum + item.height, 0) + Math.max(0, ordered.length - 1) * gap;
    if (required > bottom - top) return null;
    const blocks = [];
    let offset = 0;
    for (const item of ordered) {
      const target = item.point.y - item.height / 2 - offset;
      blocks.push({items: [{item, offset}], sum: target, count: 1});
      while (blocks.length > 1) {
        const current = blocks[blocks.length - 1];
        const previous = blocks[blocks.length - 2];
        if (previous.sum / previous.count <= current.sum / current.count) break;
        previous.items.push(...current.items);
        previous.sum += current.sum;
        previous.count += current.count;
        blocks.pop();
      }
      offset += item.height + gap;
    }
    for (const block of blocks) {
      const base = Math.max(top, Math.min(bottom - required, block.sum / block.count));
      for (const placement of block.items) placement.item.top = base + placement.offset;
    }
    return ordered;
  }

  // %% authored component anchors
  class PartAnchor {
    constructor(THREE, root, part) {
      this.THREE = THREE;
      this.root = root;
      this.part = part;
      this.names = new Set((part.meshNames || [part.body]).filter(Boolean).map(normalizedName));
      this.node = null;
      this.center = new THREE.Vector3();
      this.point = new THREE.Vector3();
      this.bounds = new THREE.Box3();
      this.childBounds = new THREE.Box3();
      this.inverse = new THREE.Matrix4();
      this.relative = new THREE.Matrix4();
    }

    /** Resolve late-loaded nodes and retain a point in the component's own frame. */
    resolve() {
      if (this.node) {
        for (let ancestor = this.node; ancestor; ancestor = ancestor.parent) {
          if (ancestor === this.root) return true;
        }
        this.node = null;
      }
      this.root.traverse(node => {
        if (this.node || (!this.names.has(normalizedName(node.name)) && !this.names.has(normalizedName(node.userData && node.userData.name)))) return;
        if (this.measure(node)) this.node = node;
      });
      return Boolean(this.node);
    }

    /** Union a component's visible material meshes in the named component frame. */
    measure(node) {
      if (this.part.position) { this.center.fromArray(this.part.position); return true; }
      if (node.geometry) {
        if (!node.geometry.boundingBox) node.geometry.computeBoundingBox();
        node.geometry.boundingBox.getCenter(this.center);
        return true;
      }
      this.bounds.makeEmpty();
      this.inverse.copy(node.matrixWorld).invert();
      node.traverse(child => {
        if (!child.geometry) return;
        for (let ancestor = child; ancestor && ancestor !== node; ancestor = ancestor.parent) {
          if (ancestor.visible === false) return;
        }
        if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
        this.relative.copy(this.inverse).multiply(child.matrixWorld);
        this.childBounds.copy(child.geometry.boundingBox).applyMatrix4(this.relative);
        this.bounds.union(this.childBounds);
      });
      if (this.bounds.isEmpty()) return false;
      this.bounds.getCenter(this.center);
      return true;
    }

    /** Follow every parent articulation while respecting authored visibility. */
    worldPoint() {
      if (!this.resolve()) return null;
      for (let ancestor = this.node; ancestor; ancestor = ancestor.parent) {
        if (ancestor.visible === false) return null;
      }
      if (!this.node.geometry && !this.measure(this.node)) return null;
      return this.point.copy(this.center).applyMatrix4(this.node.matrixWorld);
    }
  }

  // %% accessible annotation elements
  class ManualOverlay {
    constructor(options) {
      this.options = options;
      this.THREE = options.THREE;
      this.root = options.root;
      this.host = options.host;
      this.document = this.host.ownerDocument;
      this.snapshot = null;
      this.visible = false;
      this.destroyed = false;
      this.selected = null;
      this.listeners = [];
      this.element = this.make('section', 'laboratory-manual-overlay');
      this.element.setAttribute('aria-label', 'Instrument manual');
      this.element.hidden = true;
      this.lines = this.document.createElementNS(SVG_NAMESPACE, 'svg');
      this.lines.setAttribute('class', 'laboratory-manual-lines');
      this.lines.setAttribute('aria-hidden', 'true');
      this.element.appendChild(this.lines);
      this.compact = this.make('section', 'laboratory-manual-compact', this.element);
      this.compact.hidden = true;
      this.listen(this.compact, 'pointerdown', event => event.stopPropagation());
      this.navigation = this.make('nav', 'laboratory-manual-navigation', this.compact);
      this.navigation.setAttribute('aria-label', 'Instrument parts');
      this.compactTitle = this.make('strong', 'laboratory-manual-title', this.compact);
      this.compactDescription = this.make('p', 'laboratory-manual-description', this.compact);
      this.compactStatus = this.make('p', 'laboratory-manual-status', this.compact);
      this.empty = this.make('div', 'laboratory-manual-empty', this.element);
      this.empty.textContent = 'Move the instrument into view to explore its parts.';
      this.empty.hidden = true;
      this.entries = [];
      this.setParts(options.parts || []);
      this.host.appendChild(this.element);
    }

    make(tag, className, parent) {
      const element = this.document.createElement(tag);
      element.className = className;
      if (parent) parent.appendChild(element);
      return element;
    }

    listen(element, event, callback) {
      element.addEventListener(event, callback);
      this.listeners.push({element, event, callback});
    }

    button(className, part, parent) {
      const button = this.make('button', className, parent);
      button.type = 'button';
      button.setAttribute('aria-label', part.number + '. ' + part.title);
      button.setAttribute('aria-pressed', 'false');
      this.listen(button, 'pointerdown', event => event.stopPropagation());
      this.listen(button, 'click', event => {
        event.stopPropagation();
        this.selected = part.id;
        this.render();
      });
      return button;
    }

    makeEntry(part) {
      const line = this.document.createElementNS(SVG_NAMESPACE, 'polyline');
      line.setAttribute('class', 'laboratory-manual-leader');
      line.setAttribute('display', 'none');
      this.lines.appendChild(line);
      const hotspot = this.button('laboratory-manual-hotspot', part, this.element);
      hotspot.textContent = String(part.number);
      hotspot.title = part.number + '. ' + part.title + '\n' + (part.description || '');
      hotspot.hidden = true;
      const card = this.button('laboratory-manual-card', part, this.element);
      card.hidden = true;
      card.dataset.part = part.id;
      const number = this.make('span', 'laboratory-manual-number', card);
      number.textContent = String(part.number).padStart(2, '0');
      const copy = this.make('span', 'laboratory-manual-copy', card);
      const title = this.make('strong', 'laboratory-manual-title', copy);
      title.textContent = part.title;
      const description = this.make('span', 'laboratory-manual-description', copy);
      description.textContent = part.description || '';
      const status = this.make('span', 'laboratory-manual-status', copy);
      const chip = this.button('laboratory-manual-chip', part, this.navigation);
      chip.textContent = String(part.number);
      chip.title = part.title;
      chip.hidden = true;
      return {part, anchor: new PartAnchor(this.THREE, this.root, part), line, hotspot, card, title, description, status, chip, point: null};
    }

    reservation(name, fallback) {
      const value = this.options[name];
      const measurement = typeof value === 'function' ? value() : value;
      return Number.isFinite(measurement) ? Math.max(0, measurement) : fallback;
    }

    /** Replace the selected instrument without retaining its controls or observations. */
    setParts(parts) {
      if (this.destroyed) return;
      const removed = new Set(this.entries.flatMap(entry => [entry.hotspot, entry.card, entry.chip, entry.line]));
      this.listeners = this.listeners.filter(listener => {
        if (!removed.has(listener.element)) return true;
        listener.element.removeEventListener(listener.event, listener.callback);
        return false;
      });
      for (const element of removed) if (element.parentNode) element.parentNode.removeChild(element);
      this.selected = null;
      this.snapshot = null;
      this.entries = parts.map(part => this.makeEntry(part));
      if (this.visible) this.render();
    }

    /** Show the guide without changing the instrument or issuing commands. */
    setVisible(visible) {
      if (this.destroyed) return;
      this.visible = Boolean(visible);
      this.element.hidden = !this.visible;
      if (this.visible) this.render();
    }

    /** Keep live observations separate from the authored part explanation. */
    update(snapshot) {
      this.snapshot = snapshot;
      if (this.visible) this.render();
    }

    hideEntry(entry) {
      entry.point = null;
      entry.hotspot.hidden = true;
      entry.card.hidden = true;
      entry.chip.hidden = true;
      entry.line.setAttribute('display', 'none');
    }

    projectEntries(camera, width, height) {
      const projected = [];
      for (const entry of this.entries) {
        this.hideEntry(entry);
        if (entry.part.visible && !entry.part.visible(this.snapshot)) continue;
        const world = entry.anchor.worldPoint();
        const point = world && projectAnchor(this.THREE, world, camera, width, height);
        if (!point) continue;
        entry.point = point;
        entry.hotspot.hidden = false;
        entry.hotspot.style.left = point.x + 'px';
        entry.hotspot.style.top = point.y + 'px';
        entry.status.textContent = entry.part.describe ? String(entry.part.describe(this.snapshot) || '') : '';
        entry.status.hidden = !entry.status.textContent;
        entry.chip.hidden = false;
        projected.push(entry);
      }
      return projected;
    }

    selectVisible(projected) {
      const selected = projected.find(entry => entry.part.id === this.selected) || projected[0];
      this.selected = selected ? selected.part.id : null;
      for (const entry of this.entries) {
        const active = entry === selected;
        for (const button of [entry.hotspot, entry.card, entry.chip]) {
          button.setAttribute('aria-pressed', String(active));
          button.dataset.selected = String(active);
        }
      }
      if (selected) {
        this.compactTitle.textContent = selected.part.number + ' · ' + selected.part.title;
        this.compactDescription.textContent = selected.part.description || '';
        this.compactStatus.textContent = selected.status.textContent;
        this.compactStatus.hidden = !selected.status.textContent;
      }
    }

    /** Keep both the label columns and every leader close to the instrument. */
    placeRails(projected, availableWidth, height) {
      if (availableWidth < DIMENSIONS.minimumRailWidth || !projected.length) return false;
      const horizontal = projected.map(entry => entry.point.x);
      const leftEdge = Math.min(...horizontal) - DIMENSIONS.leaderGap;
      const rightEdge = Math.max(...horizontal) + DIMENSIONS.leaderGap;
      const rails = [];
      for (const side of ['left', 'right']) {
        const left = side === 'left' ? leftEdge - DIMENSIONS.cardWidth : rightEdge;
        const entries = projected.filter(entry => (entry.part.side || 'left') === side);
        if (!entries.length) continue;
        if (left < DIMENSIONS.margin || left + DIMENSIONS.cardWidth > availableWidth - DIMENSIONS.margin) return false;
        const items = entries.map(entry => {
          entry.card.style.width = DIMENSIONS.cardWidth + 'px';
          entry.card.hidden = false;
          return {entry, point: entry.point, height: entry.card.offsetHeight || DIMENSIONS.estimatedHeight};
        });
        const top = this.reservation(side === 'left' ? 'reservedTopLeft' : 'reservedTopRight', side === 'left' ? DIMENSIONS.leftTop : DIMENSIONS.rightTop);
        const positions = layoutRail(items, top, height - DIMENSIONS.margin, DIMENSIONS.gap);
        if (!positions) return false;
        const edge = side === 'left' ? left + DIMENSIONS.cardWidth : left;
        const elbow = edge + (side === 'left' ? DIMENSIONS.gap : -DIMENSIONS.gap);
        for (const item of positions) {
          const middle = item.top + item.height / 2;
          if (Math.hypot(item.point.x - elbow, item.point.y - middle) + DIMENSIONS.gap > DIMENSIONS.maximumLeader) return false;
        }
        rails.push({left, edge, elbow, positions});
      }
      for (const rail of rails) {
        for (const item of rail.positions) {
          const entry = item.entry;
          entry.card.style.left = rail.left + 'px';
          entry.card.style.top = item.top + 'px';
          const middle = item.top + item.height / 2;
          entry.line.setAttribute('points', [entry.point.x + ',' + entry.point.y, rail.elbow + ',' + middle, rail.edge + ',' + middle].join(' '));
          entry.line.setAttribute('display', '');
        }
      }
      return true;
    }

    /** Find a nearby readable card position without covering its selected marker. */
    compactPosition(point, width, cardHeight, availableWidth, height) {
      const gap = DIMENSIONS.leaderGap;
      const margin = DIMENSIONS.margin;
      const positions = [
        {left: point.x - gap - width, top: point.y - cardHeight / 2},
        {left: point.x + gap, top: point.y - cardHeight / 2},
        {left: point.x - width / 2, top: point.y + gap},
        {left: point.x - width / 2, top: point.y - gap - cardHeight},
      ];
      const candidates = [];
      for (const position of positions) {
        const left = Math.max(margin, Math.min(availableWidth - margin - width, position.left));
        const reservedTop = this.reservation(left < availableWidth / 2 ? 'reservedTopLeft' : 'reservedTopRight', 0);
        const top = Math.max(margin, reservedTop, Math.min(height - margin - cardHeight, position.top));
        if (top + cardHeight > height - margin) continue;
        const right = left + width;
        const bottom = top + cardHeight;
        if (point.x > left - margin && point.x < right + margin && point.y > top - margin && point.y < bottom + margin) continue;
        const endpoint = {x: Math.max(left, Math.min(right, point.x)), y: Math.max(top, Math.min(bottom, point.y))};
        const distance = Math.hypot(point.x - endpoint.x, point.y - endpoint.y);
        if (distance <= DIMENSIONS.maximumLeader) candidates.push({left, top, endpoint, distance});
      }
      candidates.sort((left, right) => left.distance - right.distance);
      return candidates[0] || null;
    }

    /** Show one explanation next to its part when the local columns do not fit. */
    placeCompact(projected, availableWidth, height) {
      for (const entry of projected) {
        entry.card.hidden = true;
        entry.line.setAttribute('display', 'none');
      }
      const width = Math.min(DIMENSIONS.maximumCompactWidth, availableWidth - DIMENSIONS.margin * 2);
      this.compact.hidden = !projected.length || width < DIMENSIONS.minimumCompactWidth || height < DIMENSIONS.estimatedHeight + DIMENSIONS.margin * 2;
      if (this.compact.hidden) return;
      this.compact.style.width = width + 'px';
      this.compact.style.bottom = 'auto';
      const selected = projected.find(entry => entry.part.id === this.selected);
      if (!selected) return;
      const cardHeight = this.compact.offsetHeight || DIMENSIONS.estimatedHeight;
      const position = this.compactPosition(selected.point, width, cardHeight, availableWidth, height);
      if (!position) { this.compact.hidden = true; return; }
      this.compact.style.left = position.left + 'px';
      this.compact.style.top = position.top + 'px';
      this.compact.style.bottom = height - position.top - cardHeight + 'px';
      selected.line.setAttribute('points', [selected.point, position.endpoint].map(value => value.x + ',' + value.y).join(' '));
      selected.line.setAttribute('display', '');
    }

    /** Reproject after camera movement, resize or an articulated body update. */
    render() {
      if (!this.visible || this.destroyed) return;
      const camera = typeof this.options.camera === 'function' ? this.options.camera() : this.options.camera;
      const width = this.host.clientWidth;
      const height = this.host.clientHeight;
      if (!camera || !(width > 0) || !(height > 0)) return;
      this.root.updateMatrixWorld(true);
      camera.updateMatrixWorld(true);
      const reservedRight = Math.min(width - DIMENSIONS.margin * 2, this.reservation('reservedRight', 0));
      const availableWidth = Math.max(DIMENSIONS.margin * 2, width - reservedRight);
      this.lines.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
      const projected = this.projectEntries(camera, width, height);
      this.selectVisible(projected);
      const rails = this.placeRails(projected, availableWidth, height);
      this.element.dataset.layout = rails ? Layout.RAILS : Layout.COMPACT;
      this.compact.hidden = true;
      this.empty.hidden = Boolean(projected.length);
      this.empty.style.left = availableWidth / 2 + 'px';
      if (!rails) this.placeCompact(projected, availableWidth, height);
    }

    /** Release all DOM and listeners without touching the physical scene. */
    destroy() {
      if (this.destroyed) return;
      this.destroyed = true;
      this.visible = false;
      for (const listener of this.listeners) listener.element.removeEventListener(listener.event, listener.callback);
      this.listeners.length = 0;
      if (this.element.parentNode) this.element.parentNode.removeChild(this.element);
    }
  }

  return {projectAnchor, layoutRail, create: options => new ManualOverlay(options)};
});
