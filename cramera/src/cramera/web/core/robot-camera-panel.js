/* %% robot viewpoint controls and shared renderer viewport */
(function (global) {
  'use strict';

  const WINDOW_MARGIN = 8;
  const MINIMUM_SIZE = Object.freeze({width: 220, height: 140});
  const RESIZE_CORNERS = Object.freeze([
    {name: 'top-left', horizontal: -1, vertical: -1},
    {name: 'top-right', horizontal: 1, vertical: -1},
    {name: 'bottom-left', horizontal: -1, vertical: 1},
    {name: 'bottom-right', horizontal: 1, vertical: 1},
  ]);

  class RobotCameraPanel {
    /** Mount an optional robot viewpoint on the scene's existing renderer. */
    constructor(options) {
      this.options = options;
      this.root = options.root;
      this.layer = options.layer;
      this.pose = new global.RobotCamera.Pose(options.THREE);
      this.sources = [];
      this.modelCount = -1;
      this.primary = null;
      this.selected = null;
      this.chosen = false;
      this.expanded = false;
      this.position = null;
      /** Last inset position relative to the scene container. */
      this.drag = null;
      /** Captured pointer and its offset within the window. */
      this.size = null;
      /** Chosen inset size, retained while the scene is temporarily smaller. */
      this.resizing = null;
      /** Captured resize pointer and the window geometry where it started. */
      this.listeners = [];
      this.header = this.root.querySelector('.robot-camera-head');
      this.header.title = 'Drag to move the robot view';
      this.viewport = this.root.querySelector('[data-robot-camera="viewport"]');
      this.canvas = this.root.ownerDocument.createElement('canvas');
      this.canvas.setAttribute('aria-hidden', 'true');
      this.context = this.canvas.getContext('2d', {alpha: false});
      this.viewport.appendChild(this.canvas);
      this.status = this.root.querySelector('[data-robot-camera="status"]');
      this.select = this.root.querySelector('[data-robot-camera="source"]');
      this.expand = this.root.querySelector('[data-robot-camera="expand"]');
      this.savedViewport = new options.THREE.Vector4();
      this.savedScissor = new options.THREE.Vector4();
      this.listen(this.layer, 'change', () => this.setVisible(this.layer.checked));
      this.listen(this.expand, 'click', () => this.setExpanded(!this.expanded));
      this.listen(this.root.querySelector('[data-robot-camera="close"]'), 'click', () => {
        this.setVisible(false);
        this.layer.focus();
      });
      this.listen(this.select, 'change', () => {
        this.selected = this.sources.find(source => source.id === this.select.value) || null;
        this.chosen = true;
        options.invalidate();
      });
      this.listen(this.viewport, 'dblclick', () => this.setExpanded(!this.expanded));
      this.listen(this.root, 'keydown', event => {
        if (event.key !== 'Escape') return;
        event.stopPropagation();
        if (this.expanded) this.setExpanded(false);
        else { this.setVisible(false); this.layer.focus(); }
      });
      for (const name of ['pointerdown', 'wheel', 'dblclick']) {
        this.listen(this.root, name, event => event.stopPropagation());
      }
      this.listen(this.header, 'pointerdown', event => this.startDrag(event));
      this.listen(this.header, 'pointermove', event => this.moveDrag(event));
      for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
        this.listen(this.header, name, event => this.endDrag(event));
      }
      this.addResizeHandles();
      this.observer = new options.ResizeObserver(() => {
        this.fitSize();
        this.fitPosition();
        options.invalidate();
      });
      this.observer.observe(this.viewport);
      this.observer.observe(options.container);
      this.setVisible(false);
      this.setExpanded(false);
      this.setModels([]);
    }

    /** Track event handlers so unmounting releases the panel completely. */
    listen(element, name, handler) {
      element.addEventListener(name, handler);
      this.listeners.push({element, name, handler});
    }

    /** Keep the layer checkbox and window visibility in agreement. */
    setVisible(visible) {
      if (!visible) { this.endDrag(); this.endResize(); }
      this.layer.checked = visible;
      this.root.hidden = !visible;
      if (visible) { this.fitSize(); this.fitPosition(); }
      this.options.invalidate();
    }

    /** Switch between the inset and a large view within the scene panel. */
    setExpanded(expanded) {
      this.endDrag();
      this.endResize();
      this.expanded = expanded;
      this.root.classList.toggle('expanded', expanded);
      if (expanded) {
        for (const edge of ['left', 'top', 'right', 'bottom']) this.root.style[edge] = '';
        if (this.size) { this.root.style.width = ''; this.root.style.height = ''; }
      } else { this.fitSize(); this.fitPosition(); }
      const label = expanded ? 'Restore robot view' : 'Enlarge robot view';
      this.expand.textContent = expanded ? '↙' : '⛶';
      this.expand.title = label;
      this.expand.setAttribute('aria-label', label);
      this.expand.setAttribute('aria-expanded', expanded);
      this.options.invalidate();
    }

    // %% window movement
    /** Capture a primary pointer on the title bar, leaving buttons clickable. */
    startDrag(event) {
      if (this.root.hidden || this.expanded || this.drag || this.resizing || event.button !== 0 ||
          event.isPrimary === false || event.target.closest('button')) return;
      const bounds = this.root.getBoundingClientRect();
      this.drag = {pointer: event.pointerId, offsetX: event.clientX - bounds.left,
        offsetY: event.clientY - bounds.top};
      this.header.setPointerCapture(event.pointerId);
      this.root.classList.toggle('dragging', true);
      event.preventDefault();
    }

    /** Move the inset with its captured pointer while keeping it inside the scene. */
    moveDrag(event) {
      if (!this.drag || event.pointerId !== this.drag.pointer) return;
      const container = this.options.container.getBoundingClientRect();
      this.position = {left: event.clientX - container.left - this.drag.offsetX,
        top: event.clientY - container.top - this.drag.offsetY};
      this.fitPosition();
      this.options.invalidate();
    }

    /** Release a completed or interrupted drag without moving the overview camera. */
    endDrag(event) {
      if (!this.drag || (event && event.pointerId !== this.drag.pointer)) return;
      const pointer = this.drag.pointer;
      this.drag = null;
      if (this.header.hasPointerCapture(pointer)) this.header.releasePointerCapture(pointer);
      this.root.classList.toggle('dragging', false);
    }

    /** Keep a manually placed window reachable after resizing or restoring it. */
    fitPosition() {
      if (!this.position || this.root.hidden || this.expanded) return;
      const container = this.options.container.getBoundingClientRect();
      const bounds = this.root.getBoundingClientRect();
      if (!container.width || !container.height) return;
      const horizontalMargin = Math.min(WINDOW_MARGIN, Math.max(0, (container.width - bounds.width) / 2));
      const verticalMargin = Math.min(WINDOW_MARGIN, Math.max(0, (container.height - bounds.height) / 2));
      this.position.left = Math.max(horizontalMargin, Math.min(this.position.left, container.width - bounds.width - horizontalMargin));
      this.position.top = Math.max(verticalMargin, Math.min(this.position.top, container.height - bounds.height - verticalMargin));
      this.root.style.left = this.position.left + 'px';
      this.root.style.top = this.position.top + 'px';
      this.root.style.right = 'auto';
      this.root.style.bottom = 'auto';
    }

    // %% window resizing
    /** Provide corner grips that work for both mouse and touch pointers. */
    addResizeHandles() {
      for (const corner of RESIZE_CORNERS) {
        const handle = this.root.ownerDocument.createElement('button');
        handle.type = 'button';
        handle.className = 'robot-camera-resize';
        handle.setAttribute('data-robot-camera-resize', corner.name);
        handle.setAttribute('aria-label', 'Resize robot view from ' + corner.name.replace('-', ' '));
        handle.title = 'Drag to resize the robot view';
        this.root.appendChild(handle);
        this.listen(handle, 'pointerdown', event => this.startResize(event, handle, corner));
        this.listen(handle, 'pointermove', event => this.moveResize(event));
        for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
          this.listen(handle, name, event => this.endResize(event));
        }
      }
    }

    /** Fix the opposite corner before capturing a resize gesture. */
    startResize(event, handle, corner) {
      if (this.root.hidden || this.expanded || this.drag || this.resizing ||
          event.button !== 0 || event.isPrimary === false) return;
      const bounds = this.root.getBoundingClientRect();
      const container = this.options.container.getBoundingClientRect();
      this.resizing = {pointer: event.pointerId, handle, corner,
        clientX: event.clientX, clientY: event.clientY,
        left: bounds.left - container.left, top: bounds.top - container.top,
        width: bounds.width, height: bounds.height};
      handle.setPointerCapture(event.pointerId);
      event.preventDefault();
      event.stopPropagation();
    }

    /** Resize toward the captured pointer without moving the opposite corner. */
    moveResize(event) {
      const gesture = this.resizing;
      if (!gesture || event.pointerId !== gesture.pointer) return;
      const container = this.options.container.getBoundingClientRect();
      const leftward = gesture.corner.horizontal < 0;
      const upward = gesture.corner.vertical < 0;
      const right = gesture.left + gesture.width;
      const bottom = gesture.top + gesture.height;
      const maximumWidth = Math.max(1, (leftward ? right : container.width - gesture.left) - WINDOW_MARGIN);
      const maximumHeight = Math.max(1, (upward ? bottom : container.height - gesture.top) - WINDOW_MARGIN);
      const width = gesture.width + gesture.corner.horizontal * (event.clientX - gesture.clientX);
      const height = gesture.height + gesture.corner.vertical * (event.clientY - gesture.clientY);
      this.size = {
        width: Math.max(Math.min(MINIMUM_SIZE.width, maximumWidth), Math.min(width, maximumWidth)),
        height: Math.max(Math.min(MINIMUM_SIZE.height, maximumHeight), Math.min(height, maximumHeight)),
      };
      this.position = {left: leftward ? right - this.size.width : gesture.left,
        top: upward ? bottom - this.size.height : gesture.top};
      this.fitSize();
      this.fitPosition();
      this.options.invalidate();
    }

    /** Release resize capture when the gesture or window ends. */
    endResize(event) {
      if (!this.resizing || (event && event.pointerId !== this.resizing.pointer)) return;
      const {handle, pointer} = this.resizing;
      this.resizing = null;
      if (handle.hasPointerCapture(pointer)) handle.releasePointerCapture(pointer);
    }

    /** Fit the chosen size to the current scene without losing its saved dimensions. */
    fitSize() {
      if (!this.size || this.root.hidden || this.expanded) return;
      const container = this.options.container.getBoundingClientRect();
      const maximumWidth = container.width - WINDOW_MARGIN * 2;
      const maximumHeight = container.height - WINDOW_MARGIN * 2;
      if (maximumWidth <= 0 || maximumHeight <= 0) return;
      this.root.style.width = Math.min(this.size.width, maximumWidth) + 'px';
      this.root.style.height = Math.min(this.size.height, maximumHeight) + 'px';
    }

    // %% camera selection and rendering
    /** Refresh sources after asynchronous model loads, retaining explicit choices. */
    setModels(models, primary) {
      this.modelCount = models.length;
      this.primary = primary;
      this.sources = global.RobotCamera.sources(models);
      const previous = this.chosen && this.selected
        ? this.sources.find(source => source.id === this.selected.id) : null;
      const preferred = primary ? global.RobotCamera.sources([primary])[0] : null;
      this.selected = previous || (preferred && this.sources.find(source => source.id === preferred.id)) || this.sources[0] || null;
      this.select.replaceChildren();
      for (const source of this.sources) {
        const option = this.root.ownerDocument.createElement('option');
        option.value = source.id;
        option.textContent = source.label;
        this.select.appendChild(option);
      }
      this.select.value = this.selected ? this.selected.id : '';
      this.select.disabled = !this.selected;
      this.status.textContent = 'No robot camera available in this scene.';
      this.status.hidden = !!this.selected;
      this.canvas.hidden = !this.selected;
      this.options.invalidate();
    }

    /** Resolve newly loaded models before rendering their current viewpoint. */
    refreshModels(models, primary) {
      if (this.modelCount !== models.length || this.primary !== primary) this.setModels(models, primary);
    }

    /** Draw the current robot pose without changing the overview camera or GPU context. */
    render() {
      if (this.root.hidden || !this.selected) return;
      const bounds = this.viewport.getBoundingClientRect();
      if (!this.pose.update(this.selected, bounds.width, bounds.height)) return;
      const container = this.options.container.getBoundingClientRect();
      const left = bounds.left - container.left;
      const bottom = container.bottom - bounds.bottom;
      const renderer = this.options.renderer;
      renderer.getViewport(this.savedViewport);
      renderer.getScissor(this.savedScissor);
      const scissorTest = renderer.getScissorTest();
      renderer.setViewport(left, bottom, bounds.width, bounds.height);
      renderer.setScissor(left, bottom, bounds.width, bounds.height);
      renderer.setScissorTest(true);
      renderer.clear();
      renderer.render(this.options.scene, this.pose.camera);
      this.copyViewport(renderer.domElement, renderer.getPixelRatio(), left, bottom, bounds.width, bounds.height);
      renderer.setViewport(this.savedViewport);
      renderer.setScissor(this.savedScissor);
      renderer.setScissorTest(scissorTest);
    }

    /** Cover other scene overlays with the freshly rendered camera image. */
    copyViewport(source, pixelRatio, left, bottom, width, height) {
      const pixelWidth = Math.floor(width * pixelRatio);
      const pixelHeight = Math.floor(height * pixelRatio);
      if (this.canvas.width !== pixelWidth) this.canvas.width = pixelWidth;
      if (this.canvas.height !== pixelHeight) this.canvas.height = pixelHeight;
      this.context.drawImage(source,
        Math.floor(left * pixelRatio), source.height - Math.floor(bottom * pixelRatio) - pixelHeight,
        pixelWidth, pixelHeight, 0, 0, pixelWidth, pixelHeight);
    }

    /** Release observers and input handlers when the scene is unmounted. */
    destroy() {
      this.endDrag();
      this.endResize();
      this.observer.disconnect();
      this.listeners.forEach(({element, name, handler}) => element.removeEventListener(name, handler));
      this.listeners = [];
    }
  }

  global.RobotCameraPanel = RobotCameraPanel;
})(window);
