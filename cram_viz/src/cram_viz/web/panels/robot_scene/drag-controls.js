/* ============================================================================
 * panels/robot_scene/drag-controls.js — pointer interaction for the 3D scene:
 * dragging bench objects / the place-target marker (surface-snapped so a
 * dragged object always rests on whatever is beneath it), and classifying a
 * plain click (as opposed to a drag) into an entity id for the rest of the
 * app to react to.
 *
 * Owns its own pointer listeners on `renderer.domElement`; `destroy()` removes
 * them so the panel can tear this down independently of the scene itself.
 * ==========================================================================*/
window.RobotSceneDragControls = (function () {
  const CLICK_VS_DRAG_PX = 5; // pointerup within this many px of pointerdown counts as a click, not a drag

  function DragControls(opts) {
    const camera = opts.camera, renderer = opts.renderer, worldRoot = opts.worldRoot;
    const dragRay = new THREE.Raycaster(), dragNdc = new THREE.Vector2();
    const dragStartNdc = new THREE.Vector2(), dragStartWorld = new THREE.Vector3();
    const camRight = new THREE.Vector3(), camForward = new THREE.Vector3(), camUp = new THREE.Vector3();
    const markerDelta = new THREE.Vector3();
    const dragPlane = new THREE.Plane();
    const hitPoint = new THREE.Vector3();
    const surfaceRay = new THREE.Raycaster();
    const DOWN = new THREE.Vector3(0, -1, 0);

    let dragTarget = null, dragging = false;
    let clickX = 0, clickY = 0, clickArmed = false;

    function pointerNdc(e) {
      const r = renderer.domElement.getBoundingClientRect();
      dragNdc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    }

    function draggableMeshes() {
      const list = [];
      const objectMeshes = opts.getObjectMeshes();
      for (const key in objectMeshes) {
        objectMeshes[key].traverse(function (c) { if (c.isMesh) { c.userData.simObj = key; list.push(c); } });
      }
      const marker = opts.getMarker();
      if (marker.visible) {
        marker.group.traverse(function (c) { if (c.isMesh) { c.userData.simMarker = true; list.push(c); } });
      }
      return list;
    }

    function raycastDraggable(e) {
      pointerNdc(e);
      dragRay.setFromCamera(dragNdc, camera);
      const hits = dragRay.intersectObjects(draggableMeshes(), false);
      if (!hits.length) return null;
      const o = hits[0].object;
      if (o.userData.simMarker) return { marker: true };
      return { name: o.userData.simObj };
    }

    // rest a dragged object on whatever is beneath it: cast a ray straight
    // down through the object and drop it onto the highest surface (table,
    // counter, another object, else the floor) — never floating.
    function meshBox(group) {
      const box = new THREE.Box3(); box.makeEmpty();
      group.traverse(function (c) { if (c.isMesh) box.expandByObject(c); });
      return box;
    }
    function surfaceTargets(excludeKey) {
      const targets = opts.getEnvMeshes().slice();
      const objectMeshes = opts.getObjectMeshes();
      for (const key in objectMeshes) {
        if (key !== excludeKey && objectMeshes[key].visible) {
          objectMeshes[key].traverse(function (o) { if (o.isMesh) targets.push(o); });
        }
      }
      return targets;
    }
    function snapToSurface(group, excludeKey) {
      const box = meshBox(group);
      if (!isFinite(box.min.y)) return;
      const c = box.getCenter(new THREE.Vector3());
      surfaceRay.set(new THREE.Vector3(c.x, box.max.y + 0.5, c.z), DOWN);
      surfaceRay.far = 6;
      const hits = surfaceRay.intersectObjects(surfaceTargets(excludeKey), false);
      if (!hits.length) return;
      group.position.z += hits[0].point.y - box.min.y; // worldRoot: +local z == +world y
    }

    // cursor -> world point on the horizontal drag plane (infinite, so the
    // object follows the cursor anywhere — the surface height is fixed
    // afterwards by snapToSurface). The plane is set through the grab point
    // on pointerdown.
    function surfacePointAt(e) {
      pointerNdc(e);
      surfaceRay.setFromCamera(dragNdc, camera);
      return surfaceRay.ray.intersectPlane(dragPlane, hitPoint) ? hitPoint.clone() : null;
    }

    function onPointerDown(e) {
      if (e.button !== 0) return;
      clickX = e.clientX; clickY = e.clientY; clickArmed = true;
      if (opts.isPlaying()) return;
      const hit = raycastDraggable(e);
      if (!hit) return;
      if (opts.isLive() && hit.marker) return; // the place marker has no meaning live
      dragTarget = hit; dragging = true;
      if (opts.controlsEnabled) opts.controlsEnabled(false);
      renderer.domElement.setPointerCapture(e.pointerId);
      renderer.domElement.style.cursor = 'grabbing';
      dragStartNdc.copy(dragNdc);
      const group = hit.marker ? opts.getMarker().group : opts.getObjectGroup(hit.name);
      group.getWorldPosition(dragStartWorld);
      dragPlane.setFromNormalAndCoplanarPoint(new THREE.Vector3(0, 1, 0), dragStartWorld);
      if (!hit.marker && opts.onObjectDragStart) opts.onObjectDragStart(hit.name);
      e.preventDefault();
    }

    function onPointerMove(e) {
      if (!dragging) {
        if (!opts.isPlaying() && e.buttons === 0) {
          renderer.domElement.style.cursor = raycastDraggable(e) ? 'grab' : '';
        }
        return;
      }
      if (dragTarget.name) {
        const key = dragTarget.name, group = opts.getObjectGroup(key);
        const hit = surfacePointAt(e); // world point on the drag plane
        if (hit) {
          worldRoot.worldToLocal(hit); // map frame (z-up)
          group.position.x = hit.x; group.position.y = hit.y;
          snapToSurface(group, key); // exact rest height for this object
          if (opts.onObjectDrag) opts.onObjectDrag(key, group.position.x, group.position.y, group.position.z);
        }
        return;
      }
      // marker drag: camera-basis mapping, clamped to the table bounds
      pointerNdc(e);
      const marker = opts.getMarker().group;
      const dist = camera.position.distanceTo(dragStartWorld);
      const halfH = Math.tan((camera.fov * Math.PI) / 360) * dist;
      const halfW = halfH * camera.aspect;
      camera.getWorldDirection(camForward);
      const pitch = Math.max(0.3, Math.abs(camForward.y));
      camRight.set(1, 0, 0).applyQuaternion(camera.quaternion); camRight.y = 0; camRight.normalize();
      camUp.set(0, 1, 0).applyQuaternion(camera.quaternion);
      camForward.y = 0; camUp.y = 0; camForward.add(camUp).normalize();
      markerDelta.copy(dragStartWorld)
        .addScaledVector(camRight, (dragNdc.x - dragStartNdc.x) * halfW)
        .addScaledVector(camForward, (dragNdc.y - dragStartNdc.y) * (halfH / pitch));
      worldRoot.worldToLocal(markerDelta);
      const limit = opts.getMarkerBounds() || opts.getDragBounds();
      marker.position.x = Math.min(limit.maxX, Math.max(limit.minX, markerDelta.x));
      marker.position.y = Math.min(limit.maxY, Math.max(limit.minY, markerDelta.y));
      if (opts.onMarkerDrag) opts.onMarkerDrag(marker.position.x, marker.position.y);
    }

    function endDrag() {
      if (!dragging) return;
      if (dragTarget && dragTarget.name) {
        const group = opts.getObjectGroup(dragTarget.name);
        if (opts.onObjectDragEnd) opts.onObjectDragEnd(dragTarget.name, group.position.x, group.position.y, group.position.z);
      }
      dragging = false; dragTarget = null;
      if (opts.controlsEnabled) opts.controlsEnabled(true);
      renderer.domElement.style.cursor = '';
    }

    function onPointerUp(e) {
      endDrag();
      if (!clickArmed) return;
      clickArmed = false;
      if (Math.hypot(e.clientX - clickX, e.clientY - clickY) > CLICK_VS_DRAG_PX) return;
      const hit = raycastDraggable(e);
      const id = hit ? (hit.marker ? 'place_area' : opts.objectIdFor(hit.name)) : (opts.classifyMiss ? opts.classifyMiss(e) : null);
      if (id && opts.onClick) opts.onClick(id);
    }

    function onPointerCancel() { clickArmed = false; endDrag(); }

    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('pointerup', onPointerUp);
    renderer.domElement.addEventListener('pointercancel', onPointerCancel);

    this.destroy = function () {
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      renderer.domElement.removeEventListener('pointercancel', onPointerCancel);
    };
  }

  return { DragControls: DragControls };
})();
