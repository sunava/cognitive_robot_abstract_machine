/* ============================================================================
 * panels/robot_scene/playback.js — trajectory math: joint/base/object pose
 * interpolation and the pick/place transport blend used while dragging.
 *
 * `resolveJoint` and `TransportBlender` are pure arithmetic over plain
 * numbers/URDF joint objects (no rendering), so they're unit-testable without
 * a renderer. `setPose` needs THREE's Vector3/Quaternion for lerp/slerp.
 * ==========================================================================*/
window.RobotScenePlayback = (function () {
  const ZERO = { x: 0, y: 0 };

  function smooth(u) {
    u = Math.min(1, Math.max(0, u));
    return u * u * (3 - 2 * u);
  }

  // resolve a trajectory key ('prefix/joint_name' or 'joint_name') to the
  // matching URDF joint on whichever loaded model owns that prefix — shared
  // by recorded playback (applyFrame) and live-bridge frames (applyLive), so
  // a fix to the lookup only needs to happen once.
  function resolveJoint(models, key) {
    const cut = key.indexOf('/');
    const prefix = cut < 0 ? '' : key.slice(0, cut);
    const jointName = cut < 0 ? key : key.slice(cut + 1);
    for (let i = 0; i < models.length; i++) {
      if (models[i].prefix === prefix) return models[i].obj.joints[jointName] || null;
    }
    return null;
  }

  let scratch = null;
  function setPose(obj, a, b, t) {
    if (!scratch) {
      scratch = {
        p0: new THREE.Vector3(), p1: new THREE.Vector3(),
        q0: new THREE.Quaternion(), q1: new THREE.Quaternion(),
      };
    }
    scratch.p0.set(a[0], a[1], a[2]); scratch.p1.set(b[0], b[1], b[2]);
    obj.position.copy(scratch.p0).lerp(scratch.p1, t);
    scratch.q0.set(a[3], a[4], a[5], a[6]); scratch.q1.set(b[3], b[4], b[5], b[6]);
    obj.quaternion.copy(scratch.q0).slerp(scratch.q1, t);
  }

  // Tracks the drag offset for each pickable object plus the shared
  // place-target offset, and blends an object smoothly between "resting at
  // its recorded spawn + drag offset" and "resting at the place target"
  // across the segment where the scene records it as picked/placed.
  function TransportBlender() {
    let transports = [];
    let pickDeltas = {};
    let spawn0 = {};
    const placeDelta = { x: 0, y: 0 };

    // one shared blend-weight curve used by both the object's own motion and
    // the base's compensating motion while carrying it (used to be
    // duplicated with slightly different formulas in each caller).
    function weightAt(transport, f) {
      if (f < transport.attach) return 0;
      if (f < transport.detach) return smooth((f - transport.attach) / (transport.detach - transport.attach));
      return 1;
    }

    this.configure = function (sceneObjects, segments, objectKeyById) {
      transports = [];
      pickDeltas = {};
      spawn0 = {};
      (sceneObjects || []).forEach(function (o) {
        pickDeltas[o.key] = { x: 0, y: 0, zAbs: null };
        spawn0[o.key] = o.spawn.slice();
      });
      (segments || []).forEach(function (s) {
        if (!s.picks || s.attach === undefined || s.detach === undefined) return;
        const key = objectKeyById[s.picks];
        if (key) transports.push({ obj: key, start: s.start, end: s.end, attach: s.attach, detach: s.detach });
      });
    };

    this.pickDelta = function (key) { return pickDeltas[key] || ZERO; };

    this.recordDrag = function (key, x, y, zAbs) {
      const s0 = spawn0[key];
      if (!s0 || !pickDeltas[key]) return;
      pickDeltas[key].x = x - s0[0];
      pickDeltas[key].y = y - s0[1];
      pickDeltas[key].zAbs = zAbs;
    };

    this.setPlaceDelta = function (x, y) { placeDelta.x = x; placeDelta.y = y; };

    this.restingBeforePick = function (key, f) {
      for (let i = 0; i < transports.length; i++) {
        if (transports[i].obj === key) return f < transports[i].attach;
      }
      return true; // never picked -> always resting
    };

    this.objOffsetAt = function (key, f) {
      for (let i = 0; i < transports.length; i++) {
        const transport = transports[i];
        if (transport.obj !== key) continue;
        const pickDelta = pickDeltas[key] || ZERO;
        const u = weightAt(transport, f);
        return { x: pickDelta.x + (placeDelta.x - pickDelta.x) * u, y: pickDelta.y + (placeDelta.y - pickDelta.y) * u };
      }
      return pickDeltas[key] || ZERO;
    };

    this.baseOffsetAt = function (f) {
      for (let i = 0; i < transports.length; i++) {
        const transport = transports[i];
        if (f < transport.start || f >= transport.end) continue;
        const pickDelta = pickDeltas[transport.obj] || ZERO;
        if (f < transport.attach) {
          const w = smooth((f - transport.start) / Math.max(1, transport.attach - transport.start));
          return { x: pickDelta.x * w, y: pickDelta.y * w };
        }
        const u = weightAt(transport, f);
        if (f < transport.detach) {
          return { x: pickDelta.x + (placeDelta.x - pickDelta.x) * u, y: pickDelta.y + (placeDelta.y - pickDelta.y) * u };
        }
        const w2 = 1 - smooth((f - transport.detach) / Math.max(1, transport.end - transport.detach));
        return { x: placeDelta.x * w2, y: placeDelta.y * w2 };
      }
      return ZERO;
    };
  }

  return { smooth: smooth, resolveJoint: resolveJoint, setPose: setPose, TransportBlender: TransportBlender, ZERO: ZERO };
})();
