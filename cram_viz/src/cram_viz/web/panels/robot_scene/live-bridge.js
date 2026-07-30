/* ============================================================================
 * panels/robot_scene/live-bridge.js — HTTP polling client for the live-viz
 * bridge (cram-viz-live): probes whether a bridge is running, polls its
 * world state while attached, and posts dragged-object moves back to it.
 *
 * Pure networking/timers — knows nothing about three.js or the scene graph.
 * `destroy()` clears its timers so the panel can tear this down on unmount.
 * ==========================================================================*/
window.RobotSceneLiveBridge = (function () {
  const LIVE_POLL_INTERVAL_MS = 66;        // ~15 Hz render updates
  const LIVE_PROBE_INTERVAL_MS = 3000;     // how often to check for a bridge before attaching
  const LIVE_FAIL_LIMIT = 30;              // consecutive poll failures before auto-detach
  const LIVE_MOVE_THROTTLE_MS = 100;       // min gap between non-final drag move posts
  const LIVE_CATALOG_RECONCILE_EVERY = 45; // polls between object-catalog reconciles (~3s)

  function round3(v) { return Math.round(v * 1000) / 1000; }

  function LiveBridge(opts) {
    // opts: { onProbeResult(available), onFrame(state), onAttach(), onDetach(), onReconcileTick() }
    let on = false, pollTimer = null, probeTimer = null, fails = 0, lastSeq = -1, polls = 0;
    let lastMovePostAt = 0;

    function url() {
      const m = /[?&]live=([\w.:-]+)/.exec(window.location.search);
      return 'http://' + (m ? m[1] : (window.location.hostname + ':8765'));
    }

    function probe() {
      fetch(url() + '/info').then(function (r) { return r.json(); })
        .then(function (info) { if (!on && opts.onProbeResult) opts.onProbeResult(!!info); })
        .catch(function () { if (!on && opts.onProbeResult) opts.onProbeResult(false); });
    }

    function detach() {
      on = false;
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
      if (opts.onDetach) opts.onDetach();
    }

    function poll() {
      if (++polls % LIVE_CATALOG_RECONCILE_EVERY === 0 && opts.onReconcileTick) opts.onReconcileTick();
      fetch(url() + '/state').then(function (r) { return r.json(); })
        .then(function (st) {
          fails = 0;
          if (st.seq !== lastSeq) { lastSeq = st.seq; if (opts.onFrame) opts.onFrame(st); }
        })
        .catch(function () { if (++fails > LIVE_FAIL_LIMIT) detach(); });
    }

    this.url = url;
    this.isOn = function () { return on; };

    this.startProbing = function () {
      probe();
      probeTimer = setInterval(probe, LIVE_PROBE_INTERVAL_MS);
    };

    this.attach = function () {
      on = true; lastSeq = -1; polls = 0;
      pollTimer = setInterval(poll, LIVE_POLL_INTERVAL_MS);
      if (opts.onAttach) opts.onAttach();
    };

    this.detach = detach;

    this.postMove = function (key, x, y, z, final) {
      const now = performance.now();
      if (!final && now - lastMovePostAt < LIVE_MOVE_THROTTLE_MS) return;
      lastMovePostAt = now;
      fetch(url() + '/move', {
        method: 'POST',
        body: JSON.stringify({ object: key, position: [round3(x), round3(y), round3(z)], final: !!final }),
      }).catch(function () {});
    };

    this.fetchObjects = function () { return fetch(url() + '/objects').then(function (r) { return r.json(); }); };

    this.destroy = function () {
      if (pollTimer) clearInterval(pollTimer);
      if (probeTimer) clearInterval(probeTimer);
    };
  }

  return { LiveBridge: LiveBridge };
})();
