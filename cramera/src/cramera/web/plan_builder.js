/* ============================================================================
 * plan_builder.js — the Plan Builder page: compose a plan by drag-and-drop and
 * place objects in a top-down scene, then generate a runnable coraplex demo file.
 * A page script (owns the whole document), like models_page.js.
 * ==========================================================================*/
(function () {
  'use strict';

  // ---- available object meshes (coraplex/resources/objects) ----
  // kitchen items + an industrial/factory set (a robot carries these A->B on a shop floor)
  const MESHES = ['milk.stl', 'bowl.stl', 'spoon.stl', 'breakfast_cereal.stl', 'jeroen_cup.stl',
    'Static_CokeBottle.stl', 'big-knife.stl', 'whisk.stl', 'bread.stl', 'apartment_bowl.stl',
    'wrench.stl', 'axle.stl', 'plate.stl', 'base.stl', 'open_crate.stl'];
  const OBJ_COLORS = ['#e6ecff', '#e6c07f', '#9aa1ad', '#8fd6c8', '#c9a0ff', '#ff9db1', '#9ecb6b'];

  // ---- action blocks ----
  const BLOCKS = {
    park_arms: { name: 'Park arms', color: '#b98cff', params: { arm: 'BOTH' } },
    move_torso: { name: 'Move torso', color: '#ff9db1', params: { torso: 'HIGH' } },
    navigate: { name: 'Navigate', color: '#8fd6c8', params: { x: 2.6, y: 1.8, z: 0.0, yaw: 0.0 } },
    transport: { name: 'Transport object', color: '#5b8cff', params: { object: '', x: 5.0, y: 3.3, z: 0.8, yaw: 1.57, arm: 'LEFT', targetMode: 'semantic', surfaceType: 'CounterTop', surfaceName: '' } },
  };
  const ARMS = ['LEFT', 'RIGHT', 'BOTH'];
  const TORSO = ['HIGH', 'MID', 'LOW'];
  // selectable robots -> the class + import to emit; RobotSpecification derives the drive
  // from the robot's mobile base, so no drive type needs spelling out here.
  const ROBOTS = {
    PR2: { cls: 'PR2', module: 'pr2' },
    Garmi: { cls: 'Garmi', module: 'garmi' },
    HSRB: { cls: 'HSRB', module: 'hsrb' },
    Tiago: { cls: 'Tiago', module: 'tiago' },
    Stretch: { cls: 'Stretch', module: 'stretch' },
    Armar7: { cls: 'Armar7', module: 'armar7' },
    Justin: { cls: 'Justin', module: 'justin' },
    ICub3: { cls: 'ICub3', module: 'icub3' },
    MMPDresden: { cls: 'MMPDresden', module: 'mmp_dresden' },
  };
  Object.keys(ROBOTS).forEach(function (k) {
    const r = ROBOTS[k]; r.import = 'from semantic_digital_twin.robots.' + r.module + ' import ' + r.cls;
  });
  function robotInfo() { const v = ($('pb-robot') && $('pb-robot').value) || 'PR2'; return ROBOTS[v] || ROBOTS.PR2; }
  // semantic place targets: supporting surfaces ("on") and case containers ("in").
  // Both expose HasSupportingSurface.sample_points_from_surface, so resolution is identical.
  const SEMANTIC_SURFACES = ['CounterTop', 'Table', 'ShelfLayer', 'Floor', 'Sofa'];
  const SEMANTIC_CONTAINERS = ['Drawer', 'Fridge', 'Cabinet', 'Cupboard', 'Dresser', 'Dishwasher'];
  const SEMANTIC_TYPES = SEMANTIC_SURFACES.concat(SEMANTIC_CONTAINERS);
  function isContainer(t) { return SEMANTIC_CONTAINERS.indexOf(t) >= 0; }
  function prep(t) { return isContainer(t) ? 'in' : 'on'; }
  const DEFAULT_START = { x: 2.4, y: 2.2, z: 0.95, yaw: 0.0 };   // start pose used when an object was never placed/captured
  let liveSurfaces = [];   // [{type, name}] fetched from the live world when the scene runs

  // ---- constraints: natural language -> giskardpy goal (same rule-based mapping as the Plan view) ----
  let CONSTRAINTS = [
    { id: 'c1', text: 'Milk must always stay upright' },
    { id: 'c2', text: 'Robot must look where it operates' },
    { id: 'c3', text: 'Keep the bowl above the table' },
  ];
  let conSeq = 4;
  function objIn(text, node) {
    const m = String(text).toLowerCase().match(/\b(milk|bowl|spoon|fork|knife|plate|cup|mug|tray|bottle|flask|vial|beaker|tube|rack|sample|cereal|box|jar|glass|can|whisk|bread)\b/);
    if (m) return m[1];
    if (node && node.object) return String(node.object).replace(/\.(stl|obj|dae)$/i, '');
    return 'object';
  }
  function lenIn(text) {
    const m = String(text).toLowerCase().match(/(\d+(?:\.\d+)?)\s*(mm|cm|centimet(?:er|re)s?|m\b|met(?:er|re)s?)/);
    if (!m) return null;
    const v = parseFloat(m[1]), u = m[2];
    if (u.indexOf('mm') === 0) return v / 1000;
    if (u.indexOf('c') === 0) return v / 100;
    return v;
  }
  // node = the step's params (so a Transport step's `object` is the fallback body)
  function compileConstraint(text, node) {
    const t = String(text).toLowerCase();
    const o = objIn(text, node);
    const d = lenIn(t);
    if (/upright|stand up|stay up|vertical|straight up|tip over|tips?\b|tilt|spill|level|flat|horizontal|steady|balanc|no spill|don.?t (tip|spill|tilt)/.test(t))
      return { goal: 'VectorsAligned', params: { root_link: 'map', tip_link: o, tip_normal: [0, 0, 1], goal_normal: [0, 0, 1], threshold: 0.1 } };
    if (/look|watch|gaze|point (at|the camera)|face the|observ|keep .*(in view|an eye)|focus on|keep sight|see the|where it (operat|work)/.test(t))
      return { goal: 'PointingAt', params: { tip_link: 'head_camera', root_link: 'map', pointing_axis: [0, 0, 1], goal_point: '@operation_target', goal_point_body: o, threshold: 0.05 } };
    if (/above|higher|over the|off the (table|ground|surface|bench)|keep .*(high|up high|elevated)|lift(ed)? (up|above)?/.test(t))
      return { goal: 'HeightMonitor', params: { tip_link: o, lower_limit: (d != null ? d : 0.05), upper_limit: 2.0 } };
    if (/below|under(neath)?|lower than|keep .*(low|down|close to the (table|surface|ground))/.test(t))
      return { goal: 'HeightMonitor', params: { tip_link: o, lower_limit: 0.0, upper_limit: (d != null ? d : 0.1) } };
    if (/away from|keep .*clear|clearance|distance|avoid|don.?t (hit|touch|collide|bump)|too close|stay .*away|far from|min(imum)? distance/.test(t))
      return { goal: 'DistanceMonitor', params: { tip_link: o, lower_limit: (d != null ? d : 0.05), upper_limit: 5.0 } };
    return { goal: null, params: {} };
  }
  const CON_INFO_ROWS = [
    ['upright, level, flat, tilt, spill, steady, balanced', 'VectorsAligned', "keep the object's up-axis aligned with world up"],
    ['look, watch, observe, "keep in view", gaze, face', 'PointingAt', 'aim the head camera at the object'],
    ['above, higher, "off the table", "keep high", lift', 'HeightMonitor', 'keep the object at/above a height'],
    ['below, under, "lower than", "keep low"', 'HeightMonitor', 'keep the object below a height'],
    ['"away from", clearance, distance, avoid, "keep clear"', 'DistanceMonitor', 'keep a minimum distance / clearance'],
  ];

  // ---- state ----
  let steps = [];       // [{type, params:{...}}]
  let objects = [];      // [{id, mesh, name, x, y, z, yaw, color}]
  let objSeq = 1, stepSeq = 1;
  let robotXY = { x: 1.5, y: 2.5 };   // robot spawn (draggable in the scene)
  let liveOn = false;                 // true while the scaffold scene is up (constraints can be pushed live)

  // scene mapping: origin offset so the typical apartment area sits centred
  const SCALE = 40, ORIGIN_X = 2.5, ORIGIN_Y = 2.0;
  const $ = function (id) { return document.getElementById(id); };

  // ---------- palette ----------
  function renderBlocks() {
    const el = $('pb-blocks'); el.innerHTML = '';
    Object.keys(BLOCKS).forEach(function (k) {
      const b = BLOCKS[k];
      const d = document.createElement('div');
      d.className = 'pb-block'; d.draggable = true; d.dataset.block = k;
      d.innerHTML = '<span class="ic" style="background:' + b.color + '"></span>' + b.name;
      d.addEventListener('dragstart', function (e) { e.dataTransfer.setData('text/plain', 'block:' + k); });
      el.appendChild(d);
    });
    const meshSel = $('pb-mesh'); meshSel.innerHTML = MESHES.map(function (m) { return '<option>' + m + '</option>'; }).join('');
    const robotSel = $('pb-robot');
    if (robotSel) robotSel.innerHTML = Object.keys(ROBOTS).map(function (k) { return '<option value="' + k + '">' + k + '</option>'; }).join('');
  }

  // ---------- objects ----------
  // A clear, visible staging pose for a freshly added object: a row beside the robot,
  // lifted above typical furniture so it never spawns hidden inside a box/cabinet. The
  // robot's own spot is collision-free, so its surroundings are a safe place to appear;
  // you then drag the object onto its real target (the drag snaps it to the surface).
  function stagingPose() {
    const n = objects.length;
    const perRow = 5, gap = 0.28;
    return {
      x: robotXY.x + 0.7 + (n % perRow) * gap,   // a row extending beside the robot
      y: robotXY.y - 0.7 - Math.floor(n / perRow) * gap,
      z: 1.35,                                    // above counters/tables, so it's visible
    };
  }
  function addObject(mesh, opts) {
    opts = opts || {};
    const stage = stagingPose();
    const o = { id: 'o' + (objSeq++), mesh: mesh, name: mesh,
      x: opts.x != null ? opts.x : stage.x, y: opts.y != null ? opts.y : stage.y,
      z: opts.z != null ? opts.z : stage.z, yaw: opts.yaw != null ? opts.yaw : 0.0,
      color: OBJ_COLORS[(objSeq) % OBJ_COLORS.length] };
    objects.push(o); renderObjects(); renderScene(); refreshObjectSelects();
    return o;
  }
  function renderObjects() {
    const el = $('pb-objects'); el.innerHTML = '';
    objects.forEach(function (o) {
      const d = document.createElement('div'); d.className = 'pb-obj';
      d.innerHTML =
        '<div class="row1"><span class="pb-swatch" style="background:' + o.color + '"></span>' +
        '<span class="oname" title="' + o.mesh + '">' + o.name + '</span>' +
        '<span class="ocap" data-cap="' + o.id + '" title="drag the object to its start position in the 3D scene, then click to capture that as its start pose">⟳ capture</span>' +
        '<span class="oreset" data-reset="' + o.id + '" title="move the object in the 3D scene back to these coordinates (undo a bad drag/snap)">⟲</span>' +
        '<span class="odel" data-del="' + o.id + '">×</span></div>' +
        '<div class="fields">' +
        field(o, 'x') + field(o, 'y') + field(o, 'z') + field(o, 'yaw') + '</div>';
      el.appendChild(d);
    });
    el.querySelectorAll('.pb-num').forEach(function (inp) {
      inp.addEventListener('input', function () {
        const o = objects.find(function (x) { return x.id === inp.dataset.oid; });
        if (o) { o[inp.dataset.k] = parseFloat(inp.value) || 0; renderScene(); }
      });
    });
    el.querySelectorAll('.odel').forEach(function (x) {
      x.addEventListener('click', function () { objects = objects.filter(function (o) { return o.id !== x.dataset.del; }); renderObjects(); renderScene(); refreshObjectSelects(); });
    });
    el.querySelectorAll('.ocap').forEach(function (x) {
      x.addEventListener('click', function () { captureObject(x.dataset.cap); });
    });
    el.querySelectorAll('.oreset').forEach(function (x) {
      x.addEventListener('click', function () { resetObject(x.dataset.reset); });
    });
  }
  // move an object in the live 3D scene back to its builder coordinates (undo a bad snap)
  function resetObject(oid) {
    const o = objects.find(function (x) { return x.id === oid; }); if (!o) return;
    // tell the embedded 3D scene to move the mesh back (the idle sim won't apply a
    // queued /move, so a visual reset must go through the viewer itself)
    const f = $('pb-3d');
    if (f && f.contentWindow) f.contentWindow.postMessage(
      { type: 'cramera-reset-object', key: o.mesh, position: [o.x, o.y, o.z] }, '*');
    // also update the bridge's last-move overlay so a later capture reads the reset pose
    fetch(bridgeUrl() + '/move', { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ object: o.mesh, position: [o.x, o.y, o.z], final: true }) })
      .then(function () { status('reset ' + o.name + ' in the 3D scene → (' + o.x + ', ' + o.y + ', ' + o.z + ')', 'ok'); })
      .catch(function () { status('reset failed — start the live scene first', 'err'); });
  }
  function resetAllObjects() { objects.forEach(function (o) { resetObject(o.id); }); }
  function field(o, k) {
    return '<label>' + k.toUpperCase() + '<input class="pb-num" data-oid="' + o.id + '" data-k="' + k + '" type="number" step="0.05" value="' + o[k] + '"' + (o.base && k !== 'yaw' ? '' : '') + '></label>';
  }

  // ---------- constraints palette ----------
  function renderConstraints() {
    const el = $('pb-cons'); if (!el) return;
    el.innerHTML = CONSTRAINTS.map(function (c) {
      const comp = compileConstraint(c.text, null);
      const badge = comp.goal ? '<span class="pb-con-goal" title="translates to giskardpy ' + comp.goal + '">' + comp.goal + '</span>'
        : '<span class="pb-con-goal nomatch" title="no rule matched — this text will not translate to a goal">no match</span>';
      return '<div class="pb-con" draggable="true" data-cid="' + c.id + '">' +
        '<span class="pb-con-grip">⠿</span><span class="pb-con-txt">' + c.text + '</span>' + badge +
        '<span class="pb-con-del" data-del="' + c.id + '">×</span></div>';
    }).join('');
    el.querySelectorAll('.pb-con').forEach(function (card) {
      card.addEventListener('dragstart', function (e) { e.dataTransfer.setData('text/plain', 'con:' + card.dataset.cid); e.dataTransfer.effectAllowed = 'copy'; });
    });
    el.querySelectorAll('.pb-con-del').forEach(function (x) {
      x.addEventListener('click', function (e) { e.stopPropagation(); CONSTRAINTS = CONSTRAINTS.filter(function (c) { return c.id !== x.dataset.del; }); renderConstraints(); });
    });
  }
  function addConstraintText(txt) {
    const v = String(txt || '').trim(); if (!v) return;
    CONSTRAINTS.push({ id: 'c' + (conSeq++), text: v }); renderConstraints();
  }
  // attach a constraint (by palette id) to a plan step
  function attachConstraint(stepId, cid) {
    const s = steps.find(function (x) { return x.id === stepId; });
    const c = CONSTRAINTS.find(function (x) { return x.id === cid; });
    if (!s || !c) return;
    const comp = compileConstraint(c.text, s.params);
    if (!comp.goal) { status('“' + c.text + '” — no rule matched, not attached', 'err'); return; }
    s.constraints = s.constraints || [];
    if (s.constraints.some(function (a) { return a.text === c.text; })) { status('already attached to this step', ''); return; }
    s.constraints.push({ text: c.text, goal: comp.goal, params: comp.params });
    renderSteps();
    if (liveOn) pushConstraintLive(s, { text: c.text, goal: comp.goal, params: comp.params });
    else status('attached “' + c.text + '” → ' + comp.goal + ' (start the live scene to apply it)', 'ok');
  }
  function detachConstraint(stepId, idx) {
    const s = steps.find(function (x) { return x.id === stepId; }); if (!s || !s.constraints) return;
    s.constraints.splice(idx, 1); renderSteps();
  }
  // push a constraint to the running scaffold's bridge (same endpoint the Plan view uses)
  function pushConstraintLive(s, a) {
    const b = BLOCKS[s.type];
    const body = { op: 'attach_monitor', text: a.text, apply: 'next_activation',
      target_plan_node: { id: s.id, kind: s.type, label: (b ? b.name : s.type) },
      giskard_node: { type: a.goal, params: a.params } };
    fetch(bridgeUrl() + '/constraint', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (j && j.ok) status('attached “' + a.text + '” → ' + a.goal + ' — queued in the live plan (next activation)', 'ok');
        else status('live attach failed: ' + ((j && j.error) || '?'), 'err');
      })
      .catch(function (e) { status('live attach failed: ' + e, 'err'); });
  }
  function conInfoHtml() {
    const rows = CON_INFO_ROWS.map(function (r) {
      return '<tr><td>' + r[0] + '</td><td class="goal">' + r[1] + '</td><td>' + r[2] + '</td></tr>';
    }).join('');
    return '<div class="ci-h">How constraints are translated <span class="ci-note">(rule-based, not an LLM)</span></div>' +
      '<table class="ci-table"><thead><tr><th>Phrasing</th><th>giskardpy goal</th><th>Effect</th></tr></thead><tbody>' + rows + '</tbody></table>' +
      '<div class="ci-foot">A length in the text (<code>10 cm</code>, <code>0.1 m</code>) sets the thresholds. ' +
      'The object comes from the sentence or, on a Transport step, its transported object. ' +
      'Applied to the running plan on the next motion activation.</div>';
  }

  // ---------- scene (top-down) ----------
  function worldToPx(x, y) {
    const sc = $('pb-scene'); const w = sc.clientWidth, h = sc.clientHeight;
    return { px: w / 2 + (x - ORIGIN_X) * SCALE, py: h / 2 - (y - ORIGIN_Y) * SCALE };
  }
  function pxToWorld(px, py) {
    const sc = $('pb-scene'); const w = sc.clientWidth, h = sc.clientHeight;
    return { x: ORIGIN_X + (px - w / 2) / SCALE, y: ORIGIN_Y - (py - h / 2) / SCALE };
  }
  function renderScene() {
    const sc = $('pb-scene');
    if (!sc) return;   // the 2D scene was replaced by the live 3D capture flow
    sc.querySelectorAll('.pb-marker,.pb-tmarker').forEach(function (m) { m.remove(); });
    // robot spawn (draggable)
    const rp = worldToPx(robotXY.x, robotXY.y);
    const rm = document.createElement('div'); rm.className = 'pb-marker pb-rm';
    rm.style.left = rp.px + 'px'; rm.style.top = rp.py + 'px'; rm.style.background = '#38405c'; rm.style.fontSize = '13px';
    rm.innerHTML = '<span class="lbl">robot</span>🤖';
    rm.addEventListener('mousedown', function (e) {
      dragMarker(e, rm, function (x, y) { robotXY.x = x; robotXY.y = y; }, function () {});
    });
    sc.appendChild(rm);
    // objects (draggable)
    objects.forEach(function (o) {
      const p = worldToPx(o.x, o.y);
      const m = document.createElement('div'); m.className = 'pb-marker';
      m.style.left = p.px + 'px'; m.style.top = p.py + 'px'; m.style.background = o.color;
      m.innerHTML = '<span class="lbl">' + o.name.replace(/\.stl$/i, '') + '</span>' + o.name.charAt(0).toUpperCase();
      m.addEventListener('mousedown', function (e) {
        dragMarker(e, m, function (x, y) { o.x = x; o.y = y; syncNum(o.id); }, function () { renderObjects(); });
      });
      sc.appendChild(m);
    });
    // transport destinations (ghost target, draggable) — "where the object should go"
    steps.forEach(function (s, i) {
      if (s.type !== 'transport') return;
      const p = worldToPx(s.params.x, s.params.y);
      const m = document.createElement('div'); m.className = 'pb-marker pb-tmarker';
      m.style.left = p.px + 'px'; m.style.top = p.py + 'px';
      m.innerHTML = '<span class="lbl">→ ' + (s.params.object ? s.params.object.replace(/\.stl$/i, '') : 'step ' + (i + 1)) + '</span>◎';
      m.addEventListener('mousedown', function (e) {
        dragMarker(e, m, function (x, y) { s.params.x = x; s.params.y = y; syncStepNum(s.id); }, function () { renderSteps(); });
      });
      sc.appendChild(m);
    });
  }
  function dragMarker(e, m, apply, onEnd) {
    e.preventDefault();
    const sc = $('pb-scene');
    function move(ev) {
      const r = sc.getBoundingClientRect();
      const px = Math.max(0, Math.min(r.width, ev.clientX - r.left));
      const py = Math.max(0, Math.min(r.height, ev.clientY - r.top));
      m.style.left = px + 'px'; m.style.top = py + 'px';
      const w = pxToWorld(px, py); apply(Math.round(w.x * 100) / 100, Math.round(w.y * 100) / 100);
    }
    function up() { document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); onEnd(); }
    document.addEventListener('mousemove', move); document.addEventListener('mouseup', up);
  }
  function syncNum(oid) {
    document.querySelectorAll('.pb-num[data-oid="' + oid + '"]').forEach(function (inp) {
      const o = objects.find(function (x) { return x.id === oid; }); if (!o) return;
      if (inp.dataset.k === 'x') inp.value = o.x; if (inp.dataset.k === 'y') inp.value = o.y;
    });
  }
  function syncStepNum(sid) {
    document.querySelectorAll('.pb-num[data-sid="' + sid + '"]').forEach(function (inp) {
      const s = steps.find(function (x) { return x.id === sid; }); if (!s) return;
      if (inp.dataset.k === 'x') inp.value = s.params.x; if (inp.dataset.k === 'y') inp.value = s.params.y;
    });
  }

  // ---------- plan steps ----------
  function addStep(type) {
    const b = BLOCKS[type]; if (!b) return;
    const params = Object.assign({}, b.params);
    if (type === 'transport' && !params.object && objects.length) params.object = objects[0].mesh;
    steps.push({ id: 's' + (stepSeq++), type: type, params: params });
    renderSteps();
  }
  function renderSteps() {
    const el = $('pb-steps');
    $('pb-step-count').textContent = steps.length ? '(' + steps.length + ')' : '';
    if (!steps.length) { el.innerHTML = '<div class="pb-drop-hint">Drop action blocks here to build the sequence</div>'; renderScene(); return; }
    el.innerHTML = '';
    steps.forEach(function (s, i) {
      const b = BLOCKS[s.type];
      const d = document.createElement('div'); d.className = 'pb-step'; d.style.borderLeftColor = b.color;
      d.dataset.sid = s.id;
      d.innerHTML =
        '<div class="sh"><span class="snum">' + (i + 1) + '</span><span class="sname">' + b.name + '</span>' +
        '<span class="sctl"><button data-up="' + s.id + '" title="Move up">↑</button>' +
        '<button data-down="' + s.id + '" title="Move down">↓</button>' +
        '<button data-del="' + s.id + '" title="Remove">×</button></span></div>' +
        stepChips(s) +
        '<div class="sparams">' + stepParams(s) + '</div>';
      el.appendChild(d);
    });
    wireStepEvents();
    renderScene();
  }
  function stepChips(s) {
    const cs = s.constraints || [];
    if (!cs.length) return '';
    return '<div class="sconstraints">' + cs.map(function (a, idx) {
      return '<span class="scon-chip" title="' + a.text + ' → giskardpy ' + a.goal + '">⛓ ' + a.text +
        '<span class="scon-goal">' + a.goal + '</span>' +
        '<span class="scon-x" data-scon-del="' + s.id + '" data-scon-idx="' + idx + '">×</span></span>';
    }).join('') + '</div>';
  }
  function row(html) { return '<div class="sparam-row">' + html + '</div>'; }
  function stepParams(s) {
    if (s.type === 'park_arms') return row(sel(s, 'arm', ARMS));
    if (s.type === 'move_torso') return row(sel(s, 'torso', TORSO));
    if (s.type === 'navigate') return row('<span class="pb-group-lbl">go to →</span>' + num(s, 'x') + num(s, 'y') + num(s, 'z') + num(s, 'yaw'));
    if (s.type === 'transport') {
      const mode = s.params.targetMode || 'semantic';
      const dropRow = (mode === 'semantic')
        ? row('<span class="pb-group-lbl">place →</span>' + semanticTypeSel(s) + surfaceInstanceSel(s))
        : row('<span class="pb-group-lbl">drop-off (to) →</span>' + num(s, 'x') + num(s, 'y') + num(s, 'z') + num(s, 'yaw') +
          '<button class="pb-capbtn" data-capstep="' + s.id + '" title="drag the object to its drop-off in the 3D scene, then capture that pose as this step\'s target">◎ capture</button>');
      return (
        row(objSel(s)) +
        row('<span class="pb-group-lbl start">start (from) →</span>' +
          '<button class="pb-capbtn start" data-capstart="' + s.id + '" title="drag the object to its START in the 3D scene, then capture that as its start pose (shown on the object card)">◎ capture</button>') +
        row('<span class="pb-group-lbl">target →</span>' + modeSel(s)) +
        dropRow +
        row(sel(s, 'arm', ARMS))
      );
    }
    return '';
  }
  function num(s, k) { return '<label>' + k.toUpperCase() + '<input class="pb-num xyz" data-sid="' + s.id + '" data-k="' + k + '" type="number" step="0.05" value="' + s.params[k] + '"></label>'; }
  function sel(s, k, opts) { return '<label>' + k + '<select class="pb-sel" data-sid="' + s.id + '" data-k="' + k + '">' + opts.map(function (o) { return '<option' + (s.params[k] === o ? ' selected' : '') + '>' + o + '</option>'; }).join('') + '</select></label>'; }
  // a select whose option values differ from their labels: pairs = [[value, label], ...]
  function selPairs(s, k, pairs) {
    return '<select class="pb-sel" data-sid="' + s.id + '" data-k="' + k + '">' + pairs.map(function (p) {
      return '<option value="' + p[0] + '"' + ((s.params[k] || '') === p[0] ? ' selected' : '') + '>' + p[1] + '</option>';
    }).join('') + '</select>';
  }
  function modeSel(s) { return selPairs(s, 'targetMode', [['semantic', 'semantic location'], ['pose', 'exact pose (XYZ)']]); }
  // semantic type dropdown, grouped into "on a surface" / "in a container"
  function semanticTypeSel(s) {
    function grp(label, types) {
      return '<optgroup label="' + label + '">' + types.map(function (t) {
        return '<option value="' + t + '"' + ((s.params.surfaceType || '') === t ? ' selected' : '') + '>' + prep(t) + ' ' + t + '</option>';
      }).join('') + '</optgroup>';
    }
    return '<select class="pb-sel" data-sid="' + s.id + '" data-k="surfaceType">' +
      grp('on a surface', SEMANTIC_SURFACES) + grp('in a container', SEMANTIC_CONTAINERS) + '</select>';
  }
  // instance dropdown: "first found" + any live-enumerated instances of the chosen type
  function surfaceInstanceSel(s) {
    const t = s.params.surfaceType || 'CounterTop';
    const inst = liveSurfaces.filter(function (x) { return x.type === t; });
    const pairs = [['', 'first found']].concat(inst.map(function (x) { return [x.name, x.name]; }));
    return selPairs(s, 'surfaceName', pairs);
  }
  function objSel(s) {
    // keep the param in sync with the visibly-selected first option, so capture works
    // even for a Transport step whose object dropdown was never touched
    if (!s.params.object && objects.length) s.params.object = objects[0].mesh;
    const opts = objects.map(function (o) { return '<option value="' + o.mesh + '"' + (s.params.object === o.mesh ? ' selected' : '') + '>' + o.name + '</option>'; }).join('');
    return '<label>object<select class="pb-sel" data-sid="' + s.id + '" data-k="object">' + (opts || '<option value="">— add an object —</option>') + '</select></label>';
  }
  function wireStepEvents() {
    const el = $('pb-steps');
    el.querySelectorAll('.pb-num,.pb-sel').forEach(function (inp) {
      inp.addEventListener('input', function () {
        const s = steps.find(function (x) { return x.id === inp.dataset.sid; }); if (!s) return;
        const k = inp.dataset.k;
        const v = inp.classList.contains('pb-num') ? (parseFloat(inp.value) || 0) : inp.value;
        s.params[k] = v;
        // switching target mode / surface type swaps which fields are shown -> re-render
        if (k === 'surfaceType') { s.params.surfaceName = ''; renderSteps(); }
        else if (k === 'targetMode') { renderSteps(); }
        else { renderScene(); }
      });
    });
    el.querySelectorAll('[data-del]').forEach(function (b) { b.addEventListener('click', function () { steps = steps.filter(function (s) { return s.id !== b.dataset.del; }); renderSteps(); }); });
    el.querySelectorAll('[data-up]').forEach(function (b) { b.addEventListener('click', function () { moveStep(b.dataset.up, -1); }); });
    el.querySelectorAll('[data-down]').forEach(function (b) { b.addEventListener('click', function () { moveStep(b.dataset.down, 1); }); });
    el.querySelectorAll('[data-capstep]').forEach(function (b) { b.addEventListener('click', function (e) { e.preventDefault(); captureStepTarget(b.dataset.capstep); }); });
    el.querySelectorAll('[data-capstart]').forEach(function (b) { b.addEventListener('click', function (e) { e.preventDefault(); captureStepStart(b.dataset.capstart); }); });
    // remove an attached constraint chip
    el.querySelectorAll('[data-scon-del]').forEach(function (x) {
      x.addEventListener('click', function (e) { e.stopPropagation(); detachConstraint(x.dataset.sconDel, parseInt(x.dataset.sconIdx, 10)); });
    });
    // each step is a drop target for a constraint card
    el.querySelectorAll('.pb-step').forEach(function (st) {
      st.addEventListener('dragover', function (e) {
        // the dragged payload isn't readable during dragover, so allow the drop and
        // decide on drop() below (only con: payloads actually attach)
        e.preventDefault(); st.classList.add('con-drop');
      });
      st.addEventListener('dragleave', function () { st.classList.remove('con-drop'); });
      st.addEventListener('drop', function (e) {
        st.classList.remove('con-drop');
        const d = e.dataTransfer.getData('text/plain') || '';
        if (d.indexOf('con:') === 0) { e.preventDefault(); e.stopPropagation(); attachConstraint(st.dataset.sid, d.slice(4)); }
      });
    });
  }
  function moveStep(id, dir) {
    const i = steps.findIndex(function (s) { return s.id === id; }); const j = i + dir;
    if (i < 0 || j < 0 || j >= steps.length) return;
    const t = steps[i]; steps[i] = steps[j]; steps[j] = t; renderSteps();
  }
  function refreshObjectSelects() { renderSteps(); }

  // drop zone
  const stepsEl = $('pb-steps');
  stepsEl.addEventListener('dragover', function (e) { e.preventDefault(); stepsEl.classList.add('drop-ok'); });
  stepsEl.addEventListener('dragleave', function () { stepsEl.classList.remove('drop-ok'); });
  stepsEl.addEventListener('drop', function (e) {
    e.preventDefault(); stepsEl.classList.remove('drop-ok');
    const d = e.dataTransfer.getData('text/plain') || '';
    if (d.indexOf('block:') === 0) addStep(d.slice(6));
  });

  // ---------- code generation ----------
  function py(v) { return (Math.round(v * 1000) / 1000).toString(); }
  function jsonStr(s) { return JSON.stringify(String(s)); }
  // python literal for a constraint param value (list / string / number)
  function jsonPy(v) {
    if (Array.isArray(v)) return '[' + v.map(jsonPy).join(', ') + ']';
    if (typeof v === 'object' && v) return '{' + Object.keys(v).map(function (k) { return jsonStr(k) + ': ' + jsonPy(v[k]); }).join(', ') + '}';
    if (typeof v === 'string') return jsonStr(v);
    return String(v);
  }
  function pyKwargs(params) {
    return Object.keys(params).map(function (k) { return k + '=' + jsonPy(params[k]); }).join(', ');
  }
  // --- "place on a surface": symbolic target resolution via semantic_digital_twin ---
  function surfaceSteps(useSteps) {
    return useSteps.filter(function (s) { return s.type === 'transport' && s.params.targetMode === 'semantic'; });
  }
  function surfaceTypesUsed(useSteps) {
    const set = {}; surfaceSteps(useSteps).forEach(function (s) { set[s.params.surfaceType || 'CounterTop'] = 1; });
    return Object.keys(set);
  }
  // lines that resolve each surface-mode transport into a `_target_<id>` Pose, given `world`.
  // Fails with a clear message (not a bare IndexError/StopIteration) when the surface is
  // missing, so a mismatched environment is obvious.
  function surfaceResolveLines(useSteps, indent) {
    const L = [];
    useSteps.forEach(function (s, i) {
      if (!(s.type === 'transport' && s.params.targetMode === 'semantic')) return;
      const T = s.params.surfaceType || 'CounterTop';
      const mesh = s.params.object || 'object';
      const id = s.id;
      const where = 'step ' + (i + 1) + ' (transport ' + mesh + ')';
      L.push(indent + '# place "' + mesh + '" ' + prep(T) + ' a ' + T + ' — pose sampled by semantic_digital_twin');
      if (s.params.surfaceName) {
        L.push(indent + '_surface_' + id + ' = next(');
        L.push(indent + '    (s for s in world.get_semantic_annotations_by_type(' + T + ')');
        L.push(indent + '     if str(s.root.name) == ' + jsonStr(s.params.surfaceName) + '),');
        L.push(indent + '    None,');
        L.push(indent + ')');
        L.push(indent + 'if _surface_' + id + ' is None:');
        L.push(indent + '    raise RuntimeError(');
        L.push(indent + '        "' + T + ' ' + jsonStr(s.params.surfaceName).slice(1, -1) +
          ' not found in this world for ' + where + '. "');
        L.push(indent + '        "See the Plan Builder\'s live /surfaces list for available surfaces."');
        L.push(indent + '    )');
      } else {
        L.push(indent + '_surfaces_' + id + ' = world.get_semantic_annotations_by_type(' + T + ')');
        L.push(indent + 'if not _surfaces_' + id + ':');
        L.push(indent + '    raise RuntimeError(');
        L.push(indent + '        "no ' + T + ' in this world for ' + where + '. "');
        L.push(indent + '        "This environment may not carry that annotation — "');
        L.push(indent + '        "see the Plan Builder\'s live /surfaces list for what is available."');
        L.push(indent + '    )');
        L.push(indent + '_surface_' + id + ' = _surfaces_' + id + '[0]');
      }
      L.push(indent + '_pts_' + id + ' = _surface_' + id + '.sample_points_from_surface(');
      L.push(indent + '    body_to_sample_for=' + body(mesh) + ')');
      L.push(indent + 'if not _pts_' + id + ':');
      L.push(indent + '    raise RuntimeError(');
      L.push(indent + '        "could not sample a free place pose ' + prep(T) + ' ' + T + ' for ' + where + ' "');
      L.push(indent + '        "(surface full or too small for ' + mesh + ')."');
      L.push(indent + '    )');
      L.push(indent + '_target_' + id + ' = Pose(_pts_' + id + '[0], reference_frame=_pts_' + id + '[0].reference_frame)');
    });
    return L;
  }
  // objects to spawn: the placed ones, plus any object a transport step references but that
  // was never placed/captured — spawned at DEFAULT_START so the demo still runs.
  function effectiveObjects(useSteps) {
    const list = objects.slice();
    const have = {}; list.forEach(function (o) { have[o.mesh] = 1; });
    useSteps.forEach(function (s) {
      if (s.type === 'transport' && s.params.object && !have[s.params.object]) {
        have[s.params.object] = 1;
        list.push({ mesh: s.params.object, name: s.params.object,
          x: DEFAULT_START.x, y: DEFAULT_START.y, z: DEFAULT_START.z, yaw: DEFAULT_START.yaw,
          color: '#cccccc', _defaulted: true });
      }
    });
    return list;
  }
  function surfaceImportLine(useSteps) {
    const types = surfaceTypesUsed(useSteps);
    if (!types.length) return null;
    return 'from semantic_digital_twin.semantic_annotations.semantic_annotations import ' + types.sort().join(', ');
  }
  // the constraints-metadata block (comment + CONSTRAINTS list), shared by both output styles
  function constraintBlock(useSteps) {
    const withCon = useSteps.filter(function (s) { return (s.constraints || []).length; });
    if (!withCon.length) return [];
    const L = [];
    L.push('# --- constraints (natural language -> giskardpy goals) ---');
    L.push('# Attached in the Plan Builder. When this demo runs under `cramera-live`, the');
    L.push('# viewer applies them to the motion statechart on the next activation of the');
    L.push('# step (via the live bridge /constraint endpoint). Listed here as plan metadata.');
    useSteps.forEach(function (s, i) {
      (s.constraints || []).forEach(function (a) {
        L.push('#   step ' + (i + 1) + ' ' + (BLOCKS[s.type] ? BLOCKS[s.type].name : s.type) + ': "' + a.text + '"');
        L.push('#     -> ' + a.goal + '(' + pyKwargs(a.params) + ')');
      });
    });
    L.push('CONSTRAINTS = [');
    useSteps.forEach(function (s, i) {
      (s.constraints || []).forEach(function (a) {
        L.push('    {"step": ' + (i + 1) + ', "text": ' + jsonStr(a.text) +
          ', "goal": ' + jsonStr(a.goal) + ', "params": ' + jsonPy(a.params) + '},');
      });
    });
    L.push(']');
    L.push('');
    return L;
  }
  function pose(p) { return 'Pose.from_xyz_rpy(' + py(p.x) + ', ' + py(p.y) + ', ' + py(p.z) + ', yaw=' + py(p.yaw) + ', reference_frame=world.root)'; }
  function body(mesh) { return 'world.get_body_by_name("' + mesh + '")'; }
  function generate(stepsOverride) {
    const useSteps = stepsOverride || steps;
    const added = effectiveObjects(useSteps);
    const env = ($('pb-env') && $('pb-env').value) || 'apartment.urdf';
    const R = robotInfo();
    const L = [];
    L.push('"""Generated by the cramera Plan Builder."""');
    L.push('import os');
    L.push('from coraplex.datastructures.dataclasses import Context');
    L.push('from coraplex.datastructures.enums import Arms, VisualizationBackend');
    L.push('from coraplex.execution_environment import simulated_robot');
    L.push('from coraplex.plans.factories import sequential');
    L.push('from coraplex.visualization import WorldVisualization');
    L.push('from coraplex.robot_plans.actions.composite.transporting import TransportAction');
    L.push('from coraplex.robot_plans.actions.core.navigation import NavigateAction');
    L.push('from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction');
    L.push('from semantic_digital_twin.adapters.mesh import DAEParser, OBJParser, STLParser');
    L.push('from semantic_digital_twin.adapters.urdf import URDFParser');
    L.push('from semantic_digital_twin.datastructures.definitions import TorsoState');
    L.push('from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner');
    L.push(R.import);
    L.push('from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix');
    L.push('from semantic_digital_twin.spatial_types.spatial_types import Pose');
    L.push('from semantic_digital_twin.world_description.geometry import Color');
    const _surfImp = surfaceImportLine(useSteps);
    if (_surfImp) L.push(_surfImp);
    L.push('');
    L.push('_HERE = os.path.dirname(__file__)');
    L.push('_WORLDS = os.path.join(_HERE, "..", "..", "resources", "worlds")');
    L.push('_OBJECTS = os.path.join(_HERE, "..", "..", "resources", "objects")');
    L.push('_MESH_PARSERS = {".stl": STLParser, ".obj": OBJParser, ".dae": DAEParser}');
    L.push('');
    L.push('');
    L.push('def _parse_mesh(mesh):');
    L.push('    """Parse an object mesh into a world, picking the parser by file extension."""');
    L.push('    ext = os.path.splitext(mesh)[1].lower()');
    L.push('    return _MESH_PARSERS.get(ext, STLParser)(os.path.join(_OBJECTS, mesh)).parse()');
    L.push('');
    L.push('');
    L.push('def build_world(env_file, robot_xy):');
    L.push('    """Parse the chosen environment + ' + R.cls + ' and spawn the robot at robot_xy."""');
    L.push('    robot_world = URDFParser.from_file(' + R.cls + '.get_ros_file_path()).parse()');
    L.push('    world = URDFParser.from_file(os.path.join(_WORLDS, env_file)).parse()');
    L.push('    with world.modify_world():');
    L.push('        robot_root = robot_world.get_body_by_name(' + R.cls + '._get_root_body_name())');
    L.push('        drive = ' + R.cls + '.get_drive_connection_type().create_with_dofs(');
    L.push('            parent=world.root, child=robot_root, world=world)');
    L.push('        world.merge_world(robot_world, drive)');
    L.push('        drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(robot_xy[0], robot_xy[1], 0)');
    L.push('    standing = max(0.0, -world.height_of_lowest_collision_point_of_branch(robot_root))');
    L.push('    with world.modify_world():');
    L.push('        drive.parent_T_connection_expression = HomogeneousTransformationMatrix.from_xyz_rpy(');
    L.push('            z=standing, reference_frame=world.root)');
    L.push('    return world');
    L.push('');
    L.push('');
    L.push('world = build_world("' + env + '", (' + py(robotXY.x) + ', ' + py(robotXY.y) + '))');
    L.push('visualization = WorldVisualization.from_environment(');
    L.push('    world, default_backend=VisualizationBackend.CRAMERA).start()');
    L.push('');
    if (added.length) {
      L.push('# --- objects placed in the Plan Builder ---');
      added.forEach(function (o, i) {
        L.push('_obj' + i + ' = _parse_mesh("' + o.mesh + '")');
      });
      L.push('with world.modify_world():');
      added.forEach(function (o, i) {
        L.push('    world.merge_world_at_pose(_obj' + i + ', HomogeneousTransformationMatrix.from_xyz_rpy(');
        L.push('        ' + py(o.x) + ', ' + py(o.y) + ', ' + py(o.z) + ', yaw=' + py(o.yaw) + ', reference_frame=world.root))');
      });
      added.forEach(function (o) {
        const c = hexToRgb(o.color);
        L.push(body(o.mesh) + '.visual.shapes[0].color = Color(' + c[0] + ', ' + c[1] + ', ' + c[2] + ')');
      });
      L.push('');
    }
    L.push('robot = ' + R.cls + '.from_world(world)');
    L.push('context = Context(world=world, robot=robot, _debug=False, ros_node=visualization.ros_node)');
    L.push('with world.modify_world():');
    L.push('    WorldReasoner(world).reason()');
    L.push('context.evaluate_conditions = False');
    L.push('');
    surfaceResolveLines(useSteps, '').forEach(function (ln) { L.push(ln); });
    if (surfaceSteps(useSteps).length) L.push('');
    L.push('plan = sequential([');
    useSteps.forEach(function (s) { L.push('    ' + stepCode(s) + ','); });
    L.push('], context=context).plan');
    L.push('visualization.attach_plan(plan)');
    L.push('');
    constraintBlock(useSteps).forEach(function (ln) { L.push(ln); });
    L.push('with simulated_robot:');
    L.push('    plan.perform()');
    L.push('');
    return L.join('\n');
  }
  function stepCode(s) {
    const p = s.params;
    if (s.type === 'park_arms') return 'ParkArmsAction(Arms.' + p.arm + ')';
    if (s.type === 'move_torso') return 'MoveTorsoAction(TorsoState.' + p.torso + ')';
    if (s.type === 'navigate') return 'NavigateAction(' + pose(p) + ')';
    if (s.type === 'transport') {
      const target = (p.targetMode === 'semantic') ? ('_target_' + s.id) : pose(p);
      return 'TransportAction(' + body(p.object || 'object') + ', ' + target + ', Arms.' + p.arm + ')';
    }
    return 'None';
  }
  function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255].map(function (v) { return Math.round(v * 100) / 100; }); }

  // PascalCase class name from the demo name (e.g. "my_demo" -> "MyDemoDemonstration")
  function className() {
    const base = ($('pb-name').value || 'my_demo').replace(/[^a-z0-9_\- ]/gi, '').replace(/[_\- ]+/g, ' ').trim();
    const cc = base.split(' ').map(function (w) { return w ? w.charAt(0).toUpperCase() + w.slice(1) : ''; }).join('');
    return (cc || 'MyDemo') + 'Demonstration';
  }
  // ---- output style: a coraplex.demonstrations.RobotDemonstration subclass ----
  function generateClass(stepsOverride) {
    const useSteps = stepsOverride || steps;
    const added = effectiveObjects(useSteps);
    const env = ($('pb-env') && $('pb-env').value) || 'apartment.urdf';
    const cls = className();
    const R = robotInfo();
    const L = [];
    L.push('"""Generated by the cramera Plan Builder — a RobotDemonstration subclass."""');
    L.push('import os');
    L.push('from dataclasses import dataclass');
    L.push('');
    L.push('from coraplex.datastructures.dataclasses import Context');
    L.push('from coraplex.datastructures.enums import Arms, VisualizationBackend');
    L.push('from coraplex.demonstrations import RobotDemonstration');
    L.push('from coraplex.plans.factories import sequential');
    L.push('from coraplex.plans.plan_node import PlanNode');
    L.push('from coraplex.robot_plans.actions.composite.transporting import TransportAction');
    L.push('from coraplex.robot_plans.actions.core.navigation import NavigateAction');
    L.push('from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction');
    L.push('from semantic_digital_twin.api import (');
    L.push('    BodySpecification,');
    L.push('    Connection6DoFSpecification,');
    L.push('    RobotSpecification,');
    L.push('    WorldSpecification,');
    L.push(')');
    L.push('from semantic_digital_twin.datastructures.definitions import TorsoState');
    L.push('from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner');
    L.push(R.import);
    L.push('from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix');
    L.push('from semantic_digital_twin.spatial_types.spatial_types import Pose');
    L.push('from semantic_digital_twin.world import World');
    L.push('from semantic_digital_twin.world_description.geometry import Color');
    const _surfImpC = surfaceImportLine(useSteps);
    if (_surfImpC) L.push(_surfImpC);
    L.push('');
    L.push('_HERE = os.path.dirname(__file__)');
    L.push('_WORLDS = os.path.join(_HERE, "..", "..", "resources", "worlds")');
    L.push('_OBJECTS = os.path.join(_HERE, "..", "..", "resources", "objects")');
    L.push('');
    L.push('ENV_FILE = "' + env + '"');
    L.push('ROBOT_XY = (' + py(robotXY.x) + ', ' + py(robotXY.y) + ')');
    L.push('');
    L.push('# objects placed in the Plan Builder: (mesh, x, y, z, yaw, (r, g, b))');
    L.push('OBJECTS = [');
    added.forEach(function (o) {
      const c = hexToRgb(o.color);
      L.push('    ("' + o.mesh + '", ' + py(o.x) + ', ' + py(o.y) + ', ' + py(o.z) + ', ' + py(o.yaw) +
        ', (' + c[0] + ', ' + c[1] + ', ' + c[2] + ')),');
    });
    L.push(']');
    L.push('');
    constraintBlock(useSteps).forEach(function (ln) { L.push(ln); });
    L.push('');
    L.push('@dataclass(kw_only=True)');
    L.push('class ' + cls + '(RobotDemonstration):');
    L.push('    """A demonstration composed in the cramera Plan Builder."""');
    L.push('');
    L.push('    def build_simulated_world(self) -> World:');
    L.push('        return WorldSpecification.from_urdf(');
    L.push('            os.path.join(_WORLDS, ENV_FILE),');
    L.push('            robots=[');
    L.push('                RobotSpecification(');
    L.push('                    semantic_annotation_type=self.used_robot,');
    L.push('                    world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(');
    L.push('                        ROBOT_XY[0], ROBOT_XY[1], 0.0),');
    L.push('                ),');
    L.push('            ],');
    L.push('        ).to_domain_object()');
    L.push('');
    L.push('    def is_scene_populated(self, world: World) -> bool:');
    L.push('        for spec in OBJECTS:');
    L.push('            try:');
    L.push('                world.get_body_by_name(spec[0])');
    L.push('            except Exception:');
    L.push('                return False');
    L.push('        return bool(OBJECTS)');
    L.push('');
    L.push('    def populate_scene(self, world: World) -> None:');
    L.push('        # each object is free to move (Connection6DoF), so the robot can transport it');
    L.push('        for mesh, x, y, z, yaw, rgb in OBJECTS:');
    L.push('            BodySpecification.mesh(');
    L.push('                mesh,');
    L.push('                os.path.join(_OBJECTS, mesh),');
    L.push('                color=Color(*rgb),');
    L.push('                parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(');
    L.push('                    x, y, z, yaw=yaw),');
    L.push('                connection_specification=Connection6DoFSpecification(),');
    L.push('            ).spawn(world)');
    L.push('');
    L.push('    def build_context(self, world: World) -> Context:');
    L.push('        with world.modify_world():');
    L.push('            WorldReasoner(world).reason()');
    L.push('        robot = world.get_semantic_annotations_by_type(self.used_robot)[0]');
    L.push('        context = Context(world=world, robot=robot, _debug=False, ros_node=self.ros_node)');
    L.push('        context.evaluate_conditions = False');
    L.push('        return context');
    L.push('');
    L.push('    def build_plan(self, context: Context) -> PlanNode:');
    L.push('        world = context.world  # bodies/poses below are resolved against it');
    surfaceResolveLines(useSteps, '        ').forEach(function (ln) { L.push(ln); });
    L.push('        return sequential([');
    useSteps.forEach(function (s) { L.push('            ' + stepCode(s) + ','); });
    L.push('        ], context=context).plan');
    L.push('');
    L.push('');
    L.push('def main() -> None:');
    L.push('    """Run the demonstration.');
    L.push('');
    L.push('    RobotDemonstration.run() acquires the world, starts the visualization backend,');
    L.push('    attaches the plan and performs it. The backend defaults to CRAMERA (the browser');
    L.push('    viewer); CORAPLEX_VISUALIZATION overrides it, so `cramera-live` works unchanged');
    L.push('    and you can also force RVIZ / NONE from the outside.');
    L.push('    """');
    L.push('    ' + cls + '(');
    L.push('        used_robot=' + R.cls + ',');
    L.push('        default_visualization_backend=VisualizationBackend.CRAMERA,');
    L.push('    ).run()');
    L.push('');
    L.push('');
    L.push('if __name__ == "__main__":');
    L.push('    main()');
    L.push('');
    return L.join('\n');
  }
  // pick the generator by the selected output style
  function outputStyle() { const s = $('pb-style'); return s ? s.value : 'script'; }
  function generateSelected() { return outputStyle() === 'class' ? generateClass() : generate(); }

  function showCode() { $('pb-code').textContent = generateSelected(); status('', ''); }
  function status(msg, cls) { const el = $('pb-status'); el.textContent = msg; el.className = 'pb-status ' + (cls || ''); }
  function fileName() { return (($('pb-name').value || 'my_demo').replace(/[^a-z0-9_\-]/gi, '_')) + '.py'; }

  function download() {
    const code = generateSelected();
    const blob = new Blob([code], { type: 'text/x-python' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = fileName(); a.click();
    URL.revokeObjectURL(a.href); status('downloaded ' + fileName(), 'ok');
  }
  function save() {
    const code = generateSelected();
    fetch('/api/plan/save', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ name: fileName(), code: code }) })
      .then(function (r) { return r.json(); })
      .then(function (j) { if (j.ok) status('saved → ' + j.path + '  (run: cramera-live ' + j.path + ')', 'ok'); else status('save failed: ' + (j.error || '?'), 'err'); })
      .catch(function (e) { status('save failed: ' + e, 'err'); });
  }

  // ---------- live 3D capture ----------
  function bridgeUrl() { return 'http://' + window.location.hostname + ':8765'; }
  function quatToYaw(q) { // q = [qx,qy,qz,qw] -> yaw
    return Math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]));
  }
  function liveStatus(msg, cls) { const el = $('pb-live-status'); el.textContent = msg; el.className = 'pb-live-status ' + (cls || ''); }
  function fetchCaptured() {
    return fetch(bridgeUrl() + '/captured_objects').then(function (r) { return r.json(); }).then(function (d) { return (d && d.objects) || {}; });
  }
  function poseFromCaptured(objs, mesh) {
    const p = objs[mesh]; if (!p || p.length < 7) return null;
    return { x: Math.round(p[0] * 100) / 100, y: Math.round(p[1] * 100) / 100, z: Math.round(p[2] * 100) / 100, yaw: Math.round(quatToYaw(p.slice(3)) * 1000) / 1000 };
  }
  function captureObject(oid) {
    const o = objects.find(function (x) { return x.id === oid; }); if (!o) return;
    fetchCaptured().then(function (objs) {
      const pz = poseFromCaptured(objs, o.mesh);
      if (!pz) { status('no live pose for ' + o.mesh + ' — is the scene running?', 'err'); return; }
      o.x = pz.x; o.y = pz.y; o.z = pz.z; o.yaw = pz.yaw; renderObjects();
      status('captured ' + o.name + ' → (' + pz.x + ', ' + pz.y + ', ' + pz.z + ')', 'ok');
    }).catch(function () { status('capture failed — start the live scene first', 'err'); });
  }
  function captureStepStart(sid) {
    // capture the transported object's live pose as ITS start pose (the "from")
    const s = steps.find(function (x) { return x.id === sid; }); if (!s) return;
    if (!s.params.object) { status('pick an object for this Transport step first', 'err'); return; }
    const o = objects.find(function (x) { return x.mesh === s.params.object; });
    if (!o) { status('object ' + s.params.object + ' is not in the objects list', 'err'); return; }
    fetchCaptured().then(function (objs) {
      const pz = poseFromCaptured(objs, o.mesh);
      if (!pz) { status('no live pose for ' + o.mesh + ' — is the scene running?', 'err'); return; }
      o.x = pz.x; o.y = pz.y; o.z = pz.z; o.yaw = pz.yaw;
      renderObjects();   // left panel
      renderSteps();     // the start (from) fields on the step read from the object
      status('captured start for ' + o.name + ' → (' + pz.x + ', ' + pz.y + ', ' + pz.z + ')', 'ok');
    }).catch(function () { status('capture failed — start the live scene first', 'err'); });
  }
  function captureStepTarget(sid) {
    const s = steps.find(function (x) { return x.id === sid; }); if (!s) return;
    if (!s.params.object) { status('pick an object for this Transport step first', 'err'); return; }
    fetchCaptured().then(function (objs) {
      const pz = poseFromCaptured(objs, s.params.object);
      if (!pz) { status('no live pose for ' + s.params.object, 'err'); return; }
      s.params.x = pz.x; s.params.y = pz.y; s.params.z = pz.z; s.params.yaw = pz.yaw; renderSteps();
      status('captured target for ' + s.params.object + ' → (' + pz.x + ', ' + pz.y + ', ' + pz.z + ')', 'ok');
    }).catch(function () { status('capture failed — start the live scene first', 'err'); });
  }
  function startLive() {
    const code = generate([{ type: 'park_arms', params: { arm: 'BOTH' } }]);   // scaffold: world + objects, idle
    liveStatus('starting… (first run parses meshes, ~1 min)', '');
    fetch('/api/plan/scaffold', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ code: code }) })
      .then(function (r) { return r.json(); })
      .then(function (j) { if (!j.ok) { liveStatus('failed: ' + (j.error || '?'), 'err'); return; } pollLive(0); })
      .catch(function (e) { liveStatus('failed: ' + e, 'err'); });
  }
  // run the built plan itself (not the idle scaffold) and watch the robot perform it:
  // the full generated demo ends in `plan.perform()`, launched through the same endpoint
  function runPlan() {
    if (!steps.length) { liveStatus('add plan steps first', 'err'); return; }
    const code = generate();   // full demo, real steps, ends with plan.perform()
    liveStatus('running plan… (first run parses meshes, ~1 min)', '');
    fetch('/api/plan/scaffold', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ code: code }) })
      .then(function (r) { return r.json(); })
      .then(function (j) { if (!j.ok) { liveStatus('failed: ' + (j.error || '?'), 'err'); return; } pollLive(0, '● running — watch the robot in the 3D view'); })
      .catch(function (e) { liveStatus('failed: ' + e, 'err'); });
  }
  function pollLive(n, okMsg) {
    fetch(bridgeUrl() + '/captured_objects').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (d) { liveOn = true; liveStatus(okMsg || '● live — drag objects in the 3D view, then capture', 'ok'); const f=$('pb-3d'); if (f && f.src.indexOf('index.html')<0) f.src='index.html?scene'; fetchSurfaces(); }
        else if (n < 40) { setTimeout(function () { pollLive(n + 1, okMsg); }, 3000); }
        else liveStatus('scene did not come up — check the terminal', 'err');
      })
      .catch(function () { if (n < 40) setTimeout(function () { pollLive(n + 1, okMsg); }, 3000); else liveStatus('scene did not come up', 'err'); });
  }
  // enumerate placement surfaces from the live world (for the "on a surface" target mode)
  function fetchSurfaces() {
    fetch(bridgeUrl() + '/surfaces').then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        const next = (d && d.surfaces) || [];
        const changed = JSON.stringify(next) !== JSON.stringify(liveSurfaces);
        liveSurfaces = next;
        if (changed && steps.some(function (s) { return s.type === 'transport' && s.params.targetMode === 'semantic'; })) renderSteps();
      }).catch(function () {});
  }
  // reload ONLY the embedded 3D view (it sometimes loads partially) without touching the
  // plan/objects/constraints on this page. Cache-busts so a stuck load is force-refreshed.
  function reloadScene() {
    const f = $('pb-3d'); if (!f) return;
    f.src = 'index.html?scene&r=' + Date.now();
    liveStatus('reloading 3D view…', '');
  }
  function stopLive() {
    liveOn = false; liveSurfaces = [];
    fetch('/api/plan/scaffold/stop', { method: 'POST' }).then(function () { liveStatus('stopped', ''); const f=$('pb-3d'); if (f) f.src='about:blank'; }).catch(function () {});
  }

  // ---------- boot ----------
  renderBlocks();
  renderConstraints();
  $('pb-add-obj').addEventListener('click', function () {
    const o = addObject($('pb-mesh').value);
    // objects are spawned when the scaffold is built, so one added mid-session needs a
    // (re)start to appear; it will show at its staging spot beside the robot, in the open.
    if (liveOn) status('added ' + o.name + ' — click “Start live scene” to (re)spawn it in the 3D view (staged beside the robot)', 'ok');
  });
  $('pb-con-add').addEventListener('click', function () { const inp = $('pb-con-in'); addConstraintText(inp.value); inp.value = ''; inp.focus(); });
  $('pb-con-in').addEventListener('keydown', function (e) { if (e.key === 'Enter') { e.preventDefault(); addConstraintText(this.value); this.value = ''; } });
  (function () {
    const box = $('pb-con-info-box'); if (box) box.innerHTML = conInfoHtml();
    const btn = $('pb-con-info'); if (btn && box) btn.addEventListener('click', function () { box.classList.toggle('open'); });
  })();
  $('pb-env').addEventListener('change', renderScene);
  $('pb-run').addEventListener('click', runPlan);
  $('pb-live-start').addEventListener('click', startLive);
  $('pb-live-stop').addEventListener('click', stopLive);
  $('pb-reset-all').addEventListener('click', resetAllObjects);
  $('pb-reload-3d').addEventListener('click', reloadScene);
  $('pb-rx').addEventListener('input', function () { robotXY.x = parseFloat(this.value) || 0; });
  $('pb-ry').addEventListener('input', function () { robotXY.y = parseFloat(this.value) || 0; });
  addObject('milk.stl', { x: 2.5, y: 2.3, z: 0.9 });
  addObject('bowl.stl', { x: 2.4, y: 2.0, z: 0.95 });
  renderSteps();
  // a friendly starter plan
  addStep('park_arms'); addStep('move_torso');
  $('pb-generate').addEventListener('click', showCode);
  function reshowIfGenerated() {
    const pre = $('pb-code'); if (pre && pre.textContent && pre.textContent.indexOf('Click') !== 0) showCode();
  }
  $('pb-style').addEventListener('change', reshowIfGenerated);
  $('pb-robot').addEventListener('change', reshowIfGenerated);
  $('pb-env').addEventListener('change', reshowIfGenerated);
  $('pb-download').addEventListener('click', download);
  $('pb-save').addEventListener('click', save);
  window.addEventListener('resize', renderScene);
})();
